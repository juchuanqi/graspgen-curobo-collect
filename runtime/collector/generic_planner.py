"""cuRobo + complete independent geometry checks for mesh collection.

Finger sphere/world terms are disabled during optimization, then all omitted
world pairs are restored by exact URDF-box/complete compound-target sweeps.
Self checks re-enable all native robot spheres and retain vendor ignore pairs.
Full held geometry is coupled to FK throughout the initial support departure;
cuRobo uses its full conservative attachment after this generic departure.
"""
from __future__ import annotations
import copy
import json
import time
from pathlib import Path
import sys

import numpy as np
import torch
from scipy.spatial.transform import Rotation
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.types import WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState
from curobo.util_file import get_robot_configs_path, join_path, load_yaml
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from geometry.api import check_robot_path
from geometry.robot import fk
from geometry.primitives import pose_matrix, part_matrix
from target_provider import screen_scene_candidates

FINGERS = ["panda_leftfinger", "panda_rightfinger"]
ATTACHMENT_SLOTS = 128
LINE_IK_SEEDS = 4


def pose_list(t):
    q = Rotation.from_matrix(t[:3, :3]).as_quat()[[3, 0, 1, 2]]
    return [*t[:3, 3].tolist(), *q.tolist()]


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def stretch(q):
    q = np.asarray(q, np.float32)
    points = [q[0]]
    for a, b in zip(q[:-1], q[1:]):
        n = max(1, int(np.ceil(np.max(np.abs(b - a)) / .0018)))
        points.extend(a + (b-a)*u for u in np.linspace(0, 1, n+1)[1:])
    return np.asarray(points, np.float32)


class PlanningBudgetExceeded(RuntimeError):
    """A completed non-preemptible stage crossed the candidate soft deadline."""
    def __init__(self, stage):
        super().__init__('candidate_planning_budget_exhausted')
        self.stage = stage


def check_deadline(deadline, stage):
    if deadline is not None and time.monotonic() >= deadline:
        raise PlanningBudgetExceeded(stage)


class Planner:
    def __init__(self, scene, output):
        self.device = TensorDeviceType()
        self.base = pose_matrix(scene['base_pose'])
        self.base_inverse = np.linalg.inv(self.base)
        self.output = Path(output); self.output.mkdir(parents=True, exist_ok=True)
        cfg = copy.deepcopy(load_yaml(join_path(get_robot_configs_path(), 'franka.yml'))['robot_cfg'])
        kin = cfg['kinematics']
        kin.update(urdf_path=scene['urdf_path'], asset_root_path=str(Path(scene['urdf_path']).parent),
                   ee_link='panda_hand_tcp', lock_joints={'panda_finger_joint1': .04},
                   mesh_link_names=[], collision_sphere_buffer=.002, extra_collision_spheres={'attached_object': ATTACHMENT_SLOTS})
        kin['extra_links']['attached_object']['parent_link_name'] = 'panda_hand_tcp'
        kin['link_names'] = [f'panda_link{i}' for i in range(1, 8)] + ['panda_hand', *FINGERS, 'panda_hand_tcp']
        self.config = copy.deepcopy(cfg)
        self.scene = scene
        self.world = self.build_world(scene)
        config = MotionGenConfig.load_from_robot_config(
            cfg, self.world, self.device, interpolation_dt=1/60, use_cuda_graph=False,
            collision_checker_type=CollisionCheckerType.PRIMITIVE, collision_cache={'obb': 32},
            num_ik_seeds=32, num_trajopt_seeds=4, position_threshold=.0002, rotation_threshold=.002,
            collision_activation_distance=.005, self_collision_check=True, self_collision_opt=True)
        self.motion_gen = MotionGen(config)
        self.joint_names = self.motion_gen.kinematics.joint_names
        if self.joint_names != scene['joint_names'][:7]:
            raise ValueError(f'Joint order mismatch {self.joint_names}')
        self.attachment = None
        self.calls = []
        self.initialized_at = time.monotonic()
        self.update(scene, output)

    def build_world(self, scene, remove_object=False):
        table = self.base_inverse @ pose_matrix(scene['table_pose']) @ pose_matrix(scene['table_shape_local_pose'])
        world = {'cuboid': {'table': {'dims': (2*np.asarray(scene['table_shape_half_size'])).tolist(), 'pose': pose_list(table)}}}
        for obstacle in scene.get('obstacles', []):
            world['cuboid'][obstacle['name']] = {'dims': obstacle['dims'], 'pose': pose_list(self.base_inverse@pose_matrix(obstacle['pose']))}
        if not remove_object:
            for i, part in enumerate(scene['object_parts']):
                dims = part['size'] if part['shape']=='box' else [2*part['radius'], 2*part['radius'], part['length']]
                world['cuboid'][f'target_{i:02d}'] = {'dims': dims, 'pose': pose_list(self.base_inverse @ pose_matrix(scene['object_pose']) @ part_matrix(part))}
        if len(world['cuboid']) > 32: raise ValueError('Complete primitive world exceeds 32 OBB cache entries')
        return world

    def update(self, scene, output, remove_object=False):
        if not np.allclose(pose_matrix(scene['base_pose']), self.base): raise ValueError('Robot base changed')
        if self.attachment is not None:
            self.motion_gen.toggle_link_collision(['attached_object'], True)
            self.motion_gen.detach_object_from_robot(); self.attachment = None
        self.scene = scene; self.output = Path(output); self.output.mkdir(parents=True, exist_ok=True)
        self.calls = []
        self.world = self.build_world(scene, remove_object)
        self.motion_gen.update_world(WorldConfig.from_dict(self.world))
        self.set_width(float(np.asarray(scene['qpos'])[7:].sum()))
        self.check_fk(scene['qpos'][:7], scene['tcp_pose'])
        save(self.output/'ROBOT_CONFIG.json', self.config)
        save(self.output/'WORLD.json', self.world)
        save(self.output/'COLLISION_POLICY.json', {
            'world_target_parts': 0 if remove_object else len(scene['object_parts']),
            'complete_target_parts': len(scene['object_parts']),
            'primitive_world_cylinders': 'conservative axis-aligned local enclosing cuboids; exact independent checker retains cylinder64 circumscription',
            'optimizer_disabled_world_links': FINGERS,
            'restored_world_pairs': 'all complete fingers/arm/palm vs entire target/table; full held target vs table',
            'self_collision': 'cuRobo native spheres, vendor self_collision_ignore retained; all native robot spheres re-enabled for full-path validity',
            'physics_self_collision': False,
            'physics_self_collision_limitation': 'PhysX native Panda self collision remains disabled as inherited; cuRobo self collision constrains all commanded paths, not dynamic deviations'})

    def set_width(self, opening):
        if hasattr(self, 'width') and abs(self.width-float(opening)) < 1e-8:
            self.motion_gen.toggle_link_collision(FINGERS, False)
            self.motion_gen.rollout_fn.robot_self_collision_constraint.enable_cost()
            return
        self.width = float(opening)
        self.motion_gen.toggle_link_collision(FINGERS, True)
        self.motion_gen.update_locked_joints({'panda_finger_joint1': self.width/2}, copy.deepcopy(self.config))
        self.motion_gen.toggle_link_collision(FINGERS, False)
        self.motion_gen.rollout_fn.robot_self_collision_constraint.enable_cost()

    def check_fk(self, q, expected_pose):
        state = self.motion_gen.kinematics.get_state(torch.tensor(np.asarray(q)[None], dtype=torch.float32, device='cuda'))
        actual = self.base @ pose_matrix(np.r_[state.ee_position[0].cpu().numpy(), state.ee_quaternion[0].cpu().numpy()])
        expected = pose_matrix(expected_pose)
        p = float(np.linalg.norm(actual[:3,3]-expected[:3,3]))
        a = float(Rotation.from_matrix(actual[:3,:3].T@expected[:3,:3]).magnitude())
        cpu = fk(np.asarray(q), self.width, self.base)['panda_hand_tcp']
        cpu_p = float(np.linalg.norm(cpu[:3,3]-expected[:3,3]))
        cpu_a = float(Rotation.from_matrix(cpu[:3,:3].T@expected[:3,:3]).magnitude())
        report = {'curobo_physx_position_m': p, 'curobo_physx_rotation_rad': a,
                  'geometry_physx_position_m': cpu_p, 'geometry_physx_rotation_rad': cpu_a,
                  'passed': p<=.0002 and a<=.002 and cpu_p<=.0002 and cpu_a<=.002}
        save(self.output/'FK.json', report)
        if not report['passed']: raise ValueError(f'FK qualification failed {report}')
        return report

    def target_pose(self, world_transform):
        return Pose.from_list(pose_list(self.base_inverse @ world_transform), self.device)

    def batch_ik(self, transforms, seed, return_seeds=1, num_seeds=32):
        positions, quaternions = [], []
        for t in transforms:
            p = pose_list(self.base_inverse @ t); positions.append(p[:3]); quaternions.append(p[3:])
        goals = Pose(position=torch.tensor(positions, device='cuda', dtype=torch.float32),
                     quaternion=torch.tensor(quaternions, device='cuda', dtype=torch.float32))
        q = torch.tensor(np.repeat(np.asarray(seed)[None], len(positions), axis=0), dtype=torch.float32, device='cuda')
        result = self.motion_gen.ik_solver.solve_batch(goals, retract_config=q, seed_config=q[:,None,:],
                                                       return_seeds=return_seeds, num_seeds=num_seeds)
        torch.cuda.synchronize()
        return result.success.cpu().numpy().reshape(len(transforms), -1), result.solution.cpu().numpy().reshape(len(transforms), return_seeds, 7)

    def self_valid(self, path, opening, independent_fingers=None):
        if abs(self.width-float(opening)) > 1e-8:
            self.set_width(opening)
        self.motion_gen.toggle_link_collision(FINGERS, True)
        self.motion_gen.rollout_fn.robot_self_collision_constraint.enable_cost()
        try:
            qt = torch.tensor(path, dtype=torch.float32, device='cuda')
            state = self.motion_gen.kinematics.get_state(qt)
            spheres = state.link_spheres_tensor.clone().unsqueeze(0)
            asymmetry = np.zeros(2) if independent_fingers is None else np.abs(np.asarray(independent_fingers)-opening/2)
            for finger,error in zip(FINGERS,asymmetry):
                if error:
                    indices=self.motion_gen.kinematics.kinematics_config.get_sphere_index_from_link_name(finger)
                    spheres[...,indices,3]+=float(error)
            value = self.motion_gen.rollout_fn.robot_self_collision_constraint.forward(spheres)
            bad = (value>0).reshape(-1).cpu().numpy()
            report = {'valid': not bool(bad.any()), 'samples': len(path), 'invalid_indices': np.flatnonzero(bad).tolist(),
                      'total_opening_m': float(opening), 'all_native_finger_spheres_enabled': True,
                      'independent_finger_radius_inflation_m':asymmetry.tolist(),
                      'finger_representation':'vendor spheres at symmetric mean, radii inflated by actual independent finger displacement bound',
                      'configuration': self.config['kinematics']['self_collision_ignore']}
        finally:
            self.motion_gen.toggle_link_collision(FINGERS, False)
        return report

    def validate(self, path, opening, label, allow_pad=False, attached_to_tcp=None, deadline=None):
        check_deadline(deadline, label + ":before_geometry")
        kwargs = {} if attached_to_tcp is None else {'attached_to_tcp': attached_to_tcp}
        if attached_to_tcp is not None:
            kwargs['pad_penetration_m']=.001  # measured grasp: existing physical contact tolerance
        independent_fingers=np.asarray(self.scene['qpos'][7:9]) if attached_to_tcp is not None else None
        exact_path=path if independent_fingers is None else np.c_[path,np.repeat(independent_fingers[None],len(path),axis=0)]
        exact = check_robot_path(exact_path, opening, pose_matrix(self.scene['object_pose']), self.scene['object_id'],
                                 base_pose=self.base, allow_pad_contact=allow_pad, **kwargs)
        check_deadline(deadline, label + ":after_geometry")
        own = self.self_valid(path, opening, independent_fingers)
        check_deadline(deadline, label + ":after_self_collision")
        report = {'valid': exact['valid'] and own['valid'], 'geometry': exact, 'self': own,
                  'contact_mode':'measured_grasp_1mm' if attached_to_tcp is not None else 'planned_approach_0.5mm'}
        save(self.output/f'CHECK_{label}.json', report)
        return report

    def motion(self, start, goal, timeout, deadline=None):
        check_deadline(deadline, "motion:before_plan")
        if timeout<=0: return None, {'reason':'candidate_planning_budget_exhausted'}
        t=time.monotonic()
        self.motion_gen.rollout_fn.robot_self_collision_constraint.enable_cost()
        result=self.motion_gen.plan_single(JointState.from_position(torch.tensor([start],device='cuda',dtype=torch.float32),joint_names=self.joint_names),
                    self.target_pose(goal), MotionGenPlanConfig(max_attempts=2,timeout=float(timeout),enable_graph_attempt=1))
        torch.cuda.synchronize()
        report={'success': bool(result.success.item()),'status':str(result.status),'wall_s':time.monotonic()-t}
        if deadline is not None and time.monotonic() >= deadline:
            report.update(success=False, solver_success=bool(result.success.item()),
                          reason='candidate_planning_budget_exhausted',
                          softdeadline_overrun_s=max(0.,time.monotonic()-deadline))
            return None,report
        path=stretch(result.get_interpolated_plan().position.cpu().numpy()) if report['success'] else None
        check_deadline(deadline, 'motion:after_interpolation')
        return path,report

    def line(self, start_q, start_tcp, end_tcp, deadline=None):
        check_deadline(deadline, "line:before_setup")
        t=time.monotonic()
        if Rotation.from_matrix(start_tcp[:3,:3].T@end_tcp[:3,:3]).magnitude()>.002:
            return None, {'reason':'linear_stage_orientation_changed'}
        distance=np.linalg.norm(end_tcp[:3,3]-start_tcp[:3,3])
        count=max(2,int(np.ceil(distance/.005))+1)
        targets=np.repeat(start_tcp[None],count,axis=0)
        targets[:,:3,3]=np.linspace(start_tcp[:3,3],end_tcp[:3,3],count)
        check_deadline(deadline, "line:before_batch_ik")
        valid,solutions=self.batch_ik(targets[1:],start_q,return_seeds=LINE_IK_SEEDS,num_seeds=LINE_IK_SEEDS)
        check_deadline(deadline, "line:after_batch_ik")
        q=[np.asarray(start_q)]
        for ok, candidates in zip(valid,solutions):
            eligible=np.flatnonzero(ok)
            if not len(eligible):return None, {'reason':'cartesian_waypoint_ik_failure','wall_s':time.monotonic()-t}
            i=eligible[np.argmin(np.linalg.norm(candidates[eligible]-q[-1],axis=1))]
            if np.max(np.abs(candidates[i]-q[-1]))>.2:return None, {'reason':'cartesian_ik_branch_jump','wall_s':time.monotonic()-t}
            q.append(candidates[i])
        check_deadline(deadline, "line:before_fk")
        path=stretch(q)
        state=self.motion_gen.kinematics.get_state(torch.tensor(path,device='cuda',dtype=torch.float32))
        p=state.ee_position.cpu().numpy()@self.base[:3,:3].T+self.base[:3,3]
        delta=end_tcp[:3,3]-start_tcp[:3,3]
        projection=np.clip((p-start_tcp[:3,3])@delta/(distance**2),0,1)
        error=np.linalg.norm(p-(start_tcp[:3,3]+projection[:,None]*delta),axis=1)
        progress=(p-start_tcp[:3,3])@delta/distance
        rotations=Rotation.from_quat(state.ee_quaternion.cpu().numpy()[:,[1,2,3,0]])
        goal=Rotation.from_matrix((self.base_inverse@end_tcp)[:3,:3])
        angle=(goal.inv()*rotations).magnitude()
        report={'wall_s':time.monotonic()-t,'waypoints':count,'steps':len(path),'max_line_deviation_m':float(error.max()),
                'max_orientation_error_rad':float(angle.max()),'min_progress_increment_m':float(np.diff(progress).min()),
                'valid':bool(error.max()<=.001 and angle.max()<=.003 and np.diff(progress).min()>=-.0002)}
        check_deadline(deadline, "line:after_fk_and_checks")
        if not report['valid']:return None,report
        return path,report

    def select(self, candidates, budget_s=30, maximum=5):
        scene_valid,report=screen_scene_candidates(candidates,self.scene,maximum=50)
        report.update(ik_queries=0,full_candidates=[])
        if not scene_valid:
            save(self.output/'CANDIDATE_REPORT.json',report);return {'success':False,'reason':'no_scene_geometry_candidate','candidate_report':report}
        # A single fixed-width batch IK screen, with finger spheres omitted and
        # complete exact geometry checked at the selected width afterward.
        self.set_width(.08)
        t=time.monotonic()
        ik_ok,ik_solutions=self.batch_ik([np.asarray(c['pregrasp_world']) for c in scene_valid], self.scene['qpos'][:7])
        report['ik_queries']=len(scene_valid);report['ik_wall_s']=time.monotonic()-t
        report['ik_passed']=int(ik_ok[:,0].sum())
        report['scene_ik_results']=[{'original_rank':int(c['original_rank']),'ik_passed':bool(ok)} for c,ok in zip(scene_valid,ik_ok[:,0])]
        # Rank physically equivalent representations by robot motion demand.
        # Parent network scores describe parent poses only; no new confidence.
        initial=np.asarray(self.scene['qpos'][:7])
        costs=[]
        for index,candidate in enumerate(scene_valid):
            delta=np.abs(ik_solutions[index,0]-initial)
            bound=float(delta.max()/.0018) if ik_ok[index,0] else float('inf')
            total=float(delta.sum()) if ik_ok[index,0] else float('inf')
            costs.append((bound,total,index))
            candidate['pregrasp_joint_motion_estimate']={'max_joint_delta_rad':float(delta.max()),
                'sum_joint_delta_rad':float(delta.sum()),'control_steps_lower_bound':None if not ik_ok[index,0] else bound}
        order=[index for _,_,index in sorted(costs)]
        report['ordering_policy']='reachable pregrasp IK maximum joint travel / 0.0018, then total joint travel; fixed parent score not treated as refined confidence'
        report['candidate_planning_order']=[{'derived_rank':scene_valid[i]['original_rank'],
            'parent_original_rank':scene_valid[i].get('parent_original_rank'),
            'symmetry_variant':scene_valid[i].get('symmetry_variant'),
            'motion_estimate':scene_valid[i]['pregrasp_joint_motion_estimate']} for i in order]
        scene_valid=[scene_valid[i] for i in order];ik_ok=ik_ok[order]
        for candidate,ok in zip(scene_valid,ik_ok[:,0]):
            if not ok:
                report.setdefault('rejections',[]).append({'original_rank':candidate['original_rank'],'rejected':'pregrasp_ik'});continue
            if len(report['full_candidates'])>=maximum:break
            started=time.monotonic();deadline=started+budget_s
            attempt={'original_rank':candidate['original_rank'],'opening_m':candidate['opening_m'],
                     'planning_budget_s':float(budget_s),'budget_type':'soft deadline; non-preemptible stages may overrun and are rejected'}
            report['full_candidates'].append(attempt)
            try:
                check_deadline(deadline, 'candidate:before_width_update')
                self.set_width(candidate['opening_m'])
                check_deadline(deadline, 'candidate:after_width_update')
                pre=np.asarray(candidate['pregrasp_world']);grasp=np.asarray(candidate['grasp_world'])
                approach,motion_report=self.motion(self.scene['qpos'][:7],pre,min(12.,deadline-time.monotonic()),deadline=deadline)
                attempt['approach_plan']=motion_report
                if approach is None:
                    attempt['rejected']=motion_report.get('reason','pregrasp_planning');continue
                check=self.validate(approach,candidate['opening_m'],'approach_'+str(candidate['original_rank']),deadline=deadline)
                attempt['approach_check']=check
                if not check['valid']:attempt['rejected']='pregrasp_full_geometry_or_self_collision';continue
                descend,line_report=self.line(approach[-1],pre,grasp,deadline=deadline)
                attempt['cartesian_approach']=line_report
                if descend is None:attempt['rejected']='straight_approach_ik';continue
                check=self.validate(descend,candidate['opening_m'],'descend_'+str(candidate['original_rank']),deadline=deadline)
                attempt['descend_check']=check
                if not check['valid']:attempt['rejected']='descend_full_geometry_or_self_collision';continue
                prefix_steps=30+len(approach)+20+len(descend)+20+135
                attempt['minimum_control_steps_with_hold']=prefix_steps+120
                if prefix_steps+120>2400:
                    attempt['rejected']='planned_prefix_leaves_insufficient_hold_steps';continue
                check_deadline(deadline, 'candidate:before_path_save')
                path_file=self.output/'SELECTED_PATHS.npz'
                np.savez_compressed(path_file,approach=approach,descend=descend)
                check_deadline(deadline, 'candidate:after_path_save')
            except PlanningBudgetExceeded as error:
                attempt.update(rejected='candidate_planning_budget_exhausted',budget_exhausted_stage=error.stage)
                continue
            finally:
                finished=time.monotonic()
                attempt['planning_wall_s']=finished-started
                attempt['softdeadline_overrun_s']=max(0.,finished-deadline)
                if finished>=deadline:
                    attempt['rejected']='candidate_planning_budget_exhausted'
                save(self.output/'CANDIDATE_REPORT.json',report)
            if attempt.get('rejected'):continue
            candidate['planning_wall_s']=attempt['planning_wall_s']
            candidate['remaining_budget_s']=max(0.,deadline-time.monotonic())
            candidate['path_file']=str(path_file)
            if candidate['remaining_budget_s']<=0:
                attempt.update(rejected='candidate_planning_budget_exhausted',
                               softdeadline_overrun_s=max(0.,time.monotonic()-deadline))
                continue
            save(self.output/'SELECTED.json',candidate);save(self.output/'CANDIDATE_REPORT.json',report)
            return {'success':True,'selected':candidate,'candidate_report':report}
        save(self.output/'CANDIDATE_REPORT.json',report)
        return {'success':False,'reason':'no_complete_candidate_within_budget','candidate_report':report}

    def attach_full(self,tcp_pose,object_pose):
        cells=[];counts=[];cell_size=.008
        geometry=[]
        for part in self.scene['object_parts']:
            dims=np.asarray(part['size'] if part['shape']=='box' else [2*part['radius'],2*part['radius'],part['length']])
            geometry.append((part_matrix(part),dims))
        while sum(int(np.prod(np.maximum(1,np.ceil(d/cell_size).astype(int)))) for _,d in geometry)>ATTACHMENT_SLOTS:cell_size*=1.1
        for t,dims in geometry:
            count=np.maximum(1,np.ceil(dims/cell_size).astype(int))
            axes=[np.linspace(-d/2+d/(2*n),d/2-d/(2*n),n) for d,n in zip(dims,count)]
            centers=np.stack(np.meshgrid(*axes,indexing='ij'),axis=-1).reshape(-1,3)
            centers=centers@t[:3,:3].T+t[:3,3]
            cells.append(np.c_[centers,np.full(len(centers),np.linalg.norm(dims/count)/2)])
            counts.append(len(centers))
        spheres=np.concatenate(cells);relative=np.linalg.inv(tcp_pose)@object_pose
        spheres[:,:3]=spheres[:,:3]@relative[:3,:3].T+relative[:3,3]
        padded=np.zeros((ATTACHMENT_SLOTS,4),np.float32);padded[:,3]=-10;padded[:len(spheres)]=spheres
        self.motion_gen.attach_spheres_to_robot(sphere_tensor=torch.tensor(padded,device='cuda'),link_name='attached_object')
        worldcenters=spheres[:,:3]@tcp_pose[:3,:3].T+tcp_pose[:3,3]
        sphere_bottom=(worldcenters[:,2]-spheres[:,3]).min()
        table_top=float(self.scene['table_pose'][2])+float(self.scene['table_shape_half_size'][2])
        clearance=max(.008,table_top-sphere_bottom+.002)
        self.attachment={'conservative_full_primitive_cover':True,'part_count':len(geometry),'sphere_count':len(spheres),
            'per_part_sphere_count':counts,'spheres_tcp':spheres.tolist(),'tcp_from_object':relative.tolist(),
            'support_clipped_sphere_count':0,'sphere_support_initial_liftoff_m':float(clearance),
            'initial_liftoff_collision': 'full target geometry coupled to TCP and checked by independent exact sweep; full attachment spheres are disabled only for this support departure, then restored'}
        save(self.output/'ATTACHMENT.json',self.attachment)
        return relative,float(clearance)

    def lift(self, remaining_s):
        started=time.monotonic();deadline=started+float(remaining_s)
        report={'planning_budget_s':float(remaining_s),
                'budget_type':'soft deadline; non-preemptible stages may overrun and are rejected'}
        reply=None
        try:
            check_deadline(deadline, 'lift:before_attachment')
            tcp=pose_matrix(self.scene['tcp_pose']);obj=pose_matrix(self.scene['object_pose'])
            relative,clearance=self.attach_full(tcp,obj)
            check_deadline(deadline, 'lift:after_attachment')
            if clearance>.03:
                reply={'success':False,'reason':'attachment_support_clearance_above_frozen_limit','clearance_m':clearance}
                return reply
            self.motion_gen.toggle_link_collision(['attached_object'],False)
            first=tcp.copy();first[2,3]+=clearance
            liftoff,line=self.line(self.scene['qpos'][:7],tcp,first,deadline=deadline)
            report['support_departure']=line
            if liftoff is None:
                reply={'success':False,'reason':'support_departure_ik','report':report};return reply
            check=self.validate(liftoff,self.width,'liftoff',allow_pad=True,attached_to_tcp=relative,deadline=deadline)
            report['support_departure_check']=check
            self.motion_gen.toggle_link_collision(['attached_object'],True)
            if not check['valid']:
                reply={'success':False,'reason':'support_departure_geometry','report':report};return reply
            goal=tcp.copy();goal[2,3]+=.12
            rest,motion=self.motion(liftoff[-1],goal,deadline-time.monotonic(),deadline=deadline)
            report['lift_motion']=motion
            if rest is None:
                reply={'success':False,'reason':motion.get('reason','lift_planning'),'report':report};return reply
            path=np.concatenate([liftoff,rest[1:]])
            check=self.validate(path,self.width,'lift',allow_pad=True,attached_to_tcp=relative,deadline=deadline)
            report['lift_check']=check
            if not check['valid']:
                reply={'success':False,'reason':'lift_complete_geometry','report':report};return reply
            check_deadline(deadline, 'lift:before_path_save')
            file=self.output/'LIFT_PATH.npz';np.savez_compressed(file,execution_q=path)
            check_deadline(deadline, 'lift:after_path_save')
            reply={'success':True,'path_file':str(file),'report':report}
            return reply
        except PlanningBudgetExceeded as error:
            report['budget_exhausted_stage']=error.stage
            reply={'success':False,'reason':'candidate_planning_budget_exhausted','report':report}
            return reply
        finally:
            finished=time.monotonic()
            report['wall_s']=finished-started
            report['softdeadline_overrun_s']=max(0.,finished-deadline)
            if reply is not None:
                reply['planning_wall_s']=report['wall_s']
                reply['softdeadline_overrun_s']=report['softdeadline_overrun_s']
                if finished>=deadline:
                    reply.update(success=False,reason='candidate_planning_budget_exhausted')
            save(self.output/'LIFT_REPORT.json',report)
