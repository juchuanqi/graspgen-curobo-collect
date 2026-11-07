"""An actual Isaac Lab DirectRLEnv for Panda pickup and raw dual-camera capture."""

import json
from pathlib import Path

import isaaclab.sim as sim_utils
import numpy as np
import torch
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import Camera, CameraCfg, ContactSensor, ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import combine_frame_transforms
from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade
from scipy.spatial.transform import Rotation

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from occlusion_geometry import design_occluder
from isaaclab.utils.math import quat_apply_inverse

ROOT = Path(__file__).resolve().parents[1]
LOCAL_ROOT = Path(__file__).resolve().parent
TRANSFER = json.loads((ROOT / "config/scene.json").read_text())
URDF = Path(__file__).resolve().parents[1] / "robot/panda_v2.urdf"

NAMES = [f"panda_joint{i}" for i in range(1, 8)] + [
    "panda_finger_joint1",
    "panda_finger_joint2",
]
INITIAL = [0, np.pi / 8, 0, -5 * np.pi / 8, 0, 3 * np.pi / 4, np.pi / 4, 0.04, 0.04]
LINKS = [f"panda_link{i}" for i in range(1, 8)] + [
    "panda_hand",
    "panda_leftfinger",
    "panda_rightfinger",
]


def raw_contact_centroids(points_w, valid):
    """Unweighted endpoint means, matching ContactSensor's object filter.

    The raw buffers already contain only the object contacts for each finger.
    Preserve NaN for a finger with no contacts, as ContactSensor does.
    """
    counts = valid.sum(dim=-1, keepdim=True)
    totals = torch.where(valid[..., None], points_w, 0.0).sum(dim=-2)
    means = totals / counts.clamp_min(1)
    return torch.where(counts > 0, means, torch.full_like(means, float("nan")))


@configclass
class PickupCfg(DirectRLEnvCfg):
    decimation = 8
    episode_length_s = 100.0
    action_space = 9
    observation_space = 18
    state_space = 0
    sim = sim_utils.SimulationCfg(
        dt=1 / 480,
        render_interval=8,
        device="cuda:0",
        render=sim_utils.RenderCfg(antialiasing_mode="Off", enable_dlssg=False),
        physx=sim_utils.PhysxCfg(solver_type=0),
    )
    scene = InteractiveSceneCfg(
        num_envs=1, env_spacing=3.0, replicate_physics=True, lazy_sensor_update=False
    )
    robot_cfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(ROOT / "robot/usd/panda.usd"),
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True, max_depenetration_velocity=1.0
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=4,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(joint_pos=dict(zip(NAMES, INITIAL))),
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[1-7]"], stiffness=10000.0, damping=200.0
            ),
            "fingers": ImplicitActuatorCfg(
                joint_names_expr=["panda_finger_joint.*"],
                stiffness=400.0,
                damping=20.0,
            ),
        },
    )


def camera_config(path, eye, look, up):
    transform = np.asarray(
        Gf.Matrix4d()
        .SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*look), Gf.Vec3d(*up))
        .GetInverse()
    ).T
    q = Rotation.from_matrix(transform[:3, :3]).as_quat()[[3, 0, 1, 2]]
    return CameraCfg(
        prim_path=path,
        # Fetch explicitly once after each rendered control step in snapshot().
        # Float32 periodic scheduling can miss a frame after long preflight.
        update_period=1e9,
        height=256,
        width=256,
        data_types=["rgb"],
        update_latest_camera_pose=False,
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=18.0,
            horizontal_aperture=20.955,
            vertical_aperture=20.955,
            clipping_range=(0.01, 10.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=tuple(transform[:3, 3]), rot=tuple(q), convention="opengl"
        ),
    )


class PickupEnv(DirectRLEnv):
    contact_sensor_type = ContactSensor

    def __init__(self, cfg, object_id, asset, scenario="clear", occlusion="none", **kwargs):
        self.asset = asset
        self.bounds_min = np.asarray(asset["bounds_min"], dtype=float)
        self.bounds_max = np.asarray(asset["bounds_max"], dtype=float)
        self._pad_force = self._nonpad_peak = self._max_penetration = None
        self._physics_apply_calls = 0
        if occlusion not in ("none", "base_partial"):
            raise ValueError(f"Unsupported visual occlusion: {occlusion}")
        self.occlusion = occlusion
        self.visual_occlusion_spec = {"mode": occlusion, "configured": False}
        self._visual_occluder_path = "/World/envs/env_0/VisualOccluder"
        self.scenario = scenario
        self.object_id = object_id
        self.size = (self.bounds_max-self.bounds_min).tolist()
        self.case_xy = [0.0, 0.0]
        self.case_yaw = 0.0
        self.reset_seed = 2026110360
        super().__init__(cfg, **kwargs)
        self.joint_ids, names = self.robot.find_joints(NAMES, preserve_order=True)
        assert names == NAMES
        self.hand_id = self.robot.find_bodies("panda_hand")[0][0]
        self.offset = torch.tensor([[0.0, 0.0, 0.1034]], device=self.device)
        materials={}
        for name,asset in (("robot",self.robot),("object",self.object)):
            values=asset.root_physx_view.get_material_properties()
            materials[name]=values.cpu().numpy().tolist()
        mu=max(float(np.max(np.asarray(values)[...,:2])) for values in materials.values())
        if not np.isfinite(mu) or mu<0:raise RuntimeError("Cannot bound nonpad friction with invalid native material")
        # All PhysX friction combine choices (min/max/average/product) are
        # bounded by this value. The normal sums include every nonpad point.
        mu_bound=1. if mu<=1. else max(mu,mu*mu)
        self.nonpad_force_multiplier=float(np.sqrt(1+mu_bound*mu_bound))
        self.contact_force_bound={"native_materials":materials,"largest_native_mu":mu,
            "friction_combine_upper_bound":mu_bound,"normal_to_total_force_multiplier":self.nonpad_force_multiplier,
            "method":"Coulomb triangle upper bound: sum(nonpad normal force)*sqrt(1+mu_bound^2)"}
        import omni.physx
        import omni.usd
        from pxr import PhysicsSchemaTools, Sdf
        self._sleep_interface=omni.physx.get_physx_simulation_interface()
        self._sleep_stage_id=omni.usd.get_context().get_stage_id()
        self._sleep_body_path=PhysicsSchemaTools.sdfPathToInt(Sdf.Path('/World/envs/env_0/Object'))

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        table = sim_utils.CuboidCfg(
            size=(1.2, 1.2, 0.05),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True, disable_gravity=True
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.8, dynamic_friction=0.7
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.3, 0.34, 0.4)),
        )
        self.table = RigidObject(
            RigidObjectCfg(
                prim_path="/World/envs/env_.*/Table",
                spawn=table,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.495, 0.0, -0.025)),
            )
        )
        object_spawn = sim_utils.UsdFileCfg(
            usd_path=str(self.asset["usd_path"]), activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=8, solver_velocity_iteration_count=4,
                max_depenetration_velocity=1.0))
        self.object = RigidObject(RigidObjectCfg(
            prim_path="/World/envs/env_.*/Object", spawn=object_spawn,
            init_state=RigidObjectCfg.InitialStateCfg(pos=(.615,0.,-float(self.bounds_min[2])+.002))))
        self.wall = RigidObject(RigidObjectCfg(
            prim_path="/World/envs/env_.*/Wall",
            spawn=sim_utils.CuboidCfg(
                size=(0.02, 0.25, 0.18),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.8, dynamic_friction=0.7),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.65,0.65,0.65))),
            init_state=RigidObjectCfg.InitialStateCfg(pos=((0.47,0.,0.09) if self.scenario=="wall" else (2.,0.,0.09)))))
        self.scene.rigid_objects["wall"] = self.wall
        self.scene.clone_environments(copy_from_source=False)
        if self.occlusion != "none":
            # A unit cube under an Xform; configure_visual_occlusion supplies
            # its final transform after the object settles. No physics APIs.
            panel = sim_utils.CuboidCfg(
                size=(1.0, 1.0, 1.0),
                rigid_props=None,
                collision_props=None,
                mass_props=None,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.18, 0.18, 0.18), roughness=1.0
                ),
            )
            panel.func(
                self._visual_occluder_path, panel, translation=(0.0, 0.0, -10.0)
            )
        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["object"] = self.object
        self.scene.rigid_objects["table"] = self.table
        self.cameras = {}
        for name in ("base_camera", "wrist_camera"):
            calibration = TRANSFER["cameras"][name]
            cfg = CameraCfg(
                prim_path=("/World/envs/env_.*/BaseCamera" if name == "base_camera"
                           else "/World/envs/env_.*/Robot/panda_hand/WristCamera"),
                update_period=1e9, height=256, width=256, data_types=["rgb"],
                update_latest_camera_pose=False,
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=float(calibration["intrinsic_cv"][0][0]) * 20.955 / 256,
                    horizontal_aperture=20.955, vertical_aperture=20.955,
                    clipping_range=(0.01, 10.0)),
                offset=CameraCfg.OffsetCfg(pos=tuple(calibration["pos"]),
                                           rot=tuple(calibration["rot"]), convention="opengl"))
            self.cameras[name] = Camera(cfg)
            self.scene.sensors[name] = self.cameras[name]
        self.contacts = {}
        for link in LINKS:
            sensor = self.contact_sensor_type(
                ContactSensorCfg(
                    prim_path=f"/World/envs/env_.*/Robot/{link}",
                    update_period=0.0,
                    history_length=8,
                    # snapshot reuses the already captured raw finger points.
                    track_contact_points=False,
                    filter_prim_paths_expr=[
                        "/World/envs/env_.*/Object",
                        "/World/envs/env_.*/Table",
                        "/World/envs/env_.*/Wall",
                    ],
                    max_contact_data_count_per_prim=128,
                )
            )
            self.contacts[link] = sensor
            self.scene.sensors[link] = sensor
        self.object_table = self.contact_sensor_type(
            ContactSensorCfg(
                prim_path="/World/envs/env_.*/Object",
                update_period=0.0,
                history_length=8,
                filter_prim_paths_expr=["/World/envs/env_.*/Table"],
                max_contact_data_count_per_prim=128,
            )
        )
        self.scene.sensors["object_table"] = self.object_table
        self.object_wall = self.contact_sensor_type(ContactSensorCfg(
            prim_path="/World/envs/env_.*/Object", update_period=0., history_length=8,
            filter_prim_paths_expr=["/World/envs/env_.*/Wall"], max_contact_data_count_per_prim=128))
        self.scene.sensors["object_wall"] = self.object_wall
        light = sim_utils.DomeLightCfg(intensity=1200.0)
        light.func("/World/Light", light)

    def _pre_physics_step(self, actions):
        self.targets = actions.clone()
        self._begin_contact_window()

    def _begin_contact_window(self):
        self._nonpad_peak = torch.zeros(2,device=self.device)
        self._max_penetration = torch.zeros((),device=self.device)
        self._physics_apply_calls = 0
        self._contact_sample_count = 0
        self._last_contact_tick = None
        self._contact_window = {
            "points_w": torch.zeros((8,2,128,3),device=self.device),
            "normals_w": torch.zeros((8,2,128,3),device=self.device),
            "normal_force_n": torch.zeros((8,2,128),device=self.device),
            "separation_m": torch.zeros((8,2,128),device=self.device),
            "valid": torch.zeros((8,2,128),device=self.device,dtype=torch.bool),
            "pad_mask": torch.zeros((8,2,128),device=self.device,dtype=torch.bool),
            "pad_force_n": torch.zeros((8,2),device=self.device),
            "finger_pose_w": torch.zeros((8,2,7),device=self.device),
            "physics_tick": torch.full((8,),-1,device=self.device,dtype=torch.int64),
            "object_pose_w": torch.zeros((8,7),device=self.device),
            "object_velocity_w": torch.zeros((8,6),device=self.device),
            "cached_object_pose_w": torch.zeros((8,7),device=self.device),
            "hand_pose_w": torch.zeros((8,7),device=self.device),
            "tcp_pose_w": torch.zeros((8,7),device=self.device),
        }

    def _capture_pad_contacts(self, physics_tick=None):
        # PhysX returns individual contacts; classify every point, not a link
        # centroid. Before apply calls 2..8 the preceding substep is current;
        # snapshot captures substep 8. Apply call 1 must skip the old interval.
        if not hasattr(self,"_contact_window"):
            self._begin_contact_window()
        tick = self._sim_step_counter if physics_tick is None else physics_tick
        if tick == self._last_contact_tick:
            return
        sample = self._contact_sample_count
        if sample >= 8:
            raise RuntimeError("More than eight distinct physics samples in one control interval")
        pad_force, nonpad, point_sets, force_sets, masks = [], [], [], [], []
        for finger_index,link in enumerate(("panda_leftfinger", "panda_rightfinger")):
            sensor=self.contacts[link]
            normal,points,normals,separations,counts,starts=sensor.contact_physx_view.get_contact_data(dt=self.cfg.sim.dt)
            if tuple(counts.shape) != (1,3) or tuple(starts.shape) != (1,3):
                raise RuntimeError("Expected one finger body and ordered object/table/wall filters")
            if int(counts.sum().item()) >= len(normal):
                raise RuntimeError("Raw contact buffer saturated; refusing incomplete classification")
            count=int(counts.reshape(-1)[0].item())
            start=int(starts.reshape(-1)[0].item())
            if not 0 <= count <= 128 or start+count > len(points):
                raise RuntimeError("Invalid raw contact buffer range")
            pts=points[start:start+count]
            force=normal[start:start+count].reshape(-1).abs()
            body=self.robot.data.body_link_state_w[0,self.robot.find_bodies(link)[0][0],:7]
            local=quat_apply_inverse(body[3:7].expand(count,4),pts-body[:3]) if count else pts
            inner_y = (-.00002 if finger_index == 0 else .00002)
            mask=(torch.abs(local[:,0])<=.00925)&(torch.abs(local[:,1]-inner_y)<=.001)&(local[:,2]>=.0355)&(local[:,2]<=.055)
            pad_force.append(force[mask].sum())
            nonpad.append(force[~mask].sum())
            if count:
                depth=torch.clamp(-separations[start:start+count].reshape(-1),min=0).max()
                self._max_penetration=torch.maximum(self._max_penetration,depth)
            if count>128:raise RuntimeError("Raw finger contact capacity exceeded; refusing incomplete contact classification")
            raw=torch.zeros((128,3),device=self.device)
            rawforce=torch.zeros(128,device=self.device)
            rawvalid=torch.zeros(128,device=self.device,dtype=torch.bool)
            raw[:count]=pts;rawforce[:count]=force;rawvalid[:count]=True
            point_sets.append(raw);force_sets.append(rawforce);masks.append(rawvalid)
            self._contact_window["points_w"][sample,finger_index,:count]=pts
            self._contact_window["normals_w"][sample,finger_index,:count]=normals[start:start+count]
            self._contact_window["normal_force_n"][sample,finger_index,:count]=force
            self._contact_window["separation_m"][sample,finger_index,:count]=separations[start:start+count].reshape(-1)
            self._contact_window["valid"][sample,finger_index,:count]=True
            self._contact_window["pad_mask"][sample,finger_index,:count]=mask
            self._contact_window["pad_force_n"][sample,finger_index]=force[mask].sum()
            self._contact_window["finger_pose_w"][sample,finger_index]=body
        self._pad_force=torch.stack(pad_force)
        self._nonpad_peak=torch.maximum(self._nonpad_peak,torch.stack(nonpad))
        self._raw_contact_points=torch.stack(point_sets)
        self._raw_contact_forces=torch.stack(force_sets)
        self._raw_contact_valid=torch.stack(masks)
        self._raw_contact_pad_mask=self._contact_window["pad_mask"][sample]
        # Direct PhysX transforms are xyzw; records use xyz,wxyz. These
        # observations add no pose writes, forces or physics/control changes.
        raw_object=self.object.root_physx_view.get_transforms()[0]
        raw_hand=self.robot.root_physx_view.get_link_transforms()[0,self.hand_id]
        object_pose=torch.cat((raw_object[:3],raw_object[[6,3,4,5]]))
        hand_pose=torch.cat((raw_hand[:3],raw_hand[[6,3,4,5]]))
        tcp_p,tcp_q=combine_frame_transforms(hand_pose[None,:3],hand_pose[None,3:7],self.offset)
        self._contact_window["object_pose_w"][sample]=object_pose
        self._contact_window["object_velocity_w"][sample]=self.object.root_physx_view.get_velocities()[0]
        self._contact_window["cached_object_pose_w"][sample]=self.object.data.root_state_w[0,:7]
        self._contact_window["hand_pose_w"][sample]=hand_pose
        self._contact_window["tcp_pose_w"][sample]=torch.cat((tcp_p,tcp_q),-1)[0]
        self._contact_window["physics_tick"][sample]=int(tick)
        self._contact_sample_count += 1
        self._last_contact_tick = tick


    def _apply_action(self):
        if self._nonpad_peak is not None and hasattr(self,"joint_ids") and self._physics_apply_calls > 0:
            self._capture_pad_contacts(physics_tick=self._sim_step_counter-1)
        self._physics_apply_calls += 1
        self.robot.set_joint_position_target(self.targets, joint_ids=self.joint_ids)

    def _get_observations(self):
        return {
            "policy": torch.cat(
                (
                    self.robot.data.joint_pos[:, self.joint_ids],
                    self.robot.data.joint_vel[:, self.joint_ids],
                ),
                -1,
            )
        }

    def _get_rewards(self):
        return torch.zeros(self.num_envs, device=self.device)

    def _get_dones(self):
        # The collector owns terminal scoring and stops before any reset.
        done = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        return done, done.clone()

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        q = torch.as_tensor(TRANSFER["initial_qpos"], dtype=torch.float32, device=self.device)[None].clone()
        self.robot.write_joint_state_to_sim(q, torch.zeros_like(q), joint_ids=self.joint_ids, env_ids=env_ids)
        self.robot.set_joint_position_target(q, joint_ids=self.joint_ids, env_ids=env_ids)
        self.targets = q.clone()
        pose = self.object.data.default_root_state[env_ids].clone()
        p = [TRANSFER["object_reference_xy_m"][0] + self.case_xy[0],
             TRANSFER["object_reference_xy_m"][1] + self.case_xy[1], -float(self.bounds_min[2]) + 0.002]
        qobj = [np.cos(self.case_yaw/2), 0., 0., np.sin(self.case_yaw/2)]
        pose[..., :7] = torch.tensor([*p, *qobj], device=self.device)
        pose[..., 7:] = 0
        self.object.write_root_state_to_sim(pose, env_ids=env_ids)
        self.yaw = self.case_yaw
        panel = self.sim.stage.GetPrimAtPath(self._visual_occluder_path)
        if panel.IsValid():
            UsdGeom.Imageable(panel).MakeInvisible()

    def park_object(self):
        """Remove old held geometry before teleporting the robot for reset."""
        state = self.object.data.default_root_state.clone()
        state[:, :7] = torch.tensor([2., 2., .5, 1., 0., 0., 0.], device=self.device)
        state[:, 7:] = 0.
        self.object.write_root_state_to_sim(state)
        for _ in range(2):
            self.step(self.targets.clone())

    def configure_visual_occlusion(self):
        """Place one render-only panel using the settled initial object pose.

        The requested fraction describes geometric projection, excluding other
        occluders such as the robot. The panel remains fixed in world space.
        """
        camera = TRANSFER["cameras"]["base_camera"]
        spec = design_occluder(
            camera["pos"], camera["rot"], camera["intrinsic_cv"],
            self.object.data.root_state_w[0, :7].cpu().numpy(), self.size,
            mode=self.occlusion,
        )
        spec.update(
            target_camera="base_camera",
            physics="none",
            planner_obstacle=False,
            placement="fixed_world_after_initial_settling",
            configured=True,
            usd_collision_api_prims=[],
            usd_rigid_body_api_prims=[],
        )
        if spec["panel"] is not None:
            panel = spec["panel"]
            prim = self.sim.stage.GetPrimAtPath(self._visual_occluder_path)
            if not prim.IsValid():
                raise RuntimeError("Visual occluder prim was not created")
            # Gf uses row-vector matrix composition: local scale then rotation.
            qw, qx, qy, qz = panel["rot_wxyz"]
            matrix = (
                Gf.Matrix4d().SetScale(Gf.Vec3d(*panel["size_m"]))
                * Gf.Matrix4d().SetRotate(Gf.Quatd(qw, Gf.Vec3d(qx, qy, qz)))
            )
            matrix.SetTranslateOnly(Gf.Vec3d(*panel["pos"]))
            UsdGeom.Xformable(prim).MakeMatrixXform().Set(matrix)
            UsdGeom.Imageable(prim).MakeVisible()
            for child in Usd.PrimRange(prim):
                if child.HasAPI(UsdPhysics.CollisionAPI):
                    spec["usd_collision_api_prims"].append(str(child.GetPath()))
                if child.HasAPI(UsdPhysics.RigidBodyAPI):
                    spec["usd_rigid_body_api_prims"].append(str(child.GetPath()))
            if spec["usd_collision_api_prims"] or spec["usd_rigid_body_api_prims"]:
                raise RuntimeError("Visual-only panel unexpectedly has physics APIs")
        self.visual_occlusion_spec = spec
        return spec

    def snapshot(self, capture_rgb=True):
        self._capture_pad_contacts()
        for camera in (self.cameras.values() if capture_rgb else []):
            period = camera.cfg.update_period
            camera.cfg.update_period = 0.0
            camera.update(0.0, force_recompute=True)
            camera.cfg.update_period = period
        hand = self.robot.data.body_link_state_w[:, self.hand_id]
        tcp_p, tcp_q = combine_frame_transforms(hand[:, :3], hand[:, 3:7], self.offset)
        forces, peaks = [], []
        for link in LINKS:
            force = self.contacts[link].data.force_matrix_w[0, 0]
            peak = (
                torch.linalg.vector_norm(
                    self.contacts[link].data.force_matrix_w_history[0, :, 0], dim=-1
                )
                .max(0)
                .values
            )
            forces.append(force)
            peaks.append(peak)
        cameras = {
            name: camera.data.output["rgb"][0, ..., :3].cpu().numpy().copy()
            for name, camera in self.cameras.items()
        } if capture_rgb else {}
        # Under GPU Fabric the USD query used by Camera.data can retain a rest
        # pose. Derive extrinsics from the live PhysX link pose and calibrated
        # local camera offset instead.
        wrist_offset = self.cameras["wrist_camera"].cfg.offset
        wrist_p, wrist_q = combine_frame_transforms(
            hand[:, :3],
            hand[:, 3:7],
            torch.tensor([wrist_offset.pos], device=self.device, dtype=hand.dtype),
            torch.tensor([wrist_offset.rot], device=self.device, dtype=hand.dtype),
        )
        base_offset = self.cameras["base_camera"].cfg.offset
        base_camera = torch.tensor(
            [[*base_offset.pos, *base_offset.rot]], device=self.device, dtype=hand.dtype
        )
        row = {
            "finger_contact_pos_w": raw_contact_centroids(self._raw_contact_points, self._raw_contact_valid).cpu().numpy().copy(),
            "finger_link_pose_w": self.robot.data.body_link_state_w[0, [self.robot.find_bodies(link)[0][0] for link in ("panda_leftfinger", "panda_rightfinger")], :7].cpu().numpy().copy(),
            "qpos": self.robot.data.joint_pos[0, self.joint_ids].cpu().numpy().copy(),
            "qvel": self.robot.data.joint_vel[0, self.joint_ids].cpu().numpy().copy(),
            "position_targets": self.targets[0].cpu().numpy().copy(),
            "tcp_pose": torch.cat((tcp_p, tcp_q), -1)[0].cpu().numpy().copy(),
            "object_pose": self.object.data.root_state_w[0, :7].cpu().numpy().copy(),
            "object_linear_velocity": self.object.data.root_lin_vel_w[0]
            .cpu()
            .numpy()
            .copy(),
            "object_angular_velocity": self.object.data.root_ang_vel_w[0]
            .cpu()
            .numpy()
            .copy(),
            "object_sleeping": np.asarray(self._sleep_interface.is_sleeping(self._sleep_stage_id,self._sleep_body_path),dtype=bool),
            "object_raw_tensor_velocity_w": self.object.root_physx_view.get_velocities()[0].cpu().numpy().copy(),
            "link_contact_force": torch.stack(forces).cpu().numpy().copy(),
            "link_contact_peak_n": torch.stack(peaks).cpu().numpy().copy(),
            "object_table_peak_n": self.object_table.data.force_matrix_w_history[0]
            .norm(dim=-1)
            .max()
            .cpu()
            .numpy()
            .copy(),
            "object_wall_peak_n": self.object_wall.data.force_matrix_w_history[0].norm(dim=-1).max().cpu().numpy().copy(),
            "hand_pose": hand[0, :7].cpu().numpy().copy(),
            "camera_frame": torch.stack([cam.frame[0] for cam in self.cameras.values()])
            .cpu()
            .numpy()
            .copy(),
            "camera_pose": torch.cat(
                (base_camera, torch.cat((wrist_p, wrist_q), -1)), 0
            )
            .cpu()
            .numpy()
            .copy(),
            "camera_intrinsics": torch.cat(
                [cam.data.intrinsic_matrices for cam in self.cameras.values()], 0
            )
            .cpu()
            .numpy()
            .copy(),
        }
        row.update(
            pad_object_force_n=self._pad_force.cpu().numpy().copy(),
            nonpad_object_peak_n=self._nonpad_peak.cpu().numpy().copy(),
            nonpad_object_force_bound_n=(self._nonpad_peak*self.nonpad_force_multiplier).cpu().numpy().copy(),
            max_contact_penetration_m=self._max_penetration.cpu().numpy().copy(),
            raw_finger_contact_points_w=self._raw_contact_points.cpu().numpy().copy(),
            raw_finger_contact_normal_force_n=self._raw_contact_forces.cpu().numpy().copy(),
            raw_finger_contact_valid=self._raw_contact_valid.cpu().numpy().copy(),
            raw_finger_contact_pad_mask=self._raw_contact_pad_mask.cpu().numpy().copy(),
            contact_substep_sample_count=np.array(self._contact_sample_count,dtype=np.int64),
            **{"contact_substep_"+key:values.cpu().numpy().copy()
               for key,values in self._contact_window.items()})
        return row, cameras
