"""Object-independent target interface consumed by the cuRobo collector.

Input: qualified local grasp pool and an observed object transform.
Output: local TCP poses, world grasp/pregrasp poses and explicit provenance.
The candidate generator owns region priors. No object-specific target offset
or pose is specified in this provider.
"""
from pathlib import Path
import hashlib,json
import numpy as np
from geometry.primitives import pose_matrix


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_candidate_source(candidate_root,method,object_id):
    folder=Path(candidate_root)/method/object_id
    path=folder/'REPORT.json'
    report=json.loads(path.read_text())
    if report['object_id']!=object_id:raise ValueError('Candidate/object identity mismatch')
    cache=path.parent / report['source_cache']
    if sha(cache)!=report['source_cache_sha256']:raise ValueError('Candidate source changed')
    candidates=report['candidates']
    if len(candidates)>200:raise ValueError('Candidate source exceeds the fixed single-call200-pose pool')
    for c in candidates:
        transform=np.asarray(c['T_object_tcp'],float)
        if transform.shape!=(4,4) or not np.isfinite(transform).all():raise ValueError('Invalid local target')
        if not np.allclose(transform[3],[0,0,0,1],atol=1e-6):raise ValueError('Invalid homogeneous row')
    provenance={'source':report['source_kind'],'method':method,'object_id':object_id,
        'report_path':str(path),'report_sha256':sha(path),'candidate_cache_path':str(cache),
        'candidate_cache_sha256':sha(cache),'region_prior':report.get('point_region_prior'),
        'refinement_metadata':report.get('refinement_metadata'),
        'transform_convention':'T_world_tcp = T_world_object @ T_object_tcp; the local TCP pose already contains the single fixed gripper adapter',
        'geometry_report':report}
    return candidates,provenance


def world_targets(candidates,object_pose,approach_distance_m=.1):
    obj=pose_matrix(object_pose)
    if not np.isfinite(obj).all():raise ValueError('Nonfinite observed object pose')
    if not 0<approach_distance_m<=.3:raise ValueError('Invalid approach distance')
    result=[]
    for candidate in candidates:
        grasp=obj@np.asarray(candidate['T_object_tcp'],float)
        pre=grasp.copy();pre[:3,3]-=float(approach_distance_m)*grasp[:3,2]
        result.append({**candidate,'grasp_world':grasp.tolist(),'pregrasp_world':pre.tolist()})
    return result


def screen_scene_candidates(candidates,scene,maximum=50):
    """Spend the scene budget only on fully collision-clear gripper sweeps.

    The source pool is the original single200-pose model call. Score order,
    exact gripper/target/table geometry, openings and contact policy are kept.
    IK and full-arm path validation remain the planner's responsibility.
    """
    import time
    from geometry.api import OPENINGS_M,check_gripper_path
    if len(candidates)>200:raise ValueError('Scene input exceeds the fixed200-pose source pool')
    if not 1<=maximum<=50:raise ValueError('Scene-qualified candidate limit must be1..50')
    started=time.monotonic();obj=pose_matrix(scene['object_pose']);valid=[]
    report={'candidates_received':len(candidates),'scene_geometric_candidates':0,
        'scene_candidates_screened':0,'scene_candidate_limit':maximum,
        'candidate_budget_policy':'single200-pose cache; dedup and object geometry first; retain up to50 complete scene-clear sweeps in source score order',
        'rejections':[]}
    for candidate in world_targets(candidates,obj):
        report['scene_candidates_screened']+=1
        grasp=np.asarray(candidate['grasp_world']);pre=np.asarray(candidate['pregrasp_world'])
        span=candidate.get('pad_span')
        minimum=2*max(abs(span['positive_y_m']),abs(span['negative_y_m']))+.001 if span else .001
        chosen=None;width_reports=[]
        for opening in OPENINGS_M:
            if opening<minimum:continue
            check=check_gripper_path(np.stack((pre,grasp)),opening,obj,scene['object_id'],allow_pad_contact=False)
            width_reports.append({'opening_m':opening,**check})
            if check['valid']:chosen=opening;break
        candidate={**candidate,'scene_width_checks':width_reports}
        if chosen is None:
            candidate['rejected']='scene_gripper_collision';report['rejections'].append(candidate);continue
        candidate['opening_m']=chosen;valid.append(candidate)
        if len(valid)>=maximum:break
    report['scene_geometric_candidates']=len(valid)
    report['geometric_filter_wall_s']=time.monotonic()-started
    return valid,report
