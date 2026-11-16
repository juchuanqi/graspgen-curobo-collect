"""Replay unchanged fixed-capacity contact math; native reads stay outside capture."""
import time
import torch


class ContactGraph:
    def __init__(self, template, finger_ids, offset, quat_apply_inverse, combine_frame_transforms):
        started = time.perf_counter()
        if any(not value.is_cuda for value in template.values()):
            raise ValueError('Contact graph requires CUDA inputs')
        self.inputs = {key:value.clone() for key,value in template.items()}
        self.layout = {key:(value.shape, value.dtype, value.device) for key,value in template.items()}
        self.finger_ids = finger_ids
        self.offset = offset.clone()
        self.quat_apply_inverse = quat_apply_inverse
        self.combine_frame_transforms = combine_frame_transforms
        self.index = torch.arange(128, device=offset.device, dtype=torch.long)
        self.pose_order = torch.tensor([6,3,4,5],device=offset.device,dtype=torch.long)
        # Warm JIT math on a separate stream before capture. No simulator work
        # or physical steps occur here, and this setup is reported separately.
        stream = torch.cuda.Stream(device=offset.device)
        stream.wait_stream(torch.cuda.current_stream(offset.device))
        with torch.cuda.stream(stream):
            for _ in range(6):
                self.outputs = self.compute()
        torch.cuda.current_stream(offset.device).wait_stream(stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=stream):
            self.outputs = self.compute()
        torch.cuda.current_stream(offset.device).wait_stream(stream)
        torch.cuda.synchronize(offset.device)
        self.initialization_s = time.perf_counter()-started

    def compute(self):
        x = self.inputs
        rows = {key:[] for key in ['points_w','normals_w','normal_force_n','separation_m',
                                  'valid','pad_mask','pad_force_n','finger_pose_w','errors']}
        nonpad = []
        penetration = x['previous_penetration']
        for finger in range(2):
            normal,points,normals,separations,counts,starts = [x[f'{finger}_{i}'] for i in range(6)]
            count = counts.reshape(-1)[0]
            start = starts.reshape(-1)[0]
            error = ((counts.sum() >= len(normal)) | (count < 0) | (count > 128) |
                     (start < 0) | (start+count > len(points)))
            valid = self.index < count
            take = (start+self.index).clamp(0,len(points)-1)
            pts = torch.where(valid[:,None], points.index_select(0,take), 0.)
            force = torch.where(valid,normal.reshape(-1).index_select(0,take).abs(),0.)
            body = x['bodies'][self.finger_ids[finger]]
            local = self.quat_apply_inverse(body[3:7].expand(128,4),pts-body[:3])
            inner_y = -.00002 if finger == 0 else .00002
            mask = (valid & (torch.abs(local[:,0]) <= .00925) &
                    (torch.abs(local[:,1]-inner_y) <= .001) &
                    (local[:,2] >= .0355) & (local[:,2] <= .055))
            pad_force = torch.where(mask,force,0.).sum()
            nonpad.append(torch.where(valid & ~mask,force,0.).sum())
            separation = torch.where(valid,separations.reshape(-1).index_select(0,take),0.)
            penetration = torch.maximum(penetration,(-separation).clamp_min(0).max())
            for key,value in dict(points_w=pts,
                    normals_w=torch.where(valid[:,None],normals.index_select(0,take),0.),
                    normal_force_n=force,separation_m=separation,valid=valid,pad_mask=mask,
                    pad_force_n=pad_force,finger_pose_w=body,errors=error).items():
                rows[key].append(value)
        result = {key:torch.stack(values) for key,values in rows.items()}
        result['nonpad_peak'] = torch.maximum(x['previous_nonpad'],torch.stack(nonpad))
        result['penetration'] = penetration
        raw_object,raw_hand = x['raw_object'],x['raw_hand']
        result['object_pose_w'] = torch.cat((raw_object[:3],raw_object.index_select(0,self.pose_order)))
        hand = torch.cat((raw_hand[:3],raw_hand.index_select(0,self.pose_order)))
        tcp_p,tcp_q = self.combine_frame_transforms(hand[None,:3],hand[None,3:7],self.offset)
        result['hand_pose_w'] = hand
        result['tcp_pose_w'] = torch.cat((tcp_p,tcp_q),-1)[0]
        # Clone these two fields so every result belongs to this replay.
        result['object_velocity_w'] = x['velocity'].clone()
        result['cached_object_pose_w'] = x['cached_object'].clone()
        return result

    def replay(self, inputs):
        if inputs.keys() != self.inputs.keys():
            raise RuntimeError('Contact graph input fields changed')
        for key,value in inputs.items():
            if (value.shape,value.dtype,value.device) != self.layout[key]:
                raise RuntimeError('Contact graph input layout changed: '+key)
            self.inputs[key].copy_(value)
        self.graph.replay()
        return self.outputs
