"""Concurrency gate: no simulator work on the RPC thread, always drain both jobs."""
from concurrent.futures import ThreadPoolExecutor
import time

def run_overlap(request, check, budget_s):
    events={}
    def rpc():
        events['rpc_started_ns']=time.monotonic_ns()
        try: return request()
        finally: events['rpc_ended_ns']=time.monotonic_ns()
    with ThreadPoolExecutor(max_workers=1,thread_name_prefix='lift-rpc') as pool:
        future=pool.submit(rpc)
        events['geometry_started_ns']=time.monotonic_ns()
        try:
            checked=check()
        finally:
            events['geometry_ended_ns']=time.monotonic_ns()
        reply=future.result()
    geometry_s=(events['geometry_ended_ns']-events['geometry_started_ns'])/1e9
    # Includes init/update, lifecycle readback and all native non-preemptible work.
    service_s=float(reply['wall_s'])
    within_budget=geometry_s+service_s <= float(budget_s)
    events.update(geometry_s=geometry_s,service_s=service_s,candidate_remaining_s=float(budget_s),
                  accounted_cost_s=geometry_s+service_s,budget_passed=within_budget,
                  both_completed_before_execution=True,
                  overlap_s=max(0,min(events['geometry_ended_ns'],events['rpc_ended_ns'])-max(events['geometry_started_ns'],events['rpc_started_ns']))/1e9)
    if not within_budget:
        reply=dict(reply,success=False,reason='candidate_planning_budget_exhausted',speculative_reply_rejected=True)
    return checked,reply,events
