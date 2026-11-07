"""Loopback service: one locked, reusable cuRobo planner for multiple collectors."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import secrets
import socketserver
import sys
import threading
import time
import traceback

from common import write


class Engine:
    def __init__(self, factory, seed):
        self.factory, self.seed = factory, seed
        self.planner = None
        self.lock = threading.Lock()
        self.index = 0

    def execute(self, payload):
        queued = time.monotonic()
        with self.lock:
            started = time.monotonic()
            request = payload["request"]
            output = Path(request["out"])
            output.mkdir(parents=True, exist_ok=True)
            self.index += 1
            try:
                self.seed(int(request["seed"]))
                if payload.get("fresh"):
                    self.planner = None
                reused = self.planner is not None
                if not reused:
                    self.planner = self.factory(request["scene"], output)
                    if request["command"] != "select":
                        raise ValueError("The first request must select a grasp")
                else:
                    self.planner.update(request["scene"], output,
                                        remove_object=request["command"] == "lift")
                init_s = time.monotonic() - started
                if request["command"] == "select":
                    reply = self.planner.select(request["candidates"], budget_s=30., maximum=5)
                elif request["command"] == "lift":
                    reply = self.planner.lift(max(0., float(request["remaining_budget_s"]) - init_s))
                else:
                    raise ValueError("Unknown planner command")
                reply.update(request_index=self.index, planner_reused=reused,
                             planner_initialization_or_update_s=init_s,
                             queue_wait_s=started - queued, wall_s=time.monotonic() - started)
                write(output / "WORKER_RESULT.json", reply)
                return reply
            except Exception:
                # Discard failed mutable state. Do not silently retry this slot.
                self.planner = None
                reply = dict(success=False, transport_error=traceback.format_exc(),
                             request_index=self.index, queue_wait_s=started - queued,
                             wall_s=time.monotonic() - started)
                write(output / "WORKER_ERROR.json", reply)
                return reply


class Server(socketserver.ThreadingTCPServer):
    daemon_threads = True
    allow_reuse_address = False


class Handler(socketserver.StreamRequestHandler):
    def handle(self):
        self.connection.settimeout(600)
        try:
            raw = self.rfile.readline(16 * 1024 * 1024)
            if not raw.endswith(b"\n"):
                raise ValueError("Invalid/oversized request")
            payload = json.loads(raw)
            if not secrets.compare_digest(str(payload.get("token", "")), self.server.token):
                raise ValueError("Incorrect benchmark token")
            reply = self.server.engine.execute(payload)
        except Exception:
            reply = dict(transport_error=traceback.format_exc())
        self.wfile.write(json.dumps(reply, allow_nan=False).encode() + b"\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--ready", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.runtime / "collector"))
    import torch
    from generic_planner import Planner
    torch.set_num_threads(2)
    with Server(("127.0.0.1", 0), Handler) as server:
        server.token = secrets.token_hex(24)
        server.engine = Engine(Planner, torch.manual_seed)
        write(args.ready, dict(port=server.server_address[1], token=server.token,
                               pid=__import__("os").getpid()))
        print("PLANNER_READY", flush=True)
        server.serve_forever(poll_interval=.2)


if __name__ == "__main__":
    main()
