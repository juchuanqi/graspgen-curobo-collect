"""Verify deployable assets without importing Isaac or creating a CUDA context."""
from __future__ import annotations
import json
from pathlib import Path
import xml.etree.ElementTree as ET


def check_assets(runtime):
    runtime = Path(runtime).resolve()
    manifest = json.loads((runtime / "assets/MANIFEST.json").read_text())
    for name, asset in manifest["objects"].items():
        for field in ("usd_path", "mesh_path", "surface_path"):
            relative = Path(asset[field])
            path = (runtime / relative).resolve()
            if relative.is_absolute() or not path.is_relative_to(runtime) or not path.is_file():
                raise ValueError(f"Invalid/missing asset: {name}/{field}")
    urdf = runtime / "robot/panda_v2.urdf"
    for mesh in ET.parse(urdf).getroot().findall(".//mesh"):
        path = (urdf.parent / mesh.get("filename")).resolve()
        if not path.is_relative_to(runtime) or not path.is_file():
            raise ValueError("Invalid/missing URDF mesh: " + str(path))
    for relative in ("robot/usd/panda.usd", "config/scene.json"):
        if not (runtime / relative).is_file():
            raise FileNotFoundError(runtime / relative)
    return dict(status="passed", objects=list(manifest["objects"]))


def check_candidates(runtime, candidate_root, objects):
    from common import sha
    import sys
    sys.path.insert(0, str(Path(runtime)))
    sys.path.insert(0, str(Path(runtime) / "collector"))
    from target_provider import load_candidate_source
    from geometry.api import manifest
    for name in objects:
        folder = Path(candidate_root) / "graspgen" / name
        if not (folder / "REPORT.json").is_file():
            raise FileNotFoundError(f"No GraspGen pool for {name}. Run: bash run.sh prepare --objects {name}")
        candidates, provenance = load_candidate_source(candidate_root, "graspgen", name)
        report = provenance["geometry_report"]
        asset = manifest()["objects"][name]
        if report.get("input_mesh_sha256") != sha(asset["mesh_path"]):
            raise ValueError("Candidate pool does not match current mesh: " + name)
        key = report.get("cache_key", {})
        if key.get("surface_sha256") != sha(asset["surface_path"]) or key.get("parts") != asset["parts"]:
            raise ValueError("Candidate pool does not match current object geometry: " + name)
        if not candidates:
            raise ValueError("No geometry-qualified candidates: " + name)
    return dict(status="passed", objects=list(objects))


if __name__ == "__main__":
    import argparse
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path)
    parser.add_argument("--objects", default="block55,slender95,cylinder36h52,disk56h24")
    args = parser.parse_args()
    check_assets(args.runtime)
    if args.candidate_root:
        check_candidates(args.runtime, args.candidate_root, args.objects.split(","))
    print("Asset/candidate checks passed")
