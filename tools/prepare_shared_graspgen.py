"""Relocate an existing GraspGen venv beside the shared simulation/planning environments."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import shutil
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-venv", type=Path, required=True)
    parser.add_argument("--shared-env", type=Path, required=True)
    args = parser.parse_args()
    source, shared = args.source_venv.resolve(), args.shared_env.resolve()
    target = shared / "graspgen"
    parent = shared / "planner_parent"
    python = shared / "python/bin/python3.11"
    if not (source / "pyvenv.cfg").is_file() or not parent.is_dir() or not python.is_file():
        raise FileNotFoundError("An existing inference venv and prepared shared Python/planner dependencies are required")
    target.mkdir(exist_ok=False)
    subprocess.run(["rsync", "-a", "--exclude=__pycache__", "--exclude=*.pyc",
                    str(source)+"/", str(target)+"/"], check=True)
    for path in (target / "bin").glob("python*"):
        if path.is_symlink():
            path.unlink()
            path.symlink_to(python)
    (target / "pyvenv.cfg").write_text(
        f"home = {python.parent}\ninclude-system-site-packages = false\nversion = 3.11.14\nexecutable = {python}\n")
    packages = target / "lib/python3.11/site-packages"
    (packages / "openpi_parent.pth").write_text(str(parent)+"\n")
    # generate_candidates.py adds the current repository's vendored source.
    source_link = packages / "graspgen_source.pth"
    if source_link.exists():
        source_link.unlink()
    for path in (target / "bin").iterdir():
        if not path.is_file() or path.is_symlink():
            continue
        data = path.read_bytes()
        if data.startswith(b"#!"):
            first, separator, rest = data.partition(b"\n")
            if str(source).encode() in first:
                path.write_bytes(first.replace(str(source).encode(), str(target).encode()) + separator + rest)
    broken = [str(p) for p in target.rglob("*") if p.is_symlink() and not p.exists()]
    if broken:
        raise RuntimeError("Broken copied links: " + str(broken))
    print("Prepared", target, "(no simulation or model inference started)")


if __name__ == "__main__":
    main()
