"""Create a clean source archive, excluding generated outputs and local environments."""
from __future__ import annotations
import argparse
import hashlib
import os
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[1]
EXCLUDE = {".git", ".cache", "__pycache__", ".pytest_cache", ".venv", "venvs", "build", "dist"}
LOCAL_ONLY = {"outputs", "generated", "models", "local_env.json", ".env"}


def source_files():
    for folder, directories, filenames in os.walk(ROOT):
        base = Path(folder)
        directories[:] = sorted(d for d in directories if d not in EXCLUDE and
                                not d.endswith(".egg-info") and not (base == ROOT and d in LOCAL_ONLY))
        for name in sorted(filenames):
            path = base / name
            relative = path.relative_to(ROOT)
            if base == ROOT and name in LOCAL_ONLY:
                continue
            if path.suffix in {".log", ".pyc", ".so", ".o", ".mp4", ".tar", ".gz"}:
                continue
            if path.is_symlink():
                raise ValueError("Source archive must contain no external symlinks: " + str(path))
            yield path, relative


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT.parent / "graspgen_curobo_collect.tar.gz")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    with tarfile.open(args.output, "w:gz") as archive:
        count = 0
        for path, relative in source_files():
            archive.add(path, arcname=str(Path("graspgen_curobo_collect") / relative), recursive=False)
            count += 1
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    args.output.with_suffix(args.output.suffix+".sha256").write_text(digest+"  "+args.output.name+"\n")
    print(f"{args.output}: {count} source/asset files; sha256={digest}")


if __name__ == "__main__":
    main()
