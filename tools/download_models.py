"""Download only the three pinned official Panda model files; verify their hashes."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import HERE, read, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "models")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    model = read(HERE / "runtime/config/models.json")
    for name, spec in model["model_files"].items():
        path = args.output / "checkpoints" / name
        if not args.verify_only and not path.exists():
            from huggingface_hub import hf_hub_download
            hf_hub_download(repo_id="adithyamurali/GraspGenModels", filename="checkpoints/"+name,
                             revision=model["model_revision"], local_dir=str(args.output))
        if not path.is_file() or sha(path) != spec["sha256"]:
            raise ValueError("Missing or different model: " + str(path))
        print("VERIFIED", path)


if __name__ == "__main__":
    main()
