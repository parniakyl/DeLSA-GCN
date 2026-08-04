"""Pre-flight check: is the GPU free and is everything installed?

Checks, in order:
  1. PyTorch + CUDA available
  2. GPU model, total / used / free memory
  3. Other processes holding GPU memory (lists their PIDs)
  4. (with --config) the config exists, the TSV splits exist, and the tokenizer loads

Usage:
    python tools/check_gpu.py
    python tools/check_gpu.py --config configs/aapd_phase1.yaml
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.console import enable_utf8_output

MIN_FREE_GB = 4.0   # deberta-v3-base + AdamW + grad checkpointing
TIGHT_FREE_GB = 5.0


def get_gpu_processes() -> list[dict]:
    """Processes currently holding GPU memory, via nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        processes = []
        for line in result.stdout.strip().split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                processes.append({"pid": parts[0], "name": parts[1], "mem_mib": int(parts[2])})
        return processes
    except Exception:
        return []


def main():
    enable_utf8_output()
    parser = argparse.ArgumentParser(description="GPU / environment pre-flight check")
    parser.add_argument("--config", type=str, default=None,
                        help="Also verify this config, its data files and its tokenizer")
    args = parser.parse_args()

    print("=" * 60)
    print("PRE-FLIGHT CHECK")
    print("=" * 60)

    try:
        import torch
        print(f"\n✓ PyTorch {torch.__version__} imported")
    except ImportError:
        print("❌ PyTorch not installed. Run: pip install -r requirements.txt")
        sys.exit(1)

    if not torch.cuda.is_available():
        print("❌ CUDA not available. Training will fall back to CPU and be very slow.")
        sys.exit(1)

    n_gpus = torch.cuda.device_count()
    print(f"✓ CUDA available, {n_gpus} GPU(s) detected")

    for i in range(n_gpus):
        props = torch.cuda.get_device_properties(i)
        free, total = torch.cuda.mem_get_info(i)
        used = total - free
        print(f"\nGPU {i}: {props.name}")
        print(f"  Total memory:       {total / 1e9:.2f} GB")
        print(f"  Used memory:        {used / 1e9:.2f} GB")
        print(f"  Free memory:        {free / 1e9:.2f} GB")
        print(f"  Compute capability: {props.major}.{props.minor}")

        if used > 1e9:
            print(f"\n  ⚠ GPU {i} has {used/1e9:.2f} GB in use by other processes — "
                  f"this may cause OOM.")
            processes = get_gpu_processes()
            if processes:
                for p in processes:
                    print(f"    PID {p['pid']:>8} | {p['mem_mib']:>5} MiB | {p['name']}")
                print("\n  Free it by killing the offending PID "
                      "(taskkill /PID <pid> /F on Windows, kill -9 <pid> elsewhere).")
            else:
                print("    (could not enumerate processes; check `nvidia-smi` manually)")

        if free < MIN_FREE_GB * 1e9:
            print(f"\n  ❌ INSUFFICIENT MEMORY: only {free/1e9:.2f} GB free, "
                  f"need at least {MIN_FREE_GB:.0f} GB.")
            sys.exit(1)
        elif free < TIGHT_FREE_GB * 1e9:
            print(f"\n  ⚠ TIGHT: only {free/1e9:.2f} GB free. Consider lowering "
                  f"train.batch_size or max_seq_len.")
        else:
            print(f"\n  ✓ Sufficient memory: {free/1e9:.2f} GB free")

    if args.config:
        import yaml

        config_path = Path(args.config)
        if not config_path.is_absolute():
            config_path = PROJECT_ROOT / args.config
        if not config_path.exists():
            print(f"\n❌ Config not found: {config_path}")
            sys.exit(1)
        print(f"\n✓ Config found: {config_path}")

        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        for key in ("train_path", "val_path", "test_path"):
            path = Path(cfg["dataset"][key])
            if not path.is_absolute():
                path = PROJECT_ROOT / path
            if not path.exists():
                print(f"  ❌ {key}: {path} NOT FOUND")
                sys.exit(1)
            with open(path, encoding="utf-8") as f:
                n_lines = sum(1 for _ in f)
            print(f"  ✓ {key}: {path} ({n_lines} samples)")

        encoder_name = cfg["model"]["encoder_name"]
        print(f"\nChecking tokenizer: {encoder_name}")
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(encoder_name)
            print(f"  ✓ Tokenizer loaded (vocab={len(tok)})")
        except Exception as e:
            print(f"  ❌ Failed to load tokenizer: {e}")
            print("     The first run downloads it (~400 MB) — check your connection.")
            sys.exit(1)

    print("\n" + "=" * 60)
    print("✓ ALL CHECKS PASSED — ready to train!")
    print("=" * 60)
    if args.config:
        print(f"\nNext: python scripts/01_train_phase1.py --config {args.config}")


if __name__ == "__main__":
    main()
