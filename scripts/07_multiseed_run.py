"""Run one phase across several seeds and aggregate mean ± std.

    python scripts/07_multiseed_run.py \
        --config configs/aapd_phase3.yaml \
        --seeds 42,123,456,789,2024 \
        --tag aapd_phase3

For each seed this rewrites `train.seed` and `train.save_dir`, runs the matching
training script as a subprocess, then collects every `results.json` and reports:
  1. mean ± std / min / max per metric
  2. a one-sample t-test against the dataset's published baseline
  3. a LaTeX table ready to paste into the paper

Which script runs is decided by the CONFIG CONTENT, not by the file name:
  a `phase3:` block          -> scripts/06_train_phase3.py
  `model.head: lsa_gated`    -> scripts/03_train_phase2.py
  otherwise                  -> scripts/01_train_phase1.py
Pass `--phase {1,2,3}` to force it.

Outputs: `outputs/<tag>_seed<seed>/` per run, plus
`outputs/<tag>_aggregated.json` and `outputs/<tag>_table.tex`.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import yaml

from src.console import enable_utf8_output
from src.reference_scores import get_primary_baseline

PHASE_SCRIPTS = {
    1: "scripts/01_train_phase1.py",
    2: "scripts/03_train_phase2.py",
    3: "scripts/06_train_phase3.py",
}

METRIC_KEYS = ["accuracy", "micro_f1", "macro_f1",
               "p@1", "p@3", "p@5", "ndcg@1", "ndcg@3", "ndcg@5"]


def detect_phase(cfg: dict) -> int:
    """Infer the phase from the config's contents."""
    if "phase3" in cfg:
        return 3
    if cfg.get("model", {}).get("head") == "lsa_gated":
        return 2
    return 1


def run_single_seed(config_path: str, seed: int, tag: str, phase: int):
    """Train one seed in a subprocess and return its parsed results.json."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    cfg["train"]["seed"] = seed
    cfg["train"]["save_dir"] = f"outputs/{tag}_seed{seed}"
    script = PHASE_SCRIPTS[phase]

    # Write the resolved config to a temp dir so a crashed run never leaves
    # stray `_temp_*.yaml` files behind in configs/.
    tmp_dir = Path(tempfile.mkdtemp(prefix=f"multiseed_{tag}_"))
    temp_config = tmp_dir / f"{tag}_seed{seed}.yaml"
    try:
        with open(temp_config, "w") as f:
            yaml.safe_dump(cfg, f, indent=2)

        print(f"\n{'='*60}")
        print(f"Running {tag} (phase {phase}) with seed={seed}")
        print(f"  Script:  {script}")
        print(f"  Save to: {cfg['train']['save_dir']}")
        print(f"{'='*60}")

        result = subprocess.run(
            [sys.executable, script, "--config", str(temp_config)],
            cwd=str(PROJECT_ROOT),
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    if result.returncode != 0:
        print(f"  ⚠ Training failed for seed={seed} (exit code {result.returncode})")
        return None

    results_path = Path(cfg["train"]["save_dir"]) / "results.json"
    if not results_path.exists():
        print(f"  ⚠ Results file not found: {results_path}")
        return None
    with open(results_path) as f:
        return json.load(f)


def aggregate_results(results_list: list, tag: str, dataset_name: str):
    """Mean ± std / min / max per metric, printed against the baseline."""
    print(f"\n{'='*60}")
    print(f"=== Aggregating {tag} ({len(results_list)} seeds) ===")
    print(f"{'='*60}")

    all_metrics = [r["test_metrics"] for r in results_list if r and "test_metrics" in r]
    if not all_metrics:
        print("  No valid results to aggregate")
        return None

    summary = {}
    for key in METRIC_KEYS:
        values = [m.get(key, 0) * 100 for m in all_metrics]  # percentages
        summary[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "values": values,
        }

    ref_name, ref = get_primary_baseline(dataset_name)
    ref_label = ref_name or "baseline"
    print(f"\n{'metric':<12} {'mean':>8} {'±':>2} {'std':>6} {'min':>8} {'max':>8}"
          f"  |  {ref_label:<10} Δ")
    print("-" * 80)
    for key in METRIC_KEYS:
        s = summary[key]
        ref_v = ref.get(key)
        if ref_v is not None:
            delta = s["mean"] - ref_v
            arrow = "▲" if delta > 0 else "▼"
            ref_str = f"{ref_v:>6.2f}  {arrow} {delta:+.2f}"
        else:
            ref_str = "     -       -"
        print(f"{key:<12} {s['mean']:>8.2f} ± {s['std']:>5.2f} "
              f"{s['min']:>8.2f} {s['max']:>8.2f}  |  {ref_str}")

    return summary


def perform_ttest(results_list: list, tag: str, dataset_name: str):
    """One-sample t-test of our per-seed values against the baseline's number."""
    from scipy import stats

    ref_name, ref = get_primary_baseline(dataset_name)
    if not ref:
        print(f"\n(no published baseline for '{dataset_name}' — skipping t-test)")
        return

    print(f"\n{'='*60}")
    print(f"=== One-sample t-test: {tag} vs {ref_name} ===")
    print(f"{'='*60}")

    all_metrics = [r["test_metrics"] for r in results_list if r and "test_metrics" in r]
    print(f"\n{'metric':<12} {'our mean':>10} {ref_name:>10} {'t-stat':>8} "
          f"{'p-value':>10} {'significant':>12}")
    print("-" * 70)
    for key, ref_v in ref.items():
        our_values = np.array([m.get(key, 0) * 100 for m in all_metrics])
        if len(our_values) > 1:
            t_stat, p_value = stats.ttest_1samp(our_values, ref_v)
            sig = "✓ p<0.05" if p_value < 0.05 else "✗ p≥0.05"
        else:
            t_stat, p_value, sig = 0.0, 1.0, "need n≥2"
        print(f"{key:<12} {our_values.mean():>10.2f} {ref_v:>10.2f} "
              f"{t_stat:>8.3f} {p_value:>10.4f} {sig:>12}")


def generate_latex_table(summary: dict, tag: str, dataset_name: str, output_path: Path):
    """Write a paper-ready LaTeX table: the baseline row plus our mean ± std."""
    print("\nGenerating LaTeX table...")
    ref_name, ref = get_primary_baseline(dataset_name)
    cols = ["p@1", "p@3", "p@5", "ndcg@3", "ndcg@5", "micro_f1"]
    headers = "P@1 & P@3 & P@5 & nDCG@3 & nDCG@5 & Micro-F1"

    latex = f"""
\\begin{{table}}[t]
\\centering
\\caption{{Performance on {dataset_name.upper()} (mean $\\pm$ std over {len(summary[cols[0]]['values'])} runs). Best results in \\textbf{{bold}}.}}
\\label{{tab:results_{tag}}}
\\begin{{tabular}}{{lcccccc}}
\\toprule
Method & {headers} \\\\
\\midrule
"""
    if ref:
        cells = " & ".join(f"{ref.get(c, float('nan')):.2f}" for c in cols)
        latex += f"{ref_name} & {cells} \\\\\n"

    cells = " & ".join(
        f"\\textbf{{{summary[c]['mean']:.2f} $\\pm$ {summary[c]['std']:.2f}}}" for c in cols
    )
    latex += f"\\textbf{{Ours ({tag})}} & {cells} \\\\\n"
    latex += "\\bottomrule\n\\end{tabular}\n\\end{table}\n"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(latex)
    print(f"  Saved: {output_path}")


def main():
    enable_utf8_output()
    parser = argparse.ArgumentParser(description="Multi-seed run + aggregation")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--seeds", type=str, default="42,123,456,789,2024",
                        help="Comma-separated seeds")
    parser.add_argument("--tag", type=str, required=True,
                        help="Run tag; output goes to outputs/<tag>_seed<seed>/")
    parser.add_argument("--phase", type=int, default=None, choices=[1, 2, 3],
                        help="Force the phase instead of inferring it from the config")
    parser.add_argument("--skip-training", action="store_true",
                        help="Only aggregate results that already exist on disk")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    dataset_name = cfg.get("dataset", {}).get("name", "unknown")
    phase = args.phase or detect_phase(cfg)

    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    print(f"Config:  {args.config}")
    print(f"Dataset: {dataset_name}")
    print(f"Phase:   {phase} -> {PHASE_SCRIPTS[phase]}"
          f"{'' if args.phase else '  (inferred from config)'}")
    print(f"Seeds:   {seeds}")
    print(f"Tag:     {args.tag}")

    results_list = []
    if args.skip_training:
        print("\nSkipping training, loading existing results...")
        for seed in seeds:
            results_path = Path(f"outputs/{args.tag}_seed{seed}/results.json")
            if results_path.exists():
                with open(results_path) as f:
                    results_list.append(json.load(f))
                print(f"  Loaded: {results_path}")
            else:
                print(f"  Not found: {results_path}")
    else:
        for seed in seeds:
            result = run_single_seed(args.config, seed, args.tag, phase)
            if result:
                results_list.append(result)

    if not results_list:
        print("\nNo results to aggregate. Exiting.")
        return

    if len(results_list) < len(seeds):
        print(f"\n⚠ Only {len(results_list)}/{len(seeds)} seeds produced results — "
              f"the statistics below cover the successful runs only.")

    summary = aggregate_results(results_list, args.tag, dataset_name)
    if len(results_list) >= 2:
        perform_ttest(results_list, args.tag, dataset_name)

    Path("outputs").mkdir(parents=True, exist_ok=True)
    agg_path = Path(f"outputs/{args.tag}_aggregated.json")
    with open(agg_path, "w") as f:
        json.dump({
            "tag": args.tag,
            "dataset": dataset_name,
            "phase": phase,
            "n_seeds": len(results_list),
            "seeds": seeds,
            "summary": summary,
            "individual_results": results_list,
        }, f, indent=2)
    print(f"\nAggregated results saved to: {agg_path}")

    if summary:
        generate_latex_table(summary, args.tag, dataset_name,
                             Path(f"outputs/{args.tag}_table.tex"))

    print(f"\n{'='*60}")
    print("=== Multi-seed experiment complete ===")
    print(f"{'='*60}")
    print("\nNext steps:")
    print("  1. Repeat for the other phases (--phase 1 / 2 / 3)")
    print("  2. Generate the figures: python scripts/08_make_figures.py --help")
    print(f"  3. Paste outputs/{args.tag}_table.tex into the paper")


if __name__ == "__main__":
    main()
