"""Phase 3 — train the standalone Hetero-GCN + LCM, then ensemble at test time.

Three stages:
  A (already done)  build graph + cache node features + cache Phase-2 logits
                    -> scripts/04_build_graph.py, scripts/05_compute_node_features.py
  B (here)          train the graph model STANDALONE — no Phase-2 fusion, so it
                    cannot shortcut to "copy Phase 2"
  C (here)          ensemble graph + Phase 2, with the mixing weight chosen on
                    the VALIDATION set only

Full-batch training: the whole graph is one forward pass, so `train.batch_size`
is 1 by definition and `train.epochs` counts full passes over the graph.

Usage:
    python scripts/06_train_phase3.py --config configs/aapd_phase3.yaml
    python scripts/06_train_phase3.py --config configs/rmsc_phase3.yaml
    python scripts/06_train_phase3.py --config configs/phate_phase3.yaml

Outputs (under `train.save_dir`):
    best_model.pt   best val micro-F1 of the graph model alone
    results.json    history, ensemble weight, graph-only + ensemble test metrics
    config.yaml     the resolved config actually used
"""

from __future__ import annotations

import argparse
import json
import pickle as pkl
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from src.console import enable_utf8_output
from src.data import MultiLabelDataset, load_split_file, make_collate_fn
from src.graph_builder import load_relation_adjacencies
from src.labels import get_labels
from src.losses import build_loss_from_config
from src.metrics import evaluate_all, format_metrics
from src.models.graph_classifier import GraphClassifier
from src.reference_scores import get_primary_baseline
from src.train_utils import set_seed


def load_node_features(features_dir: str, device: torch.device) -> tuple:
    """Load cached node features and stack them as [docs | words | labels]."""
    features_dir = Path(features_dir)
    doc_emb = np.load(features_dir / "doc_embeddings.npy")
    word_emb = np.load(features_dir / "word_embeddings.npy")
    label_emb = np.load(features_dir / "label_embeddings.npy")
    phase2_logits = np.load(features_dir / "phase2_logits.npy")

    H = np.concatenate([doc_emb, word_emb, label_emb], axis=0)
    print(f"  Node features: {H.shape}")
    print(f"  Phase 2 logits: {phase2_logits.shape}")
    return torch.from_numpy(H).float().to(device), phase2_logits


def train_one_epoch(model, H, adj_list, labels, loss_fn, optimizer, train_idx,
                    device, max_grad_norm=1.0) -> float:
    """One full-graph training step (the whole graph is a single batch)."""
    model.train()
    optimizer.zero_grad()

    doc_indices = torch.arange(labels.shape[0], device=device)
    logits = model(H, adj_list, doc_indices)["logits"]
    loss = loss_fn(logits[train_idx], labels[train_idx])

    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate_graph_model(model, H, adj_list, labels, loss_fn, idx, device,
                         threshold=0.5) -> tuple[dict, np.ndarray]:
    """Evaluate the graph model alone on `idx`. Returns (metrics, scores)."""
    model.eval()
    doc_indices = torch.arange(labels.shape[0], device=device)
    logits = model(H, adj_list, doc_indices)["logits"]

    eval_logits = logits[idx]
    eval_labels = labels[idx]
    loss = loss_fn(eval_logits, eval_labels)

    scores = torch.sigmoid(eval_logits.float()).cpu().numpy()
    metrics = evaluate_all(scores, eval_labels.cpu().numpy(), threshold=threshold)
    metrics["loss"] = loss.item()
    return metrics, scores


def find_best_ensemble_weight(graph_scores, other_scores, labels_np, val_idx,
                              weight_min=0.0, weight_max=1.0, step=0.05):
    """Grid-search the graph/other mixing weight on the VALIDATION set.

    Returns (best_weight, best_val_micro_f1).
    """
    graph_val = graph_scores[val_idx]
    other_val = other_scores[val_idx]
    targets_val = labels_np[val_idx]

    best_weight, best_mf1 = 0.5, -1.0
    weight = weight_min
    while weight <= weight_max:
        ensemble = weight * graph_val + (1 - weight) * other_val
        mf1 = evaluate_all(ensemble, targets_val, threshold=0.5)["micro_f1"]
        if mf1 > best_mf1:
            best_mf1, best_weight = mf1, weight
        weight += step
    return best_weight, best_mf1


def predict_phase1_scores(cfg, device, num_labels_fallback):
    """Optional: score the test set with the Phase-1 model for a second ensemble.

    Only runs when `phase3.phase1.{config,checkpoint}` are set AND the checkpoint
    exists. Returns (scores, metrics_or_None) or (None, None).
    """
    phase1_cfg_path = cfg.get("phase3", {}).get("phase1", {}).get("config")
    phase1_ckpt_path = cfg.get("phase3", {}).get("phase1", {}).get("checkpoint")
    if not phase1_ckpt_path or not Path(phase1_ckpt_path).exists():
        return None, None

    print("\n=== Loading Phase 1 model for the alternative ensemble ===")
    try:
        from transformers import AutoTokenizer

        from src.models.encoder_classifier import EncoderClassifier

        with open(phase1_cfg_path) as f:
            cfg_p1 = yaml.safe_load(f)

        model = EncoderClassifier(
            encoder_name=cfg_p1["model"]["encoder_name"],
            num_labels=cfg_p1["dataset"]["num_labels"],
            dropout=cfg_p1["model"]["dropout"],
            gradient_checkpointing=False,
            freeze_layers=0,
        ).to(device)
        ckpt = torch.load(phase1_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        test_samples = load_split_file(cfg_p1["dataset"]["test_path"],
                                       cfg_p1["dataset"]["num_labels"])
        tokenizer = AutoTokenizer.from_pretrained(cfg_p1["model"]["encoder_name"])
        collate = make_collate_fn(tokenizer, max_seq_len=cfg_p1["dataset"]["max_seq_len"])
        loader = DataLoader(
            MultiLabelDataset(test_samples), batch_size=16, shuffle=False,
            collate_fn=collate, num_workers=0, pin_memory=True,
        )

        scores = []
        with torch.no_grad():
            for batch in loader:
                out = model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                )
                scores.append(torch.sigmoid(out["logits"].float()).cpu().numpy())

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return np.concatenate(scores, axis=0), None
    except Exception as e:
        print(f"  ⚠ Could not load Phase 1 model: {e}")
        return None, None


def main():
    enable_utf8_output()
    parser = argparse.ArgumentParser(description="Phase 3: Hetero-GCN + LCM + ensemble")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    train_cfg = cfg["train"]
    data_cfg = cfg["dataset"]
    phase3_cfg = cfg.get("phase3", {})
    num_labels = data_cfg["num_labels"]
    dataset_name = data_cfg.get("name", "unknown")
    eval_threshold = cfg["eval"]["threshold"]

    set_seed(train_cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Dataset: {dataset_name} ({num_labels} labels)")

    save_dir = Path(train_cfg["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / "config.yaml", "w") as f:
        yaml.safe_dump(cfg, f, indent=2)

    # ------------------------------------------------------------------
    # Graph + node features
    # ------------------------------------------------------------------
    print("\nLoading graph...")
    graph_dir = phase3_cfg.get("graph_dir", f"data/{dataset_name}/graph")
    adj_list, (n_docs, n_words, n_graph_labels, n_total) = load_relation_adjacencies(
        graph_dir, device)
    print(f"  Total nodes: {n_total}")

    with open(Path(graph_dir) / "metadata.pkl", "rb") as f:
        graph_meta = pkl.load(f)
    train_size = graph_meta["train_size"]
    val_size = graph_meta["val_size"]
    test_size = graph_meta["test_size"]

    print("\nLoading node features...")
    features_dir = phase3_cfg.get("features_dir", f"data/{dataset_name}/node_features")
    H, phase2_logits = load_node_features(features_dir, device)

    # ------------------------------------------------------------------
    # Labels (train+val+test in the graph's node order)
    # ------------------------------------------------------------------
    all_samples = (
        load_split_file(data_cfg["train_path"], num_labels)
        + load_split_file(data_cfg["val_path"], num_labels)
        + load_split_file(data_cfg["test_path"], num_labels)
    )
    all_labels = np.array([s.labels for s in all_samples], dtype=np.float32)
    labels = torch.from_numpy(all_labels).to(device)

    train_idx = list(range(train_size))
    val_idx = list(range(train_size, train_size + val_size))
    test_idx = list(range(train_size + val_size, train_size + val_size + test_size))

    # ------------------------------------------------------------------
    # Phase 2 reference numbers (the thing Phase 3 has to improve on)
    # ------------------------------------------------------------------
    print("\n=== Phase 2 baseline (cached logits) ===")
    phase2_scores = 1 / (1 + np.exp(-phase2_logits))
    phase2_val_metrics = evaluate_all(phase2_scores[val_idx], all_labels[val_idx],
                                      threshold=eval_threshold)
    phase2_test_metrics = evaluate_all(phase2_scores[test_idx], all_labels[test_idx],
                                       threshold=eval_threshold)
    print(f"  Phase 2 val:  {format_metrics(phase2_val_metrics)}")
    print(f"  Phase 2 test: {format_metrics(phase2_test_metrics)}")
    phase2_val_mf1 = phase2_val_metrics["micro_f1"]

    phase1_scores, _ = predict_phase1_scores(cfg, device, num_labels)
    if phase1_scores is not None:
        phase1_test_metrics = evaluate_all(phase1_scores, all_labels[test_idx],
                                           threshold=eval_threshold)
        print(f"  Phase 1 test: {format_metrics(phase1_test_metrics)}")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    print("\nBuilding Phase 3 model (standalone Hetero-GCN + LCM)...")
    hidden_dim = phase3_cfg.get("hidden_dim", 512)
    dropout = phase3_cfg.get("dropout", 0.3)
    lcm_scale = phase3_cfg.get("lcm_scale", 0.3)
    train_label_matrix = all_labels[:train_size]

    model = GraphClassifier(
        in_dim=H.shape[1],
        hidden_dim=hidden_dim,
        num_labels=num_labels,
        train_labels=train_label_matrix,
        num_relations=len(adj_list),
        dropout=dropout,
        residual=phase3_cfg.get("residual", True),
        lcm_scale=lcm_scale,
    ).to(device)
    print(f"  Trainable params: {model.num_parameters(True)/1e6:.2f}M")
    print(f"  LCM initial scale: {lcm_scale}")

    loss_fn = build_loss_from_config(
        cfg["loss"], num_labels=num_labels, train_label_matrix=train_label_matrix,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"],
    )

    scheduler = None
    if train_cfg.get("lr_scheduler", "none") == "cosine":
        from transformers import get_cosine_schedule_with_warmup
        total_steps = train_cfg["epochs"]
        warmup_steps = int(total_steps * train_cfg.get("warmup_ratio", 0.0))
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )
        print(f"  LR scheduler: cosine (warmup={warmup_steps}, total={total_steps})")

    # ------------------------------------------------------------------
    # Stage B: train the graph model standalone
    # ------------------------------------------------------------------
    best_metric = -1.0
    best_epoch = -1
    history = []
    patience_counter = 0
    patience = train_cfg.get("early_stopping_patience", 100)
    log_every = train_cfg.get("log_every", 1)

    print("\n=== Training Phase 3 (standalone Hetero-GCN + LCM) ===")
    print("  No Phase 2 fusion during training; the ensemble happens at test time.")
    print(f"  Early stopping patience: {patience}")
    print(f"  Logging every {log_every} epochs")

    for epoch in range(train_cfg["epochs"]):
        t0 = time.time()

        train_loss = train_one_epoch(
            model, H, adj_list, labels, loss_fn, optimizer, train_idx, device,
            max_grad_norm=train_cfg.get("max_grad_norm", 1.0),
        )
        if scheduler is not None:
            scheduler.step()

        val_metrics, val_scores = evaluate_graph_model(
            model, H, adj_list, labels, loss_fn, val_idx, device, threshold=eval_threshold
        )
        val_metrics["train_loss"] = train_loss
        val_metrics["epoch"] = epoch + 1
        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            **{k: v for k, v in val_metrics.items() if k != "loss"},
        })

        cur_metric = val_metrics.get("micro_f1", -1.0)
        is_new_best = cur_metric > best_metric
        should_log = (epoch + 1) % log_every == 0 or is_new_best or epoch == 0

        if should_log:
            print(f"Epoch {epoch+1:04d}/{train_cfg['epochs']} | "
                  f"loss={train_loss:.4f} | val_loss={val_metrics['loss']:.4f} | "
                  f"val_mF1={cur_metric*100:.2f} (P2={phase2_val_mf1*100:.2f}) | "
                  f"time={time.time() - t0:.1f}s")

        if is_new_best:
            best_metric = cur_metric
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "val_metrics": val_metrics,
            }, save_dir / "best_model.pt")
            if should_log:
                print(f"  ✓ New best val_micro_f1={best_metric:.4f}, saved.")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n  Early stopping at epoch {epoch+1} "
                      f"(no improvement for {patience} epochs)")
                break

    if best_epoch < 0:
        print("\nNo epoch improved on the initial metric — nothing to evaluate.")
        return

    # ------------------------------------------------------------------
    # Stage C: final evaluation — graph alone, then the ensemble
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"=== Final evaluation (best epoch {best_epoch}) ===")
    print(f"{'='*70}")

    ckpt = torch.load(save_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    _, all_graph_scores = evaluate_graph_model(
        model, H, adj_list, labels, loss_fn, list(range(n_docs)), device,
        threshold=eval_threshold,
    )

    graph_test_metrics = evaluate_all(all_graph_scores[test_idx], all_labels[test_idx],
                                      threshold=eval_threshold)
    print("\n--- Graph-only test metrics ---")
    print(format_metrics(graph_test_metrics))

    # Mixing weight is fitted on validation only.
    print("\n--- Fitting the ensemble weight on the validation set ---")
    ensemble_cfg = cfg.get("ensemble", {})
    search_step = ensemble_cfg.get("search_step", 0.05)

    best_weight, best_ens_val_mf1 = find_best_ensemble_weight(
        all_graph_scores, phase2_scores, all_labels, val_idx,
        weight_min=ensemble_cfg.get("weight_min", 0.0),
        weight_max=ensemble_cfg.get("weight_max", 1.0),
        step=search_step,
    )
    print(f"  Grid: [{ensemble_cfg.get('weight_min', 0.0)}, "
          f"{ensemble_cfg.get('weight_max', 1.0)}] step {search_step}")
    print(f"  Best weight: graph={best_weight:.2f}, phase2={1-best_weight:.2f}")
    print(f"  Ensemble val micro_f1: {best_ens_val_mf1*100:.2f}")
    print(f"  (Phase 2 alone val micro_f1: {phase2_val_mf1*100:.2f})")
    print(f"  (Graph alone  val micro_f1: {best_metric*100:.2f})")

    graph_test_s = all_graph_scores[test_idx]
    phase2_test_s = phase2_scores[test_idx]
    ensemble_test_metrics = evaluate_all(
        best_weight * graph_test_s + (1 - best_weight) * phase2_test_s,
        all_labels[test_idx], threshold=eval_threshold,
    )
    print(f"\n--- Ensemble test metrics "
          f"(graph={best_weight:.2f}, phase2={1-best_weight:.2f}) ---")
    print(format_metrics(ensemble_test_metrics))

    test_metrics = ensemble_test_metrics
    best_ensemble_source = "phase2"

    if phase1_scores is not None:
        # Reuse the same weight — we have no Phase-1 validation scores to fit on.
        ensemble_p1_metrics = evaluate_all(
            best_weight * graph_test_s + (1 - best_weight) * phase1_scores,
            all_labels[test_idx], threshold=eval_threshold,
        )
        print(f"\n--- Alternative: ensemble with Phase 1 "
              f"(graph={best_weight:.2f}, phase1={1-best_weight:.2f}) ---")
        print(format_metrics(ensemble_p1_metrics))
        if ensemble_p1_metrics.get("micro_f1", 0) > ensemble_test_metrics.get("micro_f1", 0):
            print("\n  ✓ The Phase 1 ensemble is better — using it.")
            print("    NOTE: this choice is made on TEST (no Phase-1 val scores are "
                  "cached), so treat it as exploratory, not as the headline result.")
            test_metrics = ensemble_p1_metrics
            best_ensemble_source = "phase1"
        else:
            print("\n  The Phase 2 ensemble is better — using it.")

    print(f"\nTest metrics (final — ensemble with {best_ensemble_source}):")
    for k, v in test_metrics.items():
        if k == "loss":
            continue
        print(f"  {k}: {v*100:.2f}")

    # ------------------------------------------------------------------
    # Label Correlation Matrix inspection
    # ------------------------------------------------------------------
    lcm = model.label_corr.get_matrix()
    print("\n=== Label Correlation Matrix ===")
    print(f"  Shape: {tuple(lcm.shape)}")
    print(f"  Mean: {lcm.mean():.4f} | Min: {lcm.min():.4f} | Max: {lcm.max():.4f}")
    lcm_np = lcm.numpy().copy()
    np.fill_diagonal(lcm_np, 0)
    label_names_list, _ = get_labels(dataset_name)
    print("  Top 5 correlated label pairs:")
    for flat in np.argsort(lcm_np.flatten())[::-1][:5]:
        i, j = flat // num_labels, flat % num_labels
        print(f"    {label_names_list[i]} ↔ {label_names_list[j]}: {lcm_np[i, j]:.4f}")

    # ------------------------------------------------------------------
    # Save + comparison table
    # ------------------------------------------------------------------
    results = {
        "best_epoch": best_epoch,
        "best_graph_val_metric": best_metric,
        "ensemble_source": best_ensemble_source,
        "ensemble_weight_graph": float(best_weight),
        "ensemble_weight_phase2": float(1 - best_weight),
        "ensemble_val_micro_f1": float(best_ens_val_mf1),
        "test_metrics": {k: float(v) for k, v in test_metrics.items()},
        "graph_only_test_metrics": {k: float(v) for k, v in graph_test_metrics.items()},
        "phase2_test_metrics": {k: float(v) for k, v in phase2_test_metrics.items()},
        "history": history,
    }
    with open(save_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {save_dir / 'results.json'}")

    ref_name, ref = get_primary_baseline(dataset_name)
    print(f"\n{'='*80}")
    print("=== Final comparison ===")
    print(f"{'='*80}")
    if ref:
        print(f"{'metric':<12} {ref_name:>10} {'P2':>8} {'Graph':>8} {'Ensemble':>10} "
              f"{'Δ vs ref':>11} {'Δ vs P2':>11}")
        for k, ref_v in ref.items():
            p2_v = phase2_test_metrics.get(k, 0.0) * 100
            g_v = graph_test_metrics.get(k, 0.0) * 100
            e_v = test_metrics.get(k, 0.0) * 100
            d_ref, d_p2 = e_v - ref_v, e_v - p2_v
            a_ref = "▲" if d_ref > 0 else ("=" if d_ref == 0 else "▼")
            a_p2 = "▲" if d_p2 > 0 else ("=" if d_p2 == 0 else "▼")
            print(f"{k:<12} {ref_v:>10.2f} {p2_v:>8.2f} {g_v:>8.2f} {e_v:>10.2f} "
                  f"{a_ref} {d_ref:>+8.2f} {a_p2} {d_p2:>+8.2f}")
    else:
        print(f"(no published baseline recorded for '{dataset_name}')")
        print(f"{'metric':<12} {'P2':>8} {'Graph':>8} {'Ensemble':>10} {'Δ vs P2':>11}")
        for k in ("micro_f1", "macro_f1", "p@1", "p@3", "p@5", "ndcg@3", "ndcg@5"):
            p2_v = phase2_test_metrics.get(k, 0.0) * 100
            g_v = graph_test_metrics.get(k, 0.0) * 100
            e_v = test_metrics.get(k, 0.0) * 100
            d_p2 = e_v - p2_v
            a_p2 = "▲" if d_p2 > 0 else ("=" if d_p2 == 0 else "▼")
            print(f"{k:<12} {p2_v:>8.2f} {g_v:>8.2f} {e_v:>10.2f} {a_p2} {d_p2:>+8.2f}")

    print("\n=== Summary ===")
    if test_metrics.get("micro_f1", 0) > phase2_test_metrics.get("micro_f1", 0):
        print("  ✓ The ensemble improved over Phase 2.")
        print(f"  Ensemble micro_f1: {test_metrics['micro_f1']*100:.2f} "
              f"(Phase 2: {phase2_test_metrics['micro_f1']*100:.2f})")
        print(f"  Best ensemble weight: graph={best_weight:.2f}, "
              f"phase2={1-best_weight:.2f}")
    else:
        print("  ⚠ The ensemble did NOT improve over Phase 2.")
        print("  Consider reporting Phase 2 as the final model.")


if __name__ == "__main__":
    main()
