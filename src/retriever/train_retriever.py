"""
Trains the Stage-1 BiEncoder with the Margin-Penalized InfoNCE loss.

A "batch" here is B (query, fresh_doc, stale_doc) triples. Within a batch,
each row's random in-batch negatives are automatically the *other* rows'
fresh_docs (handled inside margin_penalized_infonce via the off-diagonal of
the query-vs-positives similarity matrix) -- nothing extra needs to be
sampled or stored for that part of the loss.

Model selection uses ood_val (a domain never seen in train_domains) rather
than in_domain_val, since the point of this project is genuine temporal
generalization, not just fitting the training domains' templates.

Usage:
    python -m src.retriever.train_retriever [--config configs/stage1_retriever.yaml]
    python -m src.retriever.train_retriever --max_epochs 5   # quick smoke run
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import yaml
from torch.optim import AdamW

from src.retriever.losses import alignment, margin_penalized_infonce, uniformity
from src.retriever.model import BiEncoder
from src.utils.seed import set_seed

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_records(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def batches(records: list[dict], batch_size: int, rng: torch.Generator) -> list[list[dict]]:
    idx = torch.randperm(len(records), generator=rng).tolist()
    shuffled = [records[i] for i in idx]
    return [shuffled[i:i + batch_size] for i in range(0, len(shuffled), batch_size)]


@torch.no_grad()
def evaluate_pairwise(model: BiEncoder, records: list[dict], device: torch.device) -> dict:
    """Cheap in-training-loop metric: for each record, does the model prefer
    its own fresh_doc over its own stale_doc? Also reports alignment/uniformity
    on this split. (Full corpus-level MRR@1 lives in eval_retriever.py.)"""
    model.eval()
    q = model.encode([r["query"] for r in records], device)
    pos = model.encode([r["fresh_doc"] for r in records], device)
    hard = model.encode([r["stale_doc"] for r in records], device)

    sim_pos = (q * pos).sum(dim=1)
    sim_hard = (q * hard).sum(dim=1)
    pairwise_acc = (sim_pos > sim_hard).float().mean().item()

    return {
        "pairwise_acc": pairwise_acc,
        "alignment": alignment(q, pos).item(),
        "uniformity": uniformity(pos).item(),
        "mean_sim_pos": sim_pos.mean().item(),
        "mean_sim_hard": sim_hard.mean().item(),
    }


def train(cfg: dict) -> None:
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data_dir = REPO_ROOT / cfg["data_dir"]
    train_records = load_records(data_dir / "train.jsonl")
    in_domain_val_records = load_records(data_dir / "in_domain_val.jsonl")
    ood_val_records = load_records(data_dir / "ood_val.jsonl")
    print(f"train={len(train_records)}  in_domain_val={len(in_domain_val_records)}  ood_val={len(ood_val_records)}")

    model = BiEncoder(cfg["model_name"]).to(device)
    optimizer = AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    rng = torch.Generator().manual_seed(cfg["seed"])

    checkpoint_dir = REPO_ROOT / cfg["checkpoint_dir"]
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_dir = REPO_ROOT / cfg["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)

    # Baseline (pre-training) metrics -- the number Stage-1 training must beat.
    baseline_metrics = evaluate_pairwise(model, ood_val_records, device)
    print(f"[epoch 0 / untrained baseline] ood_val: {baseline_metrics}")

    best_ood_acc = baseline_metrics["pairwise_acc"]
    epochs_without_improvement = 0
    history = [{"epoch": 0, "train_loss": None, "ood_val": baseline_metrics}]

    accum_steps = cfg["accum_steps"]

    for epoch in range(1, cfg["max_epochs"] + 1):
        model.train()
        epoch_start = time.time()
        epoch_loss_sum, n_batches = 0.0, 0

        optimizer.zero_grad()
        for step, batch in enumerate(batches(train_records, cfg["batch_size"], rng), start=1):
            q = model.encode([r["query"] for r in batch], device, cfg["max_seq_length"])
            pos = model.encode([r["fresh_doc"] for r in batch], device, cfg["max_seq_length"])
            hard = model.encode([r["stale_doc"] for r in batch], device, cfg["max_seq_length"])

            loss = margin_penalized_infonce(q, pos, hard, tau=cfg["tau"], gamma=cfg["gamma"])
            (loss / accum_steps).backward()

            if step % accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            epoch_loss_sum += loss.item()
            n_batches += 1

        mean_train_loss = epoch_loss_sum / max(n_batches, 1)

        if epoch % cfg["eval_every"] == 0:
            id_val_metrics = evaluate_pairwise(model, in_domain_val_records, device)
            ood_val_metrics = evaluate_pairwise(model, ood_val_records, device)
            elapsed = time.time() - epoch_start
            print(
                f"[epoch {epoch}] train_loss={mean_train_loss:.4f}  "
                f"in_domain_val_acc={id_val_metrics['pairwise_acc']:.3f}  "
                f"ood_val_acc={ood_val_metrics['pairwise_acc']:.3f}  "
                f"({elapsed:.1f}s)"
            )
            history.append({
                "epoch": epoch,
                "train_loss": mean_train_loss,
                "in_domain_val": id_val_metrics,
                "ood_val": ood_val_metrics,
            })

            if ood_val_metrics["pairwise_acc"] > best_ood_acc:
                best_ood_acc = ood_val_metrics["pairwise_acc"]
                epochs_without_improvement = 0
                torch.save(model.state_dict(), checkpoint_dir / "best.pt")
                print(f"  -> new best ood_val_acc={best_ood_acc:.3f}, checkpoint saved")
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= cfg["patience"]:
                    print(f"Early stopping at epoch {epoch} (no ood_val improvement for {cfg['patience']} evals)")
                    break

    torch.save(model.state_dict(), checkpoint_dir / "last.pt")
    with open(results_dir / "retriever_train_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"Done. best ood_val_acc={best_ood_acc:.3f}")
    print(f"Checkpoints: {checkpoint_dir}/best.pt (best), {checkpoint_dir}/last.pt (final)")
    print(f"History: {results_dir}/retriever_train_history.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stage1_retriever.yaml")
    # Allow quick CLI overrides for smoke-testing without editing the yaml.
    parser.add_argument("--max_epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    args = parser.parse_args()

    with open(REPO_ROOT / args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if args.max_epochs is not None:
        cfg["max_epochs"] = args.max_epochs
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size

    train(cfg)


if __name__ == "__main__":
    main()
