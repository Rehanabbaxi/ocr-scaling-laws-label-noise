"""Spec sections 11-14 - training loop, evaluation, results logging.

One call to `train()` is one point on the scaling curve. It writes its own
result file under results/<run_name>.json; nothing appends to a shared CSV,
because two people running experiments on branches would conflict on every
merge. `aggregate_results()` collects the per-run files into one table.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import CONFIG, PATHS, build_run_name, make_dirs, set_seeds  # noqa: E402
from data import load_manifest, split_by_page, split_summary, subsample_train  # noqa: E402
from dataset import make_loader  # noqa: E402
from metrics import evaluate_strings, greedy_decode  # noqa: E402
from model import build_model, count_parameters  # noqa: E402
from noise import inject_noise, measured_corruption, save_corruption_artifacts  # noqa: E402
from vocab import build_vocab  # noqa: E402


def prepare_run(cfg: dict[str, Any]) -> dict[str, Any]:
    """Manifest -> splits, vocabulary and the (possibly corrupted) training set."""
    df = load_manifest(cfg["dataset"])
    splits = split_by_page(df, cfg["val_split"], cfg["test_split"], cfg["seed"])

    # The vocabulary is built from the FULL training split, before subsampling.
    #
    # Building it from the subsample would be defensible in isolation, but it
    # would make num_classes - and therefore the width of the output layer - a
    # function of data_fraction. That turns the scaling axis into two variables
    # at once. The alphabet is a property of the script, not of the sample, so
    # it is held fixed across every run and characters absent from a small
    # subsample simply never receive gradient.
    vocab = build_vocab(splits["train"]["text"])

    train_clean = subsample_train(
        splits["train"], cfg["data_fraction"], cfg["max_train_lines"], cfg["seed"]
    )

    train_noisy, mask = inject_noise(
        train_clean,
        cfg["noise_type"],
        cfg["noise_rate"],
        cfg["noise_seed"],
        charset=vocab.chars,
    )
    corruption = measured_corruption(train_clean, train_noisy, mask)

    return {
        "splits": splits,
        "vocab": vocab,
        "train_clean": train_clean,
        "train_noisy": train_noisy,
        "noise_mask": mask,
        "corruption": corruption,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module, loader, vocab, device: torch.device, max_examples: int = 0
) -> dict[str, Any]:
    """Greedy-decode a whole split and score it. Labels here are always clean."""
    model.eval()
    predictions: list[str] = []
    references: list[str] = []

    for batch in loader:
        images = batch["images"].to(device, non_blocking=True)
        log_probs = model(images)
        decoded = greedy_decode(log_probs, batch["input_lengths"])
        predictions.extend(vocab.decode(seq) for seq in decoded)
        references.extend(batch["texts"])

    scores = evaluate_strings(predictions, references)
    if max_examples:
        scores["examples"] = [
            {"reference": r, "prediction": p}
            for r, p in list(zip(references, predictions))[:max_examples]
        ]
    return scores


def train(cfg: dict[str, Any] | None = None, verbose: bool = True) -> dict[str, Any]:
    """Run one experiment. Returns the result row and writes it to results/."""
    cfg = dict(cfg or CONFIG)
    cfg["run_name"] = cfg.get("run_name") or build_run_name(cfg)
    run_name = cfg["run_name"]

    make_dirs()
    set_seeds(cfg["seed"])

    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    if cfg["device"] == "cuda" and device.type != "cuda":
        raise RuntimeError(
            "CONFIG['device'] is 'cuda' but no CUDA device is visible. Refusing to "
            "train on CPU; set device='cpu' explicitly if that is really intended."
        )

    prep = prepare_run(cfg)
    vocab, splits = prep["vocab"], prep["splits"]
    lines_dir = PATHS["lines_dir"] / cfg["dataset"]

    train_loader = make_loader(
        prep["train_noisy"], lines_dir, vocab, cfg["batch_size"],
        shuffle=True, num_workers=cfg["num_workers"], seed=cfg["seed"],
    )
    val_loader = make_loader(
        splits["val"], lines_dir, vocab, cfg["batch_size"],
        shuffle=False, num_workers=cfg["num_workers"],
    )
    test_loader = make_loader(
        splits["test"], lines_dir, vocab, cfg["batch_size"],
        shuffle=False, num_workers=cfg["num_workers"],
    )

    model = build_model(vocab.num_classes, cfg).to(device)
    criterion = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])
        if cfg["scheduler"] == "cosine"
        else None
    )
    use_amp = bool(cfg.get("amp", False)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    if verbose:
        print(f"run          : {run_name}")
        print(f"device       : {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'})")
        print(f"parameters   : {count_parameters(model):,}")
        print(f"classes      : {vocab.num_classes} (C={vocab.n_chars}, V={vocab.size})")
        print(f"train lines  : {len(prep['train_noisy'])}  (of {len(splits['train'])} available)")
        print(f"val / test   : {len(splits['val'])} / {len(splits['test'])}")
        print(f"noise        : {cfg['noise_type']} @ {cfg['noise_rate']}  ->  "
              f"{prep['corruption']['frac_labels_actually_corrupted']:.3f} of lines, "
              f"{prep['corruption']['frac_chars_actually_corrupted']:.3f} of chars")
        print(f"amp          : {use_amp}\n")

    history: list[dict[str, Any]] = []
    best = {"cer": float("inf"), "epoch": -1}
    epochs_without_improvement = 0
    nonfinite_batches = 0
    ckpt_path = PATHS["checkpoints_dir"] / f"{run_name}_best.pt"
    started = time.time()

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        running, seen = 0.0, 0

        for batch in train_loader:
            images = batch["images"].to(device, non_blocking=True)
            targets = batch["targets"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                log_probs = model(images)
            loss = criterion(
                log_probs, targets, batch["input_lengths"], batch["target_lengths"]
            )

            if not torch.isfinite(loss):
                # zero_infinity masks per-sample infinities; a non-finite mean
                # means something worse, so skip rather than poison the weights.
                nonfinite_batches += 1
                continue

            scaler.scale(loss).backward()
            if cfg["grad_clip"]:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            scaler.step(optimizer)
            scaler.update()

            running += loss.item() * len(batch["texts"])
            seen += len(batch["texts"])

        if scheduler is not None:
            scheduler.step()

        train_loss = running / max(1, seen)
        val = evaluate(model, val_loader, vocab, device)
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_cer": val["cer"],
            "val_wer": val["wer"],
            "lr": optimizer.param_groups[0]["lr"],
        })

        improved = val["cer"] < best["cer"]
        if improved:
            best = {"cer": val["cer"], "wer": val["wer"], "epoch": epoch}
            epochs_without_improvement = 0
            torch.save(
                {"model": model.state_dict(), "epoch": epoch,
                 "val_cer": val["cer"], "config": cfg, "vocab_chars": vocab.chars},
                ckpt_path,
            )
        else:
            epochs_without_improvement += 1

        if verbose:
            print(f"epoch {epoch:3d}/{cfg['epochs']}  loss {train_loss:7.4f}  "
                  f"val CER {val['cer']:.4f}  val WER {val['wer']:.4f}"
                  f"{'  *' if improved else ''}")

        # Written every epoch so a Colab disconnect does not lose the curve.
        pd.DataFrame(history).to_csv(
            PATHS["results_dir"] / f"{run_name}_history.csv", index=False
        )

        if epochs_without_improvement >= cfg["early_stop_patience"]:
            if verbose:
                print(f"\nearly stop: no val CER improvement for "
                      f"{cfg['early_stop_patience']} epochs")
            break

    # Final scoring uses the best checkpoint, not the last epoch's weights.
    if ckpt_path.exists():
        model.load_state_dict(torch.load(ckpt_path, map_location=device)["model"])
    val_final = evaluate(model, val_loader, vocab, device, max_examples=10)
    test_final = evaluate(model, test_loader, vocab, device, max_examples=10)

    result = {
        "run_name": run_name,
        "dataset": cfg["dataset"],
        "data_fraction": cfg["data_fraction"],
        "noise_type": cfg["noise_type"],
        "noise_rate": cfg["noise_rate"],
        "seed": cfg["seed"],
        "noise_seed": cfg["noise_seed"],
        "n_train_lines": len(prep["train_noisy"]),
        "n_train_chars": int(prep["train_noisy"]["n_chars"].sum()),
        "n_val_lines": len(splits["val"]),
        "n_test_lines": len(splits["test"]),
        "num_classes": vocab.num_classes,
        "n_parameters": count_parameters(model),
        "epochs_run": len(history),
        "best_epoch": best["epoch"],
        "val_cer": val_final["cer"],
        "val_wer": val_final["wer"],
        "val_cer_per_line_mean": val_final["cer_per_line_mean"],
        "test_cer": test_final["cer"],
        "test_wer": test_final["wer"],
        "final_train_loss": history[-1]["train_loss"] if history else None,
        "nonfinite_batches": nonfinite_batches,
        "minutes": round((time.time() - started) / 60, 2),
        **{f"measured_{k}": v for k, v in prep["corruption"].items()},
    }

    (PATHS["results_dir"] / f"{run_name}.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (PATHS["results_dir"] / f"{run_name}_config.json").write_text(
        json.dumps(cfg, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    (PATHS["results_dir"] / f"{run_name}_examples.json").write_text(
        json.dumps(
            {"val": val_final.get("examples", []), "test": test_final.get("examples", [])},
            indent=2, ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    if cfg["noise_type"] != "none" and cfg["noise_rate"] > 0:
        save_corruption_artifacts(
            prep["train_clean"], prep["train_noisy"], prep["noise_mask"],
            PATHS["results_dir"], run_name,
        )

    if verbose:
        print(f"\nbest epoch {best['epoch']}  |  val CER {result['val_cer']:.4f}  "
              f"|  test CER {result['test_cer']:.4f}  |  {result['minutes']} min")
        print(f"result -> {PATHS['results_dir'] / f'{run_name}.json'}")

    return result


def aggregate_results(results_dir: Path | None = None) -> pd.DataFrame:
    """Collect every results/<run_name>.json into one table for curve fitting."""
    results_dir = Path(results_dir or PATHS["results_dir"])
    rows = [
        json.loads(p.read_text(encoding="utf-8"))
        for p in sorted(results_dir.glob("*.json"))
        if not p.stem.endswith(("_config", "_examples"))
    ]
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    sort_cols = [c for c in ("dataset", "noise_rate", "data_fraction", "seed") if c in df]
    return df.sort_values(sort_cols).reset_index(drop=True)


def main() -> None:
    from config import use_utf8_stdout

    use_utf8_stdout()
    train(CONFIG)


if __name__ == "__main__":
    main()
