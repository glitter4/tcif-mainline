"""Run a specified TCIF checkpoint with its fixed dataset readout."""

import csv
import json
from pathlib import Path

import numpy as np
import torch

from tcif.runtime import (
    arguments,
    load_config,
    seed_all,
    make_dataset,
    make_loader,
    make_model,
    forward_batch,
    load_weights,
)
from tcif.readout import predict


def metrics(labels, predictions, dataset):
    from sklearn.metrics import f1_score

    truth = np.asarray(labels, dtype=np.float64)
    scores = np.asarray(predictions, dtype=np.float64)
    nonzero = truth != 0
    if dataset == "chsims":
        boundaries = np.array([-0.7, -0.1, 0.1, 0.7], dtype=np.float32)
        actual = np.searchsorted(
            boundaries, truth.astype(np.float32).clip(-1, 1), side="left"
        )
        predicted = np.searchsorted(
            boundaries, scores.astype(np.float32).clip(-1, 1), side="left"
        )
        accuracy_name = "Acc5"
    else:
        actual = np.rint(truth.clip(-3, 3))
        predicted = np.rint(scores.clip(-3, 3))
        accuracy_name = "Acc7"
    return {
        accuracy_name: float(100 * np.mean(actual == predicted)),
        "MAE": float(np.abs(truth - scores).mean()),
        "Acc2_nonzero": float(
            100 * np.mean((truth[nonzero] > 0) == (scores[nonzero] > 0))
        ),
        "F1_weighted_nonzero": float(
            100 * f1_score(truth[nonzero] > 0, scores[nonzero] > 0, average="weighted")
        ),
        "samples": len(truth),
    }


def main():
    parser = arguments(__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", type=int)
    cli = parser.parse_args()
    if cli.output.exists():
        raise FileExistsError(cli.output)
    config = load_config(cli)
    seed_all(config.seed, config.deterministic)
    device = torch.device(cli.device)
    dataset = make_dataset(config, cli.split)
    loader = make_loader(dataset, config, batch_size=cli.batch_size)
    model = make_model(config, device)
    load_weights(model, cli.checkpoint)
    model.eval()
    rows = []
    with torch.inference_mode():
        for batch in loader:
            predictions = (
                predict(forward_batch(model, batch, device), config).cpu().tolist()
            )
            rows.extend(zip(batch["id"], batch["raw_valence"].tolist(), predictions))
    cli.output.parent.mkdir(parents=True, exist_ok=True)
    with cli.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["id", "label", "prediction"])
        writer.writerows(rows)
    print(
        json.dumps(
            metrics([r[1] for r in rows], [r[2] for r in rows], config.dataset),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
