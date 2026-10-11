"""Train one configured TCIF model and save its specified epoch."""

import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

from tcif.runtime import (
    arguments,
    load_config,
    seed_all,
    make_dataset,
    make_loader,
    make_model,
    forward_batch,
)
from tcif.readout import continuous_to_cls7_hard, continuous_to_cls7_soft
from tcif.training import (
    _signed_centers,
    compute_regression_loss,
    _compute_tcif_context_aux_loss,
    _compute_tcif_transition_gate_loss,
    _build_optimizer_param_groups,
    _set_bert_last_layers_requires_grad,
    _set_bert_last_layers_train_mode,
)


def training_mode(model, config):
    for backbone in (model.vit, model.bert, model.hubert):
        backbone.requires_grad_(False)
    if config.full_text_encoder:
        model.bert.requires_grad_(True)
    else:
        _set_bert_last_layers_requires_grad(
            model, config.unfreeze_bert_last_n_layers, True
        )
    model.train()
    model.vit.eval()
    model.hubert.eval()
    model.bert.eval()
    if config.full_text_encoder:
        model.bert.train()
    else:
        _set_bert_last_layers_train_mode(
            model, config.unfreeze_bert_last_n_layers, True
        )


def scheduler_for(optimizer, config):
    if config.scheduler == "none":
        return None
    if config.scheduler != "cosine":
        raise ValueError(config.scheduler)
    warmup = config.lr_warmup_epochs

    def factor(index):
        if index < warmup:
            progress = float(index + 1) / float(warmup)
            return (
                config.lr_warmup_start_factor
                + (1.0 - config.lr_warmup_start_factor) * progress
            )
        decay_index = max(0, index - warmup + 1) if warmup else index
        span = max(1, config.epochs - warmup) if warmup else max(1, config.epochs - 1)
        progress = min(1.0, float(decay_index) / float(span))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return config.lr_min_factor + (1.0 - config.lr_min_factor) * cosine

    return torch.optim.lr_scheduler.LambdaLR(
        optimizer, [factor for _ in optimizer.param_groups]
    )


def loss_for(output, batch, config, device):
    target = batch["raw_valence"].to(device)
    centers = _signed_centers(config, device=device, dtype=target.dtype)
    regression_type = (
        "l1" if config.center_reg_loss_type == "l1" else config.reg_loss_type
    )
    regression = compute_regression_loss(output["y_reg"], target, regression_type)
    logits = output["cls7_logits"]
    if config.cls7_loss_type == "hard_ce":
        classification = F.cross_entropy(
            logits, continuous_to_cls7_hard(target, centers), reduction="none"
        ).mean()
    else:
        probabilities = continuous_to_cls7_soft(target, centers, config.cls7_soft_tau)
        classification = (
            -(probabilities * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
        )
    context = _compute_tcif_context_aux_loss(output, target, config)[0]
    gate = _compute_tcif_transition_gate_loss(output, batch, target, config)[0]
    return (
        regression
        + config.cls7_loss_weight * classification
        + config.tcif_context_aux_weight * context
        + config.tcif_transition_gate_loss_weight * gate
    )


def main():
    parser = arguments(__doc__)
    parser.add_argument("--output", required=True, type=Path)
    cli = parser.parse_args()
    if cli.output.exists():
        raise FileExistsError(cli.output)
    config = load_config(cli)
    if not 1 <= config.save_epoch <= config.epochs:
        raise ValueError("save_epoch must be within the configured training horizon")
    seed_all(config.seed, config.deterministic)
    device = torch.device(cli.device)
    dataset = make_dataset(config, "train")
    loader = make_loader(dataset, config, training=True)
    model = make_model(config, device)
    optimizer = torch.optim.AdamW(
        _build_optimizer_param_groups(model, None, config),
        weight_decay=config.weight_decay,
    )
    scheduler = scheduler_for(optimizer, config)
    neighbor_generator = torch.Generator().manual_seed(config.seed + 2718)
    for epoch in range(config.save_epoch):
        training_mode(model, config)
        optimizer.zero_grad()
        total = 0.0
        for step, batch in enumerate(loader):
            if config.neighbor_dropout:
                mask = batch["tcif_context_valid_mask"].bool()
                sample = torch.rand(
                    mask.shape, generator=neighbor_generator, device="cpu"
                )
                batch["tcif_context_valid_mask"] = mask & (
                    sample >= config.neighbor_dropout
                )
            output = forward_batch(model, batch, device)
            loss = loss_for(output, batch, config, device)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Nonfinite training loss at epoch {epoch + 1}, step {step + 1}"
                )
            (loss / config.grad_accum_steps).backward()
            if (step + 1) % config.grad_accum_steps == 0 or step + 1 == len(loader):
                clip_grad_norm_(
                    [p for group in optimizer.param_groups for p in group["params"]],
                    config.clip_grad_norm,
                )
                optimizer.step()
                optimizer.zero_grad()
            total += loss.item()
        print(
            f"Epoch {epoch + 1}/{config.save_epoch}: loss={total / len(loader):.6f}",
            flush=True,
        )
        if scheduler is not None:
            scheduler.step()
    cli.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), cli.output)
    print(f"Saved epoch {config.save_epoch}: {cli.output}", flush=True)


if __name__ == "__main__":
    main()
