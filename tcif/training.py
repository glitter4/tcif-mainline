import torch
import torch.nn.functional as F
from tcif.readout import get_cls7_centers, continuous_to_cls7_hard


def _signed_centers(args, device=None, dtype=torch.float32):
    return get_cls7_centers(
        device=device,
        dtype=dtype,
        num_classes=int(getattr(args, "signed_class_count", 7)),
        label_min=float(getattr(args, "label_min", -3.0)),
        label_max=float(getattr(args, "label_max", 3.0)),
    )


def compute_intensity_loss(prediction, target, loss_name: str):
    if loss_name == "mse":
        return F.mse_loss(prediction, target, reduction="mean")
    if loss_name == "smooth_l1":
        return F.smooth_l1_loss(prediction, target, reduction="mean")
    raise ValueError(f"Unsupported intensity loss type: {loss_name}")


def compute_regression_loss(prediction, target, loss_name: str):
    if loss_name == "l1":
        return F.l1_loss(prediction, target, reduction="mean")
    return compute_intensity_loss(prediction, target, loss_name)


def _compute_tcif_context_aux_loss(model_out, raw_valence, args):
    zero = raw_valence.new_zeros(())
    tcif_payload = model_out.get("extras", {}).get("tcif")
    if not isinstance(tcif_payload, dict):
        return (zero, zero, zero, 0)
    has_context = tcif_payload.get("has_context")
    context_y_reg = tcif_payload.get("context_y_reg")
    context_cls7_logits = tcif_payload.get("context_cls7_logits")
    if has_context is None or context_y_reg is None or context_cls7_logits is None:
        raise RuntimeError("Incomplete TCIF context auxiliary payload")
    has_context = has_context.to(device=raw_valence.device, dtype=torch.bool)
    valid_count = int(has_context.sum().item())
    if valid_count <= 0:
        return (zero, zero, zero, 0)
    context_reg_loss = compute_regression_loss(
        context_y_reg[has_context], raw_valence[has_context], args.reg_loss_type
    )
    context_cls7_target = continuous_to_cls7_hard(
        raw_valence[has_context],
        centers=_signed_centers(
            args, device=raw_valence.device, dtype=raw_valence.dtype
        ),
    )
    context_cls7_loss = F.cross_entropy(
        context_cls7_logits[has_context], context_cls7_target
    )
    context_total = (
        context_reg_loss + float(args.tcif_context_aux_cls7_weight) * context_cls7_loss
    )
    return (context_total, context_reg_loss, context_cls7_loss, valid_count)


def _compute_tcif_transition_gate_loss(model_out, batch, raw_valence, args):
    zero = raw_valence.new_zeros(())
    empty_stats = {
        "valid_count": 0,
        "sign_conflict_count": 0,
        "target_mean": 0.0,
        "regression_gate_mean": 0.0,
        "ordinal_gate_mean": 0.0,
        "transition_distance_mean": 0.0,
    }
    tcif_payload = model_out.get("extras", {}).get("tcif")
    if not isinstance(tcif_payload, dict):
        return (zero, empty_stats)
    context_values = batch.get("tcif_context_raw_valence")
    context_valid_mask = batch.get("tcif_context_valid_mask")
    if context_values is None or context_valid_mask is None:
        raise RuntimeError(
            "TCIF transition-gate supervision requires context raw valence and mask"
        )
    context_values = context_values.to(
        device=raw_valence.device, dtype=raw_valence.dtype
    )
    context_valid_mask = context_valid_mask.to(
        device=raw_valence.device, dtype=torch.bool
    )
    valid_sample = context_valid_mask.any(dim=1)
    valid_count = int(valid_sample.sum().item())
    if valid_count <= 0:
        return (zero, empty_stats)
    mask_float = context_valid_mask.to(dtype=raw_valence.dtype)
    context_mean = (context_values * mask_float).sum(dim=1) / mask_float.sum(
        dim=1
    ).clamp_min(1.0)
    transition_distance = (raw_valence - context_mean).abs()
    target = torch.exp(-transition_distance / float(args.tcif_transition_gate_tau))
    center_nonzero = raw_valence.abs() > 1e-12
    context_nonzero = context_values.abs() > 1e-12
    opposite_sign = torch.sign(context_values) != torch.sign(raw_valence).unsqueeze(1)
    sign_conflict = center_nonzero & (
        context_valid_mask & context_nonzero & opposite_sign
    ).any(dim=1)
    target = torch.where(
        sign_conflict,
        target.new_full(target.shape, float(args.tcif_transition_gate_conflict_target)),
        target,
    ).detach()
    regression_logits = tcif_payload.get("regression_continuation_gate_logits")
    ordinal_logits = tcif_payload.get("ordinal_continuation_gate_logits")
    regression_gate = tcif_payload.get("regression_continuation_gate")
    ordinal_gate = tcif_payload.get("ordinal_continuation_gate")
    if any(
        (
            value is None
            for value in (
                regression_logits,
                ordinal_logits,
                regression_gate,
                ordinal_gate,
            )
        )
    ):
        raise RuntimeError("Incomplete TCIF transition-gate payload")
    regression_loss = F.binary_cross_entropy_with_logits(
        regression_logits[valid_sample],
        target[valid_sample].to(dtype=regression_logits.dtype),
    )
    ordinal_loss = F.binary_cross_entropy_with_logits(
        ordinal_logits[valid_sample],
        target[valid_sample].to(dtype=ordinal_logits.dtype),
    )
    loss = 0.5 * (regression_loss + ordinal_loss)
    stats = {
        "valid_count": valid_count,
        "sign_conflict_count": int((sign_conflict & valid_sample).sum().item()),
        "target_mean": float(target[valid_sample].mean().detach().item()),
        "regression_gate_mean": float(
            regression_gate[valid_sample].mean().detach().item()
        ),
        "ordinal_gate_mean": float(ordinal_gate[valid_sample].mean().detach().item()),
        "transition_distance_mean": float(
            transition_distance[valid_sample].mean().detach().item()
        ),
    }
    return (loss, stats)


def _unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def _embedding_cache_kwargs(batch, device, use_text_cache=True):
    kwargs = {}
    for batch_key, model_key in (
        ("cached_vision", "cached_vision"),
        ("cached_text", "cached_text"),
        ("cached_audio", "cached_audio"),
        ("cached_text_mask", "cached_text_mask"),
        ("cached_audio_mask", "cached_audio_mask"),
    ):
        if not use_text_cache and batch_key in {"cached_text", "cached_text_mask"}:
            continue
        value = batch.get(batch_key)
        if value is not None:
            kwargs[model_key] = value.to(device, non_blocking=True)
    return kwargs


def _tcif_context_kwargs(batch, device, use_text_cache=True):
    kwargs = {}
    for batch_key, model_key in (
        ("tcif_context_image", "tcif_context_images"),
        ("tcif_context_input_ids", "tcif_context_input_ids"),
        ("tcif_context_attention_mask", "tcif_context_attention_mask"),
        ("tcif_context_audio_values", "tcif_context_audio_values"),
        ("tcif_context_audio_attention_mask", "tcif_context_audio_attention_mask"),
        ("tcif_context_cached_vision", "tcif_context_cached_vision"),
        ("tcif_context_cached_text", "tcif_context_cached_text"),
        ("tcif_context_cached_audio", "tcif_context_cached_audio"),
        ("tcif_context_cached_text_mask", "tcif_context_cached_text_mask"),
        ("tcif_context_cached_audio_mask", "tcif_context_cached_audio_mask"),
        ("tcif_context_valid_mask", "tcif_context_valid_mask"),
        ("tcif_context_relative_pos", "tcif_context_relative_pos"),
    ):
        if not use_text_cache and batch_key in {
            "tcif_context_cached_text",
            "tcif_context_cached_text_mask",
        }:
            continue
        value = batch.get(batch_key)
        if value is not None:
            kwargs[model_key] = value.to(device, non_blocking=True)
    return kwargs


def _get_bert_encoder_layers(model):
    base_model = _unwrap_model(model)
    bert = getattr(base_model, "bert", None)
    if bert is None:
        return []
    candidate_paths = (("encoder", "layer"), ("transformer", "layer"))
    for parent_attr, layer_attr in candidate_paths:
        parent = getattr(bert, parent_attr, None)
        layers = getattr(parent, layer_attr, None) if parent is not None else None
        if layers is not None and len(layers) > 0:
            return list(layers)
    return []


def _get_bert_last_layer_modules(model, layer_count):
    layers = _get_bert_encoder_layers(model)
    if not layers:
        return []
    layer_count = max(0, min(int(layer_count), len(layers)))
    if layer_count <= 0:
        return []
    return layers[-layer_count:]


def _set_bert_last_layers_requires_grad(model, layer_count, requires_grad):
    layers = _get_bert_last_layer_modules(model, layer_count)
    if not layers:
        return 0
    param_count = 0
    for layer in layers:
        for p in layer.parameters():
            p.requires_grad = requires_grad
            param_count += p.numel()
    return param_count


def _set_bert_last_layers_train_mode(model, layer_count, training):
    layers = _get_bert_last_layer_modules(model, layer_count)
    for layer in layers:
        layer.train(training)
    return bool(layers)


def _is_tcif_transition_gate_parameter(name):
    return name.startswith(
        ("tcif_regression.transition_gate.", "tcif_ordinal.transition_gate.")
    )


def _build_optimizer_param_groups(model, contrast_head, args):
    """Partition parameters exactly once and isolate TCIF transition gates."""
    bert_last_layer_params = []
    bert_last_layer_param_ids = set()
    if getattr(args, "full_text_encoder", False):
        bert_last_layer_params = list(_unwrap_model(model).bert.parameters())
        bert_last_layer_param_ids = {id(p) for p in bert_last_layer_params}
    elif args.unfreeze_bert_last_n_layers > 0:
        bert_last_layers = _get_bert_last_layer_modules(
            model, args.unfreeze_bert_last_n_layers
        )
        if not bert_last_layers:
            raise ValueError(
                "--unfreeze_bert_last_n_layers was set, but the text backbone exposes no supported encoder layers"
            )
        if len(bert_last_layers) != args.unfreeze_bert_last_n_layers:
            raise ValueError(
                f"Requested {args.unfreeze_bert_last_n_layers} BERT layers, but only found {len(bert_last_layers)}"
            )
        bert_last_layer_params = [
            p for layer in bert_last_layers for p in layer.parameters()
        ]
        bert_last_layer_param_ids = {id(p) for p in bert_last_layer_params}
    buckets = {
        "head": [],
        "tcif_transition_gate": [],
        "router": [],
        "bert_last_layers": bert_last_layer_params,
        "backbone": [],
    }
    parameter_names = {name: p for name, p in model.named_parameters()}
    for name, parameter in parameter_names.items():
        if id(parameter) in bert_last_layer_param_ids:
            continue
        if _is_tcif_transition_gate_parameter(name):
            buckets["tcif_transition_gate"].append(parameter)
        elif (
            ".phi" in name
            or ".scale" in name
            or name.startswith("fusion_task2_text_router.")
        ):
            buckets["router"].append(parameter)
        elif name.startswith(("vit.", "bert.", "hubert.", "resnet.")):
            buckets["backbone"].append(parameter)
        else:
            buckets["head"].append(parameter)
    if contrast_head is not None:
        buckets["head"].extend(list(contrast_head.parameters()))
    learning_rates = {
        "head": args.lr,
        "tcif_transition_gate": args.tcif_transition_gate_lr,
        "router": args.router_lr,
        "bert_last_layers": args.text_backbone_lr
        if getattr(args, "full_text_encoder", False)
        else args.lr * args.bert_last_layer_lr_ratio,
        "backbone": args.lr * args.backbone_lr_ratio,
    }
    param_groups = [
        {"name": name, "params": params, "lr": learning_rates[name]}
        for name, params in buckets.items()
        if params
    ]
    expected_ids = {id(parameter) for parameter in model.parameters()}
    if contrast_head is not None:
        expected_ids.update((id(parameter) for parameter in contrast_head.parameters()))
    grouped_ids = [
        id(parameter) for group in param_groups for parameter in group["params"]
    ]
    if len(grouped_ids) != len(set(grouped_ids)):
        raise RuntimeError("optimizer parameter groups overlap")
    if set(grouped_ids) != expected_ids:
        raise RuntimeError("optimizer parameter groups do not cover the parameter set")
    return param_groups
