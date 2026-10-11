import torch
import torch.nn.functional as F

CLS7_CENTERS = (-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0)
CHSIMS5_EXPECTED_CENTERS = (-0.9, -0.4, 0.0, 0.4, 0.9)
CHSIMS5_CLASS_BOUNDARIES = (-0.7, -0.1, 0.1, 0.7)
NEUTRAL_ACC7_MAX_ABS = 0.5


def _is_chsims5_config(num_classes: int, label_min: float, label_max: float) -> bool:
    return (
        int(num_classes) == 5
        and abs(float(label_min) + 1.0) < 1e-06
        and (abs(float(label_max) - 1.0) < 1e-06)
    )


def get_cls7_centers(
    device=None,
    dtype=torch.float32,
    num_classes: int = 7,
    label_min: float = -3.0,
    label_max: float = 3.0,
):
    num_classes = int(num_classes)
    label_min = float(label_min)
    label_max = float(label_max)
    if num_classes <= 1:
        raise ValueError(f"num_classes must be > 1, got {num_classes}")
    if label_max <= label_min:
        raise ValueError(f"label_max must be > label_min, got {label_min}..{label_max}")
    if num_classes == 7 and label_min == -3.0 and (label_max == 3.0):
        values = CLS7_CENTERS
    elif _is_chsims5_config(num_classes, label_min, label_max):
        values = CHSIMS5_EXPECTED_CENTERS
    else:
        return torch.linspace(
            label_min, label_max, steps=num_classes, device=device, dtype=dtype
        )
    return torch.tensor(values, device=device, dtype=dtype)


def continuous_to_cls7_hard(
    y: torch.Tensor, centers: torch.Tensor = None
) -> torch.Tensor:
    y = torch.as_tensor(y)
    centers = (
        get_cls7_centers(device=y.device, dtype=y.dtype)
        if centers is None
        else centers.to(device=y.device, dtype=y.dtype)
    )
    if centers.ndim != 1 or centers.numel() <= 1:
        raise ValueError(
            f"centers must be a 1D tensor with at least 2 values, got {tuple(centers.shape)}"
        )
    chsims5_centers = torch.tensor(
        CHSIMS5_EXPECTED_CENTERS, device=centers.device, dtype=centers.dtype
    )
    if centers.numel() == 5 and bool(
        torch.allclose(centers, chsims5_centers, atol=1e-06, rtol=1e-06)
    ):
        boundaries = torch.tensor(
            CHSIMS5_CLASS_BOUNDARIES, device=y.device, dtype=y.dtype
        )
        return torch.bucketize(y.clamp(-1.0, 1.0), boundaries, right=False).long()
    if centers.numel() == 7 and bool(
        torch.allclose(
            centers,
            torch.tensor(CLS7_CENTERS, device=centers.device, dtype=centers.dtype),
            atol=1e-06,
            rtol=1e-06,
        )
    ):
        rounded = torch.where(y >= 0, torch.floor(y + 0.5), torch.ceil(y - 0.5))
        return rounded.clamp(-3, 3).long() + 3
    distances = torch.abs(y.reshape(-1, 1) - centers.reshape(1, -1))
    return torch.argmin(distances, dim=-1).long()


def continuous_to_cls7_soft(
    y: torch.Tensor, centers: torch.Tensor = None, tau: float = 0.5
) -> torch.Tensor:
    if tau <= 0:
        raise ValueError(f"tau must be positive, got {tau}")
    y = torch.as_tensor(y)
    centers = (
        get_cls7_centers(device=y.device, dtype=y.dtype)
        if centers is None
        else centers.to(device=y.device, dtype=y.dtype)
    )
    distances = torch.abs(y.reshape(-1, 1) - centers.reshape(1, -1))
    logits = -distances / float(tau)
    return torch.softmax(logits, dim=-1)


def soft_cross_entropy(
    logits: torch.Tensor, target_probs: torch.Tensor
) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=-1)
    return -(target_probs.to(dtype=log_probs.dtype) * log_probs).sum(dim=-1).mean()


def cls7_expected_value_from_probs(
    probs: torch.Tensor, centers: torch.Tensor = None
) -> torch.Tensor:
    centers = (
        get_cls7_centers(device=probs.device, dtype=probs.dtype)
        if centers is None
        else centers.to(device=probs.device, dtype=probs.dtype)
    )
    return (probs * centers.reshape(1, -1)).sum(dim=-1)


def cls7_expected_value(
    logits: torch.Tensor, centers: torch.Tensor = None
) -> torch.Tensor:
    centers = (
        get_cls7_centers(device=logits.device, dtype=logits.dtype)
        if centers is None
        else centers.to(device=logits.device, dtype=logits.dtype)
    )
    probs = torch.softmax(logits, dim=-1)
    return cls7_expected_value_from_probs(probs, centers=centers)


def compute_final_prediction(y_reg, cls7_logits, eta=0.8, centers=None):
    centers = (
        get_cls7_centers(device=y_reg.device, dtype=y_reg.dtype)
        if centers is None
        else centers
    )
    return (1.0 - eta) * y_reg + eta * cls7_expected_value(cls7_logits, centers)


def predict(output, config):
    regression = output["y_reg"].clamp(config.label_min, config.label_max)
    logits = output["cls7_logits"]
    centers = get_cls7_centers(
        device=logits.device,
        dtype=logits.dtype,
        num_classes=config.signed_class_count,
        label_min=config.label_min,
        label_max=config.label_max,
    )
    if config.readout == "argmax":
        classification = centers[logits.argmax(dim=-1)]
    else:
        classification = cls7_expected_value(logits, centers)
    return 0.2 * regression + 0.8 * classification
