"""Construct the model and data pipeline from one dataset configuration."""

import argparse
import inspect
import json
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from tcif.data import (
    CMUMOSEIProcessDataset,
    CMUMOSIProcessDataset,
    CHSIMSProcessDataset,
)
from tcif.model import EmotionM4OE
from tcif.training import _embedding_cache_kwargs, _tcif_context_kwargs


def arguments(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--cache-root", required=True, type=Path)
    parser.add_argument("--text-model", required=True, type=Path)
    parser.add_argument("--vision-model", required=True, type=Path)
    parser.add_argument("--audio-model", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    return parser


def load_config(cli):
    values = json.loads(cli.config.read_text(encoding="utf-8"))
    values.update(
        dataset_root=str(cli.data_root),
        embedding_cache_root=str(cli.cache_root),
        tokenizer_path=str(cli.text_model),
        bert_model_path=str(cli.text_model),
        vit_model_path=str(cli.vision_model),
        hubert_model_path=str(cli.audio_model),
    )
    return SimpleNamespace(**values)


def seed_all(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic, warn_only=True)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def make_dataset(config, split):
    cls = {
        "cmumosei": CMUMOSEIProcessDataset,
        "cmumosi": CMUMOSIProcessDataset,
        "chsims": CHSIMSProcessDataset,
    }[config.dataset]
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    return cls(
        root=config.dataset_root,
        split=split,
        transform=transform,
        tokenizer_path=config.tokenizer_path,
        local_files_only=True,
        max_length=config.max_length,
        num_frames=config.num_frames,
        vit_context_ratio=config.vit_context_ratio,
        embedding_cache_root=config.embedding_cache_root,
        temporal_context_radius=config.tcif_context_radius,
        temporal_context_mode="neighbors",
        temporal_context_seed=config.seed,
    )


def make_loader(dataset, config, training=False, batch_size=None):
    generator = torch.Generator().manual_seed(
        config.seed if training else config.seed + 2
    )
    return DataLoader(
        dataset,
        batch_size=batch_size or config.batch_size,
        shuffle=training,
        num_workers=config.num_workers,
        worker_init_fn=seed_worker,
        generator=generator,
    )


def make_model(config, device):
    options = dict(vars(config))
    options.update(
        num_classes=2,
        num_aux_classes=2,
        depth_msoe=2,
        depth_mtoe=1,
        output_head_mode="signed_reg_cls7",
        cls7_head_type="flat",
        enable_text_shared_experts=False,
        enable_tcif=True,
        tcif_enable_transition_gate=True,
        final_pred_eta=0.8,
        enable_expert_load_output=False,
        local_files_only=True,
    )
    parameters = inspect.signature(EmotionM4OE).parameters
    return EmotionM4OE(**{k: v for k, v in options.items() if k in parameters}).to(
        device
    )


def forward_batch(model, batch, device):
    return model(
        batch["image"].to(device),
        batch["input_ids"].to(device),
        batch["attention_mask"].to(device),
        batch["audio_values"].to(device),
        batch["audio_attention_mask"].to(device),
        **_embedding_cache_kwargs(batch, device, use_text_cache=False),
        **_tcif_context_kwargs(batch, device, use_text_cache=False),
    )


def load_weights(model, path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    elif isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state, strict=True)
