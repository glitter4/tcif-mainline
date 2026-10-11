import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers.utils
import transformers.utils.import_utils as _import_utils


def _disable_sklearn():

    def _false():
        return False

    transformers.utils.is_sklearn_available = _false
    _import_utils.is_sklearn_available = _false


_disable_sklearn()
from transformers import (
    AutoModel,
    BertConfig,
    BertModel,
    ViTConfig,
    ViTModel,
    HubertConfig,
    HubertModel,
)
from tcif.readout import (
    cls7_expected_value_from_probs,
    compute_final_prediction,
    get_cls7_centers,
)
from tcif.filter import TCIF_OUTPUT_MODES, TemporalContextInnovationFilter


def softmax(x: torch.Tensor, dim) -> torch.Tensor:
    max_vals = torch.amax(x, dim=dim, keepdim=True)
    e_x = torch.exp(x - max_vals)
    sum_exp = e_x.sum(dim=dim, keepdim=True)
    return e_x / sum_exp


def _temporal_feature_mix(
    feat_task1: torch.Tensor, feat_task2: torch.Tensor, detach_task2: bool = False
) -> torch.Tensor:
    """Keep the temporal forward value fixed while optionally blocking task-2 gradients."""
    task2_source = feat_task2.detach() if detach_task2 else feat_task2
    return 0.5 * (feat_task1 + task2_source)


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features * 4
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TaskQueryPool(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        embed_dim = int(embed_dim)
        num_heads = int(num_heads)
        if embed_dim <= 0:
            raise ValueError(f"embed_dim must be positive, got {embed_dim}")
        if num_heads <= 0 or embed_dim % num_heads != 0:
            raise ValueError(
                f"num_heads must evenly divide embed_dim, got embed_dim={embed_dim}, num_heads={num_heads}"
            )
        self.query = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads=num_heads, batch_first=True
        )

    def forward(
        self,
        h: torch.Tensor,
        key_padding_mask: torch.Tensor = None,
        return_attention: bool = False,
    ) -> torch.Tensor:
        if h.ndim != 3:
            raise ValueError(f"TaskQueryPool expects [B, L, E], got {tuple(h.shape)}")
        if h.shape[1] <= 0:
            raise ValueError("TaskQueryPool received an empty token sequence")
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.to(device=h.device, dtype=torch.bool)
            if key_padding_mask.shape != h.shape[:2]:
                raise ValueError(
                    f"TaskQueryPool key_padding_mask must match [B, L]={tuple(h.shape[:2])}, got {tuple(key_padding_mask.shape)}"
                )
            all_masked = key_padding_mask.all(dim=1)
            if bool(all_masked.any().item()):
                key_padding_mask = key_padding_mask.clone()
                key_padding_mask[all_masked, 0] = False
        q = self.query.expand(h.shape[0], -1, -1)
        if return_attention:
            out, attn = self.attn(
                q,
                h,
                h,
                key_padding_mask=key_padding_mask,
                need_weights=True,
                average_attn_weights=False,
            )
            return (out, attn)
        out, _ = self.attn(
            q, h, h, key_padding_mask=key_padding_mask, need_weights=False
        )
        return out


class SharedSpecificMoELayer(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_text_specific_experts: int,
        num_audio_specific_experts: int,
        num_vision_specific_experts: int,
        num_shared_experts: int,
        normalize: bool = True,
        router_temperature: float = 0.1,
        drop: float = 0.0,
        attention_dropout: float = 0.0,
        expert_mlp_ratio: float = 1.0,
    ):
        super().__init__()
        self.dim = int(dim)
        self.num_text_specific_experts = int(num_text_specific_experts)
        self.num_audio_specific_experts = int(num_audio_specific_experts)
        self.num_vision_specific_experts = int(num_vision_specific_experts)
        self.num_shared_experts = int(num_shared_experts)
        self.num_experts = (
            self.num_text_specific_experts
            + self.num_audio_specific_experts
            + self.num_vision_specific_experts
            + self.num_shared_experts
            + 0
        )
        if self.num_experts <= 0:
            raise ValueError("SharedSpecificMoELayer requires at least one expert")
        self.slots_per_expert = 1
        self.normalize = bool(normalize)
        self.router_temperature = float(router_temperature)
        if self.router_temperature <= 0:
            raise ValueError(
                f"router_temperature must be positive, got {self.router_temperature}"
            )
        self.expert_mlp_ratio = float(expert_mlp_ratio)
        if self.expert_mlp_ratio <= 0:
            raise ValueError(
                f"expert_mlp_ratio must be positive, got {self.expert_mlp_ratio}"
            )
        self.attn_norm = nn.ModuleDict(
            {
                "vision": nn.LayerNorm(self.dim),
                "text": nn.LayerNorm(self.dim),
                "audio": nn.LayerNorm(self.dim),
            }
        )
        self.self_attn = nn.ModuleDict(
            {
                "vision": nn.MultiheadAttention(
                    self.dim,
                    num_heads=num_heads,
                    dropout=attention_dropout,
                    batch_first=True,
                ),
                "text": nn.MultiheadAttention(
                    self.dim,
                    num_heads=num_heads,
                    dropout=attention_dropout,
                    batch_first=True,
                ),
                "audio": nn.MultiheadAttention(
                    self.dim,
                    num_heads=num_heads,
                    dropout=attention_dropout,
                    batch_first=True,
                ),
            }
        )
        self.moe_norm = nn.ModuleDict(
            {
                "vision": nn.LayerNorm(self.dim),
                "text": nn.LayerNorm(self.dim),
                "audio": nn.LayerNorm(self.dim),
            }
        )
        self.phi = nn.Parameter(
            torch.zeros(self.dim, self.num_experts, self.slots_per_expert)
        )
        if self.normalize:
            self.scale = nn.Parameter(torch.ones(1))
        nn.init.normal_(self.phi, mean=0, std=1 / self.dim**0.5)
        self.experts = nn.ModuleList(
            [
                Mlp(
                    in_features=self.dim,
                    hidden_features=max(
                        1, int(round(self.dim * self.expert_mlp_ratio))
                    ),
                    out_features=self.dim,
                    drop=drop,
                )
                for _ in range(self.num_experts)
            ]
        )
        self.expert_group_layout = self._build_expert_group_layout()
        self.register_buffer(
            "text_allowed_mask", self._build_allowed_mask("text"), persistent=False
        )
        self.register_buffer(
            "audio_allowed_mask", self._build_allowed_mask("audio"), persistent=False
        )
        self.register_buffer(
            "vision_allowed_mask", self._build_allowed_mask("vision"), persistent=False
        )

    def _build_expert_group_layout(self):
        start = 0
        layout = {}
        groups = [
            ("text_specific", self.num_text_specific_experts),
            ("audio_specific", self.num_audio_specific_experts),
            ("vision_specific", self.num_vision_specific_experts),
            ("shared", self.num_shared_experts),
            ("temporal_contrast", 0),
        ]
        for name, count in groups:
            end = start + count
            layout[name] = {
                "start": int(start),
                "end": int(end),
                "count": int(count),
                "indices": list(range(start, end)),
            }
            start = end
        layout["total_experts"] = int(self.num_experts)
        return layout

    def _build_allowed_mask(self, modality: str) -> torch.Tensor:
        mask = torch.zeros(self.num_experts, dtype=torch.bool)
        if modality == "text":
            text_groups = ["text_specific"]
            for key in text_groups:
                for idx in self.expert_group_layout[key]["indices"]:
                    mask[idx] = True
        elif modality == "audio":
            groups = ["audio_specific", "shared"]
            for key in groups:
                for idx in self.expert_group_layout[key]["indices"]:
                    mask[idx] = True
        elif modality == "vision":
            groups = ["vision_specific", "shared"]
            for key in groups:
                for idx in self.expert_group_layout[key]["indices"]:
                    mask[idx] = True
        else:
            raise ValueError(f"Unknown modality: {modality}")
        if not bool(mask.any()):
            raise ValueError(f"No experts are accessible for modality '{modality}'")
        return mask

    @staticmethod
    def _coerce_token_mask(
        token_mask: torch.Tensor, x: torch.Tensor, name: str = "token_mask"
    ):
        if token_mask is None:
            return None
        token_mask = token_mask.to(device=x.device, dtype=torch.bool)
        if token_mask.shape != x.shape[:2]:
            raise ValueError(
                f"{name} must match token shape [B, L]={tuple(x.shape[:2])}, got {tuple(token_mask.shape)}"
            )
        return token_mask

    def _masked_moe(
        self,
        x: torch.Tensor,
        allowed_mask: torch.Tensor,
        token_mask: torch.Tensor = None,
    ):
        if x.ndim != 3:
            raise ValueError(f"Expected [B, L, E] tokens, got {tuple(x.shape)}")
        if x.shape[1] <= 0:
            raise ValueError("MoE received an empty token sequence")
        if x.shape[2] != self.dim:
            raise ValueError(f"Expected embedding dim {self.dim}, got {x.shape[2]}")
        token_mask = self._coerce_token_mask(token_mask, x)
        if self.normalize:
            x = F.normalize(x, dim=2)
            phi = self.scale * F.normalize(self.phi, dim=0)
        else:
            phi = self.phi
        logits = torch.einsum("bmd,dnp->bmnp", x, phi)
        logits = logits / self.router_temperature
        mask = allowed_mask.to(device=logits.device, dtype=torch.bool).view(
            1, 1, self.num_experts, 1
        )
        logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        mask_f = mask.to(dtype=logits.dtype)
        d = softmax(logits, dim=2)
        if token_mask is not None:
            token_mask_f = token_mask.to(dtype=logits.dtype).view(
                x.shape[0], x.shape[1], 1, 1
            )
            d = d * token_mask_f
            combine_logits = logits.masked_fill(
                ~token_mask.view(x.shape[0], x.shape[1], 1, 1),
                torch.finfo(logits.dtype).min,
            )
            c = softmax(combine_logits, dim=1) * token_mask_f * mask_f
        else:
            c = softmax(logits, dim=1) * mask_f
        xs = torch.einsum("bmd,bmnp->bnpd", x, d)
        ys = torch.stack(
            [expert(xs[:, idx, :, :]) for idx, expert in enumerate(self.experts)], dim=1
        )
        y = torch.einsum("bnpd,bmnp->bmd", ys, c)
        return y

    def _forward_modality(
        self, modality: str, x: torch.Tensor, token_mask: torch.Tensor = None
    ):
        token_mask = self._coerce_token_mask(
            token_mask, x, name=f"{modality}_token_mask"
        )
        key_padding_mask = None
        if token_mask is not None:
            key_padding_mask = ~token_mask
            all_masked = key_padding_mask.all(dim=1)
            if bool(all_masked.any().item()):
                key_padding_mask = key_padding_mask.clone()
                key_padding_mask[all_masked, 0] = False
        x_norm = self.attn_norm[modality](x)
        attn_out, _ = self.self_attn[modality](
            x_norm,
            x_norm,
            x_norm,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + attn_out
        x_norm = self.moe_norm[modality](x)
        allowed_mask = getattr(self, f"{modality}_allowed_mask")
        y = self._masked_moe(x_norm, allowed_mask, token_mask=token_mask)
        return x + y

    def forward(
        self,
        vision_tokens,
        text_tokens,
        audio_tokens,
        vision_mask: torch.Tensor = None,
        text_mask: torch.Tensor = None,
        audio_mask: torch.Tensor = None,
    ):
        z_v = self._forward_modality("vision", vision_tokens, token_mask=vision_mask)
        z_t = self._forward_modality("text", text_tokens, token_mask=text_mask)
        z_a = self._forward_modality("audio", audio_tokens, token_mask=audio_mask)
        return (z_v, z_t, z_a)


class EmotionM4OE(nn.Module):
    def __init__(
        self,
        embed_dim=768,
        depth_msoe=2,
        num_experts_msoe=16,
        num_experts_mtoe=32,
        num_shared_experts=2,
        num_text_specific_experts=4,
        num_audio_specific_experts=4,
        num_vision_specific_experts=4,
        temporal_projection: bool = False,
        temporal_embedding_dim: int = 128,
        vit_model_path="",
        bert_model_path="",
        hubert_model_path="",
        local_files_only=True,
        dropout=0.1,
        attention_dropout: float = 0.0,
        expert_mlp_ratio: float = 1.0,
        alignment_num_heads: int = 4,
        router_temperature: float = 0.1,
        enable_regression: bool = False,
        vit_context_ratio: float = 0.0,
        vit_context_lambda_init=None,
        vit_context_attention_dim: int = 64,
        final_pred_eta: float = 0.0,
        signed_class_count: int = 7,
        label_min: float = -3.0,
        label_max: float = 3.0,
        tcif_latent_dim: int = 128,
        tcif_context_temperature: float = 1.0,
        tcif_transition_gate_hidden_dim: int = 64,
        tcif_transition_gate_init_bias: float = 2.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.enable_regression = bool(enable_regression)
        self.final_pred_eta = float(final_pred_eta)
        self.signed_class_count = int(signed_class_count)
        self.label_min = float(label_min)
        self.label_max = float(label_max)
        if self.signed_class_count <= 1:
            raise ValueError(
                f"signed_class_count must be > 1, got {self.signed_class_count}"
            )
        if self.label_max <= self.label_min:
            raise ValueError(
                f"label_max must be > label_min, got {self.label_min}..{self.label_max}"
            )
        self.register_buffer(
            "cls7_centers",
            get_cls7_centers(
                num_classes=self.signed_class_count,
                label_min=self.label_min,
                label_max=self.label_max,
            ),
            persistent=False,
        )
        self.vit_context_ratio = float(vit_context_ratio)
        self.vit_context_attention_dim = int(vit_context_attention_dim)
        if self.vit_context_attention_dim <= 0:
            raise ValueError(
                f"vit_context_attention_dim must be positive, got {self.vit_context_attention_dim}"
            )
        self.attention_dropout = float(attention_dropout)
        self.expert_mlp_ratio = float(expert_mlp_ratio)
        if self.expert_mlp_ratio <= 0:
            raise ValueError(
                f"expert_mlp_ratio must be positive, got {self.expert_mlp_ratio}"
            )
        self.num_experts_msoe = int(num_experts_msoe)
        self.num_experts_mtoe = int(num_experts_mtoe)
        self.num_shared_experts = int(num_shared_experts)
        self.num_text_specific_experts = int(num_text_specific_experts)
        self.num_audio_specific_experts = int(num_audio_specific_experts)
        self.num_vision_specific_experts = int(num_vision_specific_experts)
        self.temporal_projection = bool(temporal_projection)
        self.temporal_embedding_dim = int(temporal_embedding_dim)
        self.tcif_latent_dim = int(tcif_latent_dim)
        self.tcif_context_temperature = float(tcif_context_temperature)
        self.tcif_transition_gate_hidden_dim = int(tcif_transition_gate_hidden_dim)
        self.tcif_transition_gate_init_bias = float(tcif_transition_gate_init_bias)
        if "posterior" not in TCIF_OUTPUT_MODES:
            raise ValueError(
                f"Unsupported tcif_output_mode={'posterior'!r}; expected one of {TCIF_OUTPUT_MODES}"
            )
        self._validate_model_files(hubert_model_path)
        self.bert = self._load_backbone(
            bert_model_path, local_files_only, BertModel(BertConfig())
        )
        self.hubert = self._load_backbone(
            hubert_model_path, local_files_only, HubertModel(HubertConfig())
        )
        self._vit_vanilla_dim = int(ViTConfig().hidden_size)
        self._bert_vanilla_dim = int(BertConfig().hidden_size)
        self._hubert_vanilla_dim = int(HubertConfig().hidden_size)
        vit_hidden_size = None
        bert_hidden_size = getattr(
            getattr(self.bert, "config", None), "hidden_size", None
        )
        hubert_hidden_size = getattr(
            getattr(self.hubert, "config", None), "hidden_size", None
        )
        bert_hidden_size = (
            int(bert_hidden_size)
            if bert_hidden_size is not None
            else self._bert_vanilla_dim
        )
        hubert_hidden_size = (
            int(hubert_hidden_size)
            if hubert_hidden_size is not None
            else self._hubert_vanilla_dim
        )
        self.vit = None
        self.resnet = None
        self.temporal_transformer = None
        self._validate_model_files(vit_model_path)
        self.vit = self._load_backbone(
            vit_model_path, local_files_only, ViTModel(ViTConfig())
        )
        vit_hidden_size = getattr(
            getattr(self.vit, "config", None), "hidden_size", None
        )
        vit_hidden_size = (
            int(vit_hidden_size)
            if vit_hidden_size is not None
            else self._vit_vanilla_dim
        )
        self.vit_projector = self._make_projector(
            vit_hidden_size, self._vit_vanilla_dim
        )
        self.visual_proj = self._make_projector(self._vit_vanilla_dim, embed_dim)
        self.vit_context_lambdas = None
        self.vit_context_lambda_gate_norm = None
        self.vit_context_lambda_gate = None
        self.vit_context_lambda_attn_norm = None
        self.vit_context_lambda_attn_q = None
        self.vit_context_lambda_attn_k = None
        if self.vit_context_ratio > 0.0:
            init_lambdas = self._parse_vit_context_lambda_init(vit_context_lambda_init)
            self.vit_context_lambdas = nn.Parameter(torch.log(init_lambdas))
            self.vit_context_lambda_attn_norm = nn.LayerNorm(embed_dim)
            self.vit_context_lambda_attn_q = nn.Linear(
                embed_dim, self.vit_context_attention_dim, bias=False
            )
            self.vit_context_lambda_attn_k = nn.Linear(
                embed_dim, self.vit_context_attention_dim, bias=False
            )
            nn.init.zeros_(self.vit_context_lambda_attn_q.weight)
            nn.init.xavier_uniform_(self.vit_context_lambda_attn_k.weight)
        self.bert_projector = self._make_projector(
            bert_hidden_size, self._bert_vanilla_dim
        )
        self.hubert_projector = self._make_projector(
            hubert_hidden_size, self._hubert_vanilla_dim
        )
        self.text_proj = nn.Linear(self._bert_vanilla_dim, embed_dim)
        self.audio_proj = self._make_projector(self._hubert_vanilla_dim, embed_dim)
        self.shared_specific_layers = nn.ModuleList(
            [
                SharedSpecificMoELayer(
                    dim=embed_dim,
                    num_heads=12,
                    num_text_specific_experts=self.num_text_specific_experts,
                    num_audio_specific_experts=self.num_audio_specific_experts,
                    num_vision_specific_experts=self.num_vision_specific_experts,
                    num_shared_experts=self.num_shared_experts,
                    normalize=True,
                    router_temperature=router_temperature,
                    drop=dropout,
                    attention_dropout=self.attention_dropout,
                    expert_mlp_ratio=self.expert_mlp_ratio,
                )
                for _ in range(depth_msoe)
            ]
        )
        self.expert_group_layout = (
            self.shared_specific_layers[0].expert_group_layout
            if len(self.shared_specific_layers) > 0
            else {
                "text_specific": {
                    "start": 0,
                    "end": self.num_text_specific_experts,
                    "count": self.num_text_specific_experts,
                    "indices": list(range(self.num_text_specific_experts)),
                },
                "audio_specific": {
                    "start": self.num_text_specific_experts,
                    "end": self.num_text_specific_experts
                    + self.num_audio_specific_experts,
                    "count": self.num_audio_specific_experts,
                    "indices": list(
                        range(
                            self.num_text_specific_experts,
                            self.num_text_specific_experts
                            + self.num_audio_specific_experts,
                        )
                    ),
                },
                "vision_specific": {
                    "start": self.num_text_specific_experts
                    + self.num_audio_specific_experts,
                    "end": self.num_text_specific_experts
                    + self.num_audio_specific_experts
                    + self.num_vision_specific_experts,
                    "count": self.num_vision_specific_experts,
                    "indices": list(
                        range(
                            self.num_text_specific_experts
                            + self.num_audio_specific_experts,
                            self.num_text_specific_experts
                            + self.num_audio_specific_experts
                            + self.num_vision_specific_experts,
                        )
                    ),
                },
                "shared": {
                    "start": self.num_text_specific_experts
                    + self.num_audio_specific_experts
                    + self.num_vision_specific_experts,
                    "end": self.num_text_specific_experts
                    + self.num_audio_specific_experts
                    + self.num_vision_specific_experts
                    + self.num_shared_experts,
                    "count": self.num_shared_experts,
                    "indices": list(
                        range(
                            self.num_text_specific_experts
                            + self.num_audio_specific_experts
                            + self.num_vision_specific_experts,
                            self.num_text_specific_experts
                            + self.num_audio_specific_experts
                            + self.num_vision_specific_experts
                            + self.num_shared_experts,
                        )
                    ),
                },
                "temporal_contrast": {
                    "start": self.num_text_specific_experts
                    + self.num_audio_specific_experts
                    + self.num_vision_specific_experts
                    + self.num_shared_experts,
                    "end": self.num_text_specific_experts
                    + self.num_audio_specific_experts
                    + self.num_vision_specific_experts
                    + self.num_shared_experts
                    + 0,
                    "count": 0,
                    "indices": list(
                        range(
                            self.num_text_specific_experts
                            + self.num_audio_specific_experts
                            + self.num_vision_specific_experts
                            + self.num_shared_experts,
                            self.num_text_specific_experts
                            + self.num_audio_specific_experts
                            + self.num_vision_specific_experts
                            + self.num_shared_experts
                            + 0,
                        )
                    ),
                },
                "total_experts": self.num_text_specific_experts
                + self.num_audio_specific_experts
                + self.num_vision_specific_experts
                + self.num_shared_experts
                + 0,
            }
        )
        self.task_pool_intensity = TaskQueryPool(
            embed_dim=embed_dim, num_heads=alignment_num_heads
        )
        self.task_pool_polarity = TaskQueryPool(
            embed_dim=embed_dim, num_heads=alignment_num_heads
        )
        self.msoe_v = nn.Sequential()
        self.msoe_t = nn.Sequential()
        self.msoe_a = nn.Sequential()
        self.mtoe_blocks = nn.ModuleList()
        self.fusion_task1 = nn.Identity()
        self.fusion_task2 = nn.Identity()
        self.fusion_task2_text_router = nn.Identity()
        self.head_input_dim = embed_dim * 4
        self.norm_task1 = nn.LayerNorm(self.head_input_dim)
        self.norm_task2 = nn.LayerNorm(self.head_input_dim)
        self.post_moe_dropout = nn.Dropout(dropout)
        self.head_signed_reg = nn.Linear(self.head_input_dim, 1)
        self.head_cls7 = nn.Linear(self.head_input_dim, self.signed_class_count)
        if self.temporal_projection:
            if self.temporal_embedding_dim <= 0:
                raise ValueError(
                    f"temporal_embedding_dim must be positive, got {self.temporal_embedding_dim}"
                )
            self.temporal_embedding_head = nn.Sequential(
                nn.LayerNorm(self.head_input_dim),
                nn.Linear(self.head_input_dim, self.head_input_dim),
                nn.GELU(),
                nn.Linear(self.head_input_dim, self.temporal_embedding_dim),
            )
        else:
            self.temporal_embedding_head = None
        self.tcif_regression = TemporalContextInnovationFilter(
            feature_dim=self.head_input_dim,
            latent_dim=self.tcif_latent_dim,
            context_temperature=self.tcif_context_temperature,
            transition_gate_hidden_dim=self.tcif_transition_gate_hidden_dim,
            transition_gate_init_bias=self.tcif_transition_gate_init_bias,
        )
        self.tcif_ordinal = TemporalContextInnovationFilter(
            feature_dim=self.head_input_dim,
            latent_dim=self.tcif_latent_dim,
            context_temperature=self.tcif_context_temperature,
            transition_gate_hidden_dim=self.tcif_transition_gate_hidden_dim,
            transition_gate_init_bias=self.tcif_transition_gate_init_bias,
        )
        self.tcif_context_reg_head = nn.Linear(self.tcif_latent_dim, 1)
        self.tcif_context_cls7_head = nn.Linear(
            self.tcif_latent_dim, self.signed_class_count
        )

    @staticmethod
    def _make_projector(in_dim: int, out_dim: int) -> nn.Module:
        if int(in_dim) == int(out_dim):
            return nn.Identity()
        proj = nn.Linear(int(in_dim), int(out_dim))
        nn.init.xavier_uniform_(proj.weight)
        if proj.bias is not None:
            nn.init.zeros_(proj.bias)
        return proj

    @staticmethod
    def _validate_model_files(model_path: str) -> None:
        if not model_path:
            return
        if not os.path.isdir(model_path):
            raise RuntimeError(f"Model path not found: {model_path}")
        has_config = os.path.exists(os.path.join(model_path, "config.json"))
        has_weights = any(
            (
                os.path.exists(os.path.join(model_path, fname))
                for fname in ["model.safetensors", "pytorch_model.bin"]
            )
        )
        if not (has_config and has_weights):
            raise RuntimeError(f"Model files incomplete in: {model_path}")

    @staticmethod
    def _maybe_convert_bin_to_safetensors(model_dir: str) -> bool:
        safetensors_path = os.path.join(model_dir, "model.safetensors")
        bin_path = os.path.join(model_dir, "pytorch_model.bin")
        if os.path.exists(safetensors_path):
            return False
        if not os.path.exists(bin_path):
            return False
        try:
            import safetensors.torch
        except Exception:
            return False
        state_dict = torch.load(bin_path, map_location="cpu", weights_only=True)
        safetensors.torch.save_file(state_dict, safetensors_path)
        return True

    @classmethod
    def _load_backbone(
        cls, model_path: str, local_files_only: bool, fallback_model: nn.Module
    ) -> nn.Module:
        if not model_path:
            return fallback_model
        try:
            return AutoModel.from_pretrained(
                model_path, local_files_only=local_files_only, use_safetensors=True
            )
        except Exception:
            pass
        try:
            return AutoModel.from_pretrained(
                model_path, local_files_only=local_files_only, use_safetensors=False
            )
        except Exception as e:
            if (
                isinstance(e, AttributeError)
                and "'NoneType' object has no attribute 'get'" in str(e)
                and os.path.isfile(os.path.join(model_path, "pytorch_model.bin"))
            ):
                return AutoModel.from_pretrained(
                    model_path, local_files_only=local_files_only, use_safetensors=False
                )
            if "serious vulnerability issue in `torch.load`" in str(e):
                if cls._maybe_convert_bin_to_safetensors(model_path):
                    return AutoModel.from_pretrained(
                        model_path,
                        local_files_only=local_files_only,
                        use_safetensors=True,
                    )
            raise

    @staticmethod
    def _extract_last_hidden_state(outputs):
        if (
            hasattr(outputs, "last_hidden_state")
            and outputs.last_hidden_state is not None
        ):
            return outputs.last_hidden_state
        if isinstance(outputs, (tuple, list)) and len(outputs) > 0:
            return outputs[0]
        raise RuntimeError("Model outputs do not contain last_hidden_state")

    @staticmethod
    def _parse_vit_context_lambda_init(vit_context_lambda_init):
        if vit_context_lambda_init is None or vit_context_lambda_init == "":
            values = [1.0 - 2e-06, 1e-06, 1e-06]
        elif isinstance(vit_context_lambda_init, str):
            values = [
                float(part.strip())
                for part in vit_context_lambda_init.split(",")
                if part.strip()
            ]
        else:
            values = [float(x) for x in vit_context_lambda_init]
        if len(values) != 3:
            raise ValueError(
                "vit_context_lambda_init must contain exactly three values: main,prev,next"
            )
        init_lambdas = torch.tensor(values, dtype=torch.float32)
        if torch.any(init_lambdas <= 0):
            raise ValueError("vit_context_lambda_init values must all be positive")
        init_lambdas = init_lambdas / init_lambdas.sum().clamp_min(1e-12)
        return init_lambdas.clamp_min(1e-08)

    def _encode_vit_tokens(self, vit_images: torch.Tensor) -> torch.Tensor:
        bsz, num_frames, channels, height, width = vit_images.shape
        vit_in = vit_images.reshape(bsz * num_frames, channels, height, width)
        vit_out = self._extract_last_hidden_state(self.vit(vit_in))
        vit_out = self.vit_projector(vit_out)
        vit_out = self.visual_proj(vit_out)
        return vit_out.reshape(bsz, num_frames, vit_out.shape[1], vit_out.shape[2])

    def _fuse_vit_context_triplet(self, vit_groups):
        if vit_groups.dim() != 5:
            raise ValueError(
                f"vit_groups must have shape [B,T,K,L,D], got {tuple(vit_groups.shape)}"
            )
        b, t, k, token_count, dim = vit_groups.shape
        center_idx = 1 if k > 1 else 0
        center_tokens = vit_groups[:, :, center_idx, :, :]
        if self.vit_context_lambdas is None or k < 3:
            return center_tokens
        prev_tokens = vit_groups[:, :, 0, :, :]
        next_tokens = vit_groups[:, :, 2, :, :]
        base_logits = self.vit_context_lambdas.to(
            device=center_tokens.device, dtype=center_tokens.dtype
        )
        center_cls = center_tokens[:, :, 0, :]
        prev_cls = prev_tokens[:, :, 0, :]
        next_cls = next_tokens[:, :, 0, :]
        frame_cls = torch.stack([center_cls, prev_cls, next_cls], dim=2)
        attn_norm_weight = getattr(self.vit_context_lambda_attn_norm, "weight", None)
        attn_dtype = getattr(attn_norm_weight, "dtype", center_tokens.dtype)
        frame_cls = self.vit_context_lambda_attn_norm(frame_cls.to(dtype=attn_dtype))
        q = self.vit_context_lambda_attn_q(frame_cls)
        k_proj = self.vit_context_lambda_attn_k(frame_cls)
        attn_logits = torch.matmul(q, k_proj.transpose(-1, -2)) / math.sqrt(
            float(self.vit_context_attention_dim)
        )
        attn_logits = attn_logits.to(dtype=center_tokens.dtype)
        center_row_logits = attn_logits[:, :, 0, :]
        logits = base_logits.view(1, 1, 3) + center_row_logits
        lambdas = softmax(logits, dim=-1)
        lambda_main = lambdas[..., 0].unsqueeze(-1).unsqueeze(-1)
        lambda_prev = lambdas[..., 1].unsqueeze(-1).unsqueeze(-1)
        lambda_next = lambdas[..., 2].unsqueeze(-1).unsqueeze(-1)
        return (
            lambda_main * center_tokens
            + lambda_prev * prev_tokens
            + lambda_next * next_tokens
        )

    @staticmethod
    def _all_valid_mask(tokens: torch.Tensor):
        return torch.ones(tokens.shape[:2], device=tokens.device, dtype=torch.bool)

    @staticmethod
    def _coerce_sequence_mask(mask: torch.Tensor, tokens: torch.Tensor, name: str):
        if mask is None:
            return EmotionM4OE._all_valid_mask(tokens)
        mask = mask.to(device=tokens.device, dtype=torch.bool)
        if mask.shape != tokens.shape[:2]:
            raise ValueError(
                f"{name} must match token shape [B, L]={tuple(tokens.shape[:2])}, got {tuple(mask.shape)}"
            )
        return mask

    def _build_audio_token_mask(
        self, audio_attention_mask: torch.Tensor, audio_embed: torch.Tensor
    ):
        get_feature_mask = getattr(
            self.hubert, "_get_feature_vector_attention_mask", None
        )
        if callable(get_feature_mask):
            try:
                feature_mask = get_feature_mask(
                    audio_embed.shape[1], audio_attention_mask
                )
                return self._coerce_sequence_mask(
                    feature_mask, audio_embed, "audio_token_mask"
                )
            except Exception:
                pass
        return self._all_valid_mask(audio_embed)

    @staticmethod
    def _masked_mean(tokens: torch.Tensor, token_mask: torch.Tensor = None):
        if token_mask is None:
            return tokens.mean(dim=1)
        token_mask = token_mask.to(device=tokens.device, dtype=torch.bool)
        weights = token_mask.to(dtype=tokens.dtype).unsqueeze(-1)
        denom = weights.sum(dim=1).clamp_min(1.0)
        return (tokens * weights).sum(dim=1) / denom

    @staticmethod
    def _build_task_head_feature(task_tokens, modality_tokens, modality_masks=None):
        pooled = [task_tokens.mean(dim=1)]
        if modality_masks is None:
            modality_masks = [None for _ in modality_tokens]
        pooled.extend(
            [
                EmotionM4OE._masked_mean(tokens, mask)
                for tokens, mask in zip(modality_tokens, modality_masks)
            ]
        )
        return torch.cat(pooled, dim=-1)

    @staticmethod
    def _maybe_cache_tensor(tensor):
        if tensor is None:
            return None
        if isinstance(tensor, torch.Tensor) and tensor.numel() == 0:
            return None
        return tensor

    @staticmethod
    def _module_param_dtype(module, fallback=torch.float32):
        for param in module.parameters(recurse=True):
            return param.dtype
        return fallback

    @staticmethod
    def _flatten_tcif_context_tensor(
        tensor, batch_size: int, context_count: int, name: str
    ):
        if tensor is None:
            return None
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a tensor, got {type(tensor).__name__}")
        if tensor.ndim < 2 or tuple(tensor.shape[:2]) != (batch_size, context_count):
            raise ValueError(
                f"{name} must start with [B,C]={(batch_size, context_count)}, got {tuple(tensor.shape)}"
            )
        return tensor.reshape(batch_size * context_count, *tensor.shape[2:])

    def _apply_tcif_to_local_output(
        self, local_output, context_output, context_valid_mask, context_relative_pos
    ):
        local_extras = local_output.get("extras", {})
        context_extras = context_output.get("extras", {})
        local_reg_feature = local_extras.get("feat_task1")
        local_cls_feature = local_extras.get("feat_task2")
        context_reg_feature = context_extras.get("feat_task1")
        context_cls_feature = context_extras.get("feat_task2")
        if any(
            (
                feature is None
                for feature in (
                    local_reg_feature,
                    local_cls_feature,
                    context_reg_feature,
                    context_cls_feature,
                )
            )
        ):
            raise RuntimeError(
                "TCIF requires task features from local and context forwards"
            )
        batch_size, context_count = context_valid_mask.shape
        context_reg_feature = context_reg_feature.reshape(batch_size, context_count, -1)
        context_cls_feature = context_cls_feature.reshape(batch_size, context_count, -1)
        regression_result = self.tcif_regression(
            local_reg_feature,
            context_reg_feature,
            context_valid_mask,
            context_relative_pos,
        )
        ordinal_result = self.tcif_ordinal(
            local_cls_feature,
            context_cls_feature,
            context_valid_mask,
            context_relative_pos,
        )
        tcif_context_y_reg = self.tcif_context_reg_head(
            regression_result["prior_mean"]
        ).squeeze(-1)
        tcif_context_cls7_logits = self.tcif_context_cls7_head(
            ordinal_result["prior_mean"]
        )
        tcif_payload = {
            "output_mode": "posterior",
            "has_context": regression_result["has_context"],
            "context_y_reg": tcif_context_y_reg,
            "context_cls7_logits": tcif_context_cls7_logits,
            "local_y_reg": local_output["y_reg"],
            "local_cls7_logits": local_output["cls7_logits"],
            "regression_attention": regression_result["context_attention"],
            "ordinal_attention": ordinal_result["context_attention"],
            "regression_innovation_norm": regression_result["innovation"].norm(dim=-1),
            "ordinal_innovation_norm": ordinal_result["innovation"].norm(dim=-1),
            "regression_prior_variance": regression_result["prior_variance"].mean(
                dim=-1
            ),
            "ordinal_prior_variance": ordinal_result["prior_variance"].mean(dim=-1),
            "regression_posterior_variance": regression_result[
                "posterior_variance"
            ].mean(dim=-1),
            "ordinal_posterior_variance": ordinal_result["posterior_variance"].mean(
                dim=-1
            ),
            "regression_continuation_gate": regression_result["continuation_gate"],
            "ordinal_continuation_gate": ordinal_result["continuation_gate"],
            "regression_continuation_gate_logits": regression_result[
                "continuation_gate_logits"
            ],
            "ordinal_continuation_gate_logits": ordinal_result[
                "continuation_gate_logits"
            ],
        }
        local_extras["tcif"] = tcif_payload
        feat_task1 = regression_result["feature"]
        feat_task2 = ordinal_result["feature"]
        y_reg = self.head_signed_reg(feat_task1).squeeze(-1)
        cls7_logits = self.head_cls7(feat_task2)
        cls7_probs = torch.softmax(cls7_logits, dim=-1)
        sign_logits = None
        temporal_embedding = None
        if self.temporal_embedding_head is not None:
            temporal_base = _temporal_feature_mix(
                feat_task1, feat_task2, detach_task2=False
            )
            temporal_embedding = F.normalize(
                self.temporal_embedding_head(temporal_base), dim=-1
            )
        centers = self.cls7_centers.to(
            device=cls7_logits.device, dtype=cls7_logits.dtype
        )
        y_cls_expected = cls7_expected_value_from_probs(cls7_probs, centers=centers)
        y_final = compute_final_prediction(
            y_reg, cls7_logits, eta=self.final_pred_eta, centers=centers
        )
        local_output["y_reg"] = y_reg
        local_output["cls7_logits"] = cls7_logits
        local_output["sign_logits"] = sign_logits
        local_extras.update(
            {
                "y_cls_expected": y_cls_expected,
                "y_final": y_final,
                "cls7_probs": cls7_probs,
                "feat_task2": feat_task2,
                "temporal_embedding": temporal_embedding,
            }
        )
        local_extras.pop("feat_task1", None)
        local_extras.pop("affective_repr", None)
        local_extras.pop("task_query_attention", None)
        local_output["extras"] = local_extras
        return local_output

    def forward(
        self,
        images,
        input_ids,
        attention_mask,
        audio_values=None,
        audio_attention_mask=None,
        cached_vision=None,
        cached_text=None,
        cached_audio=None,
        cached_text_mask=None,
        cached_audio_mask=None,
        tcif_context_images=None,
        tcif_context_input_ids=None,
        tcif_context_attention_mask=None,
        tcif_context_audio_values=None,
        tcif_context_audio_attention_mask=None,
        tcif_context_cached_vision=None,
        tcif_context_cached_text=None,
        tcif_context_cached_audio=None,
        tcif_context_cached_text_mask=None,
        tcif_context_cached_audio_mask=None,
        tcif_context_valid_mask=None,
        tcif_context_relative_pos=None,
        _tcif_skip: bool = False,
    ):
        if not _tcif_skip:
            if tcif_context_valid_mask is None or tcif_context_relative_pos is None:
                raise ValueError(
                    "TCIF is enabled but context_valid_mask/context_relative_pos were not provided"
                )
            if tcif_context_valid_mask.ndim != 2:
                raise ValueError(
                    f"tcif_context_valid_mask must have shape [B,C], got {tuple(tcif_context_valid_mask.shape)}"
                )
            batch_size, context_count = tcif_context_valid_mask.shape
            if context_count <= 0:
                raise ValueError("TCIF requires at least one context slot")
            local_output = self.forward(
                images,
                input_ids,
                attention_mask,
                audio_values,
                audio_attention_mask,
                cached_vision=cached_vision,
                cached_text=cached_text,
                cached_audio=cached_audio,
                cached_text_mask=cached_text_mask,
                cached_audio_mask=cached_audio_mask,
                _tcif_skip=True,
            )
            context_output = self.forward(
                self._flatten_tcif_context_tensor(
                    tcif_context_images,
                    batch_size,
                    context_count,
                    "tcif_context_images",
                ),
                self._flatten_tcif_context_tensor(
                    tcif_context_input_ids,
                    batch_size,
                    context_count,
                    "tcif_context_input_ids",
                ),
                self._flatten_tcif_context_tensor(
                    tcif_context_attention_mask,
                    batch_size,
                    context_count,
                    "tcif_context_attention_mask",
                ),
                self._flatten_tcif_context_tensor(
                    tcif_context_audio_values,
                    batch_size,
                    context_count,
                    "tcif_context_audio_values",
                ),
                self._flatten_tcif_context_tensor(
                    tcif_context_audio_attention_mask,
                    batch_size,
                    context_count,
                    "tcif_context_audio_attention_mask",
                ),
                cached_vision=self._flatten_tcif_context_tensor(
                    tcif_context_cached_vision,
                    batch_size,
                    context_count,
                    "tcif_context_cached_vision",
                ),
                cached_text=self._flatten_tcif_context_tensor(
                    tcif_context_cached_text,
                    batch_size,
                    context_count,
                    "tcif_context_cached_text",
                ),
                cached_audio=self._flatten_tcif_context_tensor(
                    tcif_context_cached_audio,
                    batch_size,
                    context_count,
                    "tcif_context_cached_audio",
                ),
                cached_text_mask=self._flatten_tcif_context_tensor(
                    tcif_context_cached_text_mask,
                    batch_size,
                    context_count,
                    "tcif_context_cached_text_mask",
                ),
                cached_audio_mask=self._flatten_tcif_context_tensor(
                    tcif_context_cached_audio_mask,
                    batch_size,
                    context_count,
                    "tcif_context_cached_audio_mask",
                ),
                _tcif_skip=True,
            )
            return self._apply_tcif_to_local_output(
                local_output,
                context_output,
                tcif_context_valid_mask,
                tcif_context_relative_pos,
            )
        cached_vision = self._maybe_cache_tensor(cached_vision)
        cached_text = self._maybe_cache_tensor(cached_text)
        cached_audio = self._maybe_cache_tensor(cached_audio)
        cached_text_mask = self._maybe_cache_tensor(cached_text_mask)
        cached_audio_mask = self._maybe_cache_tensor(cached_audio_mask)
        if cached_vision is None and images is None:
            raise ValueError("images or cached_vision must be provided")
        if cached_text is None and (input_ids is None or attention_mask is None):
            raise ValueError("input_ids/attention_mask or cached_text must be provided")
        if (
            cached_text is not None
            and cached_text_mask is None
            and (attention_mask is None)
        ):
            raise ValueError(
                "attention_mask or cached_text_mask must be provided with cached_text"
            )
        if cached_audio is None and audio_values is None:
            raise ValueError("audio_values or cached_audio must be provided")
        if cached_audio is None and audio_attention_mask is None:
            raise ValueError("audio_attention_mask must be provided with raw audio")
        device = None
        for tensor in (
            cached_vision,
            cached_text,
            cached_audio,
            images,
            input_ids,
            audio_values,
        ):
            if isinstance(tensor, torch.Tensor):
                device = tensor.device
                break
        if device is None:
            raise ValueError("Could not infer input device")
        bool(False)
        if cached_vision is not None:
            vision_dtype = self._module_param_dtype(
                self.vit_projector,
                self._module_param_dtype(self.visual_proj, torch.float32),
            )
            vit_groups = self.visual_proj(
                self.vit_projector(cached_vision.to(device=device, dtype=vision_dtype))
            )
            if vit_groups.dim() == 5:
                b, t, k, token_count, dim = vit_groups.shape
                center_tokens = self._fuse_vit_context_triplet(vit_groups)
                vision_embed = center_tokens.reshape(b, t * token_count, dim)
            elif vit_groups.dim() == 4:
                b, t, token_count, dim = vit_groups.shape
                vision_embed = vit_groups.reshape(b, t * token_count, dim)
            elif vit_groups.dim() == 3:
                vision_embed = vit_groups
            else:
                raise ValueError(
                    f"cached_vision must have shape [B,L,D], [B,T,L,D], or [B,T,K,L,D], got {tuple(vit_groups.shape)}"
                )
        else:
            images = images.to(device=device)
            if images.dim() == 4:
                images = images.unsqueeze(1)
            if images.shape[1] <= 0:
                raise ValueError(
                    f"images must contain at least one frame, got shape={tuple(images.shape)}"
                )
            if images.dim() == 6:
                b, t, k, c, h, w = images.shape
                vit_groups = self._encode_vit_tokens(images.reshape(b, t * k, c, h, w))
                vit_groups = vit_groups.reshape(
                    b, t, k, vit_groups.shape[2], vit_groups.shape[3]
                )
                center_tokens = self._fuse_vit_context_triplet(vit_groups)
                vision_embed = center_tokens.reshape(
                    b, t * center_tokens.shape[2], center_tokens.shape[3]
                )
            else:
                vit_out = self._encode_vit_tokens(images)
                vision_embed = vit_out.reshape(
                    vit_out.shape[0],
                    vit_out.shape[1] * vit_out.shape[2],
                    vit_out.shape[3],
                )
        if cached_text is not None:
            text_dtype = self._module_param_dtype(
                self.bert_projector,
                self._module_param_dtype(self.text_proj, torch.float32),
            )
            text_embed = self.bert_projector(
                cached_text.to(device=device, dtype=text_dtype)
            )
        else:
            input_ids = input_ids.to(device=device)
            attention_mask = attention_mask.to(device=device)
            text_embed = self._extract_last_hidden_state(
                self.bert(input_ids=input_ids, attention_mask=attention_mask)
            )
            text_embed = self.bert_projector(text_embed)
        text_embed = self.text_proj(text_embed)
        if cached_audio is not None:
            audio_dtype = self._module_param_dtype(
                self.hubert_projector,
                self._module_param_dtype(self.audio_proj, torch.float32),
            )
            audio_embed = self.hubert_projector(
                cached_audio.to(device=device, dtype=audio_dtype)
            )
        else:
            audio_dtype = (
                images.dtype
                if isinstance(images, torch.Tensor)
                else self._module_param_dtype(self.hubert_projector, torch.float32)
            )
            audio_values = audio_values.to(device=device, dtype=audio_dtype)
            audio_attention_mask = audio_attention_mask.to(
                device=device, dtype=torch.long
            )
            if audio_values.ndim != 2:
                raise ValueError(
                    f"audio_values must have shape [B, T], got {tuple(audio_values.shape)}"
                )
            if audio_attention_mask.shape != audio_values.shape:
                raise ValueError(
                    f"audio_attention_mask must match audio_values shape, got mask={tuple(audio_attention_mask.shape)} values={tuple(audio_values.shape)}"
                )
            if audio_values.shape[1] <= 0:
                raise ValueError("audio_values must contain at least one timestep")
            audio_embed = self._extract_last_hidden_state(
                self.hubert(
                    input_values=audio_values, attention_mask=audio_attention_mask
                )
            )
            audio_embed = self.hubert_projector(audio_embed)
        audio_embed = self.audio_proj(audio_embed)
        if (
            vision_embed.shape[1] <= 0
            or text_embed.shape[1] <= 0
            or audio_embed.shape[1] <= 0
        ):
            raise ValueError(
                f"Encoded modality tokens must all be non-empty, got vision={tuple(vision_embed.shape)} text={tuple(text_embed.shape)} audio={tuple(audio_embed.shape)}"
            )
        vision_mask = self._all_valid_mask(vision_embed)
        if cached_text is not None and cached_text_mask is not None:
            text_mask = self._coerce_sequence_mask(
                cached_text_mask.to(device=device), text_embed, "cached_text_mask"
            )
        else:
            text_mask = self._coerce_sequence_mask(
                attention_mask.to(device=device), text_embed, "attention_mask"
            )
        if cached_audio is not None:
            if cached_audio_mask is not None:
                audio_mask = self._coerce_sequence_mask(
                    cached_audio_mask.to(device=device),
                    audio_embed,
                    "cached_audio_mask",
                )
            else:
                audio_mask = self._all_valid_mask(audio_embed)
        else:
            audio_mask = self._build_audio_token_mask(audio_attention_mask, audio_embed)
        z_v, z_t, z_a = (vision_embed, text_embed, audio_embed)
        for layer in self.shared_specific_layers:
            z_v, z_t, z_a = layer(
                z_v,
                z_t,
                z_a,
                vision_mask=vision_mask,
                text_mask=text_mask,
                audio_mask=audio_mask,
            )
        h = torch.cat([z_v, z_t, z_a], dim=1)
        h_mask = torch.cat([vision_mask, text_mask, audio_mask], dim=1)
        task_key_padding_mask = ~h_mask
        total_reg_loss = h.new_zeros(())
        o_task1 = self.task_pool_intensity(h, key_padding_mask=task_key_padding_mask)
        o_task2 = self.task_pool_polarity(h, key_padding_mask=task_key_padding_mask)
        modality_masks = (vision_mask, text_mask, audio_mask)
        feat_task1 = self._build_task_head_feature(
            o_task1, (z_v, z_t, z_a), modality_masks
        )
        feat_task2 = self._build_task_head_feature(
            o_task2, (z_v, z_t, z_a), modality_masks
        )
        feat_task1 = self.norm_task1(feat_task1)
        feat_task2 = self.norm_task2(feat_task2)
        feat_task1 = self.post_moe_dropout(feat_task1)
        feat_task2 = self.post_moe_dropout(feat_task2)
        y_reg = self.head_signed_reg(feat_task1).squeeze(-1)
        cls7_aux = {}
        cls7_logits = self.head_cls7(feat_task2)
        cls7_probs = torch.softmax(cls7_logits, dim=-1)
        sign_logits = None
        temporal_embedding = None
        if self.temporal_embedding_head is not None:
            temporal_base = _temporal_feature_mix(
                feat_task1, feat_task2, detach_task2=False
            )
            temporal_embedding = F.normalize(
                self.temporal_embedding_head(temporal_base), dim=-1
            )
        centers = self.cls7_centers.to(
            device=cls7_logits.device, dtype=cls7_logits.dtype
        )
        y_cls_expected = cls7_expected_value_from_probs(cls7_probs, centers=centers)
        y_final = compute_final_prediction(
            y_reg, cls7_logits, eta=self.final_pred_eta, centers=centers
        )
        model_output = {
            "y_reg": y_reg,
            "cls7_logits": cls7_logits,
            "sign_logits": sign_logits,
            "tc_loss": total_reg_loss,
            "extras": {
                "y_cls_expected": y_cls_expected,
                "y_final": y_final,
                "cls7_probs": cls7_probs,
                "cls7_head_type": "flat",
                "feat_task2": feat_task2,
                "temporal_embedding": temporal_embedding,
            },
        }
        if _tcif_skip:
            model_output["extras"]["feat_task1"] = feat_task1
        model_output["extras"].update(cls7_aux)
        return model_output
