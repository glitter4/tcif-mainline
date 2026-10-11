import os
import csv
import json
import re
import numpy as np
import torch
from torch.utils.data import Dataset
import transformers.utils
import transformers.utils.import_utils as _import_utils


def _disable_sklearn():

    def _false():
        return False

    transformers.utils.is_sklearn_available = _false
    _import_utils.is_sklearn_available = _false


_disable_sklearn()
from transformers import AutoTokenizer


class MultimodalEmbeddingCache:
    def __init__(self, root, split):
        self.root = os.path.abspath(os.path.expanduser(str(root))) if root else ""
        self.split = split
        self.split_dir = os.path.join(self.root, split) if self.root else ""
        self.index = {}
        self.shard_dirs = []
        self._arrays = {}
        if not self.root:
            return
        if not os.path.isdir(self.split_dir):
            raise FileNotFoundError(
                f"Embedding cache split directory not found: {self.split_dir}"
            )
        for name in sorted(os.listdir(self.split_dir)):
            shard_dir = os.path.join(self.split_dir, name)
            if not os.path.isdir(shard_dir):
                continue
            ids_path = os.path.join(shard_dir, "ids.json")
            done_path = os.path.join(shard_dir, "done.json")
            if not os.path.isfile(ids_path) or not os.path.isfile(done_path):
                continue
            with open(ids_path, "r", encoding="utf-8") as f:
                ids = json.load(f)
            shard_idx = len(self.shard_dirs)
            self.shard_dirs.append(shard_dir)
            for row_idx, sample_id in enumerate(ids):
                self.index[str(sample_id)] = (shard_idx, row_idx)
        if not self.index:
            raise RuntimeError(
                f"No completed embedding cache shards found under {self.split_dir}"
            )

    def __len__(self):
        return len(self.index)

    def __contains__(self, sample_id):
        return str(sample_id) in self.index

    def _array(self, shard_idx, name):
        key = (shard_idx, name)
        if key not in self._arrays:
            path = os.path.join(self.shard_dirs[shard_idx], f"{name}.npy")
            self._arrays[key] = np.load(path, mmap_mode="r")
        return self._arrays[key]

    def get(self, sample_id):
        loc = self.index.get(str(sample_id))
        if loc is None:
            return None
        shard_idx, row_idx = loc

        def tensor(name):
            arr = self._array(shard_idx, name)[row_idx]
            return torch.from_numpy(np.array(arr, copy=True))

        return {
            "cached_vision": tensor("vision"),
            "cached_text": tensor("text"),
            "cached_audio": tensor("audio"),
            "cached_text_mask": tensor("text_mask").to(torch.long),
            "cached_audio_mask": tensor("audio_mask").to(torch.long),
        }


def _validate_embedding_cache(cache, ids, split):
    if cache is None:
        return
    missing = [sample_id for sample_id in ids if sample_id not in cache]
    if missing:
        preview = ", ".join((str(x) for x in missing[:5]))
        raise RuntimeError(
            f"Embedding cache for split '{split}' is incomplete: missing {len(missing)} samples. First missing ids: {preview}"
        )
    print(f"Loaded embedding cache split={split} samples={len(ids)} root={cache.root}")


_TEMPORAL_NUMBER_RE = re.compile("[-+]?\\d*\\.?\\d+")


def _coerce_temporal_position(value, default=None):
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        match = _TEMPORAL_NUMBER_RE.search(str(value))
        if match is None:
            return default
        try:
            return float(match.group(0))
        except ValueError:
            return default


def parse_temporal_sample_id(sample_id):
    sample_text = str(sample_id)
    if "_" in sample_text:
        base_id, clip_idx = sample_text.rsplit("_", 1)
    else:
        base_id, clip_idx = (sample_text, "")
    clip_pos = _coerce_temporal_position(clip_idx, default=0.0)
    return (base_id, clip_idx, float(clip_pos))


def build_temporal_metadata(sample_id, label=None, group_id=0):
    base_id, clip_idx, clip_pos = parse_temporal_sample_id(sample_id)
    label = label or {}
    temporal_pos = clip_pos
    if isinstance(label, dict):
        for key in ("start", "start_time", "start_sec", "begin", "begin_time"):
            if key in label:
                temporal_pos = _coerce_temporal_position(
                    label.get(key), default=clip_pos
                )
                break
    return {
        "base_id": base_id,
        "clip_idx": clip_idx,
        "temporal_pos": float(temporal_pos),
        "temporal_group_id": int(group_id),
    }


def build_temporal_context_index(
    ids, temporal_metadata_by_id, radius=0, mode="neighbors", seed=0
):
    """Build split-local masked-center neighbor windows."""
    radius = int(radius)
    mode = str(mode).lower()
    if radius < 0:
        raise ValueError(f"temporal context radius must be non-negative, got {radius}")
    if mode not in {"neighbors"}:
        raise ValueError(f"temporal context mode must be 'neighbors', got {mode!r}")
    if radius == 0:
        return {idx: [] for idx in range(len(ids))}
    groups = {}
    for idx, sample_id in enumerate(ids):
        metadata = temporal_metadata_by_id[sample_id]
        group_id = int(metadata["temporal_group_id"])
        groups.setdefault(group_id, []).append(idx)
    for group_indices in groups.values():
        group_indices.sort(
            key=lambda idx: (
                float(temporal_metadata_by_id[ids[idx]]["temporal_pos"]),
                str(ids[idx]),
            )
        )
    group_by_index = {}
    position_in_group = {}
    for group_id, group_indices in groups.items():
        for group_pos, idx in enumerate(group_indices):
            group_by_index[idx] = group_id
            position_in_group[idx] = group_pos
    offsets = list(range(-radius, 0)) + list(range(1, radius + 1))
    context_index = {}
    for center_idx, center_id in enumerate(ids):
        group_id = group_by_index[center_idx]
        group_indices = groups[group_id]
        center_group_pos = position_in_group[center_idx]
        center_temporal_pos = float(temporal_metadata_by_id[center_id]["temporal_pos"])
        rows = []
        for offset in offsets:
            neighbor_group_pos = center_group_pos + offset
            valid = 0 <= neighbor_group_pos < len(group_indices)
            neighbor_idx = group_indices[neighbor_group_pos] if valid else center_idx
            if valid:
                neighbor_id = ids[neighbor_idx]
                relative_pos = (
                    float(temporal_metadata_by_id[neighbor_id]["temporal_pos"])
                    - center_temporal_pos
                )
            else:
                relative_pos = 0.0
            rows.append(
                {
                    "index": int(neighbor_idx),
                    "valid": bool(valid),
                    "relative_pos": float(relative_pos),
                }
            )
        context_index[center_idx] = rows
    return context_index


def _attach_temporal_context(dataset, idx, item):
    if dataset.temporal_context_radius <= 0:
        return item
    context_rows = dataset.temporal_context_index[idx]
    if not context_rows:
        raise RuntimeError(
            "temporal_context_radius is positive but no context slots were built"
        )
    context_items = [dataset._get_single_item(row["index"]) for row in context_rows]
    stack_keys = (
        "image",
        "input_ids",
        "attention_mask",
        "audio_values",
        "audio_attention_mask",
        "raw_valence",
        "cached_vision",
        "cached_text",
        "cached_audio",
        "cached_text_mask",
        "cached_audio_mask",
    )
    for key in stack_keys:
        if all((key in context_item for context_item in context_items)):
            item[f"tcif_context_{key}"] = torch.stack(
                [context_item[key] for context_item in context_items], dim=0
            )
    item["tcif_context_valid_mask"] = torch.tensor(
        [row["valid"] for row in context_rows], dtype=torch.bool
    )
    item["tcif_context_relative_pos"] = torch.tensor(
        [row["relative_pos"] for row in context_rows], dtype=torch.float32
    )
    return item


def _pretokenize_texts(tokenizer, texts, max_length: int, batch_size: int = 256):
    texts = [str(text) for text in texts]
    if not texts:
        return []
    if tokenizer is None:
        zeros_ids = torch.zeros(max_length, dtype=torch.long)
        zeros_mask = torch.zeros(max_length, dtype=torch.long)
        return [(zeros_ids.clone(), zeros_mask.clone()) for _ in texts]
    tokenized = []
    batch_size = max(1, int(batch_size))
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        encoded = tokenizer(
            batch_texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        batch_input_ids = encoded["input_ids"]
        batch_attention_mask = encoded["attention_mask"]
        for row_idx in range(batch_input_ids.shape[0]):
            tokenized.append(
                (
                    batch_input_ids[row_idx].clone(),
                    batch_attention_mask[row_idx].clone(),
                )
            )
    return tokenized


def _valence_to_intensity(val):
    intensity = abs(float(val))
    if intensity < 0:
        intensity = 0.0
    if intensity > 3.0:
        intensity = 3.0
    return intensity


def _valence_to_polarity(val):
    v = float(val)
    if v < 0:
        return 0
    return 1


class CMUMOSEIProcessDataset(Dataset):
    def __init__(
        self,
        root="",
        split="train",
        transform=None,
        tokenizer_path="",
        max_length=128,
        local_files_only=True,
        frame_policy="middle",
        audio_sample_rate=16000,
        audio_max_seconds=6.0,
        audio_frame_ms=25.0,
        audio_hop_ms=10.0,
        audio_denoise=True,
        num_frames=4,
        openface_face_dir=None,
        vit_context_ratio=0.0,
        embedding_cache_root=None,
        temporal_context_radius=0,
        temporal_context_mode="neighbors",
        temporal_context_seed=0,
    ):
        self.root = root
        self.split = split
        self.transform = transform
        self.max_length = max_length
        self.frame_policy = frame_policy
        self.audio_sample_rate = audio_sample_rate
        self.audio_max_seconds = audio_max_seconds
        self.audio_frame_ms = audio_frame_ms
        self.audio_hop_ms = audio_hop_ms
        self.audio_denoise = audio_denoise
        self.num_frames = int(num_frames)
        self.openface_face_dir = openface_face_dir or os.path.join(
            root, "openface_face"
        )
        self.vit_context_ratio = float(vit_context_ratio)
        self.temporal_context_radius = int(temporal_context_radius)
        self.temporal_context_mode = str(temporal_context_mode).lower()
        self.temporal_context_seed = int(temporal_context_seed)
        self.embedding_cache = None
        label_npz = np.load(os.path.join(root, "label.npz"), allow_pickle=True)
        split_key = {
            "train": "train_corpus",
            "val": "val_corpus",
            "test": "test_corpus",
        }[split]
        self.labels = label_npz[split_key].item()
        trans_path = os.path.join(root, "transcription-engchi-polish.csv")
        self.text_by_id = {}
        with open(trans_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if not row:
                    continue
                if row[0] == "id":
                    continue
                if len(row) >= 3:
                    self.text_by_id[row[0]] = row[2]
                elif len(row) == 2:
                    self.text_by_id[row[0]] = row[1]
                else:
                    self.text_by_id[row[0]] = ""
        self.ids = sorted(self.labels.keys())
        group_ids = {
            base_id: group_idx
            for group_idx, base_id in enumerate(
                sorted(
                    {parse_temporal_sample_id(sample_id)[0] for sample_id in self.ids}
                )
            )
        }
        self.temporal_metadata_by_id = {
            sample_id: build_temporal_metadata(
                sample_id,
                self.labels.get(sample_id, {}),
                group_id=group_ids[parse_temporal_sample_id(sample_id)[0]],
            )
            for sample_id in self.ids
        }
        self.temporal_context_index = build_temporal_context_index(
            self.ids,
            self.temporal_metadata_by_id,
            radius=self.temporal_context_radius,
            mode=self.temporal_context_mode,
            seed=self.temporal_context_seed,
        )
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path, local_files_only=local_files_only
            )
        except Exception as e:
            print(f"Warning: Could not load tokenizer {tokenizer_path}. Error: {e}")
            self.tokenizer = None
        self.encoded_text = _pretokenize_texts(
            self.tokenizer,
            [self.text_by_id.get(sample_id, "") for sample_id in self.ids],
            self.max_length,
        )
        if embedding_cache_root:
            self.embedding_cache = MultimodalEmbeddingCache(embedding_cache_root, split)
            _validate_embedding_cache(self.embedding_cache, self.ids, split)

    def __len__(self):
        return len(self.ids)

    def _get_single_item(self, idx):
        sample_id = self.ids[idx]
        temporal_metadata = self.temporal_metadata_by_id[sample_id]
        os.path.join(self.root, "subvideo_new", f"{sample_id}.mp4")
        os.path.join(self.root, "subaudio", f"{sample_id}.wav")
        input_ids, attention_mask = self.encoded_text[idx]
        val = float(self.labels[sample_id]["val"])
        intensity = _valence_to_intensity(val)
        polarity = _valence_to_polarity(val)
        cached = (
            self.embedding_cache.get(sample_id)
            if self.embedding_cache is not None
            else None
        )
        if cached is None:
            raise RuntimeError("A complete feature cache is required")
        item = {
            "image": torch.empty(0, dtype=torch.float32),
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "raw_valence": torch.tensor(val, dtype=torch.float32),
            "intensity": torch.tensor(intensity, dtype=torch.float32),
            "polarity": torch.tensor(polarity, dtype=torch.long),
            "id": sample_id,
            "base_id": temporal_metadata["base_id"],
            "clip_idx": temporal_metadata["clip_idx"],
            "temporal_pos": torch.tensor(
                temporal_metadata["temporal_pos"], dtype=torch.float32
            ),
            "temporal_group_id": torch.tensor(
                temporal_metadata["temporal_group_id"], dtype=torch.long
            ),
            "audio_values": torch.empty(0, dtype=torch.float32),
            "audio_attention_mask": torch.empty(0, dtype=torch.long),
        }
        item.update(cached)
        return item

    def __getitem__(self, idx):
        item = self._get_single_item(idx)
        if self.temporal_context_radius <= 0:
            return item
        context_rows = self.temporal_context_index[idx]
        if not context_rows:
            raise RuntimeError(
                "temporal_context_radius is positive but no context slots were built"
            )
        context_items = [self._get_single_item(row["index"]) for row in context_rows]
        stack_keys = (
            "image",
            "input_ids",
            "attention_mask",
            "audio_values",
            "audio_attention_mask",
            "raw_valence",
            "cached_vision",
            "cached_text",
            "cached_audio",
            "cached_text_mask",
            "cached_audio_mask",
        )
        for key in stack_keys:
            if all((key in context_item for context_item in context_items)):
                item[f"tcif_context_{key}"] = torch.stack(
                    [context_item[key] for context_item in context_items], dim=0
                )
        item["tcif_context_valid_mask"] = torch.tensor(
            [row["valid"] for row in context_rows], dtype=torch.bool
        )
        item["tcif_context_relative_pos"] = torch.tensor(
            [row["relative_pos"] for row in context_rows], dtype=torch.float32
        )
        return item


class CHSIMSProcessDataset(Dataset):
    """CH-SIMS process-format loader with the same cached-token contract as MOSEI.

    CH-SIMS differs from CMU-MOSEI mainly in its [-1, 1] label range and audio
    directory layout.  Keeping a dedicated loader prevents dataset-specific path
    and transcription conventions from leaking into the model architecture.
    """

    def __init__(
        self,
        root="",
        split="train",
        transform=None,
        tokenizer_path="",
        max_length=128,
        local_files_only=True,
        frame_policy="middle",
        audio_sample_rate=16000,
        audio_max_seconds=6.0,
        audio_frame_ms=25.0,
        audio_hop_ms=10.0,
        audio_denoise=True,
        num_frames=4,
        openface_face_dir=None,
        vit_context_ratio=0.0,
        embedding_cache_root=None,
        valence_scale=1.0,
        temporal_context_radius=0,
        temporal_context_mode="neighbors",
        temporal_context_seed=0,
    ):
        self.root = root
        self.split = split
        self.transform = transform
        self.max_length = max_length
        self.frame_policy = frame_policy
        self.audio_sample_rate = audio_sample_rate
        self.audio_max_seconds = audio_max_seconds
        self.audio_frame_ms = audio_frame_ms
        self.audio_hop_ms = audio_hop_ms
        self.audio_denoise = audio_denoise
        self.num_frames = int(num_frames)
        self.openface_face_dir = openface_face_dir or os.path.join(
            root, "openface_face"
        )
        self.audio_dir = os.path.join(root, "audio")
        self.vit_context_ratio = float(vit_context_ratio)
        self.embedding_cache = None
        self.valence_scale = float(valence_scale)
        self.temporal_context_radius = int(temporal_context_radius)
        self.temporal_context_mode = str(temporal_context_mode).lower()
        self.temporal_context_seed = int(temporal_context_seed)
        if self.valence_scale <= 0:
            raise ValueError("valence_scale must be positive")
        label_npz = np.load(os.path.join(root, "label.npz"), allow_pickle=True)
        split_key = {
            "train": "train_corpus",
            "val": "val_corpus",
            "test": "test_corpus",
        }[split]
        self.labels = label_npz[split_key].item()
        trans_path = os.path.join(root, "transcription-engchi-polish.csv")
        self.text_by_id = {}
        with open(trans_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if not row or row[0] in {"id", "name", "video_id"}:
                    continue
                if len(row) >= 3:
                    self.text_by_id[row[0]] = row[2] or row[1]
                elif len(row) == 2:
                    self.text_by_id[row[0]] = row[1]
                else:
                    self.text_by_id[row[0]] = ""
        self.ids = sorted(self.labels.keys())
        group_ids = {
            base_id: group_idx
            for group_idx, base_id in enumerate(
                sorted(
                    {parse_temporal_sample_id(sample_id)[0] for sample_id in self.ids}
                )
            )
        }
        self.temporal_metadata_by_id = {
            sample_id: build_temporal_metadata(
                sample_id,
                self.labels.get(sample_id, {}),
                group_id=group_ids[parse_temporal_sample_id(sample_id)[0]],
            )
            for sample_id in self.ids
        }
        self.temporal_context_index = build_temporal_context_index(
            self.ids,
            self.temporal_metadata_by_id,
            radius=self.temporal_context_radius,
            mode=self.temporal_context_mode,
            seed=self.temporal_context_seed,
        )
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path, local_files_only=local_files_only
            )
        except Exception as e:
            print(f"Warning: Could not load tokenizer {tokenizer_path}. Error: {e}")
            self.tokenizer = None
        self.encoded_text = _pretokenize_texts(
            self.tokenizer,
            [self.text_by_id.get(sample_id, "") for sample_id in self.ids],
            self.max_length,
        )
        if embedding_cache_root:
            self.embedding_cache = MultimodalEmbeddingCache(embedding_cache_root, split)
            _validate_embedding_cache(self.embedding_cache, self.ids, split)

    def __len__(self):
        return len(self.ids)

    def _get_single_item(self, idx):
        sample_id = self.ids[idx]
        temporal_metadata = self.temporal_metadata_by_id[sample_id]
        input_ids, attention_mask = self.encoded_text[idx]
        val = float(self.labels[sample_id]["val"]) * self.valence_scale
        intensity = _valence_to_intensity(val)
        polarity = _valence_to_polarity(val)
        cached = (
            self.embedding_cache.get(sample_id)
            if self.embedding_cache is not None
            else None
        )
        if cached is None:
            raise RuntimeError("A complete feature cache is required")
        item = {
            "image": torch.empty(0, dtype=torch.float32),
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "raw_valence": torch.tensor(val, dtype=torch.float32),
            "intensity": torch.tensor(intensity, dtype=torch.float32),
            "polarity": torch.tensor(polarity, dtype=torch.long),
            "id": sample_id,
            "base_id": temporal_metadata["base_id"],
            "clip_idx": temporal_metadata["clip_idx"],
            "temporal_pos": torch.tensor(
                temporal_metadata["temporal_pos"], dtype=torch.float32
            ),
            "temporal_group_id": torch.tensor(
                temporal_metadata["temporal_group_id"], dtype=torch.long
            ),
            "audio_values": torch.empty(0, dtype=torch.float32),
            "audio_attention_mask": torch.empty(0, dtype=torch.long),
        }
        item.update(cached)
        return item

    def __getitem__(self, idx):
        return _attach_temporal_context(self, idx, self._get_single_item(idx))


class CMUMOSIProcessDataset(Dataset):
    def __init__(
        self,
        root="",
        split="train",
        transform=None,
        tokenizer_path="",
        max_length=128,
        local_files_only=True,
        frame_policy="middle",
        audio_sample_rate=16000,
        audio_max_seconds=6.0,
        audio_frame_ms=25.0,
        audio_hop_ms=10.0,
        audio_denoise=True,
        num_frames=4,
        openface_face_dir=None,
        vit_context_ratio=0.0,
        embedding_cache_root=None,
        temporal_context_radius=0,
        temporal_context_mode="neighbors",
        temporal_context_seed=0,
    ):
        self.root = root
        self.split = split
        self.transform = transform
        self.max_length = max_length
        self.frame_policy = frame_policy
        self.audio_sample_rate = audio_sample_rate
        self.audio_max_seconds = audio_max_seconds
        self.audio_frame_ms = audio_frame_ms
        self.audio_hop_ms = audio_hop_ms
        self.audio_denoise = audio_denoise
        self.num_frames = int(num_frames)
        self.openface_face_dir = openface_face_dir or os.path.join(
            root, "openface_face"
        )
        self.vit_context_ratio = float(vit_context_ratio)
        self.embedding_cache = None
        self.temporal_context_radius = int(temporal_context_radius)
        self.temporal_context_mode = str(temporal_context_mode).lower()
        self.temporal_context_seed = int(temporal_context_seed)
        label_npz = np.load(os.path.join(root, "label.npz"), allow_pickle=True)
        split_key = {
            "train": "train_corpus",
            "val": "val_corpus",
            "test": "test_corpus",
        }[split]
        self.labels = label_npz[split_key].item()
        trans_path = os.path.join(root, "transcription-engchi-polish.csv")
        self.text_by_id = {}
        with open(trans_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if not row:
                    continue
                if row[0] in {"id", "name"}:
                    continue
                if len(row) >= 3:
                    self.text_by_id[row[0]] = row[2]
                elif len(row) == 2:
                    self.text_by_id[row[0]] = row[1]
                else:
                    self.text_by_id[row[0]] = ""
        self.ids = sorted(self.labels.keys())
        group_ids = {
            base_id: group_idx
            for group_idx, base_id in enumerate(
                sorted(
                    {parse_temporal_sample_id(sample_id)[0] for sample_id in self.ids}
                )
            )
        }
        self.temporal_metadata_by_id = {
            sample_id: build_temporal_metadata(
                sample_id,
                self.labels.get(sample_id, {}),
                group_id=group_ids[parse_temporal_sample_id(sample_id)[0]],
            )
            for sample_id in self.ids
        }
        self.temporal_context_index = build_temporal_context_index(
            self.ids,
            self.temporal_metadata_by_id,
            radius=self.temporal_context_radius,
            mode=self.temporal_context_mode,
            seed=self.temporal_context_seed,
        )
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path, local_files_only=local_files_only
            )
        except Exception as e:
            print(f"Warning: Could not load tokenizer {tokenizer_path}. Error: {e}")
            self.tokenizer = None
        self.encoded_text = _pretokenize_texts(
            self.tokenizer,
            [self.text_by_id.get(sample_id, "") for sample_id in self.ids],
            self.max_length,
        )
        if embedding_cache_root:
            self.embedding_cache = MultimodalEmbeddingCache(embedding_cache_root, split)
            _validate_embedding_cache(self.embedding_cache, self.ids, split)

    def __len__(self):
        return len(self.ids)

    def _get_single_item(self, idx):
        sample_id = self.ids[idx]
        temporal_metadata = self.temporal_metadata_by_id[sample_id]
        os.path.join(self.root, "subvideo", f"{sample_id}.mp4")
        os.path.join(self.root, "subaudio", f"{sample_id}.wav")
        input_ids, attention_mask = self.encoded_text[idx]
        val = float(self.labels[sample_id]["val"])
        intensity = _valence_to_intensity(val)
        polarity = _valence_to_polarity(val)
        cached = (
            self.embedding_cache.get(sample_id)
            if self.embedding_cache is not None
            else None
        )
        if cached is None:
            raise RuntimeError("A complete feature cache is required")
        item = {
            "image": torch.empty(0, dtype=torch.float32),
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "raw_valence": torch.tensor(val, dtype=torch.float32),
            "intensity": torch.tensor(intensity, dtype=torch.float32),
            "polarity": torch.tensor(polarity, dtype=torch.long),
            "id": sample_id,
            "base_id": temporal_metadata["base_id"],
            "clip_idx": temporal_metadata["clip_idx"],
            "temporal_pos": torch.tensor(
                temporal_metadata["temporal_pos"], dtype=torch.float32
            ),
            "temporal_group_id": torch.tensor(
                temporal_metadata["temporal_group_id"], dtype=torch.long
            ),
            "audio_values": torch.empty(0, dtype=torch.float32),
            "audio_attention_mask": torch.empty(0, dtype=torch.long),
        }
        item.update(cached)
        return item

    def __getitem__(self, idx):
        return _attach_temporal_context(self, idx, self._get_single_item(idx))
