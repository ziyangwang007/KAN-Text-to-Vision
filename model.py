from typing import List

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam, optimizer

from kan import KAN
from KanTransformerEncorder import KANTransformerEncoderLayer

EPSILON = 1e-4
START_LEARNING_RATE = 1e-3
MAX_SEQ_LEN = 200


def masked_mse_loss(
    pose: torch.Tensor,
    pose_hat: torch.Tensor,
    confidence: torch.Tensor,
    model_num_steps: int = 10,
):
    sq_error = torch.pow(pose - pose_hat, 2).sum(-1)
    num_steps_norm = np.log(model_num_steps) ** 2 if model_num_steps != 1 else 1
    return (sq_error * confidence).mean() * num_steps_norm


def _normalize_encoder_type(name: str) -> str:
    if name is None:
        return "transformer"
    normalized = name.lower().replace("_", "").replace("-", "")
    if normalized in {"transformer", "text", "vanilla"}:
        return "transformer"
    if normalized in {"kan", "kansformer"}:
        return "kan"
    raise ValueError(f"Unsupported encoder type: {name}")


def _build_encoder_layer(
    encoder_type: str,
    hidden_dim: int,
    encoder_heads: int,
    encoder_dim_feedforward: int,
    kan_hidden_dim: int,
):
    encoder_type = _normalize_encoder_type(encoder_type)
    if encoder_type == "kan":
        return KANTransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=encoder_heads,
            dim_feedforward=encoder_dim_feedforward,
            hdim_kan=kan_hidden_dim,
        )
    return nn.TransformerEncoderLayer(
        d_model=hidden_dim,
        nhead=encoder_heads,
        dim_feedforward=encoder_dim_feedforward,
    )


def _build_pose_projection(
    projection_type: str,
    pose_dim: int,
    hidden_dim: int,
    num_pose_projection_layers: int,
):
    projection_type = projection_type.lower().replace("_", "").replace("-", "")
    if projection_type == "linear":
        if num_pose_projection_layers == 1:
            return nn.Linear(pose_dim, hidden_dim)
        layers = [nn.Linear(pose_dim, hidden_dim)]
        for _ in range(num_pose_projection_layers - 1):
            layers.extend([nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)])
        return nn.Sequential(*layers)

    if projection_type == "mlp":
        layers = [nn.Linear(pose_dim, hidden_dim)]
        for _ in range(max(1, num_pose_projection_layers) - 1):
            layers.extend([nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)])
        if len(layers) == 1:
            layers = [nn.Linear(pose_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)]
        return nn.Sequential(*layers)

    if projection_type == "kan":
        if num_pose_projection_layers <= 1:
            return KAN([pose_dim, hidden_dim])
        hidden_layers = [hidden_dim] * (num_pose_projection_layers - 1)
        return KAN([pose_dim, *hidden_layers, hidden_dim])

    raise ValueError(f"Unsupported pose projection type: {projection_type}")


class IterativeTextGuidedPoseGenerationModel(pl.LightningModule):
    def __init__(
        self,
        tokenizer,
        pose_dims: (int, int) = (137, 2),
        hidden_dim: int = 128,
        text_encoder_depth: int = 2,
        pose_encoder_depth: int = 4,
        encoder_heads: int = 2,
        encoder_dim_feedforward: int = 2048,
        max_seq_size: int = MAX_SEQ_LEN,
        min_seq_size: int = 20,
        num_steps: int = 10,
        tf_p: float = 0.5,
        lr: float = START_LEARNING_RATE,
        noise_epsilon: float = EPSILON,
        seq_len_weight: float = 2e-5,
        optimizer_fn: optimizer = torch.optim.Adam,
        separate_positional_embedding: bool = False,
        num_pose_projection_layers: int = 1,
        concat: bool = True,
        blend: bool = True,
        model_variant: str = "multiscale",
        text_encoder_type: str = "transformer",
        text_pose_encoder_type: str = "kan",
        pose_projection_type: str = "linear",
        kan_hidden_dim: int = 0,
    ):
        super().__init__()
        self.lr = lr
        self.noise_epsilon = noise_epsilon
        self.tf_p = tf_p
        self.seq_len_weight = seq_len_weight
        self.tokenizer = tokenizer
        self.max_seq_size = max_seq_size
        self.min_seq_size = min_seq_size
        self.num_steps = num_steps
        self.hidden_dim = hidden_dim
        self.pose_dims = pose_dims
        self.optimizer_fn = optimizer_fn
        self.separate_positional_embedding = separate_positional_embedding
        self.best_loss = np.inf
        self.concat = concat
        self.blend = blend
        self.model_variant = model_variant.lower().replace("_", "").replace("-", "")
        if self.model_variant not in {"baseline", "multiscale"}:
            raise ValueError(f"Unsupported model variant: {model_variant}")

        self.text_encoder_type = _normalize_encoder_type(text_encoder_type)
        self.text_pose_encoder_type = _normalize_encoder_type(text_pose_encoder_type)
        self.pose_projection_type = pose_projection_type
        self.kan_hidden_dim = kan_hidden_dim if kan_hidden_dim and kan_hidden_dim > 0 else max(1, hidden_dim // 2)

        pose_dim = int(np.prod(pose_dims))
        self.coarse_dim = (len(BODY_COARSE_POSE), 2)
        self.coarse_pose_dim = len(BODY_COARSE_POSE) * 2

        self.embedding = nn.Embedding(
            num_embeddings=len(tokenizer),
            embedding_dim=hidden_dim,
            padding_idx=tokenizer.pad_token_id,
        )
        self.step_embedding = nn.Embedding(num_embeddings=num_steps, embedding_dim=hidden_dim)

        if separate_positional_embedding:
            self.pos_positional_embeddings = nn.Embedding(num_embeddings=max_seq_size, embedding_dim=hidden_dim)
            self.text_positional_embeddings = nn.Embedding(num_embeddings=max_seq_size, embedding_dim=hidden_dim)
        else:
            self.positional_embeddings = nn.Embedding(num_embeddings=max_seq_size, embedding_dim=hidden_dim)
            self.alpha_pose = nn.Parameter(torch.randn(1))
            self.alpha_text = nn.Parameter(torch.randn(1))

        self.pose_projection = _build_pose_projection(
            projection_type=pose_projection_type,
            pose_dim=pose_dim,
            hidden_dim=hidden_dim,
            num_pose_projection_layers=num_pose_projection_layers,
        )

        text_encoder_layer = _build_encoder_layer(
            encoder_type=self.text_encoder_type,
            hidden_dim=hidden_dim,
            encoder_heads=encoder_heads,
            encoder_dim_feedforward=encoder_dim_feedforward,
            kan_hidden_dim=self.kan_hidden_dim,
        )
        pose_encoder_layer = _build_encoder_layer(
            encoder_type=self.text_pose_encoder_type,
            hidden_dim=hidden_dim,
            encoder_heads=encoder_heads,
            encoder_dim_feedforward=encoder_dim_feedforward,
            kan_hidden_dim=self.kan_hidden_dim,
        )

        self.text_encoder = nn.TransformerEncoder(text_encoder_layer, num_layers=text_encoder_depth)
        self.pose_encoder = nn.TransformerEncoder(pose_encoder_layer, num_layers=pose_encoder_depth)

        self.step_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        self.seq_length = nn.Linear(hidden_dim, 1)

        if self.model_variant == "multiscale":
            self.coarse_pose_projection = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, self.coarse_pose_dim),
            )
            self.fine_pose_projection = nn.Sequential(
                nn.Linear(hidden_dim + self.coarse_pose_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, pose_dim),
            )
        else:
            self.pose_diff_projection = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, pose_dim),
            )

    def predict_coarse_pose(self, encoded_sequence):
        return self.coarse_pose_projection(encoded_sequence)

    def predict_fine_pose(self, encoded_sequence, coarse_pose):
        fine_input = torch.cat([encoded_sequence, coarse_pose], dim=-1)
        return self.fine_pose_projection(fine_input)

    def encode_text(self, texts: List[str]):
        tokenized = self.tokenizer(texts, device=self.device)
        if self.separate_positional_embedding:
            positional_embedding = self.text_positional_embeddings(tokenized["positions"])
        else:
            positional_embedding = self.alpha_text * self.positional_embeddings(tokenized["positions"])

        embedding = self.embedding(tokenized["tokens_ids"]) + positional_embedding
        encoded = self.text_encoder(
            embedding.transpose(0, 1),
            src_key_padding_mask=tokenized["attention_mask"],
        ).transpose(0, 1)

        seq_length = self.seq_length(encoded).mean(axis=1)
        return {"data": encoded, "mask": tokenized["attention_mask"]}, seq_length

    def forward(self, text: str, first_pose: torch.Tensor, sequence_length: int = -1):
        text_encoding, seq_len = self.encode_text([text])
        seq_len = round(float(seq_len))
        seq_len = max(min(seq_len, self.max_seq_size), self.min_seq_size)
        sequence_length = seq_len if sequence_length == -1 else sequence_length
        pose_sequence = {
            "data": first_pose.expand(1, sequence_length, *self.pose_dims),
            "mask": torch.zeros([1, sequence_length], dtype=torch.bool, device=self.device),
        }

        if self.num_steps == 1:
            pred = self.refine_pose_sequence(pose_sequence, text_encoding)[0]
            yield pred
        else:
            step_num = 0
            while True:
                yield pose_sequence["data"][0]
                pose_sequence["data"] = self.refinement_step(step_num, pose_sequence, text_encoding)[0]
                step_num += 1

    def refinement_step(self, step_num, pose_sequence, text_encoding):
        batch_size = pose_sequence["data"].shape[0]
        pose_sequence["data"] = pose_sequence["data"].detach()
        batch_step_num = torch.repeat_interleave(torch.LongTensor([step_num]), batch_size).unsqueeze(1).to(self.device)
        step_encoding = self.step_encoder(self.step_embedding(batch_step_num))
        change_pred, coarse_pred = self.refine_pose_sequence(pose_sequence, text_encoding, step_encoding)
        cur_step_size = self.get_step_size(step_num + 1)
        prev_step_size = self.get_step_size(step_num) if step_num > 0 else 0
        step_size = cur_step_size - prev_step_size
        if self.blend:
            pred = (1 - step_size) * pose_sequence["data"] + step_size * change_pred
        else:
            pred = pose_sequence["data"] + step_size * change_pred
        return pred, coarse_pred, cur_step_size

    def embed_pose(self, pose_sequence_data):
        batch_size, seq_length, _, _ = pose_sequence_data.shape
        flat_pose_data = pose_sequence_data.reshape(batch_size, seq_length, -1)

        positions = torch.arange(0, seq_length, dtype=torch.long, device=self.device)
        if self.separate_positional_embedding:
            positional_embedding = self.pos_positional_embeddings(positions)
        else:
            positional_embedding = self.alpha_pose * self.positional_embeddings(positions)

        return self.pose_projection(flat_pose_data) + positional_embedding

    def encode_pose(self, pose_sequence, text_encoding, step_encoding=None):
        batch_size, seq_length, _, _ = pose_sequence["data"].shape
        pose_embedding = self.embed_pose(pose_sequence["data"])

        if step_encoding is not None:
            step_mask = torch.zeros([step_encoding.size(0), 1], dtype=torch.bool, device=self.device)
            pose_text_sequence = torch.cat([pose_embedding, text_encoding["data"], step_encoding], dim=1)
            pose_text_mask = torch.cat([pose_sequence["mask"], text_encoding["mask"], step_mask], dim=1)
        else:
            pose_text_sequence = torch.cat([pose_embedding, text_encoding["data"]], dim=1)
            pose_text_mask = torch.cat([pose_sequence["mask"], text_encoding["mask"]], dim=1)

        pose_encoding = self.__get_text_pose_encoder()(
            pose_text_sequence.transpose(0, 1),
            src_key_padding_mask=pose_text_mask,
        ).transpose(0, 1)[:, :seq_length, :]
        return pose_encoding

    def __get_text_pose_encoder(self):
        if hasattr(self, "text_pose_encoder"):
            return self.text_pose_encoder
        return self.pose_encoder

    def refine_pose_sequence(self, pose_sequence, text_encoding, step_encoding=None):
        batch_size, seq_length, _, _ = pose_sequence["data"].shape
        pose_encoding = self.encode_pose(pose_sequence, text_encoding, step_encoding)

        if self.model_variant == "multiscale":
            coarse_pose_pred = self.predict_coarse_pose(pose_encoding)
            fine_pose_pred = self.predict_fine_pose(pose_encoding, coarse_pose_pred)
            return (
                fine_pose_pred.reshape(batch_size, seq_length, *self.pose_dims),
                coarse_pose_pred.reshape(batch_size, seq_length, *self.coarse_dim),
            )

        flat_pose_projection = self.pose_diff_projection(pose_encoding)
        return flat_pose_projection.reshape(batch_size, seq_length, *self.pose_dims), None

    def get_step_size(self, step_num):
        if step_num < 2:
            return 0.1
        return np.log(step_num) / np.log(self.num_steps)

    def training_step(self, batch, *unused_args):
        return self.step(batch, *unused_args, phase="train")

    def validation_step(self, batch, *unused_args):
        return self.step(batch, *unused_args, phase="validation")

    def step(self, batch, *unused_args, phase: str):
        text_encoding, sequence_length = self.encode_text(batch["text"])
        pose = batch["pose"]

        batch_size, pose_seq_length, _, _ = pose["data"].shape
        pose_sequence = {
            "data": torch.stack([pose["data"][:, 0]] * pose_seq_length, dim=1),
            "mask": torch.logical_not(pose["inverse_mask"]),
        }

        coarse_pose_gt = extract_gt_coarse_pose(batch) if self.model_variant == "multiscale" else None

        if self.num_steps == 1:
            pred, coarse_pred = self.refine_pose_sequence(pose_sequence, text_encoding)
            l1_gold = pose["data"]
            refinement_loss = masked_mse_loss(l1_gold, pred, pose["confidence"], self.num_steps)
            coarse_loss = F.mse_loss(coarse_pose_gt, coarse_pred) if coarse_pose_gt is not None else 0
        else:
            refinement_loss = 0
            coarse_loss = 0
            for i in range(self.num_steps):
                pred, coarse_pred, step_size = self.refinement_step(i, pose_sequence, text_encoding)
                l1_gold = step_size * pose["data"] + (1 - step_size) * pose_sequence["data"]
                refinement_loss += masked_mse_loss(l1_gold, pred, pose["confidence"], self.num_steps)
                if coarse_pose_gt is not None:
                    coarse_loss += F.mse_loss(coarse_pose_gt, coarse_pred)

                teacher_forcing_step_level = np.random.rand(1)[0] < self.tf_p
                pose_sequence["data"] = l1_gold if phase == "validation" or teacher_forcing_step_level else pred

                if phase == "train":
                    pose_sequence["data"] = pose_sequence["data"] + torch.randn_like(pose_sequence["data"]) * self.noise_epsilon

        sequence_length_loss = F.mse_loss(sequence_length, pose["length"])
        loss = refinement_loss + self.seq_len_weight * sequence_length_loss + coarse_loss

        self.log(phase + "_seq_length_loss", sequence_length_loss, batch_size=batch_size)
        self.log(phase + "_refinement_loss", refinement_loss, batch_size=batch_size)
        if coarse_pose_gt is not None:
            self.log(phase + "_coarse_loss", coarse_loss, batch_size=batch_size)
        self.log(phase + "_loss", loss, batch_size=batch_size)

        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr)


BODY_COARSE_POSE = {
    "RIGHT_ARM_UPPER": range(2, 4),
    "RIGHT_ARM_LOWER": range(3, 5),
    "LEFT_ARM_UPPER": range(5, 7),
    "LEFT_ARM_LOWER": range(6, 8),
    "HEAD": range(25, 95),
    "LEFT_THUMB_LOWER": [95, 96, 97],
    "LEFT_THUMB_UPPER": [97, 98, 99],
    "LEFT_INDEX_LOWER": [95, 100, 101],
    "LEFT_INDEX_UPPER": [101, 102, 103],
    "LEFT_MIDDLE_LOWER": [95, 104, 105],
    "LEFT_MIDDLE_UPPER": [105, 106, 107],
    "LEFT_RING_LOWER": [95, 108, 109],
    "LEFT_RING_UPPER": [109, 110, 111],
    "LEFT_LITTLE_LOWER": [95, 112, 113],
    "LEFT_LITTLE_UPPER": [113, 114, 115],
    "RIGHT_THUMB_LOWER": [116, 117, 118],
    "RIGHT_THUMB_UPPER": [118, 119, 120],
    "RIGHT_INDEX_LOWER": [116, 121, 122],
    "RIGHT_INDEX_UPPER": [122, 123, 124],
    "RIGHT_MIDDLE_LOWER": [116, 125, 126],
    "RIGHT_MIDDLE_UPPER": [126, 127, 128],
    "RIGHT_RING_LOWER": [116, 129, 130],
    "RIGHT_RING_UPPER": [130, 131, 132],
    "RIGHT_LITTLE_LOWER": [116, 133, 134],
    "RIGHT_LITTLE_UPPER": [134, 135, 136],
}


def extract_gt_coarse_pose(batch):
    pose_xy = batch["pose"]["data"]
    conf = batch["pose"]["confidence"]
    coarse_pose = []
    for body_part in [
        "RIGHT_ARM_UPPER",
        "RIGHT_ARM_LOWER",
        "LEFT_ARM_UPPER",
        "LEFT_ARM_LOWER",
        "HEAD",
        "LEFT_THUMB_LOWER",
        "LEFT_THUMB_UPPER",
        "LEFT_INDEX_LOWER",
        "LEFT_INDEX_UPPER",
        "LEFT_MIDDLE_LOWER",
        "LEFT_MIDDLE_UPPER",
        "LEFT_RING_LOWER",
        "LEFT_RING_UPPER",
        "LEFT_LITTLE_LOWER",
        "LEFT_LITTLE_UPPER",
        "RIGHT_THUMB_LOWER",
        "RIGHT_THUMB_UPPER",
        "RIGHT_INDEX_LOWER",
        "RIGHT_INDEX_UPPER",
        "RIGHT_MIDDLE_LOWER",
        "RIGHT_MIDDLE_UPPER",
        "RIGHT_RING_LOWER",
        "RIGHT_RING_UPPER",
        "RIGHT_LITTLE_LOWER",
        "RIGHT_LITTLE_UPPER",
    ]:
        idx = list(BODY_COARSE_POSE[body_part])
        part_xy = pose_xy[:, :, idx, :]
        part_w = conf[:, :, idx].unsqueeze(-1)

        num = (part_xy * part_w).sum(dim=2)
        den = part_w.sum(dim=2).clamp_min(1e-6)
        part_pose = num / den
        coarse_pose.append(part_pose)

    return torch.stack(coarse_pose, dim=2)
