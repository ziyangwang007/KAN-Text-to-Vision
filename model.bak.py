from typing import List
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
from torch.optim import optimizer, Adam
from kan import KAN
from KanTransformerEncorder import KANTransformerEncoderLayer

EPSILON = 1e-4
START_LEARNING_RATE = 1e-3
MAX_SEQ_LEN = 200


def masked_mse_loss(pose: torch.Tensor, pose_hat: torch.Tensor, confidence: torch.Tensor, model_num_steps: int = 10):
    # Loss by confidence. If missing joint, no loss. If less likely joint, less gradients.
    sq_error = torch.pow(pose - pose_hat, 2).sum(-1)
    num_steps_norm = np.log(model_num_steps) ** 2 if model_num_steps != 1 else 1  # normalization of the loss by the
    # model's step number
    return (sq_error * confidence).mean() * num_steps_norm


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
            blend: bool = True
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
        self.coarse_dim = (len(BODY_COARSE_POSE), 2)

        pose_dim = int(np.prod(pose_dims))

        # Embedding layers

        self.embedding = nn.Embedding(
            num_embeddings=len(tokenizer),
            embedding_dim=hidden_dim,
            padding_idx=tokenizer.pad_token_id,
        )

        self.step_embedding = nn.Embedding(
            num_embeddings=num_steps, embedding_dim=hidden_dim
        )

        if separate_positional_embedding:
            self.pos_positional_embeddings = nn.Embedding(
                num_embeddings=max_seq_size, embedding_dim=hidden_dim
            )
            self.text_positional_embeddings = nn.Embedding(
                num_embeddings=max_seq_size, embedding_dim=hidden_dim
            )

        else:
            self.positional_embeddings = nn.Embedding(
                num_embeddings=max_seq_size, embedding_dim=hidden_dim
            )

            # positional embedding scalars
            self.alpha_pose = nn.Parameter(torch.randn(1))
            self.alpha_text = nn.Parameter(torch.randn(1))

        text_encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=encoder_heads,
                                                        dim_feedforward=encoder_dim_feedforward)

        # encoding layers
        encoder_layer = KANTransformerEncoderLayer(d_model=hidden_dim, nhead=encoder_heads,
                                                   dim_feedforward=encoder_dim_feedforward)

        self.text_encoder = nn.TransformerEncoder(text_encoder_layer, num_layers=text_encoder_depth)
        self.pose_encoder = nn.TransformerEncoder(encoder_layer, num_layers=pose_encoder_depth)

        # step encoder
        self.step_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU()
        )

        # Predict sequence length
        self.seq_length = nn.Linear(hidden_dim, 1)

        self.coarse_pose_dim = len(BODY_COARSE_POSE) * 2

        self.pose_projection = nn.Linear(pose_dim, hidden_dim)

        self.coarse_pose_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.coarse_pose_dim),
        )
        self.fine_pose_projection = nn.Sequential(
            nn.Linear(hidden_dim + self.coarse_pose_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, pose_dim),
        )  # predict fine pose from coarse pose and fine pose

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
        encoded = self.text_encoder(embedding.transpose(0, 1),
                                    src_key_padding_mask=tokenized["attention_mask"]).transpose(0, 1)

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
            pred, _ = self.refine_pose_sequence(pose_sequence, text_encoding)
            yield pred
        else:
            step_num = 0
            while True:
                yield pose_sequence["data"][0]
                pose_sequence["data"] = self.refinement_step(step_num, pose_sequence, text_encoding)[0]
                step_num += 1

    def refinement_step(self, step_num, pose_sequence, text_encoding):
        batch_size = pose_sequence["data"].shape[0]
        pose_sequence["data"] = pose_sequence["data"].detach()  # Detach from graph
        batch_step_num = torch.repeat_interleave(torch.LongTensor([step_num]),
                                                 batch_size).unsqueeze(1).to(self.device)
        step_encoding = self.step_encoder(self.step_embedding(batch_step_num))
        change_pred, coarse_pred = self.refine_pose_sequence(pose_sequence, text_encoding, step_encoding)
        cur_step_size = self.get_step_size(step_num + 1)
        prev_step_size = self.get_step_size(step_num) if step_num > 0 else 0
        step_size = cur_step_size - prev_step_size
        if self.blend:
            pred = (1 - step_size) * pose_sequence["data"] + step_size * change_pred
        else:
            pred = pose_sequence["data"] + step_size * change_pred  # add
        return pred, coarse_pred, cur_step_size

    def embed_pose(self, pose_sequence_data):
        batch_size, seq_length, _, _ = pose_sequence_data.shape
        flat_pose_data = pose_sequence_data.reshape(batch_size, seq_length, -1)

        positions = torch.arange(0, seq_length, dtype=torch.long, device=self.device)
        if self.separate_positional_embedding:
            positional_embedding = self.pos_positional_embeddings(positions)
        else:
            positional_embedding = self.alpha_pose * self.positional_embeddings(positions)

        # Encode pose sequence
        pose_embedding = self.pose_projection(flat_pose_data) + positional_embedding
        return pose_embedding

    def encode_pose(self, pose_sequence, text_encoding, step_encoding=None):
        batch_size, seq_length, _, _ = pose_sequence["data"].shape

        # Encode pose sequence
        pose_embedding = self.embed_pose(pose_sequence["data"])

        if step_encoding is not None:
            step_mask = torch.zeros([step_encoding.size(0), 1], dtype=torch.bool, device=self.device)

        pose_text_sequence = torch.cat([pose_embedding, text_encoding["data"], step_encoding], dim=1)
        pose_text_mask = torch.cat(
            [pose_sequence["mask"], text_encoding["mask"], step_mask], dim=1
        )

        pose_encoding = self.__get_text_pose_encoder()(
            pose_text_sequence.transpose(0, 1), src_key_padding_mask=pose_text_mask
        ).transpose(0, 1)[:, :seq_length, :]

        return pose_encoding

    def __get_text_pose_encoder(self):
        if hasattr(self, "text_pose_encoder"):
            return self.text_pose_encoder
        else:
            return self.pose_encoder

    def refine_pose_sequence(self, pose_sequence, text_encoding, step_encoding=None):
        batch_size, seq_length, _, _ = pose_sequence["data"].shape
        pose_encoding = self.encode_pose(pose_sequence, text_encoding, step_encoding)
        # predict coarse pose dim : (batch_size, seq_length, 2*len(BODY_COARSE_POSE))
        coarse_pose_pred = self.predict_coarse_pose(pose_encoding)
        # predict fine pose dim: (batch_size, seq_length, num_keypoints, pose_dim)
        fine_pose_pred = self.predict_fine_pose(pose_encoding, coarse_pose_pred)

        return fine_pose_pred.reshape(batch_size, seq_length, *self.pose_dims), coarse_pose_pred.reshape(batch_size,
                                                                                                         seq_length,
                                                                                                         *self.coarse_dim)

    def get_step_size(self, step_num):
        if step_num < 2:
            return 0.1
        else:
            return np.log(step_num) / np.log(self.num_steps)

    def training_step(self, batch, *unused_args):
        return self.step(batch, *unused_args, phase="train")

    def validation_step(self, batch, *unused_args):
        return self.step(batch, *unused_args, phase="validation")

    def step(self, batch, *unused_args, phase: str):
        """
        @param batch: data batch
        @param phase: either "train" or "validation"
        """
        text_encoding, sequence_length = self.encode_text(batch["text"])
        pose = batch["pose"]

        # Repeat the first frame for initial prediction
        batch_size, pose_seq_length, num_keypoints, _ = pose["data"].shape

        pose_sequence = {
            "data": torch.stack([pose["data"][:, 0]] * pose_seq_length, dim=1),
            "mask": torch.logical_not(pose["inverse_mask"])
        }
        coarse_pose_gt = extract_gt_coarse_pose(batch)
        if self.num_steps == 1:
            pred, coarse_pred = self.refine_pose_sequence(pose_sequence, text_encoding)
            l1_gold = pose["data"]
            coarse_loss = F.mse_loss(coarse_pose_gt, coarse_pred)
            refinement_loss = masked_mse_loss(l1_gold, pred, pose["confidence"], self.num_steps)
        else:
            refinement_loss = 0
            coarse_loss = 0
            for i in range(self.num_steps):
                pred, coarse_pred, step_size = self.refinement_step(i, pose_sequence, text_encoding)
                l1_gold = step_size * pose["data"] + (1 - step_size) * pose_sequence["data"]
                refinement_loss += masked_mse_loss(l1_gold, pred, pose["confidence"], self.num_steps)
                # print(coarse_pose_gt.shape, coarse_pred.shape)
                # Coarse pose loss
                coarse_loss += F.mse_loss(coarse_pose_gt, coarse_pred)

                teacher_forcing_step_level = np.random.rand(1)[0] < self.tf_p
                pose_sequence["data"] = l1_gold if phase == "validation" or teacher_forcing_step_level else pred

                if phase == "train":  # add just a little noise while training
                    pose_sequence["data"] = pose_sequence["data"] + torch.randn_like(pose_sequence["data"]) * \
                                            self.noise_epsilon

        sequence_length_loss = F.mse_loss(sequence_length, pose["length"])
        loss = refinement_loss + self.seq_len_weight * sequence_length_loss + coarse_loss

        self.log(phase + "_seq_length_loss", sequence_length_loss, batch_size=batch_size)
        self.log(phase + "_refinement_loss", refinement_loss, batch_size=batch_size)
        self.log(phase + "_coarse_loss", coarse_loss, batch_size=batch_size)
        self.log(phase + "_loss", loss, batch_size=batch_size)

        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr)


# Point 0: Wrist
# Points 1-4: Thumb (from palm to tip)
# Points 5-8: Index finger
# Points 9-12: Middle finger
# Points 13-16: Ring finger
# Points 17-20: Little finger

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
    # "LEFT_HAND": range(95, 116),
    # "RIGHT_HAND": range(116, 137)
}


def extract_gt_coarse_pose(batch):
    # print(batch["pose"]["data"].shape) torch.Size([16, 146, 137, 2])
    pose_data = batch["pose"]["data"]
    coarse_pose = []
    for body_part in [
        "RIGHT_ARM_UPPER",
        "RIGHT_ARM_LOWER",
        "LEFT_ARM_UPPER",
        "LEFT_ARM_LOWER",
        "HEAD",

        "LEFT_THUMB_LOWER", "LEFT_THUMB_UPPER",
        "LEFT_INDEX_LOWER", "LEFT_INDEX_UPPER",
        "LEFT_MIDDLE_LOWER", "LEFT_MIDDLE_UPPER",
        "LEFT_RING_LOWER", "LEFT_RING_UPPER",
        "LEFT_LITTLE_LOWER", "LEFT_LITTLE_UPPER",

        "RIGHT_THUMB_LOWER", "RIGHT_THUMB_UPPER",
        "RIGHT_INDEX_LOWER", "RIGHT_INDEX_UPPER",
        "RIGHT_MIDDLE_LOWER", "RIGHT_MIDDLE_UPPER",
        "RIGHT_RING_LOWER", "RIGHT_RING_UPPER",
        "RIGHT_LITTLE_LOWER", "RIGHT_LITTLE_UPPER"
    ]:
        # 对所有部位计算范围内点的平均值
        range_indices = BODY_COARSE_POSE[body_part]
        part_pose = pose_data[:, :, range_indices, :].mean(dim=2)
        coarse_pose.append(part_pose)
    return torch.stack(coarse_pose, dim=2)


if __name__ == '__main__':
    data = [[0.0116, -0.6531],
            [0.0113, 0.0193],
            [-0.4860, -0.0157],
            [-0.8990, 0.6426],
            [-0.8009, 1.0906],
            [0.5016, 0.0544],
            [0.8514, 0.7197],
            [1.0056, 1.1819],
            [0.0000, 0.0000],
            [0.0000, 0.0000],
            [0.0000, 0.0000],
            [0.0000, 0.0000],
            [0.0000, 0.0000],
            [0.0000, 0.0000],
            [0.0000, 0.0000],
            [-0.0936, -0.7370],
            [0.1302, -0.7300],
            [-0.2198, -0.6109],
            [0.2775, -0.5831],
            [0.0000, 0.0000],
            [0.0000, 0.0000],
            [0.0000, 0.0000],
            [0.0000, 0.0000],
            [0.0000, 0.0000],
            [0.0000, 0.0000],
            [-0.2315, -0.6910],  # start
            [-0.2288, -0.6310],
            [-0.2124, -0.5819],
            [-0.2097, -0.5191],
            [-0.2069, -0.4563],
            [-0.1660, -0.4100],
            [-0.1169, -0.3854],
            [-0.0541, -0.3663],
            [0.0114, -0.3636],
            [0.0769, -0.3663],
            [0.1396, -0.3827],
            [0.1888, -0.4072],
            [0.2324, -0.4509],
            [0.2515, -0.5027],
            [0.2542, -0.5655],
            [0.2706, -0.6255],
            [0.2733, -0.6856],
            [-0.1660, -0.7592],
            [-0.1414, -0.7838],
            [-0.1005, -0.8002],
            [-0.0568, -0.8002],
            [-0.0241, -0.7811],
            [0.0687, -0.7783],
            [0.1014, -0.7893],
            [0.1424, -0.7893],
            [0.1833, -0.7783],
            [0.2079, -0.7511],
            [0.0277, -0.7374],
            [0.0250, -0.7101],
            [0.0223, -0.6747],
            [0.0250, -0.6501],
            [-0.0214, -0.6010],
            [-0.0023, -0.5901],
            [0.0196, -0.5873],
            [0.0441, -0.5901],
            [0.0632, -0.5955],
            [-0.1251, -0.7183],
            [-0.1005, -0.7347],
            [-0.0732, -0.7374],
            [-0.0432, -0.7156],
            [-0.0732, -0.7129],
            [-0.1005, -0.7129],
            [0.0851, -0.7129],
            [0.1178, -0.7320],
            [0.1424, -0.7320],
            [0.1669, -0.7129],
            [0.1424, -0.7047],
            [0.1178, -0.7074],
            [-0.0705, -0.5109],
            [-0.0323, -0.5328],
            [-0.0050, -0.5409],
            [0.0168, -0.5382],
            [0.0414, -0.5382],
            [0.0741, -0.5218],
            [0.1014, -0.5000],
            [0.0741, -0.4973],
            [0.0414, -0.4945],
            [0.0141, -0.4945],
            [-0.0077, -0.4973],
            [-0.0350, -0.4973],
            [-0.0541, -0.5136],
            [-0.0050, -0.5191],
            [0.0168, -0.5164],
            [0.0387, -0.5191],
            [0.0905, -0.5082],
            [0.0414, -0.5191],
            [0.0168, -0.5191],
            [-0.0050, -0.5191],
            [-0.0841, -0.7292],
            [0.1314, -0.7183],  # end
            [1.0179, 1.1938],
            [0.9517, 1.2711],
            [0.9379, 1.4337],
            [0.9710, 1.4696],
            [0.7890, 1.4585],
            [0.9793, 1.4503],
            [0.9876, 1.4751],
            [1.0013, 1.4806],
            [1.0482, 1.5137],
            [1.0124, 1.3179],
            [0.9820, 1.4751],
            [1.0041, 1.4861],
            [1.0592, 1.4944],
            [1.0262, 1.3124],
            [0.9710, 1.4751],
            [1.0041, 1.4779],
            [1.0648, 1.4779],
            [1.0868, 1.3234],
            [0.9903, 1.4779],
            [1.0041, 1.4779],
            [1.0758, 1.4117],
            [-0.8227, 1.0902],
            [-0.7828, 1.2641],
            [-0.7201, 1.3896],
            [-0.7315, 1.4352],
            [-0.7258, 1.4751],
            [-0.7087, 1.3867],
            [-0.6802, 1.4523],
            [-0.6574, 1.4608],
            [-0.6659, 1.4951],
            [-0.7828, 1.2755],
            [-0.6631, 1.1815],
            [-0.6488, 1.4580],
            [-0.6659, 1.4951],
            [-0.7971, 1.2983],
            [-0.6631, 1.1843],
            [-0.6460, 1.4580],
            [-0.6460, 1.4779],
            [-0.8199, 1.3269],
            [-0.5890, 1.2242],
            [-0.7800, 1.4038],
            [-0.7828, 1.4038]]
    # tonumpy
    data = np.array(data)
    result = []
    for part, range_indices in BODY_COARSE_POSE.items():
        part_pose = data[range_indices, :].mean(axis=0)
        print(part, data[range_indices, :])
        result.append(part_pose)
    import matplotlib.pyplot as plt

    # vituralize the result
    result = np.array(result)
    print(result)
    x = result[:, 0]
    y = result[:, 1]

    # Create the plot
    plt.figure(figsize=(10, 10))
    plt.scatter(x, y, c='blue', s=50)

    # Add labels for each point
    for i, (x_i, y_i) in enumerate(zip(x, y)):
        plt.annotate(f'Point {i}', (x_i, y_i), xytext=(5, 5), textcoords='offset points')
    plt.show()

    # LEFT_ARM_UPPER [-0.6925   0.31345]
    # LEFT_ARM_LOWER [-0.84995  0.8666 ]
    # RIGHT_ARM_UPPER [0.6765  0.38705]
    # RIGHT_ARM_LOWER [0.9285 0.9508]
    # LEFT_ARM_HANDCONNECTION [0.0113 0.0193]
    # RIGHT_ARM_HANDCONNECTION [0.0113 0.0193]
    # HEAD [ 0.01924857 -0.60452429]
    # LEFT_HAND [0.99831905 1.42638571]
    # RIGHT_HAND [-0.71331429  1.3636381 ]

    data[:, 0] = data[:, 0] * 640 + 640
    data[:, 1] = data[:, 1] * 340 + 340

    # Create a new figure
    plt.figure(figsize=(12, 8))

    # Plot all points
    # plt.scatter(data[:, 0], data[:, 1], c='gray', alpha=0.5, label='All points')

    # Plot and label points for each body part
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan', 'magenta']
    for (part, range_indices), color in zip(BODY_COARSE_POSE.items(), colors):
        part_data = data[range_indices]
        plt.scatter(part_data[:, 0], part_data[:, 1], c=color, label=part)

        # Label points with their values
        for i, (x, y) in enumerate(part_data):
            plt.annotate(f'({x:.1f}, {y:.1f})', (x, y), xytext=(5, 5),
                         textcoords='offset points', fontsize=8,
                         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.7))

    # Set plot limits and labels
    plt.xlim(0, 1280)
    plt.ylim(0, 680)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Body Pose Visualization')
    plt.legend()

    # Show the plot
    plt.tight_layout()
    plt.show()
