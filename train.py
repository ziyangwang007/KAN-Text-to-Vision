import os
import pickle
import statistics
import yaml
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import sys
import time
import torch
from pytorch_lightning.callbacks import Callback


rootdir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, rootdir)

from data.collator import zero_pad_collator
from args import args
from data.data import get_dataset
from model import IterativeTextGuidedPoseGenerationModel
from tokenizer.hamnosys.hamnosys_tokenizer import HamNoSysTokenizer
from predict import pred
from constants import num_steps_to_batch_size, batch_size_to_accumulate, DATASET_SIZE
import os



def get_optimizer(opt_str):
    if opt_str == "Adam":
        return Adam
    elif opt_str == "SGD":
        return SGD
    else:
        raise Exception("optimizer not supported. use Adam or SGD.")


def get_model_args(args, num_pose_joints, num_pose_dims):
    model_args = dict(tokenizer=HamNoSysTokenizer(),
                      pose_dims=(num_pose_joints, num_pose_dims),
                      hidden_dim=args["hidden_dim"],
                      text_encoder_depth=args["text_encoder_depth"],
                      pose_encoder_depth=args["pose_encoder_depth"],
                      encoder_heads=args["encoder_heads"],
                      max_seq_size=args["max_seq_size"],
                      num_steps=args["num_steps"],
                      tf_p=args["tf_p"],
                      seq_len_weight=args["seq_len_weight"],
                      noise_epsilon=args["noise_epsilon"],
                      optimizer_fn=get_optimizer(args["optimizer"]),
                      separate_positional_embedding=args["separate_positional_embedding"],
                      encoder_dim_feedforward=args["encoder_dim_feedforward"],
                      num_pose_projection_layers=args["num_pose_projection_layers"],
                      model_variant=args["model_variant"],
                      text_encoder_type=args["text_encoder_type"],
                      text_pose_encoder_type=args["text_pose_encoder_type"],
                      pose_projection_type=args["pose_projection_type"],
                      kan_hidden_dim=args["kan_hidden_dim"],
                      )

    return model_args

class EfficiencyCallback(Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        self.epoch_start_time = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def on_train_epoch_end(self, trainer, pl_module):
        epoch_time = time.perf_counter() - self.epoch_start_time
        peak_mem_gb = 0.0
        if torch.cuda.is_available():
            peak_mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)

        print(
            f"[EFFICIENCY] epoch={trainer.current_epoch + 1} "
            f"time={epoch_time:.2f}s "
            f"peak_train_mem={peak_mem_gb:.3f}GB"
        )

        if trainer.logger is not None:
            trainer.logger.log_metrics({
                "efficiency/train_epoch_time_sec": epoch_time,
                "efficiency/train_peak_mem_gb": peak_mem_gb,
            }, step=trainer.current_epoch + 1)


def benchmark_inference(model, dataloader, warmup=3, steps=10):
    model.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    batch = next(iter(dataloader))

    def move_to_device(obj):
        if isinstance(obj, dict):
            return {k: move_to_device(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [move_to_device(v) for v in obj]
        elif isinstance(obj, tuple):
            return tuple(move_to_device(v) for v in obj)
        elif hasattr(obj, "to"):
            return obj.to(device)
        return obj

    batch = move_to_device(batch)

    print("model device:", next(model.parameters()).device)
    print("pose device:", batch["pose"]["data"].device)
    print("batch size:", batch["pose"]["data"].shape[0])

    @torch.no_grad()
    def run_once():
        text_encoding, _ = model.encode_text(batch["text"])
        pose = batch["pose"]
        _, pose_seq_length, _, _ = pose["data"].shape

        pose_sequence = {
            "data": torch.stack([pose["data"][:, 0]] * pose_seq_length, dim=1),
            "mask": torch.logical_not(pose["inverse_mask"])
        }

        if model.num_steps == 1:
            pred, coarse_pred = model.refine_pose_sequence(pose_sequence, text_encoding)
            return pred, coarse_pred
        else:
            pred = None
            coarse_pred = None
            for i in range(model.num_steps):
                pred, coarse_pred, _ = model.refinement_step(i, pose_sequence, text_encoding)
                pose_sequence["data"] = pred
            return pred, coarse_pred

    for _ in range(warmup):
        _ = run_once()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    times = []
    for _ in range(steps):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()

        _ = run_once()

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end = time.perf_counter()
        times.append(end - start)

    peak_alloc_gb = 0.0
    peak_reserved_gb = 0.0
    if torch.cuda.is_available():
        peak_alloc_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
        peak_reserved_gb = torch.cuda.max_memory_reserved() / (1024 ** 3)

    mean_ms = statistics.mean(times) * 1000
    std_ms = statistics.pstdev(times) * 1000 if len(times) > 1 else 0.0

    print(f"[EFFICIENCY] inference_latency_mean={mean_ms:.2f}ms")
    print(f"[EFFICIENCY] inference_latency_std={std_ms:.2f}ms")
    print(f"[EFFICIENCY] inference_peak_alloc={peak_alloc_gb:.3f}GB")
    print(f"[EFFICIENCY] inference_peak_reserved={peak_reserved_gb:.3f}GB")

    return {
        "inference_latency_mean_ms": mean_ms,
        "inference_latency_std_ms": std_ms,
        "inference_peak_alloc_gb": peak_alloc_gb,
        "inference_peak_reserved_gb": peak_reserved_gb,
    }

if __name__ == '__main__':
    args = vars(args).copy()
    if args["config_file"]:  # override args with yaml config file
        with open(args["config_file"], 'r') as f:
            args.update(yaml.safe_load(f) or {})

    LOGGER = None
    # if not args["no_wandb"]:
    #     LOGGER = WandbLogger(project="ham2pose", log_model=False, offline=False, id=args["model_name"])
    #     if LOGGER.experiment.sweep_id is None:
    #         LOGGER.log_hyperparams(args)

    args["batch_size"] = num_steps_to_batch_size[args["num_steps"]]
    test_size = int(0.1*DATASET_SIZE)
    train_split = f'test[{test_size}:]+train'
    test_split = f'test[:{test_size}]'

    # if args["leave_out"] != "":
    #     train_dataset, test_dataset = get_dataset(name=args["dataset"], poses=args["pose"], fps=args["fps"],
    #                                 components=args["pose_components"], leave_out=args["leave_out"],
    #                                 max_seq_size=args["max_seq_size"], split=train_split)
    # else:
    #     train_dataset = get_dataset(name=args["dataset"], poses=args["pose"], fps=args["fps"],
    #                                components=args["pose_components"], max_seq_size=args["max_seq_size"],
    #                                 split=train_split)
    #     test_dataset = get_dataset(name=args["dataset"], poses=args["pose"], fps=args["fps"],
    #                                components=args["pose_components"], max_seq_size=args["max_seq_size"],
    #                                split=test_split)
    # save train and test datasets
    # os.makedirs(f"./temp/train", exist_ok=True)
    # with open("temp/dwpose/train_dataset_dwpose.pkl", "wb") as f:
    #     pickle.dump(train_dataset, f)
    # # Saving test_dataset
    # with open("temp/dwpose/test_dataset_dwpose.pkl", "wb") as f:
    #     pickle.dump(test_dataset, f)
    # print("Datasets saved successfully.")
    # load train and test datasets
    # with open("./temp/train/train_dataset_old.pkl", "rb") as f:
    #     train_dataset = pickle.load(f)
    with open("./temp/train/test_dataset_old.pkl", "rb") as f:
        test_dataset = pickle.load(f)
        train_dataset = test_dataset
    print("Load dataset from test_dataset_old.pkl")
    # print(train_dataset[0])
    # print(train_dataset[0]['pose']['data'][0])

    train_loader = DataLoader(train_dataset, batch_size=args["batch_size"],
                              shuffle=True, collate_fn=zero_pad_collator)
    test_loader = DataLoader(test_dataset, batch_size=args["batch_size"],
                             collate_fn=zero_pad_collator)

    _, num_pose_joints, num_pose_dims = train_dataset[0]["pose"]["data"].shape

    model_args = get_model_args(args, num_pose_joints, num_pose_dims)


    if os.path.isfile(f"./models/{args['model_name']}/{args['ckpt']}/model.ckpt"):
        model = IterativeTextGuidedPoseGenerationModel.load_from_checkpoint(f"./models/{args['model_name']}/"
                                                                            f"{args['ckpt']}/model.ckpt", **model_args)
    else:
        model = IterativeTextGuidedPoseGenerationModel(**model_args)

    callbacks = [EfficiencyCallback()]
    if LOGGER is not None:
        os.makedirs(f"./models/{args['model_name']}", exist_ok=True)
        callbacks.append(ModelCheckpoint(
            dirpath=f"./models/{args['model_name']}",
            filename="model",
            verbose=True,
            save_top_k=3,
            monitor='train_loss',
            mode='min'
        ))

    trainer = pl.Trainer(
        max_epochs=args["max_epochs"],
        logger=LOGGER,
        callbacks=callbacks,
        accelerator='gpu',
        devices=args['num_gpus'],
        accumulate_grad_batches=batch_size_to_accumulate[args['batch_size']],
    )
    trainer.fit(model, train_dataloaders=train_loader)
    metrics = benchmark_inference(model, test_loader, warmup=3, steps=10)
    print(metrics)
    # evaluate
    model = IterativeTextGuidedPoseGenerationModel.load_from_checkpoint(f"./models/{args['model_name']}/"
                                                                        f"{args['ckpt']}/model.ckpt", **model_args)
    model.eval()

    # test seq_len_predictor
    diffs = []
    for d in test_dataset:
        _, seq_len = model.encode_text([d["text"]])
        real_seq_len = len(d["pose"]["data"])
        diff = np.abs(real_seq_len-seq_len.item())
        diffs.append(diff)
    print(np.mean(diffs), np.median(diffs), np.max(diffs))

    pred(model, train_dataset, os.path.join(f"./models/{args['model_name']}", args['output_dir'], "train"))
    pred(model, test_dataset, os.path.join(f"./models/{args['model_name']}", args['output_dir'], "test"))
