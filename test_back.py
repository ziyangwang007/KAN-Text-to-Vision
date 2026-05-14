import os
import json
import pickle
import random
import time
import torch.multiprocessing as mp
from functools import partial
import tqdm
import yaml
import numpy as np
import matplotlib.pyplot as plt
import torch
import sys
from fastdtw import fastdtw

rootdir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, rootdir)
import warnings

warnings.filterwarnings('ignore')
from args import args
from data.data import get_dataset
from constants import DATASET_SIZE, num_steps_to_batch_size
from model import IterativeTextGuidedPoseGenerationModel
from train import get_model_args
from predict import pred, predict_pose
from metrics import get_poses_ranks, __compare_pred_to_video, __compare_pred_to_video_dtw

random.seed(42)

def combine_results(experiment_name, results_path):
    results = dict()
    for file in os.listdir(results_path):
        if experiment_name in file:
            with open(os.path.join(results_path, file)) as f:
                results.update(json.load(f))
    return np.mean(list(results.values())), np.median(list(results.values()))


def get_lang(sign_id):
    if "pjm" in sign_id:
        return "pjm"
    elif "gsl" in sign_id:
        return "gsl"
    elif sign_id.isnumeric():
        return "dgs"
    else:
        return "lsf"


def get_results_by_language(filename, num_files=5):
    paths = [os.path.join(filename + f"_{i}.txt") for i in range(num_files)]
    languages = {"pjm", "dgs", "gsl", "lsf"}
    all_results = {lang: {"pred_rank1": 0, "pred_rank5": 0, "pred_rank10": 0, "gt_rank1": 0, "gt_rank5": 0,
                          "gt_rank10": 0} for lang in languages}
    lang2count = {lang: 0 for lang in languages}
    for path in paths:
        with open(path, 'r') as f:
            lines = f.readlines()
            for i in range(1, len(lines), 2):
                if lines[i].startswith("rank"):
                    break
                lang = get_lang(lines[i].split(" ")[0])
                lang2count[lang] += 1
                dist, pred_rank1, pred_rank5, pred_rank10, gt_rank1, gt_rank5, gt_rank10 = lines[i + 1].strip().split(
                    ", ")
                all_results[lang]["pred_rank1"] += int(pred_rank1 == "True")
                all_results[lang]["pred_rank5"] += int(pred_rank5 == "True")
                all_results[lang]["pred_rank10"] += int(pred_rank10 == "True")
                all_results[lang]["gt_rank1"] += int(gt_rank1 == "True")
                all_results[lang]["gt_rank5"] += int(gt_rank5 == "True")
                all_results[lang]["gt_rank10"] += int(gt_rank10 == "True")

    for lang, ranks in all_results.items():
        for rank, rank_count in ranks.items():
            rank_mean = rank_count / lang2count[lang]
            print(f"{rank} of {lang} is: {rank_count}/{lang2count[lang]}= {rank_mean}")


def test_seq_len(model, dataset, model_name, save_plots=True):
    """
    Evaluate sequence length predictor.

    Reports:
      - raw predictor length (from encode_text)
      - inference-time length (round + clip to [min_seq_size, max_seq_size]) consistent with forward()

    Metrics:
      - MAE/MSE in frames
      - relative MAE/MSE (%)
      - mean/median (raw/clipped) for convenience

    Saves:
      - per-sample diffs json (raw and clipped)
      - summary json (overall + per-language)
      - histograms (optional)
    """
    os.makedirs(f"models/{model_name}/results", exist_ok=True)

    # Per-sample records
    rec = {}  # id -> dict

    # Buckets
    langs = {"pjm", "dgs", "gsl", "lsf"}
    bucket = {l: [] for l in langs}  # list of per-sample dicts

    model_min = int(getattr(model, "min_seq_size", 20))
    model_max = int(getattr(model, "max_seq_size", 200))

    def _clip_inference_len(x: float) -> int:
        # match forward(): round then clip
        x_int = int(round(float(x)))
        return max(min(x_int, model_max), model_min)

    # Collect errors
    raw_err_frames = []
    clip_err_frames = []
    raw_err_rel = []
    clip_err_rel = []

    for d in dataset:
        # predicted
        _, seq_len = model.encode_text([d["text"]])  # seq_len is tensor shape [1] or [1,1]
        pred_raw = float(seq_len.item())
        pred_clip = float(_clip_inference_len(pred_raw))

        # ground truth
        gt = float(d["pose"]["length"].item())
        if gt <= 0:
            continue

        # errors (frames)
        e_raw = pred_raw - gt
        e_clip = pred_clip - gt

        # relative errors
        r_raw = e_raw / gt
        r_clip = e_clip / gt

        sid = d["id"]
        lang = get_lang(sid)

        one = {
            "id": sid,
            "lang": lang,
            "gt": gt,
            "pred_raw": pred_raw,
            "pred_clip": pred_clip,
            "err_raw_frames": e_raw,
            "err_clip_frames": e_clip,
            "err_raw_rel": r_raw,
            "err_clip_rel": r_clip,
        }
        rec[sid] = one
        if lang in bucket:
            bucket[lang].append(one)

        raw_err_frames.append(e_raw)
        clip_err_frames.append(e_clip)
        raw_err_rel.append(r_raw)
        clip_err_rel.append(r_clip)

    def _metrics(err_frames, err_rel):
        err_frames = np.array(err_frames, dtype=np.float64)
        err_rel = np.array(err_rel, dtype=np.float64)

        mae_frames = float(np.mean(np.abs(err_frames))) if len(err_frames) else float("nan")
        mse_frames = float(np.mean(err_frames ** 2)) if len(err_frames) else float("nan")
        mean_frames = float(np.mean(err_frames)) if len(err_frames) else float("nan")
        median_frames = float(np.median(err_frames)) if len(err_frames) else float("nan")

        mae_rel = float(np.mean(np.abs(err_rel))) if len(err_rel) else float("nan")
        mse_rel = float(np.mean(err_rel ** 2)) if len(err_rel) else float("nan")
        mean_rel = float(np.mean(err_rel)) if len(err_rel) else float("nan")
        median_rel = float(np.median(err_rel)) if len(err_rel) else float("nan")

        return {
            "count": int(len(err_frames)),
            "MAE_frames": mae_frames,
            "MSE_frames2": mse_frames,
            "mean_frames": mean_frames,
            "median_frames": median_frames,
            "MAE_rel": mae_rel,
            "MSE_rel2": mse_rel,
            "mean_rel": mean_rel,
            "median_rel": median_rel,
        }

    overall = {
        "raw": _metrics(raw_err_frames, raw_err_rel),
        "clipped": _metrics(clip_err_frames, clip_err_rel),
        "min_seq_size": model_min,
        "max_seq_size": model_max,
    }

    per_lang = {}
    for lang in sorted(langs):
        raw_f = [x["err_raw_frames"] for x in bucket[lang]]
        raw_r = [x["err_raw_rel"] for x in bucket[lang]]
        clip_f = [x["err_clip_frames"] for x in bucket[lang]]
        clip_r = [x["err_clip_rel"] for x in bucket[lang]]
        per_lang[lang] = {
            "raw": _metrics(raw_f, raw_r),
            "clipped": _metrics(clip_f, clip_r),
        }

    summary = {"overall": overall, "by_language": per_lang}

    # Print concise summary (good for logs)
    print("[SeqLen] Overall RAW  : "
          f"MAE={overall['raw']['MAE_frames']:.3f} frames, "
          f"MSE={overall['raw']['MSE_frames2']:.3f} frames^2, "
          f"MAE%={overall['raw']['MAE_rel']*100:.2f}%")
    print("[SeqLen] Overall CLIP : "
          f"MAE={overall['clipped']['MAE_frames']:.3f} frames, "
          f"MSE={overall['clipped']['MSE_frames2']:.3f} frames^2, "
          f"MAE%={overall['clipped']['MAE_rel']*100:.2f}%")

    # Save jsons
    with open(f"models/{model_name}/results/seq_len_records.json", "w") as f:
        json.dump(rec, f, indent=2)

    with open(f"models/{model_name}/results/seq_len_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Backward compatible outputs if you still want them
    diffs_rel_raw = {k: float(v["err_raw_rel"]) for k, v in rec.items()}
    diffs_rel_clip = {k: float(v["err_clip_rel"]) for k, v in rec.items()}
    diffs_abs_raw = {k: float(abs(v["err_raw_frames"])) for k, v in rec.items()}
    diffs_abs_clip = {k: float(abs(v["err_clip_frames"])) for k, v in rec.items()}

    with open(f"models/{model_name}/results/seq_len_diffs_rel_raw.json", "w") as f:
        json.dump(diffs_rel_raw, f, indent=2)
    with open(f"models/{model_name}/results/seq_len_diffs_rel_clipped.json", "w") as f:
        json.dump(diffs_rel_clip, f, indent=2)
    with open(f"models/{model_name}/results/seq_len_abs_diffs_raw.json", "w") as f:
        json.dump(diffs_abs_raw, f, indent=2)
    with open(f"models/{model_name}/results/seq_len_abs_diffs_clipped.json", "w") as f:
        json.dump(diffs_abs_clip, f, indent=2)

    if save_plots:
        # Histogram: relative error (%)
        plt.hist([v * 100 for v in raw_err_rel], bins=80, alpha=0.6, label="raw")
        plt.hist([v * 100 for v in clip_err_rel], bins=80, alpha=0.6, label="clipped")
        plt.xticks(ticks=[-50, 0, 50, 100, 150], labels=["-50%", "0%", "50%", "100%", "150%"])
        plt.xlabel("sequence length error percentage")
        plt.ylabel("Count")
        plt.legend()
        plt.savefig(f"models/{model_name}/results/seq_len_diff_hist_raw_vs_clipped.png")
        plt.clf()

        # Histogram: absolute frame error
        plt.hist([abs(v) for v in raw_err_frames], bins=20, alpha=0.6, label="raw")
        plt.hist([abs(v) for v in clip_err_frames], bins=20, alpha=0.6, label="clipped")
        plt.xlabel("absolute frame error (FPS=25)")
        plt.ylabel("Count")
        plt.legend()
        plt.savefig(f"models/{model_name}/results/seq_len_abs_diff_hist_raw_vs_clipped.png")
        plt.clf()

    return summary


def test_seq_len_old(model, dataset, model_name):
    abs_diffs = dict()
    diffs = dict()
    for d in dataset:
        _, seq_len = model.encode_text([d["text"]])
        real_seq_len = d["pose"]["length"]
        diff = seq_len.item() - real_seq_len.item()
        abs_diffs[d["id"]] = np.abs(diff)
        diffs[d["id"]] = diff / real_seq_len.item()
    print(f"mean diff: {np.mean(list(diffs.values()))}, median: {np.median(list(diffs.values()))}")
    print(f"mean absolute diff: {np.mean(list(abs_diffs.values()))}, median:"
          f" {np.median(list(abs_diffs.values()))}")

    plt.hist([v * 100 for v in diffs.values()], bins=80)
    plt.xticks(ticks=[-50, 0, 50, 100, 150], labels=["-50%", "0%", "50%", "100%", "150%"])
    # plt.title("Predicted vs. real sequence length difference")
    plt.xlabel('sequence length error percentage')
    plt.ylabel('Count')
    plt.savefig(f"models/{model_name}/results/seq_len_diff_hist.png")
    plt.clf()

    plt.hist(list(abs_diffs.values()), bins=10)
    # plt.title("Predicted vs. real sequence length absolute difference")
    plt.xlabel('frame number difference (FPS=25)')
    plt.ylabel('Count')
    plt.savefig(f"models/{model_name}/results/seq_len_abs_diff_hist.png")
    plt.clf()

    with open(f"models/{model_name}/results/seq_len_diffs.json", 'w') as f:
        json.dump(diffs, f)
    with open(f"models/{model_name}/results/seq_len_abs_diffs.json", 'w') as f:
        json.dump(abs_diffs, f)


def worker_predict_and_rank(model_state_dict, model_args, device, datum, pose_header, keypoints_path, data_ids,
                            num_samples, dataset):
    # 在每个进程中加载模型并转移到指定设备（GPU）
    model = IterativeTextGuidedPoseGenerationModel(**model_args)
    model.load_state_dict(model_state_dict)
    model.to(device)
    model.eval()

    with torch.no_grad():
        # 运行predict_pose
        predicted_pose = predict_pose(model, datum, pose_header)
        # 运行get_poses_ranks
        pred2label_distance, rank_1_pred, rank_5_pred, rank_10_pred, rank_1_label, rank_5_label, rank_10_label = get_poses_ranks(
            predicted_pose, datum["id"], keypoints_path, data_ids, model=model, pose_header=pose_header,
            ds=dataset, num_samples=num_samples)

    return datum[
        "id"], pred2label_distance, rank_1_pred, rank_5_pred, rank_10_pred, rank_1_label, rank_5_label, rank_10_label


def update_progress_bar(pbar):
    """用于在每个任务完成后更新进度条"""
    pbar.update()


def test_distance(model, model_name, dataset, keypoints_path, num_samples=20):
    keypoints_dirs = os.listdir(keypoints_path)
    with open("data/hamnosys/data.json", 'r') as f:
        data = json.load(f)
        data_ids = list(filter(lambda x: x in keypoints_dirs, data.keys()))

    model = model.cuda()
    with torch.no_grad():
        pred2label_distances = dict()
        pred2label_distances_dtw = dict()
        for datum in dataset:
            if len(datum["pose"]["data"]) == 0:
                continue
            predicted_pose = predict_pose(model, datum, pose_header)
            pred2label_distance = __compare_pred_to_video(predicted_pose, keypoints_path, datum["id"],
                                                          distance_function=fastdtw)
            pred2label_distances_dtw[datum["id"]] = __compare_pred_to_video_dtw(predicted_pose, keypoints_path, datum["id"],
                                                          distance_function=fastdtw)
            pred2label_distances[datum["id"]] = pred2label_distance
            print(f"{datum['id']} ndtw distance: {pred2label_distance}")
            print(f"{datum['id']} dtw distance: {pred2label_distances_dtw[datum['id']]}")

        print(f"ndtw mean distance between pred and label: {np.mean(list(pred2label_distances.values()))}")
        print(f"ndtw median distance between pred and label: {np.median(list(pred2label_distances.values()))}")
        print(f"dtw mean distance between pred and label: {np.mean(list(pred2label_distances_dtw.values()))}")
        print(f"dtw median distance between pred and label: {np.median(list(pred2label_distances_dtw.values()))}")

    print(pred2label_distances)
    print(pred2label_distances_dtw)

def test_distance_ranks(model, model_name, dataset, keypoints_path, num_samples=20):
    keypoints_dirs = os.listdir(keypoints_path)
    with open("data/hamnosys/data.json", 'r') as f:
        data = json.load(f)
        data_ids = list(filter(lambda x: x in keypoints_dirs, data.keys()))

    model_state_dict = model.state_dict()
    model_args = get_model_args(args, dataset[0]["pose"]["data"].shape[1], dataset[0]["pose"]["data"].shape[2])

    with mp.Pool(processes=8) as pool:
        with tqdm.tqdm(total=len(dataset), desc="Processing") as pbar:
            results = []
            for datum in dataset:  # get datum inside the dataset
                if len(datum["pose"]["data"]) == 0:
                    continue
                res = pool.apply_async(
                    worker_predict_and_rank,
                    args=(model_state_dict, model_args, 'cuda:0', datum,
                          dataset[0]["pose"]["obj"].header, keypoints_path, data_ids, num_samples, dataset),
                    callback=lambda _: update_progress_bar(pbar)
                )
                results.append(res)

            # wait all process done
            results = [res.get() for res in results]

    # handle process
    pred2label_distances = {}
    rank_1_pred_sum = rank_5_pred_sum = rank_10_pred_sum = 0
    rank_1_label_sum = rank_5_label_sum = rank_10_label_sum = 0

    for res in results:
        datum_id, pred2label_distance, rank_1_pred, rank_5_pred, rank_10_pred, rank_1_label, rank_5_label, rank_10_label = res
        pred2label_distances[datum_id] = pred2label_distance
        rank_1_pred_sum += int(rank_1_pred)
        rank_5_pred_sum += int(rank_5_pred)
        rank_10_pred_sum += int(rank_10_pred)
        rank_1_label_sum += int(rank_1_label)
        rank_5_label_sum += int(rank_5_label)
        rank_10_label_sum += int(rank_10_label)

    num_samples = len(dataset)
    print(f"rank 1 pred sum: {rank_1_pred_sum} / {num_samples}: {rank_1_pred_sum / num_samples}")
    print(f"rank 5 pred sum: {rank_5_pred_sum} / {num_samples}: {rank_5_pred_sum / num_samples}")
    print(f"rank 10 pred sum: {rank_10_pred_sum} / {num_samples}: {rank_10_pred_sum / num_samples}")

    print(f"rank 1 label sum: {rank_1_label_sum} / {num_samples}: {rank_1_label_sum / num_samples}")
    print(f"rank 5 label sum: {rank_5_label_sum} / {num_samples}: {rank_5_label_sum / num_samples}")
    print(f"rank 10 label sum: {rank_10_label_sum} / {num_samples}: {rank_10_label_sum / num_samples}")

    with open(f"models/{model_name}/results/pred2label_distances_NDTW_pred_label_gallery.json", 'w') as f:
        json.dump(pred2label_distances, f)

    print(f"mean distance between pred and label: {np.mean(list(pred2label_distances.values()))}")
    print(f"median distance between pred and label: {np.median(list(pred2label_distances.values()))}")

    plt.hist(list(pred2label_distances.values()))
    plt.title("DTW distance between ground truth and predicted pose")
    plt.savefig(f"models/{model_name}/results/pred2label_distances_hist.png")




def test(model, model_name, dataset, test_seq_len_predictor=True, test_ranks=True, output_dir="",
         keypoints_path="", test_dis=True):
    os.makedirs(f"models/{model_name}/results", exist_ok=True)

    if output_dir != "":
        pred(model, dataset, f"models/{model_name}/{output_dir}")

    if test_seq_len_predictor:
        test_seq_len(model, dataset, model_name)

    if test_ranks:
        test_distance_ranks(model, model_name, dataset, keypoints_path)
    if test_dis:
        test_distance(model, model_name, dataset, keypoints_path)

def compute_sequence_complexity(datum, use_conf=True, eps=1e-9):
    """
    Returns complexity components and a raw score (not z-normalized yet).
    datum["pose"]["data"]: [T, K, D], D>=2 (x,y[,conf])
    """
    pose = datum["pose"]["data"]
    if isinstance(pose, torch.Tensor):
        pose = pose.detach().cpu().numpy()
    T, K, D = pose.shape
    if T < 3:
        return {"T": float(T), "E": 0.0, "J": 0.0, "raw": 0.0}

    xy = pose[:, :, :2].astype(np.float64)  # [T,K,2]

    # Optional confidence mask
    if use_conf and D >= 3:
        conf = pose[:, :, 2].astype(np.float64)  # [T,K]
        valid = conf > 0.0
    else:
        valid = np.ones((T, K), dtype=bool)

    # First differences (velocity proxy)
    v = xy[1:] - xy[:-1]  # [T-1,K,2]
    v_norm = np.linalg.norm(v, axis=-1)  # [T-1,K]
    valid_v = valid[1:] & valid[:-1]

    # Motion energy: mean per-frame per-kpt displacement
    denom_E = np.maximum(valid_v.sum(), 1.0)
    E = float((v_norm * valid_v).sum() / denom_E)

    # Second differences (acceleration/jerk proxy)
    a = v[1:] - v[:-1]  # [T-2,K,2]
    a_norm = np.linalg.norm(a, axis=-1)  # [T-2,K]
    valid_a = valid[2:] & valid[1:-1] & valid[:-2]
    denom_J = np.maximum(valid_a.sum(), 1.0)
    J = float((a_norm * valid_a).sum() / denom_J)

    # Length
    T_float = float(T)

    # Raw score (log-compressed; z-normalize later across dataset)
    raw = np.log1p(T_float) + np.log1p(E + eps) + np.log1p(J + eps)

    return {"T": T_float, "E": E, "J": J, "raw": float(raw)}

def bucket_by_complexity(dataset):
    # Collect per-language scores
    lang2items = {"pjm": [], "dgs": [], "gsl": [], "lsf": []}
    for d in dataset:
        if len(d["pose"]["data"]) == 0:
            continue
        lang = get_lang(d["id"])
        c = compute_sequence_complexity(d)
        lang2items[lang].append((d["id"], c))

    # Compute thresholds and assign buckets
    out = {}
    for lang, items in lang2items.items():
        if not items:
            continue
        raws = np.array([c["raw"] for _, c in items], dtype=np.float64)
        q1, q2 = np.quantile(raws, [0.33, 0.66])
        for sid, c in items:
            if c["raw"] <= q1:
                b = "simple"
            elif c["raw"] <= q2:
                b = "medium"
            else:
                b = "complex"
            out[sid] = {**c, "lang": lang, "bucket": b, "q1": float(q1), "q2": float(q2)}
    return out

def plot_bucket_counts(comp_dict, out_dir, filename_prefix="complexity_bucket_counts"):
    """
    Plot per-dataset bucket counts (simple/medium/complex) for PJM/DGS/GSL/LSF.
    comp_dict: dict[sample_id] -> {"lang":..., "bucket":...}
    Saves:
      - stacked bar plot: <out_dir>/<prefix>_stacked.png
      - grouped bar plot: <out_dir>/<prefix>_grouped.png
      - summary json:      <out_dir>/<prefix>.json
    """
    os.makedirs(out_dir, exist_ok=True)

    langs = ["pjm", "dgs", "gsl", "lsf"]
    buckets = ["simple", "medium", "complex"]

    # Count
    cnt = {lang: {b: 0 for b in buckets} for lang in langs}
    for _, info in comp_dict.items():
        lang = info.get("lang")
        b = info.get("bucket")
        if lang in cnt and b in cnt[lang]:
            cnt[lang][b] += 1

    # Save summary json
    with open(os.path.join(out_dir, f"{filename_prefix}.json"), "w", encoding="utf-8") as f:
        json.dump(cnt, f, indent=2)

    x = np.arange(len(langs))

    # 1) Stacked bar
    bottom = np.zeros(len(langs), dtype=np.int64)
    plt.figure()
    for b in buckets:
        vals = np.array([cnt[lang][b] for lang in langs], dtype=np.int64)
        plt.bar(x, vals, bottom=bottom, label=b)
        bottom += vals
    plt.xticks(x, langs)
    plt.ylabel("Number of sequences")
    plt.title("Complexity bucket counts per dataset")
    plt.legend()
    plt.savefig(os.path.join(out_dir, f"{filename_prefix}_stacked.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # 2) Grouped bar
    width = 0.25
    plt.figure()
    for i, b in enumerate(buckets):
        vals = np.array([cnt[lang][b] for lang in langs], dtype=np.int64)
        plt.bar(x + (i - 1) * width, vals, width=width, label=b)
    plt.xticks(x, langs)
    plt.ylabel("Number of sequences")
    plt.title("Complexity bucket counts per dataset")
    plt.legend()
    plt.savefig(os.path.join(out_dir, f"{filename_prefix}_grouped.png"), dpi=200, bbox_inches="tight")
    plt.close()

    print(f"[ComplexityPlot] saved to {out_dir}: {filename_prefix}_stacked.png, {filename_prefix}_grouped.png")


if __name__ == "__main__":
    args = vars(args)
    if args["config_file"]:  # override args with yaml config file
        with open(args["config_file"], 'r') as f:
            args = yaml.safe_load(f)
    args["batch_size"] = num_steps_to_batch_size[args["num_steps"]]
    test_size = int(0.1 * DATASET_SIZE)
    # if args["leave_out"] != "":
    #     _, dataset = get_dataset(name=args["dataset"], poses=args["pose"], fps=args["fps"],
    #                              components=args["pose_components"], leave_out=args["leave_out"],
    #                              max_seq_size=args["max_seq_size"], split='test')
    # else:
    #     dataset = get_dataset(name=args["dataset"], poses=args["pose"], fps=args["fps"],
    #                           components=args["pose_components"], max_seq_size=args["max_seq_size"],
    #                           split=f'test[:{test_size}]')
    #
    # with open("./temp/test/train_dataset.pkl", "wb") as f:
    #     pickle.dump(dataset, f)

    import pickle

    with open("./temp/train/test_dataset_old.pkl", "rb") as f:
        dataset = pickle.load(f)
    _, num_pose_joints, num_pose_dims = dataset[0]["pose"]["data"].shape
    pose_header = dataset.data[0]["pose"].header

    model_args = get_model_args(args, num_pose_joints, num_pose_dims)

    ckpt = f"./models/{args['model_name']}/{args['ckpt']}/model.ckpt"
    model = IterativeTextGuidedPoseGenerationModel.load_from_checkpoint(ckpt, **model_args)
    model.eval()

    # run_fasterkan_ffn_interpret(
    #     model,
    #     out_dir=f"models/{args['model_name']}/results/kan_interpret",
    #     topk=10,
    #     curves_per_layer=6
    # )
    #
    # comp = bucket_by_complexity(dataset)
    # with open(f"models/{args['model_name']}/results/test_complexity_buckets.json", "w") as f:
    #     json.dump(comp, f, indent=2)
    #
    # out_dir = f"models/{args['model_name']}/results"
    # plot_bucket_counts(comp, out_dir, filename_prefix="test_complexity_bucket_counts")

    test(model, args["model_name"], dataset, test_seq_len_predictor=True, test_ranks=False, test_dis=True,
         output_dir=args["output_dir"], keypoints_path="data/hamnosys/keypoints")
    # test(model, args["model_name"], dataset, test_seq_len_predictor=False, test_ranks=True,
    #      output_dir="", keypoints_path="data/hamnosys/keypoints")
    # test(model, args["model_name"], dataset, test_seq_len_predictor=True, test_ranks=False,
    #      output_dir="", keypoints_path="data/hamnosys/keypoints")