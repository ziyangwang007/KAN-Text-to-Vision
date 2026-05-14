# interpret_kan.py
import os
from typing import List, Tuple, Dict

import numpy as np
import torch
import matplotlib.pyplot as plt


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _collect_pose_encoder_fasterkans(model) -> List[Tuple[int, torch.nn.Module]]:
    """
    返回 [(layer_idx, fasterkan_module), ...] for model.pose_encoder.layers[i].kan
    只针对 Transformer FFN 的 FasterKAN。
    """
    if not hasattr(model, "pose_encoder"):
        raise AttributeError("Model has no attribute 'pose_encoder'.")

    enc = model.pose_encoder
    if not hasattr(enc, "layers"):
        raise AttributeError("model.pose_encoder has no attribute 'layers'.")

    kans = []
    for i, layer in enumerate(enc.layers):
        if hasattr(layer, "kan"):
            kans.append((i, layer.kan))
    if not kans:
        raise RuntimeError("No FFN FasterKAN found under model.pose_encoder.layers[*].kan")
    return kans


def _fasterkan_layers(fasterkan_module: torch.nn.Module) -> List[torch.nn.Module]:
    """
    FasterKAN 内部通常有 .layers (ModuleList)，每个元素是 FasterKANLayer
    """
    if hasattr(fasterkan_module, "layers"):
        return list(fasterkan_module.layers)
    # 兜底：遍历 named_modules 找 FasterKANLayer
    layers = []
    for _, m in fasterkan_module.named_modules():
        if m.__class__.__name__ == "FasterKANLayer":
            layers.append(m)
    return layers


def _weight_group_importance(fasterkan_layer: torch.nn.Module) -> np.ndarray:
    """
    对 FasterKANLayer 的 spline_linear.weight 做按输入维度分组的重要性。
    weight: [out_dim, in_dim*num_grids]  -> reshape [out_dim, in_dim, num_grids]
    importance per input dim: L2 norm over (out_dim, num_grids)
    """
    W = fasterkan_layer.spline_linear.weight.detach()  # [out, in*num_grids]
    in_dim = fasterkan_layer.layernorm.normalized_shape[0]
    num_grids = W.shape[1] // in_dim
    Wg = W.view(W.shape[0], in_dim, num_grids)
    imp = torch.norm(Wg, p=2, dim=(0, 2))  # [in_dim]
    return imp.cpu().numpy()


@torch.no_grad()
def _plot_importance(imp: np.ndarray, title: str, out_path: str, topk: int = 20):
    idx = np.argsort(-imp)[:topk]
    vals = imp[idx]

    plt.figure(figsize=(5, 4))
    plt.bar(np.arange(len(idx)), vals)
    plt.xticks(np.arange(len(idx)), [str(i) for i in idx], rotation=60, ha="right")
    plt.title(title)
    plt.xlabel("Input dimension index")
    plt.ylabel("Importance (L2 norm)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


@torch.no_grad()
def _plot_1d_response(fasterkan_layer: torch.nn.Module, in_idx: int, out_idx: int, out_path: str,
                      x_min: float = -2.0, x_max: float = 2.0, steps: int = 400):
    """
    画单个 FasterKANLayer 的 1D 响应：只变化 input[in_idx]，其他维度为 0。
    注意：为了直接对应 learned 1D mapping，我们这里跳过 LayerNorm（否则 LN 会引入跨维耦合）。
    """
    W = fasterkan_layer.spline_linear.weight.detach()
    in_dim = fasterkan_layer.layernorm.normalized_shape[0]
    num_grids = W.shape[1] // in_dim
    Wg = W.view(W.shape[0], in_dim, num_grids)  # [out, in, grids]

    xs = torch.linspace(x_min, x_max, steps)
    X = torch.zeros(steps, in_dim)
    X[:, in_idx] = xs

    basis = fasterkan_layer.rbf(X)  # 期望 [steps, in, grids]
    y = (basis[:, in_idx, :] * Wg[out_idx, in_idx, :]).sum(-1)  # [steps]

    plt.figure(figsize=(6, 4))
    plt.plot(xs.cpu().numpy(), y.cpu().numpy())
    plt.title(f"1D response: in={in_idx} -> out={out_idx}")
    plt.xlabel(f"x[{in_idx}]")
    plt.ylabel("response")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def run_fasterkan_ffn_interpret(model, out_dir: str, topk: int = 20, curves_per_layer: int = 6):
    """
    主入口：对 pose_encoder 的每一层 FFN FasterKAN 做：
    - global importance 图（每个内部 FasterKANLayer 一张）
    - 1D response 曲线图（基于 Top importance 维度）
    """
    _ensure_dir(out_dir)
    model.eval()

    kans = _collect_pose_encoder_fasterkans(model)
    avg_imp_accum: List[np.ndarray] = []

    for enc_layer_idx, fasterkan in kans:
        layers = _fasterkan_layers(fasterkan)
        layer_dir = os.path.join(out_dir, f"pose_encoder_layer_{enc_layer_idx:02d}")
        _ensure_dir(layer_dir)

        for j, fk_layer in enumerate(layers):
            imp = _weight_group_importance(fk_layer)
            avg_imp_accum.append(imp)

            # 1) importance bar
            _plot_importance(
                imp,
                title=f"Global importance (pose_enc_layer={enc_layer_idx}, fk_layer={j})",
                out_path=os.path.join(layer_dir, f"importance_fk_layer_{j}.png"),
                topk=topk
            )

            # 2) 1D curves for top dims
            top_dims = np.argsort(-imp)[:curves_per_layer]
            curve_dir = os.path.join(layer_dir, f"curves_fk_layer_{j}")
            _ensure_dir(curve_dir)

            # 选一个稳定的 out_idx：用 0 号输出通道（你也可以改成多个 out_idx）
            out_idx = 0
            for rank, in_idx in enumerate(top_dims):
                _plot_1d_response(
                    fk_layer,
                    in_idx=int(in_idx),
                    out_idx=out_idx,
                    out_path=os.path.join(curve_dir, f"curve_rank{rank:02d}_in{int(in_idx)}_out{out_idx}.png")
                )

    # 跨层平均重要性
    if avg_imp_accum:
        # 对齐维度（理论上都相同：d_model）
        min_len = min(x.shape[0] for x in avg_imp_accum)
        avg_imp = np.stack([x[:min_len] for x in avg_imp_accum], axis=0).mean(axis=0)

        _plot_importance(
            avg_imp,
            title="Global importance (averaged over all pose-encoder FasterKAN layers)",
            out_path=os.path.join(out_dir, "importance_global_average.png"),
            topk=topk
        )

    print(f"[KAN interpret] Saved results to: {out_dir}")
