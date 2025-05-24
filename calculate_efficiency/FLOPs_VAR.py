################## 1. Download checkpoints and build models
import os
import os.path as osp
import torch
import random
import numpy as np
import json
import time
from absl import app, flags
from tqdm import tqdm
from torchVAR.models import build_vae_var
from torchVAR.utils.imagenet_utils import (
    get_imagenet_class_mapping,
    save_batch_to_imagenet_structure,
)
from torch.profiler import (
    profile,
    record_function,
    ProfilerActivity,
    tensorboard_trace_handler,
)

setattr(
    torch.nn.Linear, "reset_parameters", lambda self: None
)  # disable default parameter init for faster speed
setattr(
    torch.nn.LayerNorm, "reset_parameters", lambda self: None
)  # disable default parameter init for faster speed


FLAGS = flags.FLAGS

flags.DEFINE_integer("model_depth", 16, help="model depth")
flags.DEFINE_string("vae_ckpt", "vae_ch160v4096z32.pth", help="vae checkpoint")
flags.DEFINE_string("var_ckpt", "var_d16.pth", help="var checkpoint")
flags.DEFINE_integer("seed", 0, help="seed")
flags.DEFINE_float("cfg", 4.0, help="classifier-free guidance")
flags.DEFINE_bool("more_smooth", False, help="more smooth")
flags.DEFINE_bool("debug", False, help="debug")
flags.DEFINE_bool("flash_attn", False, help="flash_attn")
flags.DEFINE_bool("fused_mlp", False, help="fused_mlp")
flags.DEFINE_list("return_sizes", [16], help="return sizes")
flags.DEFINE_integer("batch_size", 64, help="batch size")


def calculate_var_flops(var, B, patch_nums):
    """Calculate theoretical FLOPs for VAR model inference.

    Args:
        var: VAR model instance
        B: batch size
        patch_nums: tuple of patch sizes used in the model

    Returns:
        total_flops: total number of FLOPs
        flops_breakdown: dictionary with FLOPs breakdown by component
    """
    flops_breakdown = {}
    total_flops = 0

    # 1. Initial embeddings and processing
    C = var.C  # embedding dimension
    D = var.D  # condition dimension
    num_heads = var.num_heads
    head_dim = C // num_heads

    # Class embedding
    class_emb_flops = B * C * 2  # multiply-add operations
    flops_breakdown["class_embedding"] = class_emb_flops
    total_flops += class_emb_flops

    # Position embeddings
    pos_emb_flops = B * C * 2  # multiply-add operations
    flops_breakdown["position_embedding"] = pos_emb_flops
    total_flops += pos_emb_flops

    # 2. Transformer blocks
    for stage_idx, pn in enumerate(patch_nums):
        L = pn * pn  # sequence length for this stage

        # Self-attention
        # Q, K, V projections
        qkv_proj_flops = B * L * C * (3 * C) * 2  # multiply-add operations
        # Attention matrix computation
        attn_matrix_flops = B * num_heads * L * L * head_dim * 2
        # Output projection
        out_proj_flops = B * L * C * C * 2

        attn_flops = qkv_proj_flops + attn_matrix_flops + out_proj_flops
        flops_breakdown[f"stage_{stage_idx}_attention"] = attn_flops
        total_flops += attn_flops

        # MLP
        mlp_hidden = C * 4  # MLP ratio is 4
        mlp_flops = (
            B * L * (C * mlp_hidden * 2 + mlp_hidden * C * 2)
        )  # two linear layers
        flops_breakdown[f"stage_{stage_idx}_mlp"] = mlp_flops
        total_flops += mlp_flops

        # Layer norms
        norm_flops = B * L * C * 4  # 4 operations per element (mean, var, normalize)
        flops_breakdown[f"stage_{stage_idx}_norm"] = norm_flops
        total_flops += norm_flops

    # 3. Final head
    head_flops = B * C * var.V * 2  # multiply-add operations
    flops_breakdown["final_head"] = head_flops
    total_flops += head_flops

    return total_flops, flops_breakdown


def sample_var(argv):
    MODEL_DEPTH = FLAGS.model_depth
    assert MODEL_DEPTH in {16, 20, 24, 30}

    # download checkpoint
    hf_home = "https://huggingface.co/FoundationVision/var/resolve/main"
    vae_ckpt, var_ckpt = FLAGS.vae_ckpt, FLAGS.var_ckpt
    if not osp.exists(vae_ckpt):
        os.system(f"wget {hf_home}/{vae_ckpt}")
    if not osp.exists(var_ckpt):
        os.system(f"wget {hf_home}/{var_ckpt}")

    # build vae, var
    patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
    return_sizes = [int(x) for x in FLAGS.return_sizes]

    assert all(x in patch_nums for x in return_sizes), (
        f"return_sizes must be a subset of patch_nums: {patch_nums}"
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if "vae" not in globals() or "var" not in globals():
        vae, var = build_vae_var(
            V=4096,
            Cvae=32,
            ch=160,
            share_quant_resi=4,  # hard-coded VQVAE hyperparameters
            device=device,
            patch_nums=patch_nums,
            num_classes=1000,
            depth=MODEL_DEPTH,
            shared_aln=False,
            flash_if_available=FLAGS.flash_attn,
            fused_if_available=FLAGS.fused_mlp,
        )

    # load checkpoints
    vae.load_state_dict(torch.load(vae_ckpt, map_location="cpu"), strict=True)
    var.load_state_dict(torch.load(var_ckpt, map_location="cpu"), strict=True)
    vae.eval(), var.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    for p in var.parameters():
        p.requires_grad_(False)
    print(f"prepare finished.")

    ############################# 2. Sample with classifier-free guidance

    # set args
    seed = FLAGS.seed
    torch.manual_seed(seed)
    cfg = FLAGS.cfg
    more_smooth = FLAGS.more_smooth

    # seed
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # run faster
    tf32 = True
    torch.backends.cudnn.allow_tf32 = bool(tf32)
    torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
    torch.set_float32_matmul_precision("high" if tf32 else "highest")

    class_labels = [0] * FLAGS.batch_size
    B = len(class_labels)
    label_B: torch.LongTensor = torch.tensor(class_labels, device=device)

    print(f"Sampling {B} images")

    # Calculate theoretical FLOPs
    total_flops, flops_breakdown = calculate_var_flops(var, B, patch_nums)
    print(f"\nTheoretical FLOPs Analysis:")
    print(f"Total FLOPs: {total_flops / 1e9:.2f} GFLOPs")
    print("\nBreakdown by component:")
    for component, flops in flops_breakdown.items():
        print(f"{component}: {flops / 1e9:.2f} GFLOPs")

    # Profile the inference
    print("\nRunning PyTorch Profiler...")

    # Create a directory for profiling results
    os.makedirs("profiler_results", exist_ok=True)

    with torch.inference_mode():
        with torch.autocast(
            "cuda", enabled=True, dtype=torch.float16, cache_enabled=True
        ):
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
                with_flops=True,
                with_modules=True,
                schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
            ) as prof:
                for _ in range(5):  # Run multiple iterations for better profiling
                    with record_function("model_inference"):
                        recon_B3HW = var.autoregressive_infer_cfg(
                            B=B,
                            label_B=label_B,
                            cfg=cfg,
                            top_k=900,
                            top_p=0.95,
                            g_seed=None,
                            more_smooth=more_smooth,
                        )
                    prof.step()

    # Print profiling results
    print("\nPyTorch Profiler Results:")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

    # Print total FLOPs from profiler
    total_flops = sum(evt.flops for evt in prof.key_averages())
    print(f"\nTotal FLOPs from profiler: {total_flops / 1e9:.2f} GFLOPs")

    # Print memory usage
    print("\nMemory Usage:")
    print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=20))

    # Print detailed FLOPs breakdown
    print("\nDetailed FLOPs Breakdown:")
    flops_by_op = {}
    for evt in prof.key_averages():
        if evt.flops > 0:
            flops_by_op[evt.key] = flops_by_op.get(evt.key, 0) + evt.flops

    for op, flops in sorted(flops_by_op.items(), key=lambda x: x[1], reverse=True):
        print(f"{op}: {flops / 1e9:.2f} GFLOPs")

    # Save profiling results
    prof.export_chrome_trace("profiler_results/trace.json")
    print("\nProfiling results saved to profiler_results/ directory")


if __name__ == "__main__":
    app.run(sample_var)
