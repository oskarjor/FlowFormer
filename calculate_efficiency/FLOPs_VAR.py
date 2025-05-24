################## 1. Download checkpoints and build models
import os
import os.path as osp
import torch
import random
import numpy as np
from absl import app, flags
from torchVAR.models import build_vae_var
import fvcore.nn as fnn


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

    class VARWrapper(torch.nn.Module):
        def __init__(self, var):
            super().__init__()
            self.var = var

        def forward(self, B, label_B, cfg, top_k, top_p, g_seed, more_smooth):
            return self.var.autoregressive_infer_cfg(
                B=B,
                label_B=label_B,
                cfg=cfg,
                top_k=top_k,
                top_p=top_p,
                g_seed=g_seed,
                more_smooth=more_smooth,
            )

    var_wrapper = VARWrapper(var)

    with torch.inference_mode():
        with torch.autocast(
            "cuda", enabled=True, dtype=torch.float16, cache_enabled=True
        ):
            # Simplified FLOPs calculation
            inputs_for_flops = (B, label_B, cfg, 900, 0.95, None, more_smooth)
            analysis = fnn.FlopCountAnalysis(var_wrapper, inputs_for_flops)
            flop_count_str = fnn.flop_count_table(analysis)
            flop_count = analysis.total()
            print(flop_count_str)


if __name__ == "__main__":
    app.run(sample_var)
