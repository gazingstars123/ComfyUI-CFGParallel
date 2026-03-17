import torch
import time
import logging

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

logger = logging.getLogger("VerboseDebug")


def _fmt_bytes(b):
    if b < 1024:
        return f"{b} B"
    elif b < 1024 ** 2:
        return f"{b / 1024:.1f} KB"
    elif b < 1024 ** 3:
        return f"{b / 1024 ** 2:.1f} MB"
    else:
        return f"{b / 1024 ** 3:.2f} GB"


def _gpu_mem_info(device):
    if device is not None and device.type == "cuda":
        return (torch.cuda.memory_allocated(device),
                torch.cuda.memory_reserved(device))
    return (0, 0)


def _system_ram_mb():
    if _HAS_PSUTIL:
        return psutil.Process().memory_info().rss / 1024 ** 2
    return None


def _fmt_delta(before, after):
    delta = after - before
    sign = "+" if delta >= 0 else "-"
    return f"{sign}{_fmt_bytes(abs(delta))}"


class VerboseDebug:
    """
    Verbose Debug Logger: logs per-step timing, VRAM, memory, and tensor info
    during sampling. Single GPU, standalone node.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
            },
            "optional": {
                "enabled": ("BOOLEAN", {"default": True, "tooltip": "Toggle debug logging on/off without disconnecting the node."}),
                "log_vram": ("BOOLEAN", {"default": True, "tooltip": "Log per-GPU VRAM allocated/reserved before and after each step."}),
                "log_system_ram": ("BOOLEAN", {"default": True, "tooltip": "Log process RSS memory (requires psutil)."}),
                "log_tensor_shapes": ("BOOLEAN", {"default": True, "tooltip": "Log input tensor shape, dtype, device, and cond_or_uncond info."}),
                "log_timing": ("BOOLEAN", {"default": True, "tooltip": "Log forward pass wall time with cuda.synchronize for accuracy."}),
                "log_every_n_steps": ("INT", {"default": 1, "min": 1, "max": 1000, "tooltip": "Log every Nth step. Set to 1 to log every step."}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "setup"
    CATEGORY = "sampling/debug"
    DESCRIPTION = "Logs detailed per-step debug info during sampling: timing, VRAM, system RAM, tensor shapes. Place before KSampler."

    def setup(self, model, enabled=True, log_vram=True, log_system_ram=True,
              log_tensor_shapes=True, log_timing=True, log_every_n_steps=1):

        if not enabled:
            return (model,)

        cloned = model.clone()

        # Shared mutable state for the closure
        state = {
            "call_count": 0,       # total forward pass calls
            "denoise_step": 0,     # actual denoising step (groups cond+uncond)
            "last_sigma": None,    # track sigma to detect new denoising step
            "step_t0": None,       # wall time at start of denoising step
            "step_alloc0": None,   # VRAM at start of denoising step
            "step_resv0": None,
            "step_ram0": None,
            "pass_times": [],      # per-pass timings within current denoising step
            "pass_labels": [],     # "cond" / "uncond" / "batched" labels
        }

        opts = {
            "vram": log_vram,
            "ram": log_system_ram,
            "shapes": log_tensor_shapes,
            "timing": log_timing,
            "every_n": max(1, log_every_n_steps),
        }

        def _detect_new_denoise_step(timestep):
            """Detect if this is a new denoising step by checking sigma value."""
            sigma_val = timestep.flatten()[0].item()
            if state["last_sigma"] is None or sigma_val != state["last_sigma"]:
                state["last_sigma"] = sigma_val
                return True
            return False

        def _pass_label(cond_or_uncond):
            """Label a forward pass based on cond_or_uncond content."""
            conds = set(cond_or_uncond) if cond_or_uncond else set()
            if conds == {0}:
                return "cond"
            elif conds == {1}:
                return "uncond"
            elif conds == {0, 1}:
                return "batched"
            return f"unknown({list(cond_or_uncond)})"

        def _log_denoise_step_summary(device):
            """Log the summary for a completed denoising step."""
            step_num = state["denoise_step"]
            should_log = (step_num % opts["every_n"]) == 0 or step_num == 1
            if not should_log or not state["pass_times"]:
                return

            total_time = time.perf_counter() - state["step_t0"]

            # Pass breakdown
            parts = []
            for label, t in zip(state["pass_labels"], state["pass_times"]):
                parts.append(f"{label}={t * 1000:.1f}ms")
            logger.info(f"  [DEBUG]   Passes: {' + '.join(parts)} = {total_time * 1000:.1f}ms total")

            if opts["vram"] and state["step_alloc0"] is not None:
                alloc1, resv1 = _gpu_mem_info(device)
                logger.info(f"  [DEBUG]   VRAM ({device}): alloc={_fmt_bytes(alloc1)} resv={_fmt_bytes(resv1)} Δalloc={_fmt_delta(state['step_alloc0'], alloc1)} Δresv={_fmt_delta(state['step_resv0'], resv1)}")
                for i in range(torch.cuda.device_count()):
                    dev_i = torch.device(f"cuda:{i}")
                    if dev_i != device:
                        a1, r1 = _gpu_mem_info(dev_i)
                        if a1 > 0:
                            logger.info(f"  [DEBUG]   VRAM ({dev_i}): alloc={_fmt_bytes(a1)} resv={_fmt_bytes(r1)}")

            if opts["ram"] and state["step_ram0"] is not None:
                ram1 = _system_ram_mb()
                if ram1 is not None:
                    logger.info(f"  [DEBUG]   System RAM: {ram1:.1f} MB (Δ{ram1 - state['step_ram0']:+.1f} MB)")

            logger.info(f"  [DEBUG] ── Denoise step {step_num} end ({total_time * 1000:.1f}ms) ──")

        def verbose_wrapper(apply_model_func, kwargs):
            input_x = kwargs["input"]
            timestep = kwargs["timestep"]
            c = kwargs["c"]
            cond_or_uncond = kwargs.get("cond_or_uncond", [])

            state["call_count"] += 1
            device = input_x.device
            is_new_step = _detect_new_denoise_step(timestep)
            pass_label = _pass_label(cond_or_uncond)

            # ── New denoising step ──
            if is_new_step:
                state["denoise_step"] += 1
                state["pass_times"] = []
                state["pass_labels"] = []
                state["step_t0"] = time.perf_counter()

                step_num = state["denoise_step"]
                should_log = (step_num % opts["every_n"]) == 0 or step_num == 1

                if should_log:
                    sigma_val = timestep.flatten()[0].item()
                    logger.info(f"  [DEBUG] ── Denoise step {step_num} begin (σ={sigma_val:.6f}) ──")

                    if opts["shapes"]:
                        logger.info(f"  [DEBUG]   input_x: {list(input_x.shape)} {input_x.dtype} on {input_x.device}")
                        for key, val in c.items():
                            if isinstance(val, torch.Tensor):
                                logger.info(f"  [DEBUG]   c[{key}]: {list(val.shape)} {val.dtype} on {val.device}")
                            elif isinstance(val, dict):
                                for k2, v2 in val.items():
                                    if isinstance(v2, torch.Tensor):
                                        logger.info(f"  [DEBUG]   c[{key}][{k2}]: {list(v2.shape)} {v2.dtype}")

                    if opts["vram"]:
                        state["step_alloc0"], state["step_resv0"] = _gpu_mem_info(device)
                        logger.info(f"  [DEBUG]   VRAM ({device}): alloc={_fmt_bytes(state['step_alloc0'])} resv={_fmt_bytes(state['step_resv0'])}")
                        for i in range(torch.cuda.device_count()):
                            dev_i = torch.device(f"cuda:{i}")
                            if dev_i != device:
                                a, r = _gpu_mem_info(dev_i)
                                if a > 0:
                                    logger.info(f"  [DEBUG]   VRAM ({dev_i}): alloc={_fmt_bytes(a)} resv={_fmt_bytes(r)}")

                    if opts["ram"]:
                        state["step_ram0"] = _system_ram_mb()
                        if state["step_ram0"] is not None:
                            logger.info(f"  [DEBUG]   System RAM: {state['step_ram0']:.1f} MB")

            # ── Forward pass ──
            step_num = state["denoise_step"]
            should_log = (step_num % opts["every_n"]) == 0 or step_num == 1

            if opts["timing"]:
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                t0 = time.perf_counter()

            output = apply_model_func(input_x, timestep, **c)

            if opts["timing"]:
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                elapsed = time.perf_counter() - t0
                state["pass_times"].append(elapsed)
                state["pass_labels"].append(pass_label)
                if should_log:
                    logger.info(f"  [DEBUG]   {pass_label}: {elapsed * 1000:.1f}ms | output: {list(output.shape)} {output.dtype}")
            else:
                state["pass_times"].append(0)
                state["pass_labels"].append(pass_label)

            return output

        def post_cfg_callback(args):
            """Emit the summary for the current denoising step after CFG is applied."""
            device = args["denoised"].device
            _log_denoise_step_summary(device)
            return args["denoised"]

        cloned.set_model_unet_function_wrapper(verbose_wrapper)
        cloned.set_model_sampler_post_cfg_function(post_cfg_callback)
        logger.info("VerboseDebug: wrapper installed, ready for sampling")
        return (cloned,)


NODE_CLASS_MAPPINGS = {
    "VerboseDebug": VerboseDebug,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VerboseDebug": "Verbose Debug Logger",
}
