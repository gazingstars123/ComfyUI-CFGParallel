import torch
import copy
import gc
import queue
import threading
import weakref
import logging
import time
import comfy.model_management
from contextlib import contextmanager

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

logger = logging.getLogger("CfgParallel")


def _fmt_bytes(b):
    """Format bytes to human-readable string."""
    if b < 1024:
        return f"{b} B"
    elif b < 1024 ** 2:
        return f"{b / 1024:.1f} KB"
    elif b < 1024 ** 3:
        return f"{b / 1024 ** 2:.1f} MB"
    else:
        return f"{b / 1024 ** 3:.2f} GB"


def _tensor_bytes(t):
    """Total bytes of a tensor."""
    return t.nelement() * t.element_size()


def _count_transfer_bytes(value):
    """Recursively count total bytes of all tensors in a nested structure."""
    if isinstance(value, torch.Tensor):
        return _tensor_bytes(value)
    if isinstance(value, dict):
        return sum(_count_transfer_bytes(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_count_transfer_bytes(v) for v in value)
    return 0


def _gpu_mem_info(device):
    """Return (allocated, reserved) bytes for a CUDA device."""
    if device is not None and device.type == "cuda":
        return (torch.cuda.memory_allocated(device),
                torch.cuda.memory_reserved(device))
    return (0, 0)


def _system_ram_mb():
    """Return process RSS in MB, or None if psutil unavailable."""
    if _HAS_PSUTIL:
        return psutil.Process().memory_info().rss / 1024 ** 2
    return None

@contextmanager
def safe_rope_fallback():
    patched = False
    original_apply_rope1 = None
    original_apply_rope = None
    try:
        import comfy.ldm.flux.math as flux_math
        if hasattr(flux_math, 'q_apply_rope1') and hasattr(flux_math, '_apply_rope1'):
            original_apply_rope1 = flux_math.q_apply_rope1
            original_apply_rope = getattr(flux_math, 'q_apply_rope', None)
            
            def safe_apply_rope1(x, freqs_cis):
                try:
                    return original_apply_rope1(x, freqs_cis)
                except Exception as e:
                    # Fallback to pure PyTorch implementation if comfy_kitchen fails
                    return flux_math._apply_rope1(x, freqs_cis)
            
            def safe_apply_rope(xq, xk, freqs_cis):
                try:
                    if original_apply_rope is not None:
                        return original_apply_rope(xq, xk, freqs_cis)
                    return safe_apply_rope1(xq, freqs_cis), safe_apply_rope1(xk, freqs_cis)
                except Exception as e:
                    return flux_math._apply_rope1(xq, freqs_cis), flux_math._apply_rope1(xk, freqs_cis)

            flux_math.q_apply_rope1 = safe_apply_rope1
            if hasattr(flux_math, 'q_apply_rope'):
                flux_math.q_apply_rope = safe_apply_rope
            patched = True
    except Exception as e:
        logger.warning(f"Failed to patch safe rope fallback: {e}")
        pass
    
    try:
        yield
    finally:
        if patched:
            import comfy.ldm.flux.math as flux_math
            flux_math.q_apply_rope1 = original_apply_rope1
            if original_apply_rope is not None:
                flux_math.q_apply_rope = original_apply_rope

DIFFUSION_PREFIX = "diffusion_model."


def _convert_tensor(extra, dtype, device):
    if hasattr(extra, "dtype"):
        if extra.dtype != torch.int and extra.dtype != torch.long:
            extra = comfy.model_management.cast_to_device(extra, device, dtype)
        else:
            extra = comfy.model_management.cast_to_device(extra, device, None)
    return extra


def _move_to_device(value, device, non_blocking=False):
    """Recursively move tensors to device, preserving structure."""
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=non_blocking) if value.device != device else value
    if isinstance(value, dict):
        return {k: _move_to_device(v, device, non_blocking) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_move_to_device(item, device, non_blocking) for item in value)
    return value


class CfgParallelState:
    """Manages the secondary diffusion model, weight sync, and compilation."""

    def __init__(self, secondary_diff_model, secondary_device, compile_kwargs,
                 diffusion_model, cloned_ref, base_model_ref, verbose_debug=False):
        self.secondary_diff_model = secondary_diff_model
        self.secondary_device = secondary_device
        self.compile_kwargs = compile_kwargs
        self.diffusion_model = diffusion_model  # primary diffusion model (direct ref, not owned)
        self.cloned_ref = cloned_ref            # weakref to ModelPatcher
        self.base_model_ref = base_model_ref    # weakref to BaseModel
        self.verbose = verbose_debug

        self.neg_work_q = queue.Queue()
        self.neg_result_q = queue.Queue()
        self.pos_work_q = queue.Queue()
        self.pos_result_q = queue.Queue()
        self.gen_id = 0
        self.step_count = 0
        self.weights_synced = False
        self.compiled = False
        self.last_synced_hooks = object()  # sentinel, guarantees first sync
        self.last_synced_keys = set()

        # Deferred parallel state for sequential-cond models (e.g. Lumina2)
        self.deferred_cond_future = None
        self.deferred_uncond_future = None
        self.deferred_sigma = None
        self.deferred_cond_count = 0
        self.deferred_mode = False

    def _get_patched_keys(self):
        """Get the set of secondary model keys that have patches applied."""
        cloned = self.cloned_ref()
        if cloned is None:
            return set()

        patched_keys = set()

        # Regular LoRA patches (keys like "diffusion_model.X.weight")
        for key in getattr(cloned, 'patches', {}):
            if key.startswith(DIFFUSION_PREFIX):
                patched_keys.add(key[len(DIFFUSION_PREFIX):])

        # Hook patches (nested: hook_ref -> key -> patches)
        for hook_ref_patches in getattr(cloned, 'hook_patches', {}).values():
            for key in hook_ref_patches:
                if key.startswith(DIFFUSION_PREFIX):
                    patched_keys.add(key[len(DIFFUSION_PREFIX):])

        return patched_keys

    def copy_weights_all(self):
        """Copy ALL weights from primary to secondary (initial sync)."""
        sec = self.secondary_diff_model
        if sec is None:
            return 0
        src_sd = self.diffusion_model.state_dict()
        sec_sd = sec.state_dict()
        updated = 0
        for key in sec_sd:
            if key in src_sd:
                sec_sd[key].copy_(src_sd[key].to(sec_sd[key].device))
                updated += 1
        return updated

    def copy_weights_selective(self, keys):
        """Copy only specified weight keys from primary to secondary."""
        sec = self.secondary_diff_model
        if sec is None or not keys:
            return 0
        src_sd = self.diffusion_model.state_dict()
        sec_sd = sec.state_dict()
        updated = 0
        for key in keys:
            if key in src_sd and key in sec_sd:
                sec_sd[key].copy_(src_sd[key].to(sec_sd[key].device))
                updated += 1
        return updated

    def sync_weights(self):
        """Initial full sync (one-time)."""
        if self.weights_synced:
            return
        t0 = time.perf_counter()
        updated = self.copy_weights_all()
        elapsed = time.perf_counter() - t0
        self.weights_synced = True
        # Set current hooks so resync_if_hooks_changed doesn't double-fire
        cloned = self.cloned_ref()
        if cloned is not None:
            self.last_synced_hooks = getattr(cloned, 'current_hooks', None)
            self.last_synced_keys = self._get_patched_keys()
        logger.info(f"  Synced {updated} weight tensors to secondary model")
        if self.verbose:
            logger.info(f"  [DEBUG] Full weight sync took {elapsed:.3f}s")

    def compile_secondary(self):
        """Apply torch.compile to secondary model (one-time, after first sync)."""
        if self.compiled or self.compile_kwargs is None:
            return
        sec = self.secondary_diff_model
        if sec is None:
            return

        t0 = time.perf_counter()
        compile_args = dict(self.compile_kwargs)
        layer_types = ["double_blocks", "single_blocks", "layers",
                       "transformer_blocks", "blocks"]
        compiled_count = 0
        for layer_name in layer_types:
            if hasattr(sec, layer_name):
                blocks = getattr(sec, layer_name)
                for i in range(len(blocks)):
                    blocks[i] = torch.compile(blocks[i], **compile_args)
                    compiled_count += 1
        if compiled_count > 0:
            logger.info(f"  Compiled {compiled_count} transformer blocks on secondary model")
        else:
            self.secondary_diff_model = torch.compile(sec, **compile_args)
            logger.info("  Compiled entire secondary diffusion model")
        torch._dynamo.reset()
        self.compiled = True
        if self.verbose:
            elapsed = time.perf_counter() - t0
            logger.info(f"  [DEBUG] torch.compile setup took {elapsed:.3f}s | kwargs={compile_args}")

    def resync_if_hooks_changed(self):
        """Re-sync only patched weights when hooks change."""
        cloned = self.cloned_ref()
        if cloned is None:
            return
        current_hooks = getattr(cloned, 'current_hooks', None)
        if current_hooks is self.last_synced_hooks:
            return

        t0 = time.perf_counter()
        # Union of previously synced keys + currently patched keys
        current_patched = self._get_patched_keys()
        keys_to_sync = current_patched | self.last_synced_keys

        updated = self.copy_weights_selective(keys_to_sync)
        self.last_synced_hooks = current_hooks
        self.last_synced_keys = current_patched
        elapsed = time.perf_counter() - t0
        if updated:
            logger.info(f"  Re-synced {updated}/{len(keys_to_sync)} patched weight tensors (hooks changed)")
        if self.verbose:
            logger.info(f"  [DEBUG] Hook resync took {elapsed:.3f}s | keys={len(keys_to_sync)} updated={updated}")

    def apply_model(self, x, t, c_concat=None, c_crossattn=None, control=None,
                    transformer_options={}, **kwargs):
        """Run inference on the secondary diffusion model."""
        sec = self.secondary_diff_model
        if sec is None:
            raise RuntimeError("Secondary model has been cleaned up")

        bm = self.base_model_ref()
        if bm is None:
            raise RuntimeError("Base model has been garbage collected")
        ms = bm.model_sampling
        sigma = t
        xc = ms.calculate_input(sigma, x)

        if c_concat is not None:
            xc = torch.cat([xc] + [comfy.model_management.cast_to_device(c_concat, xc.device, xc.dtype)], dim=1)

        context = c_crossattn
        dtype = bm.get_dtype()
        if bm.manual_cast_dtype is not None:
            dtype = bm.manual_cast_dtype

        xc = xc.to(dtype)
        device = xc.device
        t = ms.timestep(t).float()
        if context is not None:
            context = comfy.model_management.cast_to_device(context, device, dtype)

        def _convert_extra(v):
            if isinstance(v, torch.Tensor):
                return _convert_tensor(v, dtype, device)
            if isinstance(v, dict):
                return {k: _convert_extra(val) for k, val in v.items()}
            if isinstance(v, (list, tuple)):
                return type(v)(_convert_extra(item) for item in v)
            return v

        extra_conds = {o: _convert_extra(kwargs[o]) for o in kwargs}
        t = bm.process_timestep(t, x=x, **extra_conds)

        with torch.cuda.device(self.secondary_device), safe_rope_fallback():
            model_output = sec(xc, t, context=context, control=control,
                               transformer_options=transformer_options, **extra_conds)

        return ms.calculate_denoised(sigma, model_output.float(), x)

    def cleanup(self):
        """Free the secondary model and stop worker threads."""
        logger.info("CFG Parallel: cleaning up secondary model")

        self.neg_work_q.put(None)
        self.pos_work_q.put(None)

        sec_model = self.secondary_diff_model
        if sec_model is not None:
            dev = self.secondary_device
            self.secondary_diff_model = None
            del sec_model
            gc.collect()
            if dev is not None and dev.type == "cuda":
                torch.cuda.empty_cache()
            logger.info("  Secondary model freed")


class CfgParallel:
    """
    CFG Parallel: runs positive conditioning on primary GPU and negative
    conditioning on a secondary GPU simultaneously
    """

    @classmethod
    def INPUT_TYPES(cls):
        devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
        if not devices:
            devices = ["cuda:0"]
        return {
            "required": {
                "model": ("MODEL",),
                "secondary_device": (devices, {"default": devices[-1] if len(devices) > 1 else devices[0]}),
            },
            "optional": {
                "disable_dynamic_vram": ("BOOLEAN", {"default": True, "tooltip": "Disable dynamic VRAM on the model. Recommended on with torch.compile or dynamic weight changes such as LoRA."}),
                "verbose_debug": ("BOOLEAN", {"default": False, "tooltip": "Enable detailed per-step debug logging: timing, VRAM, transfers, bandwidth, memory."}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "setup"
    CATEGORY = "sampling/parallel"
    DESCRIPTION = "Runs positive conditioning on primary GPU and negative conditioning on a secondary GPU simultaneously. For torch.compile support, place this node AFTER TorchCompileModel in the chain."

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Force re-execution when any option changes (verbose_debug, disable_dynamic_vram, etc.)
        return float("NaN")

    def setup(self, model, secondary_device, disable_dynamic_vram=False, verbose_debug=False):
        logger.info(f"CFG Parallel: setup called | disable_dynamic_vram={disable_dynamic_vram} verbose_debug={verbose_debug}")
        secondary_device = torch.device(secondary_device)
        if disable_dynamic_vram:
            try:
                cloned = model.clone(disable_dynamic=True)
                logger.info("CFG Parallel: dynamic VRAM disabled")
            except TypeError:
                logger.warning("CFG Parallel: this ComfyUI version does not support disable_dynamic, using normal clone")
                cloned = model.clone()
        else:
            cloned = model.clone()

        base_model = cloned.model
        diffusion_model = base_model.diffusion_model

        logger.info(f"CFG Parallel: cloning diffusion model to {secondary_device}")

        secondary_diff_model = copy.deepcopy(diffusion_model)
        secondary_diff_model.to(secondary_device)

        logger.info(f"  Secondary model on {secondary_device}")

        compile_kwargs = cloned.model_options.get("torch_compile_kwargs", None)

        state = CfgParallelState(
            secondary_diff_model=secondary_diff_model,
            secondary_device=secondary_device,
            compile_kwargs=compile_kwargs,
            diffusion_model=diffusion_model,
            cloned_ref=weakref.ref(cloned),
            base_model_ref=weakref.ref(base_model),
            verbose_debug=verbose_debug,
        )

        # Worker thread for secondary GPU (uncond)
        def _neg_worker():
            torch.cuda.set_device(secondary_device)
            with torch.inference_mode():
                while True:
                    item = state.neg_work_q.get()
                    if item is None:
                        break
                    gen_id, func, args, kwargs = item
                    try:
                        if state.verbose:
                            torch.cuda.synchronize(secondary_device)
                            t0 = time.perf_counter()
                        res = func(*args, **kwargs)
                        if state.verbose:
                            torch.cuda.synchronize(secondary_device)
                            elapsed = time.perf_counter() - t0
                            state.neg_result_q.put((gen_id, "ok", res, elapsed))
                        else:
                            state.neg_result_q.put((gen_id, "ok", res, None))
                    except Exception as e:
                        state.neg_result_q.put((gen_id, "err", e, None))

        neg_worker = threading.Thread(target=_neg_worker, daemon=True)
        neg_worker.start()

        # Worker thread for primary GPU (cond in deferred mode)
        def _pos_worker():
            primary_dev = torch.device("cuda:0")  # will be set properly on first use
            with torch.inference_mode():
                while True:
                    item = state.pos_work_q.get()
                    if item is None:
                        break
                    gen_id, func, args, kwargs = item
                    try:
                        if state.verbose:
                            dev = args[0].device if args else primary_dev
                            torch.cuda.synchronize(dev)
                            t0 = time.perf_counter()
                        res = func(*args, **kwargs)
                        if state.verbose:
                            dev = res.device if isinstance(res, torch.Tensor) else primary_dev
                            torch.cuda.synchronize(dev)
                            elapsed = time.perf_counter() - t0
                            state.pos_result_q.put((gen_id, "ok", res, elapsed))
                        else:
                            state.pos_result_q.put((gen_id, "ok", res, None))
                    except Exception as e:
                        state.pos_result_q.put((gen_id, "err", e, None))

        pos_worker = threading.Thread(target=_pos_worker, daemon=True)
        pos_worker.start()

        weakref.finalize(cloned, CfgParallelState.cleanup, state)

        def cfg_parallel_wrapper(apply_model_func, kwargs):
            input_x = kwargs["input"]
            timestep = kwargs["timestep"]
            c = kwargs["c"]
            cond_or_uncond = kwargs.get("cond_or_uncond", [])
            verbose = state.verbose

            if state.secondary_diff_model is None:
                return apply_model_func(input_x, timestep, **c)

            state.sync_weights()
            state.compile_secondary()
            state.resync_if_hooks_changed()

            cond_indices = [i for i, v in enumerate(cond_or_uncond) if v == 0]
            uncond_indices = [i for i, v in enumerate(cond_or_uncond) if v == 1]

            primary_dev = input_x.device

            if not cond_indices or not uncond_indices:
                # Single-type pass (cond-only or uncond-only, e.g. Lumina2)
                # Use deferred parallel: dispatch to worker, return dummy, fix in post_cfg
                is_cond_only = cond_indices and not uncond_indices
                is_uncond_only = uncond_indices and not cond_indices

                sigma_val = timestep.flatten()[0].item()

                if is_cond_only:
                    # ── Cond-only pass ──
                    # Check if this is a new sigma (new denoising step) or repeat at same sigma
                    if state.deferred_sigma is None or sigma_val != state.deferred_sigma:
                        # First cond at new sigma → dispatch to primary worker, return dummy
                        state.deferred_sigma = sigma_val
                        state.deferred_cond_count = 1
                        state.deferred_mode = True
                        state.deferred_uncond_future = None

                        state.gen_id += 1
                        current_gen = state.gen_id

                        if verbose:
                            state.step_count += 1
                            logger.info(f"  [DEBUG] ── Deferred step {state.step_count} begin (σ={sigma_val:.6f}) ──")
                            logger.info(f"  [DEBUG]   Dispatching cond to primary worker thread")

                        # Dispatch cond to primary worker thread
                        state.pos_work_q.put((current_gen, apply_model_func, (input_x, timestep), c))
                        state.deferred_cond_future = current_gen

                        return torch.zeros_like(input_x)
                    else:
                        # Second+ cond at same sigma → area conditioning detected
                        # Abort deferred mode, wait for first cond result, run sequentially
                        state.deferred_cond_count += 1

                        if state.deferred_cond_count == 2 and state.deferred_cond_future is not None:
                            # Wait for first cond result from worker
                            if verbose:
                                logger.info(f"  [DEBUG]   Area conditioning detected (2nd cond at same σ), falling back to sequential")
                            while True:
                                gen_id, status, result, elapsed = state.pos_result_q.get()
                                if gen_id == state.deferred_cond_future:
                                    break
                            if status == "err":
                                raise result
                            state.deferred_mode = False
                            state.deferred_cond_future = None
                            # Return the first cond result we waited for? No — that was for
                            # the FIRST cond call which already returned a dummy zero.
                            # ComfyUI has already used the dummy. We can't fix it retroactively.
                            # For area conditioning, just fall through to sequential from now on.

                        # Run this cond call sequentially on primary
                        return apply_model_func(input_x, timestep, **c)

                elif is_uncond_only:
                    # ── Uncond-only pass ──
                    if state.deferred_mode and sigma_val == state.deferred_sigma:
                        # Expected uncond after deferred cond → dispatch to secondary, return dummy
                        dev = secondary_device

                        if verbose:
                            logger.info(f"  [DEBUG]   Dispatching uncond to secondary worker thread")

                        x_sec = _move_to_device(input_x, dev, non_blocking=True)
                        t_sec = _move_to_device(timestep, dev, non_blocking=True)
                        c_sec = _move_to_device(c, dev, non_blocking=True)

                        state.gen_id += 1
                        current_gen = state.gen_id
                        state.neg_work_q.put((current_gen, state.apply_model, (x_sec, t_sec), c_sec))
                        state.deferred_uncond_future = current_gen

                        return torch.zeros_like(input_x)
                    else:
                        # Not in deferred mode, run uncond on secondary sequentially
                        if verbose:
                            state.step_count += 1
                            logger.info(f"  [DEBUG] ── Pass {state.step_count} [uncond (secondary)] ──")
                            torch.cuda.synchronize(primary_dev)
                            t0 = time.perf_counter()

                        dev = secondary_device
                        x_sec = _move_to_device(input_x, dev, non_blocking=True)
                        t_sec = _move_to_device(timestep, dev, non_blocking=True)
                        c_sec = _move_to_device(c, dev, non_blocking=True)
                        out = state.apply_model(x_sec, t_sec, **c_sec)
                        out = _move_to_device(out, primary_dev)

                        if verbose:
                            torch.cuda.synchronize(primary_dev)
                            elapsed = time.perf_counter() - t0
                            logger.info(f"  [DEBUG]   uncond (secondary): {elapsed * 1000:.1f}ms")
                            logger.info(f"  [DEBUG] ── Pass {state.step_count} end ({elapsed * 1000:.1f}ms) ──")

                        return out

                # Shouldn't reach here, but fallback
                return apply_model_func(input_x, timestep, **c)

            # ── Step start ──
            state.step_count += 1
            step_num = state.step_count
            if verbose:
                step_t0 = time.perf_counter()
                pri_alloc0, pri_resv0 = _gpu_mem_info(primary_dev)
                sec_alloc0, sec_resv0 = _gpu_mem_info(secondary_device)
                ram0 = _system_ram_mb()
                logger.info(f"  [DEBUG] ── Step {step_num} begin ──")
                logger.info(f"  [DEBUG]   input_x: {list(input_x.shape)} {input_x.dtype} on {input_x.device}")
                logger.info(f"  [DEBUG]   cond_or_uncond: {cond_or_uncond} → cond_idx={cond_indices} uncond_idx={uncond_indices}")
                logger.info(f"  [DEBUG]   VRAM primary ({primary_dev}): alloc={_fmt_bytes(pri_alloc0)} resv={_fmt_bytes(pri_resv0)}")
                logger.info(f"  [DEBUG]   VRAM secondary ({secondary_device}): alloc={_fmt_bytes(sec_alloc0)} resv={_fmt_bytes(sec_resv0)}")
                if ram0 is not None:
                    logger.info(f"  [DEBUG]   System RAM: {ram0:.1f} MB")

            num_entries = len(cond_or_uncond)
            total_batch = input_x.shape[0]

            def _split_tensor(t, indices):
                if isinstance(t, torch.Tensor) and t.dim() > 0 and t.shape[0] == total_batch:
                    chunks = t.chunk(num_entries)
                    return torch.cat([chunks[i] for i in indices])
                return t

            def _split_dict(d, indices):
                out = {}
                for k, v in d.items():
                    if isinstance(v, torch.Tensor):
                        out[k] = _split_tensor(v, indices)
                    elif isinstance(v, dict):
                        out[k] = _split_dict(v, indices)
                    elif isinstance(v, (list, tuple)) and len(v) == num_entries:
                        out[k] = type(v)(v[i] for i in indices)
                    else:
                        out[k] = v
                return out

            cond_x = _split_tensor(input_x, cond_indices)
            cond_t = _split_tensor(timestep, cond_indices)
            cond_c = _split_dict(c, cond_indices)

            uncond_x = _split_tensor(input_x, uncond_indices)
            uncond_t = _split_tensor(timestep, uncond_indices)
            uncond_c = _split_dict(c, uncond_indices)

            dev = secondary_device

            # ── Transfer to secondary device ──
            if verbose:
                transfer_bytes = _count_transfer_bytes(uncond_x) + _count_transfer_bytes(uncond_t) + _count_transfer_bytes(uncond_c)
                torch.cuda.synchronize(primary_dev)
                xfer_t0 = time.perf_counter()

            uncond_x = _move_to_device(uncond_x, dev, non_blocking=True)
            uncond_t = _move_to_device(uncond_t, dev, non_blocking=True)
            uncond_c = _move_to_device(uncond_c, dev, non_blocking=True)

            if verbose:
                torch.cuda.synchronize(dev)
                xfer_elapsed = time.perf_counter() - xfer_t0
                xfer_bw = transfer_bytes / xfer_elapsed if xfer_elapsed > 0 else 0
                logger.info(f"  [DEBUG]   Transfer → secondary: {_fmt_bytes(transfer_bytes)} in {xfer_elapsed * 1000:.1f}ms ({_fmt_bytes(xfer_bw)}/s)")

            # ── Dispatch neg pass to secondary GPU worker thread ──
            state.gen_id += 1
            current_gen = state.gen_id
            state.neg_work_q.put((current_gen, state.apply_model, (uncond_x, uncond_t), uncond_c))

            if verbose:
                torch.cuda.synchronize(primary_dev)
                pos_t0 = time.perf_counter()

            res_pos = apply_model_func(cond_x, cond_t, **cond_c)

            if verbose:
                torch.cuda.synchronize(primary_dev)
                pos_elapsed = time.perf_counter() - pos_t0

            # Wait for neg result, discarding stale results from interrupted runs
            if verbose:
                wait_t0 = time.perf_counter()
            while True:
                gen_id, status_neg, res_neg, neg_elapsed = state.neg_result_q.get()
                if gen_id == current_gen:
                    break
            if verbose:
                wait_elapsed = time.perf_counter() - wait_t0
            if status_neg == "err":
                raise res_neg

            # ── Transfer result back to primary ──
            if verbose:
                ret_bytes = _count_transfer_bytes(res_neg)
                torch.cuda.synchronize(secondary_device)
                ret_t0 = time.perf_counter()

            res_neg = _move_to_device(res_neg, primary_dev, non_blocking=True)
            # Ensure the non-blocking copy completes before we use the result
            torch.cuda.current_stream(primary_dev).synchronize()

            if verbose:
                ret_elapsed = time.perf_counter() - ret_t0
                ret_bw = ret_bytes / ret_elapsed if ret_elapsed > 0 else 0

            # Reassemble output: cond/uncond chunks in original order
            res_pos_chunks = res_pos.chunk(len(cond_indices))
            res_neg_chunks = res_neg.chunk(len(uncond_indices))
            output_chunks = [None] * num_entries
            cond_idx = 0
            uncond_idx = 0
            for i, v in enumerate(cond_or_uncond):
                if v == 0:
                    output_chunks[i] = res_pos_chunks[cond_idx]
                    cond_idx += 1
                else:
                    output_chunks[i] = res_neg_chunks[uncond_idx]
                    uncond_idx += 1
            output = torch.cat(output_chunks)

            # ── Step summary ──
            if verbose:
                step_elapsed = time.perf_counter() - step_t0
                pri_alloc1, pri_resv1 = _gpu_mem_info(primary_dev)
                sec_alloc1, sec_resv1 = _gpu_mem_info(secondary_device)
                ram1 = _system_ram_mb()

                # Overlap efficiency: how much of the slower path was hidden
                if neg_elapsed is not None:
                    max_compute = max(pos_elapsed, neg_elapsed)
                    sequential_time = pos_elapsed + neg_elapsed
                    overlap_pct = (1.0 - max_compute / sequential_time) * 100 if sequential_time > 0 else 0
                else:
                    neg_elapsed = 0
                    overlap_pct = 0

                logger.info(f"  [DEBUG]   Cond  (primary)  : {pos_elapsed * 1000:.1f}ms")
                logger.info(f"  [DEBUG]   Uncond (secondary): {neg_elapsed * 1000:.1f}ms")
                logger.info(f"  [DEBUG]   Overlap efficiency: {overlap_pct:.1f}%")
                logger.info(f"  [DEBUG]   Wait for uncond  : {wait_elapsed * 1000:.1f}ms")
                logger.info(f"  [DEBUG]   Transfer ← primary: {_fmt_bytes(ret_bytes)} in {ret_elapsed * 1000:.1f}ms ({_fmt_bytes(ret_bw)}/s)")
                pri_delta = pri_alloc1 - pri_alloc0
                sec_delta = sec_alloc1 - sec_alloc0
                pri_sign = "+" if pri_delta >= 0 else "-"
                sec_sign = "+" if sec_delta >= 0 else "-"
                logger.info(f"  [DEBUG]   VRAM primary  Δalloc={pri_sign}{_fmt_bytes(abs(pri_delta)):>10s}  now={_fmt_bytes(pri_alloc1)} resv={_fmt_bytes(pri_resv1)}")
                logger.info(f"  [DEBUG]   VRAM secondary Δalloc={sec_sign}{_fmt_bytes(abs(sec_delta)):>10s}  now={_fmt_bytes(sec_alloc1)} resv={_fmt_bytes(sec_resv1)}")
                if ram0 is not None and ram1 is not None:
                    logger.info(f"  [DEBUG]   System RAM: {ram1:.1f} MB (Δ{ram1 - ram0:+.1f} MB)")
                logger.info(f"  [DEBUG]   Total step time: {step_elapsed * 1000:.1f}ms")
                logger.info(f"  [DEBUG] ── Step {step_num} end ──")

            return output

        def parallel_post_cfg(args):
            """Replace garbage CFG output with correct result when in deferred parallel mode."""
            if not state.deferred_mode:
                return args["denoised"]

            verbose = state.verbose

            if verbose:
                post_t0 = time.perf_counter()
                logger.info(f"  [DEBUG]   post_cfg: collecting deferred parallel results")

            # Wait for cond result from primary worker
            real_cond = None
            if state.deferred_cond_future is not None:
                while True:
                    gen_id, status, result, pos_elapsed = state.pos_result_q.get()
                    if gen_id == state.deferred_cond_future:
                        break
                if status == "err":
                    raise result
                real_cond = result

            # Wait for uncond result from secondary worker
            real_uncond = None
            if state.deferred_uncond_future is not None:
                while True:
                    gen_id, status, result, neg_elapsed = state.neg_result_q.get()
                    if gen_id == state.deferred_uncond_future:
                        break
                if status == "err":
                    raise result
                real_uncond = result
                if real_cond is not None:
                    real_uncond = _move_to_device(real_uncond, real_cond.device, non_blocking=True)
                    torch.cuda.current_stream(real_cond.device).synchronize()

            if real_cond is None or real_uncond is None:
                # Incomplete deferred (shouldn't happen in normal flow)
                if verbose:
                    logger.warning(f"  [DEBUG]   post_cfg: incomplete deferred results (cond={real_cond is not None}, uncond={real_uncond is not None}), passing through")
                state.deferred_mode = False
                state.deferred_cond_future = None
                state.deferred_uncond_future = None
                return args["denoised"]

            # Recompute CFG with real results
            cond_scale = args["cond_scale"]
            x = args["input"]
            model_options = args["model_options"]

            if "sampler_cfg_function" in model_options:
                # Respect custom CFG function (e.g. Rescale CFG)
                cfg_fn = model_options["sampler_cfg_function"]
                cfg_args = {
                    "cond": x - real_cond,
                    "uncond": x - real_uncond,
                    "cond_scale": cond_scale,
                    "timestep": args["sigma"],
                    "input": x,
                    "sigma": args["sigma"],
                    "cond_denoised": real_cond,
                    "uncond_denoised": real_uncond,
                    "model": args["model"],
                    "model_options": model_options,
                    "input_cond": args.get("cond", None),
                    "input_uncond": args.get("uncond", None),
                }
                cfg_result = x - cfg_fn(cfg_args)
            else:
                cfg_result = real_uncond + (real_cond - real_uncond) * cond_scale

            if verbose:
                post_elapsed = time.perf_counter() - post_t0
                if pos_elapsed is not None and neg_elapsed is not None:
                    max_compute = max(pos_elapsed, neg_elapsed)
                    sequential_time = pos_elapsed + neg_elapsed
                    overlap_pct = (1.0 - max_compute / sequential_time) * 100 if sequential_time > 0 else 0
                    logger.info(f"  [DEBUG]   Cond  (primary)  : {pos_elapsed * 1000:.1f}ms")
                    logger.info(f"  [DEBUG]   Uncond (secondary): {neg_elapsed * 1000:.1f}ms")
                    logger.info(f"  [DEBUG]   Overlap efficiency: {overlap_pct:.1f}%")
                logger.info(f"  [DEBUG]   post_cfg correction: {post_elapsed * 1000:.1f}ms")
                logger.info(f"  [DEBUG] ── Deferred step {state.step_count} end ──")

            # Reset deferred state
            state.deferred_mode = False
            state.deferred_cond_future = None
            state.deferred_uncond_future = None

            return cfg_result

        cloned.set_model_unet_function_wrapper(cfg_parallel_wrapper)
        cloned.set_model_sampler_post_cfg_function(parallel_post_cfg, disable_cfg1_optimization=True)

        logger.info("CFG Parallel: wrapper + post_cfg hook installed, ready for sampling")
        return (cloned,)


NODE_CLASS_MAPPINGS = {
    "CfgParallel": CfgParallel,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CfgParallel": "CFG Parallel 2nd GPU Loader",
}
