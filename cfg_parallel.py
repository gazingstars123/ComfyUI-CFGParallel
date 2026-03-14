import torch
import copy
import gc
import queue
import threading
import weakref
import logging
import comfy.model_management

logger = logging.getLogger("CfgParallel")

def _convert_tensor(extra, dtype, device):
    if hasattr(extra, "dtype"):
        if extra.dtype != torch.int and extra.dtype != torch.long:
            extra = comfy.model_management.cast_to_device(extra, device, dtype)
        else:
            extra = comfy.model_management.cast_to_device(extra, device, None)
    return extra


def _cleanup_parallel(state):
    """Free the secondary model and stop worker threads."""
    logger.info("CFG Parallel: cleaning up secondary model")

    # Stop worker thread
    neg_q = state.get("neg_work_q")
    if neg_q is not None:
        neg_q.put(None)

    # Delete secondary model and free GPU memory
    sec_model = state.get("secondary_diff_model")
    if sec_model is not None:
        sec_device = state.get("secondary_device")
        del state["secondary_diff_model"]
        del sec_model
        gc.collect()
        if sec_device is not None and sec_device.type == "cuda":
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
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "setup"
    CATEGORY = "sampling/parallel"
    DESCRIPTION = "Runs positive conditioning on primary GPU and negative conditioning on a secondary GPU simultaneously. For torch.compile support, place this node AFTER TorchCompileModel in the chain."

    def setup(self, model, secondary_device):
        secondary_device = torch.device(secondary_device)
        cloned = model.clone()

        base_model = cloned.model
        diffusion_model = base_model.diffusion_model

        logger.info(f"CFG Parallel: cloning diffusion model to {secondary_device}")

        secondary_diff_model = copy.deepcopy(diffusion_model)
        secondary_diff_model.to(secondary_device)

        logger.info(f"  Secondary model on {secondary_device}")

        compile_kwargs = cloned.model_options.get("torch_compile_kwargs", None)

        _state = {
            "secondary_diff_model": secondary_diff_model,
            "secondary_device": secondary_device,
            "neg_work_q": None,
            "weights_synced": False,
            "compile_kwargs": compile_kwargs,
            "gen_id": 0,
        }

        # Weakref to base_model to avoid circular reference
        base_model_ref = weakref.ref(base_model)

        def _sync_weights():
            """Copy patched weights from primary to secondary model.

            Called lazily on the first wrapper invocation, at which point
            ComfyUI has already applied LoRA patches to the primary model.
            """
            if _state["weights_synced"]:
                return
            sec = _state.get("secondary_diff_model")
            if sec is None:
                return
            src_sd = diffusion_model.state_dict()
            sec_sd = sec.state_dict()
            updated = 0
            for key in sec_sd:
                if key in src_sd:
                    sec_sd[key].copy_(src_sd[key].to(sec_sd[key].device))
                    updated += 1
            # Apply torch.compile to secondary model
            ckw = _state.get("compile_kwargs")
            if ckw is not None:
                compile_args = {k: v for k, v in ckw.items()}
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
                    _state["secondary_diff_model"] = torch.compile(sec, **compile_args)
                    logger.info("  Compiled entire secondary diffusion model")
                torch._dynamo.reset()

            _state["weights_synced"] = True
            logger.info(f"  Synced {updated} weight tensors to secondary model")

        def secondary_apply_model(x, t, c_concat=None, c_crossattn=None, control=None, transformer_options={}, **kwargs):
            sec = _state.get("secondary_diff_model")
            if sec is None:
                raise RuntimeError("Secondary model has been cleaned up")

            bm = base_model_ref()
            ms = bm.model_sampling if bm is not None else base_model.model_sampling
            sigma = t
            xc = ms.calculate_input(sigma, x)

            if c_concat is not None:
                xc = torch.cat([xc] + [comfy.model_management.cast_to_device(c_concat, xc.device, xc.dtype)], dim=1)

            context = c_crossattn
            dtype = bm.get_dtype() if bm is not None else torch.bfloat16
            if bm is not None and bm.manual_cast_dtype is not None:
                dtype = bm.manual_cast_dtype

            xc = xc.to(dtype)
            device = xc.device
            t = ms.timestep(t).float()
            if context is not None:
                context = comfy.model_management.cast_to_device(context, device, dtype)

            extra_conds = {}
            for o in kwargs:
                extra = kwargs[o]
                if hasattr(extra, "dtype"):
                    extra = _convert_tensor(extra, dtype, device)
                elif isinstance(extra, list):
                    extra = [_convert_tensor(ext, dtype, device) for ext in extra]
                extra_conds[o] = extra

            if bm is not None:
                t = bm.process_timestep(t, x=x, **extra_conds)

            model_output = sec(xc, t, context=context, control=control, transformer_options=transformer_options, **extra_conds)

            return ms.calculate_denoised(sigma, model_output.float(), x)

        # Persistent worker thread for secondary GPU (neg pass only)
        neg_work_q = queue.Queue()
        neg_result_q = queue.Queue()
        _state["neg_work_q"] = neg_work_q

        def _neg_worker():
            torch.cuda.set_device(secondary_device)
            with torch.inference_mode():
                while True:
                    item = neg_work_q.get()
                    if item is None:
                        break
                    gen_id, func, args, kwargs = item
                    try:
                        sec = _state.get("secondary_diff_model")
                        if sec is not None:
                            has_block_swap = getattr(sec, 'blocks_to_swap', None)
                            if has_block_swap and has_block_swap > 0:
                                sec.prepare_block_swap_before_forward()
                        res = func(*args, **kwargs)
                        neg_result_q.put((gen_id, "ok", res))
                    except Exception as e:
                        neg_result_q.put((gen_id, "err", e))

        worker_neg = threading.Thread(target=_neg_worker, daemon=True)
        worker_neg.start()

        # free the secondary model and stop workers
        weakref.finalize(cloned, _cleanup_parallel, _state)

        def _move_to_device(tensor, device):
            if isinstance(tensor, torch.Tensor) and tensor.device != device:
                return tensor.to(device)
            return tensor

        def _move_dict_to_device(d, device):
            out = {}
            for k, v in d.items():
                if isinstance(v, torch.Tensor):
                    out[k] = _move_to_device(v, device)
                elif isinstance(v, dict):
                    out[k] = _move_dict_to_device(v, device)
                elif isinstance(v, (list, tuple)):
                    out[k] = type(v)(_move_to_device(x, device) if isinstance(x, torch.Tensor) else x for x in v)
                else:
                    out[k] = v
            return out

        def cfg_parallel_wrapper(apply_model_func, kwargs):
            input_x = kwargs["input"]
            timestep = kwargs["timestep"]
            c = kwargs["c"]
            cond_or_uncond = kwargs.get("cond_or_uncond", [])

            # If secondary model was cleaned up, fall back to primary only
            if _state.get("secondary_diff_model") is None:
                return apply_model_func(input_x, timestep, **c)

            # Lazy sync
            _sync_weights()

            cond_indices = [i for i, v in enumerate(cond_or_uncond) if v == 0]
            uncond_indices = [i for i, v in enumerate(cond_or_uncond) if v == 1]

            primary_dev = input_x.device

            if not cond_indices or not uncond_indices:
                if uncond_indices and not cond_indices:
                    dev = secondary_device
                    x_sec = _move_to_device(input_x, dev)
                    t_sec = _move_to_device(timestep, dev)
                    c_sec = _move_dict_to_device(c, dev)
                    out = secondary_apply_model(x_sec, t_sec, **c_sec)
                    return _move_to_device(out, primary_dev)
                return apply_model_func(input_x, timestep, **c)

            num_entries = len(cond_or_uncond)
            total_batch = input_x.shape[0]

            def _split_tensor(t, indices):
                if isinstance(t, torch.Tensor) and t.dim() > 0 and t.shape[0] == total_batch:
                    # Split into per-entry chunks, select by indices, re-concatenate
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
            uncond_x = _move_to_device(uncond_x, dev)
            uncond_t = _move_to_device(uncond_t, dev)
            uncond_c = _move_dict_to_device(uncond_c, dev)

            # Dispatch neg pass to secondary GPU worker thread
            _state["gen_id"] += 1
            current_gen = _state["gen_id"]
            neg_work_q.put((current_gen, secondary_apply_model, (uncond_x, uncond_t), uncond_c))
            res_pos = apply_model_func(cond_x, cond_t, **cond_c)

            # Wait for neg result, discarding stale results from interrupted runs
            while True:
                gen_id, status_neg, res_neg = neg_result_q.get()
                if gen_id == current_gen:
                    break
            if status_neg == "err":
                raise res_neg

            res_neg = _move_to_device(res_neg, primary_dev)

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

            return output

        cloned.set_model_unet_function_wrapper(cfg_parallel_wrapper)

        logger.info("CFG Parallel: wrapper installed, ready for sampling")
        return (cloned,)


NODE_CLASS_MAPPINGS = {
    "CfgParallel": CfgParallel,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CfgParallel": "CFG Parallel 2nd GPU Loader",
}
