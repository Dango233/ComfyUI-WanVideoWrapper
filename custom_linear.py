import logging
import os
import torch
import torch.nn as nn
from accelerate import init_empty_weights

import comfy.lora as comfy_lora

#based on https://github.com/huggingface/diffusers/blob/main/src/diffusers/quantizers/gguf/utils.py
def _replace_linear(model, compute_dtype, state_dict, prefix="", patches=None, scale_weights=None):
   
    has_children = list(model.children())
    if not has_children:
        return
    for name, module in model.named_children():
        module_prefix = prefix + name + "."
        module_prefix = module_prefix.replace("_orig_mod.", "")
        _replace_linear(module, compute_dtype, state_dict, module_prefix, patches, scale_weights)

        if isinstance(module, nn.Linear) and "loras" not in module_prefix:
            in_features = state_dict[module_prefix + "weight"].shape[1]
            out_features = state_dict[module_prefix + "weight"].shape[0]
            if scale_weights is not None:
                scale_key = f"{module_prefix}scale_weight"

            with init_empty_weights():
                model._modules[name] = CustomLinear(
                    in_features,
                    out_features,
                    module.bias is not None,
                    compute_dtype=compute_dtype,
                    scale_weight=scale_weights.get(scale_key) if scale_weights else None
                )
            #set_lora_params(model._modules[name], patches, module_prefix)
            model._modules[name].source_cls = type(module)
            # Force requires_grad to False to avoid unexpected errors
            model._modules[name].requires_grad_(False)

    return model

def set_lora_params(module, patches, module_prefix=""):
    remove_lora_from_module(module)
    # Recursively set lora_diffs and lora_strengths for all CustomLinear layers
    for name, child in module.named_children():
        child_prefix = (f"{module_prefix}{name}.")
        set_lora_params(child, patches, child_prefix)
    if isinstance(module, CustomLinear):
        key = f"diffusion_model.{module_prefix}weight"
        patch = patches.get(key, [])
        #print(f"Processing LoRA patches for {key}: {len(patch)} patches found")
        if len(patch) == 0:
            key = key.replace("_orig_mod.", "")
            patch = patches.get(key, [])
            #print(f"Processing LoRA patches for {key}: {len(patch)} patches found")
        if len(patch) != 0:
            lora_diffs = []
            for p in patch:
                lora_obj = p[1]
                if "head" in key:
                    continue  # For now skip LoRA for head layers
                elif hasattr(lora_obj, "weights"):
                    lora_diffs.append(lora_obj.weights)
                elif isinstance(lora_obj, tuple) and lora_obj[0] == "diff":
                    lora_diffs.append(lora_obj[1])
                else:
                    continue
            lora_strengths = [p[0] for p in patch]
            module.lora = (lora_diffs, lora_strengths)
            module.step = 0  # Initialize step for LoRA scheduling


logger = logging.getLogger("wan_shot_lora")


class CustomLinear(nn.Linear):
    runtime_context = None

    def __init__(
        self,
        in_features,
        out_features,
        bias=False,
        compute_dtype=None,
        device=None,
        scale_weight=None
    ) -> None:
        super().__init__(in_features, out_features, bias, device)
        self.compute_dtype = compute_dtype
        self.lora = None
        self.step = 0
        self.scale_weight = scale_weight
        self.bias_function = []
        self.weight_function = []
        self.shot_lora = []
        self.shot_lora_key = None
        self._global_debug_seen = set()
        self._global_debug_last_step = None

    def clear_shot_lora_cache(self):
        """Release any cached per-device LoRA weights to help free VRAM."""
        if not hasattr(self, "shot_lora") or not self.shot_lora:
            return
        for components in self.shot_lora:
            if not components:
                continue
            for component in components:
                if not isinstance(component, dict):
                    continue
                cache = component.get("cache")
                if isinstance(cache, dict):
                    cache.clear()

    def forward(self, input):
        if self.bias is not None:
            bias = self.bias.to(input)
        else:
            bias = None
        weight = self.weight.to(input)

        if self.scale_weight is not None:
            if weight.numel() < input.numel():
                weight = weight * self.scale_weight
            else:
                input = input * self.scale_weight

        if self.lora is not None:
            weight = self.apply_lora(weight).to(self.compute_dtype)

        output = torch.nn.functional.linear(input, weight, bias)

        if self.shot_lora and CustomLinear.runtime_context is not None:
            output = self._apply_shot_lora(output, input, weight)

        return output

    @torch.compiler.disable()
    def apply_lora(self, weight):
        debug_flag = os.environ.get("WAN_GLOBAL_LORA_DEBUG", "0").lower() in ("1", "true", "yes")
        current_step = getattr(self, "step", 0)
        if debug_flag and self._global_debug_last_step != current_step:
            self._global_debug_seen.clear()
            self._global_debug_last_step = current_step
        for lora_diff, lora_strength in zip(self.lora[0], self.lora[1]):
            if isinstance(lora_strength, list):
                lora_strength = lora_strength[self.step]
                if lora_strength == 0.0:
                    continue
            elif lora_strength == 0.0:
                continue
            patch_diff = torch.mm(
                lora_diff[0].flatten(start_dim=1).to(weight.device),
                lora_diff[1].flatten(start_dim=1).to(weight.device)
            ).reshape(weight.shape)
            alpha = lora_diff[2] / lora_diff[1].shape[0] if lora_diff[2] is not None else 1.0
            scale = lora_strength * alpha
            if debug_flag:
                module_identifier = getattr(self, "shot_lora_key", None) or hex(id(self))
                debug_key = (id(lora_diff), current_step)
                if debug_key not in self._global_debug_seen:
                    self._global_debug_seen.add(debug_key)
                    delta_mean = float((patch_diff * scale).abs().mean().item())
                    logger.info(
                        "Global LoRA applied: module=%s mean|delta|=%.6f strength=%.4f alpha=%.4f rank=%d step=%d",
                        module_identifier,
                        delta_mean,
                        float(lora_strength),
                        float(alpha),
                        int(lora_diff[1].shape[0]),
                        int(current_step),
                    )
            weight = weight.add(patch_diff, alpha=scale)
        return weight

    def _apply_shot_lora(self, output, input, weight):
        ctx = CustomLinear.runtime_context
        if ctx is None:
            return output

        token_labels = ctx.get("token_labels")
        if token_labels is None or token_labels.numel() == 0:
            return output

        if not self.shot_lora or all(len(components) == 0 or components is None for components in self.shot_lora):
            return output

        flat_input = input.reshape(-1, input.shape[-1])
        flat_output = output.reshape(-1, output.shape[-1])

        if token_labels.numel() != flat_input.shape[0]:
            return output

        device = flat_input.device
        dtype = flat_input.dtype
        current_step = ctx.get("current_step", 0)
        debug_enabled = bool(ctx.get("debug_shot_lora", False))
        module_identifier = getattr(self, "shot_lora_key", None) or hex(id(self))

        for shot_idx, components in enumerate(self.shot_lora):
            if not components:
                continue
            mask = (token_labels == shot_idx)
            if not torch.any(mask):
                continue
            indices = torch.nonzero(mask, as_tuple=False).flatten()
            if indices.numel() == 0:
                continue
            shot_input = flat_input.index_select(0, indices)
            shot_delta = None
            for component in components:
                if not isinstance(component, dict):
                    continue

                strength_entry = component.get("strength", 1.0)
                if isinstance(strength_entry, list):
                    if current_step >= len(strength_entry):
                        continue
                    strength_value = float(strength_entry[current_step])
                else:
                    strength_value = float(strength_entry)
                if strength_value == 0.0:
                    continue

                delta_weight = self._get_shot_delta_weight(
                    component,
                    weight,
                    strength_value,
                    current_step,
                    device,
                    dtype,
                )
                if delta_weight is None:
                    continue

                contribution = torch.nn.functional.linear(shot_input, delta_weight)
                shot_delta = contribution if shot_delta is None else (shot_delta + contribution)

            if shot_delta is not None:
                flat_output.index_add_(0, indices, shot_delta)
                if debug_enabled:
                    debug_seen = ctx.setdefault("debug_seen", set())
                    debug_key = (module_identifier, shot_idx)
                    if debug_key not in debug_seen:
                        debug_seen.add(debug_key)
                        delta_mean = float(shot_delta.abs().mean().item())
                        logger.info(
                            "Shot LoRA applied: module=%s shot=%d mean|delta|=%.6f",
                            module_identifier,
                            shot_idx,
                            delta_mean,
                        )

        return flat_output.view_as(output)

    def _get_shot_delta_weight(self, component, base_weight, strength_value, current_step, device, dtype):
        cache = component.setdefault("cache", {})
        cache_key = (device, dtype, strength_value, current_step)
        delta = cache.get(cache_key)

        if delta is None:
            patch_value = component.get("patch")
            strength_model = component.get("strength_model", 1.0)
            offset = component.get("offset")
            function = component.get("function")
            key = component.get("key")

            if patch_value is None or key is None:
                return None

            base_clone = base_weight.detach().clone()
            patches = [(strength_value, patch_value, strength_model, offset, function)]
            try:
                updated = comfy_lora.calculate_weight(
                    patches,
                    base_clone,
                    key,
                    intermediate_dtype=torch.float32,
                    original_weights=None,
                )
            except Exception as exc:
                logger.warning(
                    "Shot LoRA calculate_weight failed: module=%s key=%s err=%s",
                    getattr(self, "shot_lora_key", None) or hex(id(self)),
                    key,
                    exc,
                )
                component.setdefault("_failed_steps", set()).add(current_step)
                return None

            delta = (updated - base_weight).to(device=device, dtype=dtype)
            if not delta.is_contiguous():
                delta = delta.contiguous()
            cache[cache_key] = delta

        return delta

def remove_lora_from_module(module):
    for name, submodule in module.named_modules():
        if isinstance(submodule, CustomLinear):
            submodule.clear_shot_lora_cache()
        submodule.lora = None


def set_shot_lora_params(module, shot_payload, module_prefix=""):
    for name, child in module.named_children():
        child_prefix = f"{module_prefix}{name}."
        set_shot_lora_params(child, shot_payload, child_prefix)

    if isinstance(module, CustomLinear):
        module.clear_shot_lora_cache()
        key = f"diffusion_model.{module_prefix}weight"
        shot_components = shot_payload.get(key)
        if shot_components is None and "_orig_mod." in key:
            alt_key = key.replace("_orig_mod.", "")
            shot_components = shot_payload.get(alt_key)

        if shot_components is None:
            module.shot_lora = []
            module.shot_lora_key = key
        else:
            module.shot_lora = [components if components is not None else [] for components in shot_components]
            module.shot_lora_key = key
