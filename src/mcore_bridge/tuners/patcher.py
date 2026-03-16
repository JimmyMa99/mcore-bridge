# Copyright (c) ModelScope Contributors. All rights reserved.
from megatron.core.extensions.transformer_engine import TEGroupedLinear, TELayerNormColumnParallelLinear, TELinear
from megatron.core.transformer.moe.router import TopKRouter
from peft.tuners.lora import model
from peft.tuners.tuners_utils import BaseTunerLayer
from torch import nn
from typing import Optional

from .lora import LoraParallelLinear


def dispatch_megatron(
    target: nn.Module,
    adapter_name: str,
    lora_config,
    **kwargs,
) -> Optional[nn.Module]:
    new_module = None

    if isinstance(target, BaseTunerLayer):
        target_base_layer = target.get_base_layer()
    else:
        target_base_layer = target

    linear_cls = (TELayerNormColumnParallelLinear, TELinear, TEGroupedLinear, TopKRouter)
    if isinstance(target_base_layer, linear_cls):
        new_module = LoraParallelLinear(base_layer=target, adapter_name=adapter_name, **kwargs)

    return new_module


model.dispatch_megatron = dispatch_megatron
