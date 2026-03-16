# Copyright (c) ModelScope Contributors. All rights reserved.
from peft import PeftConfig, PeftModel
from torch import nn


def get_peft_model(model: nn.Module, peft_config: PeftConfig) -> PeftModel:
    pass
