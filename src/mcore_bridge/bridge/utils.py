# Copyright (c) ModelScope Contributors. All rights reserved.

from mcore_bridge.config import ModelConfig

from .gpt_bridge import GPTBridge


def get_bridge(config: ModelConfig) -> GPTBridge:
    pass
