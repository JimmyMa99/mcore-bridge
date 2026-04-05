# Copyright (c) ModelScope Contributors. All rights reserved.
import torch
from transformers import PretrainedConfig

from mcore_bridge.bridge import MultimodalGPTBridge

from ..constant import ModelType
from ..register import ModelMeta, register_model
from .utils import HuggingFaceVit


class Gemma4Bridge(MultimodalGPTBridge):
    pass


class Gemma4Vit(HuggingFaceVit):
    module_mapping = {'vision_tower': 'vision_tower', 'embed_vision': 'embed_vision'}
    _vision_tower = ['vision_tower']
    _aligner = ['embed_vision']

    def prepare_model(self, hf_config: PretrainedConfig):
        from transformers.models.gemma4.modeling_gemma4 import (
            Gemma4ForConditionalGeneration,
            Gemma4MultimodalEmbedder,
            Gemma4VisionModel,
        )

        self.vision_tower = Gemma4VisionModel._from_config(hf_config.vision_config)
        self.embed_vision = Gemma4MultimodalEmbedder(hf_config.vision_config, hf_config.text_config).to(
            self.vision_tower.dtype)
        self.model_cls = Gemma4ForConditionalGeneration

    def get_inputs_embeds(self, inputs_embeds, **kwargs):
        input_ids = kwargs['input_ids']
        pixel_values = kwargs.get('pixel_values')
        pixel_values_videos = kwargs.get('pixel_values_videos')

        image_mask, video_mask, _ = self.get_placeholder_mask(input_ids=input_ids, inputs_embeds=inputs_embeds)

        if pixel_values is None and pixel_values_videos is None:
            dummy = self._get_dummy_dependency(inputs_embeds)
            return inputs_embeds + dummy * 0.

        if pixel_values is not None:
            image_features = self.get_image_features(
                pixel_values=pixel_values,
                image_position_ids=kwargs.get('image_position_ids'),
            )
            image_features = image_features.pooler_output.to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask = image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_features)

        if pixel_values_videos is not None:
            video_features = self.get_video_features(
                pixel_values_videos=pixel_values_videos,
                video_position_ids=kwargs.get('video_position_ids'),
            )
            video_features = video_features.pooler_output.to(inputs_embeds.device, inputs_embeds.dtype)
            video_mask = video_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_features)

        return inputs_embeds

    def _get_dummy_dependency(self, inputs_embeds):
        deps = []
        for module_name in ('vision_tower', 'embed_vision'):
            module = getattr(self, module_name, None)
            if module is None:
                continue
            try:
                deps.append(next(module.parameters()).mean())
            except StopIteration:
                continue
        if not deps:
            return inputs_embeds.new_zeros(())
        return sum(dep.to(inputs_embeds.device, inputs_embeds.dtype) for dep in deps)

    def get_placeholder_mask(self, *args, **kwargs):
        with self.patch_hf_config():
            return self.model_cls.get_placeholder_mask(self, *args, **kwargs)

    def get_image_features(self, *args, **kwargs):
        with self.patch_hf_config():
            return self.model_cls.get_image_features(self, *args, **kwargs)

    def get_video_features(self, *args, **kwargs):
        with self.patch_hf_config():
            return self.model_cls.get_video_features(self, *args, **kwargs)


register_model(
    ModelMeta(
        ModelType.gemma4,
        ['gemma4'],
        bridge_cls=Gemma4Bridge,
        visual_cls=Gemma4Vit,
    ))
