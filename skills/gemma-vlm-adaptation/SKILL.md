---
name: gemma-vlm-adaptation
description: Adapt new Hugging Face text-only, VLM, or omni model families into mcore-bridge or Megatron-Core. Use when adding a new architecture, extending parser/config support, wiring loader or bridge classes, handling multimodal towers, fixing weight mapping mismatches, or validating text, vision, and audio execution paths.
---

# Gemma VLM Adaptation

Use this skill to turn a new HF model family into a usable `mcore-bridge` target without guessing. Favor small, provable increments over broad rewrites.

## Workflow

1. Classify the model family before editing code.
2. Extend config parsing so the bridge can represent the family.
3. Decide whether existing text backbone code is enough.
4. Add or update loader, bridge, and multimodal wrappers.
5. Validate in a fixed order: config, build, load, modal injection, train step.

## 1. Classify The Family

Read the HF config and determine which bucket the model belongs to.

- Text-only: no vision or audio tower, no multimodal placeholders.
- VLM: text backbone plus vision tower and projector.
- Omni: text backbone plus at least one extra modality such as audio or video.

Then list family-specific structural deltas:

- attention layout
- RoPE variant
- MLP variant
- layer-wise special modules
- MoE usage
- multimodal placeholder tokens
- processor requirements

If the family mixes multiple layer types, do not flatten them into one average config. Preserve the per-layer layout.

## 2. Extend Config Parsing First

Add parser support before touching runtime code. The target is a `ModelConfig` that can fully express the HF family.

Common fields to check:

- `layer_types`
- `head_dim`
- `global_head_dim`
- `num_key_value_heads`
- `num_global_key_value_heads`
- `sliding_window`
- `rope_scaling`
- `hidden_size_per_layer_input`
- `vocab_size_per_layer_input`
- `num_kv_shared_layers`
- `enable_moe_block`
- `num_experts`
- `top_k_experts`

For mixed-attention families:

- keep the raw `layer_types`
- derive any convenience fields such as skip frequency from that list
- do not assume one `kv_channels` value fits every layer

For MoE families:

- treat parser support and runtime support as separate milestones
- it is acceptable to land config support before weight-loading support, but state that boundary explicitly

## 3. Decide Whether The Existing Text Backbone Is Sufficient

Inspect the HF text model and compare it against existing `mcore-bridge` GPT support.

Re-use the existing GPT path only if all of these are true:

- attention projection shapes are uniform across layers
- layer norms are in expected locations
- MLP width logic is uniform
- there are no extra per-layer inputs or gates
- logits and embedding wiring match existing assumptions

Create family-specific text classes when any of these are false.

Typical customization points:

- custom self-attention class for layer-aware dimensions
- custom MLP class for family-specific width rules
- custom transformer layer for extra residual branches or gates
- custom GPT model for extra embeddings or per-layer inputs
- custom loader to choose the right layer spec

## 4. Wire Loader And Bridge Separately

Do not merge loader concerns and bridge concerns.

Loader responsibilities:

- build the right MCore model classes
- select custom transformer layer specs
- pass family-specific config toggles needed at build time

Bridge responsibilities:

- map HF module names to MCore module names
- patch per-layer runtime config when HF shapes differ by layer
- load any nonstandard embeddings, gates, or projector weights
- define TP split behavior for new tensors

When debugging load failures, first identify whether the error is:

- a bad model build
- a bad tensor shape assumption
- a bad module mapping
- a missing custom preload step

## 5. Handle Multimodal Towers As A Thin Wrapper

For VLM or omni models, keep multimodal logic thin. Reuse HF implementations when practical instead of re-implementing tower internals.

Typical wrapper duties:

- instantiate HF vision or audio tower from config
- instantiate projector or embedder modules
- expose `get_image_features`, `get_video_features`, `get_audio_features`
- expose placeholder-mask logic
- merge modal embeddings into text embeddings

Common failure mode:

- binding helper methods to the wrong HF class

Check whether placeholder and feature helpers live on:

- the top-level conditional generation class, or
- the underlying base model class

If only one modality works, compare helper ownership in the HF source before changing scatter logic.

## 6. Processor And Token Placeholder Checks

Before blaming the model, verify the processor contract.

Check:

- whether image-only calls require explicit text
- whether audio-only calls require explicit text
- actual placeholder tokens and IDs
- returned batch keys such as `mm_token_type_ids`, `image_position_ids`, `input_features_mask`

If the processor auto-expands placeholders, your bridge must consume the expanded sequence rather than the human-readable prompt.

## 7. Validation Ladder

Validate in this order and stop at the first failing stage.

1. Parser smoke
2. Single-GPU build smoke
3. Weight-loading smoke
4. Vision embedding injection smoke
5. Audio embedding injection smoke
6. Text-only train step
7. Vision train step
8. Audio train step
9. Multi-GPU smoke

Use the smallest validated model variant first. For a family with E2B, E4B, 31B, and MoE variants:

- start with the smallest dense multimodal checkpoint
- then validate the next dense sibling
- defer large and MoE checkpoints until the dense path is stable

## 8. Evidence To Capture For PRs

Record concrete proof, not just conclusions.

Capture:

- exact model IDs tested
- exact commands used
- whether validation was config-level, build-level, or weight-level
- representative tensor shapes for the first sliding and first full-attention layer
- modal smoke outputs
- known unvalidated variants

Do not claim family-wide runtime support if only one or two checkpoints were tested.

## 9. Gemma 4 Case Study

Read [references/gemma4-case-study.md](references/gemma4-case-study.md) when the new family has any of these traits:

- mixed sliding and full attention
- different local and global head dimensions
- extra per-layer input embeddings or gates
- multimodal towers that should reuse HF helper methods
- small dense multimodal variants plus larger or MoE follow-ups
