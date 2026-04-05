# Gemma 4 Case Study

Use this note when adapting another family with mixed attention, multimodal towers, or nonstandard text blocks.

## What Mattered In Gemma 4

Gemma 4 was not just "another GPT with vision".

The main deltas were:

- mixed `sliding_attention` and `full_attention` layers
- local `head_dim` and global `global_head_dim`
- small dense models with `per_layer_input` structures
- multimodal image and audio towers
- processor behavior that required explicit text in modal-only smoke calls

## Parser Lessons

The parser needed to preserve more than the usual GPT fields.

Critical fields were:

- `layer_types`
- `global_kv_channels`
- `num_global_query_groups`
- `hidden_size_per_layer_input`
- `vocab_size_per_layer_input`
- `num_kv_shared_layers`
- `use_double_wide_mlp`
- `enable_moe_block`
- `top_k_experts`

Without those, the build would appear valid but fail at weight load or forward time.

## Text Backbone Lessons

The original failure was a shape mismatch in the first `full_attention` layer during `load_weights`.

The fix was not a special-case remap in the bridge. The fix was to build the correct layer shape in the model:

- sliding layers used local `kv_channels`
- full-attention layers used `global_kv_channels`
- some small-model layers needed wider MLP behavior
- small models also needed extra per-layer input modules

That forced a custom path:

- family-specific self-attention
- family-specific MLP
- family-specific transformer layer
- family-specific GPT model
- family-specific loader

## Multimodal Lessons

The multimodal wrapper could stay thin.

The right strategy was:

- instantiate HF `vision_tower` and `audio_tower`
- instantiate HF projection modules
- reuse HF feature helper methods
- only own placeholder resolution and masked scatter logic locally

One important bug came from binding helper methods to the wrong HF class. In Gemma 4, placeholder and modal helper methods lived on the base model class rather than the top-level conditional generation class.

## Processor Lessons

The first multimodal failure was not in the bridge. It was in processor usage.

For Gemma 4:

- image-only smoke needed explicit text
- audio-only smoke also needed explicit text

So modal smoke scripts should test the processor contract early before assuming the embedding merge path is wrong.

## Validation Sequence That Worked

The order below avoided wasted time.

1. parse HF config
2. build single-GPU model
3. inspect sliding-layer and full-layer QKV shapes
4. run 2-GPU `load_weights`
5. run vision embedding injection smoke
6. run audio embedding injection smoke
7. run text-only train smoke
8. run vision train smoke

## Scope Lessons For PRs

Gemma 4 support was real for:

- `E2B-it` runtime validation
- `E4B-it` structure validation

But not yet full runtime validation for:

- `31B-it`
- `26B-A4B-it`

So the correct PR framing was "initial support", not "full family support".
