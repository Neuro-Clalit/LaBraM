"""Recording-level multiple-instance learning (MIL) for brain age.

``cache`` -- per-window encoder features, window predictions and descriptors
of a trained fine-tune run (Stage 1 input); ``stage1`` -- MIL pooling layers
trained on those cached features (``docs/b1_attention_mil_plan.md``). The
pooling layers themselves live in ``labram.layers.bag_pool``.
"""
