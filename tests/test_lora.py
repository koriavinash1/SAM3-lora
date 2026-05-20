from __future__ import annotations

from sam3_lora.lora import choose_trainable_parameter_names



def test_choose_trainable_respects_fraction_and_always_patterns():
    names = [("encoder.weight", 900), ("decoder.weight", 100), ("lora_A", 5)]
    selected = choose_trainable_parameter_names(
        named_parameter_sizes=names,
        trainable_fraction=0.1,
        always_trainable_patterns=[r"lora_"],
    )

    assert "lora_A" in selected
    assert "encoder.weight" in selected
