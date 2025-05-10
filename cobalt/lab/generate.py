# Copyright (C) 2023-2024 BlueLightAI, Inc. All Rights Reserved.
#
# Any use, distribution, or modification of this software is subject to the
# terms of the BlueLightAI Software License Agreement.

from typing import Dict, List, Optional

import torch

from cobalt.lab import model as m


def observe(model: torch.nn.Module, cacher: Dict, names: List[str]):
    """Sets up forward hooks to observe the model."""
    return m.observe(model, cacher, names)


def generate(
    input_texts: List[str],
    tokenizer,
    device: str,
    model,
    output_embedding: bool = False,
    embedding_layer: Optional[str] = None,
):
    """If output_embedding: return Tuple[List[str], torch.Tensor], else: return List[str]."""
    # TODO: actual docstring
    input_ids = tokenizer(
        input_texts, return_tensors="pt", padding=True, truncation=True
    ).input_ids.to(device)

    hooks = []
    cacher = {}

    if output_embedding:
        hooks = observe(model, cacher, embedding_layer)

    with torch.no_grad():
        output_ids = model.generate(input_ids)

    for h in hooks:
        h.remove()

    if output_embedding:
        return [
            tokenizer.decode(tokens.cpu(), skip_special_tokens=True)
            for tokens in output_ids
        ], cacher[embedding_layer].last_hidden_state.cpu()
    else:
        return [
            tokenizer.decode(tokens.cpu(), skip_special_tokens=True)
            for tokens in output_ids
        ]
