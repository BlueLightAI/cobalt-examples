# Copyright (C) 2023-2024 BlueLightAI, Inc. All Rights Reserved.
#
# Any use, distribution, or modification of this software is subject to the
# terms of the BlueLightAI Software License Agreement.

import warnings
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer


### TODO: Deprecate this.
def sbert_list_texts(
    texts: List[str], model_name: str = "all-MiniLM-L6-v2", **encoder_kwargs
) -> np.ndarray:
    """Given a list of strings, encodes each using an SBERT transformer LLM.

    The default is recommended by SBERT for speed and accuracy. See alternatives:
    https://www.sbert.net/docs/pretrained_models.html
    These can then be processed into Cobalt since they are "embeddings".
    These give an outside perspective on the data distribution going into a customer's model.
    """
    warnings.warn(
        """Please import"""
        """SentenceTransformerEmbeddingModel from cobalt.embedding_models instead.""",
        DeprecationWarning,
        stacklevel=1,
    )
    model = SentenceTransformer(model_name)
    show_progress_bar = encoder_kwargs.pop("show_progress_bar", None)
    device = encoder_kwargs.pop("device", None)
    embeddings_sbert = model.encode(
        texts, show_progress_bar=show_progress_bar, device=device, **encoder_kwargs
    )
    return embeddings_sbert
