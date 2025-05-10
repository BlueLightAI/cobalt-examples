# Copyright (C) 2023-2024 BlueLightAI, Inc. All Rights Reserved.
#
# Any use, distribution, or modification of this software is subject to the
# terms of the BlueLightAI Software License Agreement.

from typing import Callable, List, Union

import ClassifierObserver
import numpy as np
import pytorch_lightning
import torch
import torch.utils.data

import cobalt
from cobalt.schema import CobaltDataSubset


class Rerouter:
    def __init__(
        self,
        model: torch.nn.Module,
        error_classifiers: List[cobalt.schema.Classifier],
        patch_outputs: List[Union[Callable, str, int]],
        embedding_layer: str,
    ):
        if len(error_classifiers) != len(patch_outputs):
            raise Exception(
                f"len(error_classifiers) ({len(error_classifiers)})\
                            != len(patch_outputs) {len(patch_outputs)}"
            )
        self.model = model
        self.error_classifiers = error_classifiers
        self.patch_outputs = patch_outputs
        self.embedding_layer = embedding_layer
        self.pl_module = ClassifierObserver(self.model, [embedding_layer])
        self.trainer = pytorch_lightning.Trainer(
            enable_model_summary=False, enable_progress_bar=False
        )

    def predict(
        self,
        dataset: torch.utils.data.Dataset,
        batch_size: int = 256,
        verbose=False,
    ):
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False
        )
        self.pl_module = ClassifierObserver(self.model, [self.embedding_layer])
        self.trainer.validate(self.pl_module, loader, verbose=verbose)
        preds, truth, embs = self.pl_module.extract_post_validation()
        old_preds = preds.clone()

        emb = embs[self.embedding_layer].cpu().numpy()
        new_preds = preds

        for classifier, patch in zip(self.error_classifiers, self.patch_outputs):
            is_patched = classifier.apply(emb.reshape(len(dataset), -1))
            if isinstance(patch, (str, int)):
                new_preds[is_patched] = patch
            else:
                new_preds[is_patched] = patch(dataset.subset(is_patched))

        return new_preds, old_preds, truth


def patch_model_on_subsets(
    model,
    failure_modes: List[CobaltDataSubset],
    full_dataset: CobaltDataSubset,
    expert_failure_handlers: List[str],
):
    classifiers = [
        subset.get_classifier(
            "knn",
            embedding_index=0,
            global_set=full_dataset,
            params={"k": 7, "threshold": 0.5},
        )
        for subset in failure_modes
    ]
    for c in classifiers:
        c.fit()

    embedding_layer = failure_modes[0].embedding_metadata[0].name

    rerouter = Rerouter(model, classifiers, expert_failure_handlers, embedding_layer)
    return rerouter


def evaluate_patched_model(new_preds, old_preds, truth, relevant_indices=None):
    if relevant_indices is not None:
        new_preds = new_preds[relevant_indices]
        old_preds = old_preds[relevant_indices]
        truth = truth[relevant_indices]

    new_misclassified = (new_preds != truth).cpu().numpy()
    old_misclassified = (old_preds != truth).cpu().numpy()
    new_acc = 1 - np.mean(new_misclassified)
    old_acc = 1 - np.mean(old_misclassified)
    return new_acc, old_acc, new_misclassified, old_misclassified
