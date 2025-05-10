# Copyright (C) 2023-2024 BlueLightAI, Inc. All Rights Reserved.
#
# Any use, distribution, or modification of this software is subject to the
# terms of the BlueLightAI Software License Agreement.

import pytorch_lightning
import torch

from cobalt.lab.generate import observe


class ClassifierObserver(pytorch_lightning.LightningModule):
    def __init__(self, model, embedding_layers: list[str]):
        super().__init__()
        self.model = model
        self.embedding_layers = embedding_layers

        self.embeddings = {layer: [] for layer in self.embedding_layers}
        self.predictions = []
        self.truths = []

    def validation_step(self, batch, batch_idx):
        x, y = batch
        tmp = {}
        hooks = observe(self.model, tmp, self.embedding_layers)
        with torch.no_grad():
            out = self.model(x)

        self.predictions.append(out)
        self.truths.append(y)
        for hook in hooks:
            hook.remove()

        for layer in self.embedding_layers:
            self.embeddings[layer].append(tmp[layer])

    def extract_post_validation(self):
        return (
            torch.argmax(torch.concat(self.predictions, axis=0), dim=-1),
            torch.concat(self.truths),
            {
                layer: torch.concat(self.embeddings[layer])
                for layer in self.embedding_layers
            },
        )

    def reset_embeddings(self):
        self.embeddings.clear()
        self.embeddings = {layer: [] for layer in self.embedding_layers}

    def on_validation_epoch_start(self) -> None:
        self.reset_embeddings()
        self.predictions = []
        self.truths = []
        return super().on_validation_epoch_start()
