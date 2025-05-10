# Copyright (C) 2023-2024 BlueLightAI, Inc. All Rights Reserved.
#
# Any use, distribution, or modification of this software is subject to the
# terms of the BlueLightAI Software License Agreement.

from __future__ import annotations

import base64
import io
import json
from abc import abstractmethod
from typing import Dict, List, Optional, Union
from uuid import uuid4

import numpy as np
import pytorch_lightning
import pytorch_lightning as pl
import torch
from torch import nn, no_grad, optim
from torch.utils.data import DataLoader, Dataset
from torchmetrics import Accuracy
from torchvision import transforms
from torchvision.datasets import MNIST


def model_to_json(model):
    buffer = io.BytesIO()
    torch.save(model, buffer)
    encoded_data = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return json.dumps({"model": encoded_data})


def model_from_json(serialized_model):
    encoded_data = json.loads(serialized_model)["model"]
    decoded_bytes = base64.b64decode(encoded_data)
    return torch.load(io.BytesIO(decoded_bytes))


def observe(model: torch.nn.Module, cacher: Dict, names: List[str]):
    child_hooks = []

    def forward_hook(name):
        def hook(module, inp, output):
            cacher[name] = output

        return hook

    for name, child in model.named_children():
        if name in names:
            hook = child.register_forward_hook(forward_hook(name))
            child_hooks.append(hook)

    return child_hooks


class MNISTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.uuid = uuid4()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def to_json(self):
        return model_to_json(self)

    @classmethod
    def from_json(cls, serialized_string):
        return model_from_json(serialized_string)

    def __eq__(self, other):
        if not isinstance(other, MNISTModel):
            return NotImplemented
        for p_self, p_other in zip(self.parameters(), other.parameters()):
            if not torch.equal(p_self, p_other):
                return False
        return True

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self) -> int:
        return hash(self.uuid)


class MNISTModule(pl.LightningModule):
    def __init__(self, lr=1e-3):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.accuracy = Accuracy(task="multiclass", num_classes=10)
        self.lr = lr

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.cross_entropy(y_hat, y)
        self.log("train_loss", loss)
        self.log("train_acc", self.accuracy(y_hat, y))
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.cross_entropy(y_hat, y)
        self.log("val_loss", loss)
        self.log("val_acc", self.accuracy(y_hat, y))
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)

    def to_json(self):
        return model_to_json(self)

    @classmethod
    def from_json(cls, serialized_string):
        return model_from_json(serialized_string)


class LightningWrapper(pytorch_lightning.LightningModule):
    def __init__(self, model: nn.Module, lr: float, loss_function=None):
        super().__init__()
        self.model = model
        self.lr = lr
        self.loss_function = loss_function

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_function(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_function(y_hat, y)
        self.log("validation_loss", loss)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)

    @classmethod
    def from_json(cls, serialized_model: str) -> LightningWrapper:
        data = json.loads(serialized_model)
        model = model_from_json(data)
        return cls(model=model, lr=data["lr"])

    def to_json(self):
        data = {
            "model": json.loads(model_to_json(self.model))["model"],
            "lr": str(self.lr),
        }
        return json.dumps(data)


class Model:
    def __init__(self, base_model, **kwargs) -> None:
        self.base_model = base_model

    @abstractmethod
    def embed(self, inputs, **kwargs):
        pass

    @abstractmethod
    def predict(self, inputs, **kwargs):
        pass

    @abstractmethod
    def fit(self, x, y, **kwargs):
        pass

    def to_json(self):
        return model_to_json(self)

    @classmethod
    def from_json(cls, serialized_data) -> Model:
        return cls(base_model=model_from_json(serialized_data))


class ScikitModel(Model):
    def embed(self, inputs):
        return inputs

    def predict(self, inputs):
        return self.base_model.predict(inputs)

    def fit(self, dataset, **kwargs):
        self.base_model.fit(dataset.X, dataset.y)


class TorchClassificationModel(Model):
    def __init__(
        self,
        base_model: nn.Module,
        loss_function=nn.functional.cross_entropy,
        lr: Optional[float] = None,
    ) -> None:
        self.observable_model = base_model
        if not isinstance(base_model, pytorch_lightning.LightningModule):
            self.base_model = LightningWrapper(base_model, lr, loss_function)

        else:
            self.base_model = base_model

    def embed(
        self,
        inputs: Union[DataLoader, Dataset],
        embedding_keys: Union[List[str], str],
        batch_size=64,
    ) -> Union[Dict[str, np.ndarray], np.ndarray]:
        """Returns embeddings of the given inputs."""
        self.base_model.eval()

        if isinstance(embedding_keys, str):
            embedding_keys = [embedding_keys]

        d = {key: [] for key in embedding_keys}

        if isinstance(inputs, Dataset):
            loader = DataLoader(inputs, batch_size, shuffle=False)
        else:
            loader = inputs

        for x, _ in loader:
            cacher = {}

            hooks = observe(self.observable_model, cacher, embedding_keys)
            with no_grad():
                _ = self.observable_model(x)

            for h in hooks:
                h.remove()

            for key in embedding_keys:
                d[key].append(cacher[key].numpy())

        for key in embedding_keys:
            d[key] = np.concatenate(d[key])

        if len(d) == 1:
            return d[embedding_keys[0]]

        return d

    def predict(self, inputs: Union[DataLoader, Dataset], **kwargs) -> np.ndarray:
        """Returns model predictions given inputs."""
        self.base_model.eval()

        if isinstance(inputs, Dataset):
            batch_size = kwargs.get("batch_size", 64)
            loader = DataLoader(inputs, batch_size, shuffle=False)
        else:
            loader = inputs

        outs = []
        with no_grad():
            for x, _ in loader:
                outs.append(self.base_model(x).numpy())
        return np.concatenate(outs)

    def fit(self, train_loader: DataLoader, fast_dev=False, **kwargs) -> None:
        """Fits model given training loader."""
        self.base_model.train(True)
        n_epochs = kwargs["n_epochs"]
        trainer = pytorch_lightning.Trainer(
            max_epochs=n_epochs,
            fast_dev_run=fast_dev,
            enable_model_summary=False,
            enable_progress_bar=False,
        )

        trainer.fit(self.base_model, train_loader)

    def embedding_layer_options(self) -> List[str]:
        """Returns possible ways of embedding the data."""
        return [name for name, _ in self.observable_model.named_children()]

    @classmethod
    def from_json(cls, serialized_model: str) -> TorchClassificationModel:
        base_model = model_from_json(serialized_model)
        lr = json.loads(serialized_model).get("lr")
        return cls(base_model=base_model, lr=lr)

    def to_json(self):
        return model_to_json(self)


def main():
    ds = MNIST(".", train=False, download=True, transform=transforms.ToTensor())
    module = MNISTModel()

    tm = TorchClassificationModel(module, lr=1e-3)
    loader = DataLoader(ds, 64, shuffle=True)
    tm.fit(loader, fast_dev=True, n_epochs=1)

    loader = DataLoader(ds, 64, shuffle=False)

    outs_ds = tm.predict(ds)
    outs = tm.predict(loader)
    embs = tm.embed(loader, "fc1")
    embs_ds = tm.embed(ds, "fc1")

    assert np.allclose(outs, outs_ds)
    assert np.allclose(embs, embs_ds)

    assert embs.shape[0] == 10000
    assert outs.shape[0] == 10000

    print("Success.")


if __name__ == "__main__":
    main()
