# Copyright (C) 2023-2024 BlueLightAI, Inc. All Rights Reserved.
#
# Any use, distribution, or modification of this software is subject to the
# terms of the BlueLightAI Software License Agreement.

from typing import ClassVar, Dict
from uuid import uuid4


class EventBus:
    def __init__(self):
        self.callbacks = {}

    def add_callback(self, callback):
        """Add a callback (callable object) by name."""
        if callable(callback):
            name = getattr(callback, "__name__", "") + str(uuid4())
            self.callbacks[name] = callback
            return name
        else:
            raise ValueError("Callback must be callable")

    def remove_callback(self, name):
        """Remove a callback by name."""
        if name in self.callbacks:
            del self.callbacks[name]

    def run_all(self):
        """Execute all callbacks."""
        for callback in self.callbacks.values():
            callback()


class GraphEventBus(EventBus):
    pass


class GroupEventBus(EventBus):
    pass


class RunEventBus(EventBus):
    pass


class DatasetEventBus(EventBus):
    pass


class EventBusController:
    groups: ClassVar[Dict[str, GroupEventBus]] = {}
    graphs: ClassVar[Dict[str, GraphEventBus]] = {}
    runs: ClassVar[Dict[str, RunEventBus]] = {}
    datasets: ClassVar[Dict[str, DatasetEventBus]] = {}

    @classmethod
    def get_graph_event_bus(cls, workspace_id) -> GraphEventBus:
        if workspace_id not in cls.graphs:
            cls.graphs[workspace_id] = GraphEventBus()
        return cls.graphs[workspace_id]

    @classmethod
    def get_group_event_bus(cls, workspace_id) -> GroupEventBus:
        if workspace_id not in cls.groups:
            cls.groups[workspace_id] = GroupEventBus()
        return cls.groups[workspace_id]

    @classmethod
    def get_run_event_bus(cls, workspace_id) -> RunEventBus:
        if workspace_id not in cls.runs:
            cls.runs[workspace_id] = RunEventBus()
        return cls.runs[workspace_id]

    @classmethod
    def get_dataset_event_bus(cls, workspace_id) -> DatasetEventBus:
        if workspace_id not in cls.datasets:
            cls.datasets[workspace_id] = DatasetEventBus()
        return cls.datasets[workspace_id]
