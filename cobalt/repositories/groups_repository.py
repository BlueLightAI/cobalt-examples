# Copyright (C) 2023-2024 BlueLightAI, Inc. All Rights Reserved.
#
# Any use, distribution, or modification of this software is subject to the
# terms of the BlueLightAI Software License Agreement.

from abc import ABC, abstractmethod


class AbstractGroupRepository(ABC):
    @abstractmethod
    def __init__(self) -> None:
        """Initialize the repository."""
        raise NotImplementedError

    @abstractmethod
    def add_group(self, group, group_name: str):
        """Add a group with the given name."""
        raise NotImplementedError

    @abstractmethod
    def get_groups_info(self):
        """Return information about all groups."""
        raise NotImplementedError

    @abstractmethod
    def get_group(self, group_name):
        """Return a group by name, or all groups if no name is provided."""
        raise NotImplementedError

    @abstractmethod
    def get_groups(self):
        """Return a group by name, or all groups if no name is provided."""
        raise NotImplementedError

    @abstractmethod
    def is_group_name_unique(self, group_name: str):
        """Check if a group name is unique."""
        raise NotImplementedError

    @abstractmethod
    def delete_group(self, group_name):
        """Delete a group by name."""
        raise NotImplementedError

    @abstractmethod
    def rename_group(self, old_name: str, new_name: str):
        """Rename group without changing order."""
        raise NotImplementedError


class GroupRepository(AbstractGroupRepository):
    def __init__(self) -> None:
        self.saved_groups = {}
        self.groups_info = {}

    def add_group(self, group, group_name: str):
        self.groups_info[group_name] = len(group)
        self.saved_groups[group_name] = group

    def add_groups(self, groups_data):
        for group_data in groups_data:
            self.add_group(group_data["group"], group_data["group_name"])

    def get_groups_info(self):
        return self.groups_info

    def get_group(self, group_name: str):
        return self.saved_groups[group_name]

    def get_groups(self):
        return self.saved_groups

    def is_group_name_unique(self, group_name: str):
        return group_name not in self.saved_groups

    def delete_group(self, group_name):
        if group_name in self.saved_groups:
            del self.saved_groups[group_name]
        if group_name in self.groups_info:
            del self.groups_info[group_name]

    def rename_group(self, old_name: str, new_name: str):
        if not old_name:
            # ToDo: add message about. ValueError ?
            return
        if not new_name:
            # ToDo: add message about. ValueError ?
            return
        if old_name == new_name:
            return

        if old_name not in self.saved_groups:
            return

        self.saved_groups = {
            new_name if k == old_name else k: v for k, v in self.saved_groups.items()
        }
        self.groups_info = {
            new_name if k == old_name else k: v for k, v in self.groups_info.items()
        }
