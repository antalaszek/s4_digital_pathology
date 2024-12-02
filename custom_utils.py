from torch.utils.data import DataLoader, WeightedRandomSampler

import random
import re
from typing import Callable, List, Optional
import torch
from torch.utils.data import Dataset
from pathlib import Path
from collections import Counter
from pprint import pprint
import abc
import typing as t
from sklearn.model_selection import StratifiedGroupKFold
from torch import Value


class WSIEmbeddingDataset(Dataset):
    def __init__(self, data, root_dir, label_to_int, transform=None, pass_groups=False):
        self.label_to_int = label_to_int
        self.data = data
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.pass_groups = pass_groups

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        label, groups, augment, feature_file = self.data[idx]
        int_label = self.label_to_int[label]
        if self.transform:
            augment = self.transform(augment)
        feature_path = self.root_dir / label / groups / augment / feature_file
        feature_vector = torch.load(feature_path)
        if self.pass_groups:
            return (feature_vector, int_label, feature_file, groups)
        else:
            return (feature_vector, int_label, feature_file)

    @staticmethod
    def load_data(root_dir, pattern=None, debug=False):
        if pattern is not None:
            pattern = re.compile(pattern)
        root_dir = Path(root_dir)
        data = sorted(
            list(
                {
                    (
                        label_dir.name,
                        "/".join([group1_dir.name, group2_dir.name, group3_dir.name]),
                        "original",
                        feature_file.name,
                    )
                    for label_dir in sorted(root_dir.iterdir())
                    if label_dir.is_dir()
                    for group1_dir in sorted(label_dir.iterdir())
                    if group1_dir.is_dir()
                    for group2_dir in sorted(group1_dir.iterdir())
                    if group2_dir.is_dir()
                    for group3_dir in sorted(group2_dir.iterdir())
                    if group3_dir.is_dir()
                    for augment_dir in sorted(group3_dir.iterdir())
                    if augment_dir.is_dir()
                    for feature_file in sorted(augment_dir.glob("*.pt"))
                    if not pattern
                    or any(
                        (
                            pattern.match(group)
                            for group in (
                                group1_dir.name,
                                group2_dir.name,
                                group3_dir.name,
                            )
                        )
                    )
                }
            )
        )
        if debug:
            pprint(f"{data}")
        return data

    def compute_group_counts(self, group_lvl=0):
        return Counter((groups.split("/")[group_lvl] for _, groups, *_ in self.data))

    def compute_label_counts(self):
        return Counter((label for label, *_ in self.data))

    def save(self, dir, name):
        import json

        out_dir = Path(dir) / name
        out_dir.parent.mkdir(parents=True, exist_ok=True)
        with open(out_dir, "w") as f:
            json.dump(self.data, f)


class RandomAugmentSelector:
    def __init__(self, augmentations, original_probablilty=0.5):
        self.augmentations = augmentations
        self.original_probablilty = original_probablilty

    def __call__(self, original_augment):
        if not self.augmentations or (random.random() < self.original_probablilty):
            return original_augment
        else:
            return random.choice(self.augmentations)




class SplitStrategy(abc.ABC):
    @abc.abstractmethod
    def split(
        self,
        data: t.List[t.Tuple[str, str, str, str]],
    ) -> t.List[t.Tuple[t.List[int], t.List[int], t.List[int]]]:
        """
        data: label, group, augm, pt
        """
        raise NotImplementedError("this should be overriden")


class RandomStratifiedGroupKFoldTrainVal(SplitStrategy):
    def __init__(self, num_splits, group_part=-1, random_state=None) -> None:
        self.num_splits = num_splits
        self.group_part = group_part
        self.random_state = random_state

    def split(
        self,
        data: t.List[t.Tuple[str, str, str, str]],
    ) -> t.List[t.Tuple[t.List[int], t.List[int], t.List[int]]]:
        labels = [label for label, _, _, _ in data]
        patient_ids = [groups.split("/")[self.group_part] for _, groups, _, _ in data]

        skf = StratifiedGroupKFold(
            n_splits=self.num_splits, shuffle=True, random_state=self.random_state
        )
        splits = skf.split(data, labels, groups=patient_ids)

        return [(a, b, []) for a, b in splits]


class RandomStratifiedGroupKFoldTrainValTest(SplitStrategy):
    def __init__(
        self,
        primary_num_splits,
        secondary_num_splits,
        group_part=-1,
        random_state=None,
    ) -> None:
        self.primary_num_splits = primary_num_splits
        self.secondary_num_splits = secondary_num_splits
        self.group_part = group_part
        self.random_state = random_state

    def split(
        self,
        data: t.List[t.Tuple[str, str, str, str]],
    ) -> t.List[t.Tuple[t.List[int], t.List[int], t.List[int]]]:
        labels = [label for label, _, _, _ in data]
        patient_ids = [groups.split("/")[self.group_part] for _, groups, _, _ in data]

        skf_primary = StratifiedGroupKFold(
            n_splits=self.primary_num_splits,
            shuffle=True,
            random_state=self.random_state,
        )
        primary_splits = skf_primary.split(data, labels, groups=patient_ids)
        train_val_ids, test_ids = next(primary_splits)
        train_val_labels = [labels[i] for i in train_val_ids]
        train_val_patient_ids = [patient_ids[i] for i in train_val_ids]

        skf_secondary = StratifiedGroupKFold(
            n_splits=self.secondary_num_splits,
            shuffle=True,
            random_state=self.random_state,
        )

        secondary_splits = skf_secondary.split(
            train_val_ids,
            train_val_labels,
            groups=train_val_patient_ids,
        )

        return [
            (
                [train_val_ids[i] for i in train_ids_ids],
                [train_val_ids[i] for i in val_ids_ids],
                test_ids,
            )
            for train_ids_ids, val_ids_ids in secondary_splits
        ]


class RegexPatternStratifiedGroupKFoldTrainValTest(SplitStrategy):
    def __init__(
        self,
        num_splits,
        test_group_pattern,
        group_part=-1,
        random_state=None,
    ) -> None:
        self.num_splits = num_splits
        self.test_group_pattern = re.compile(test_group_pattern)
        print(f"test group pattern: {self.test_group_pattern}")
        self.group_part = group_part
        self.random_state = random_state

    def split(
        self,
        data: t.List[t.Tuple[str, str, str, str]],
    ) -> t.List[t.Tuple[t.List[int], t.List[int], t.List[int]]]:
        labels = [label for label, _, _, _ in data]
        patient_ids = [groups.split("/")[self.group_part] for _, groups, _, _ in data]
        test_group_parts = [groups.split("/") for _, groups, _, _ in data]
        test_ids = [
            idx
            for idx, group_parts in enumerate(test_group_parts)
            if any(
                (
                    self.test_group_pattern.search(group_part)
                    for group_part in group_parts
                )
            )
        ]
        train_val_indices = [idx for idx in range(len(data)) if idx not in test_ids]

        if len(train_val_indices) < len(test_ids):
            raise ValueError(
                "Something must have gone wrong, since train_val set is smaller than testing set"
            )

        print(f"{len(train_val_indices)=}, {len(test_ids)=}")

        skf = StratifiedGroupKFold(
            n_splits=self.num_splits, shuffle=True, random_state=self.random_state
        )
        splits = skf.split(
            train_val_indices,
            [labels[i] for i in train_val_indices],
            groups=[patient_ids[i] for i in train_val_indices],
        )

        return [
            (
                [train_val_indices[i] for i in train_ids_ids],
                [train_val_indices[i] for i in val_ids_ids],
                test_ids,
            )
            for train_ids_ids, val_ids_ids in splits
        ]


def data_summary(data, dataset_name):
    return f"{dataset_name} dataset: len={len(data)}, groups={set(['/'.join(group.split('/')[:-1]) for _,group, *_ in data])}"


class WSIDataModule:
    def __init__(
        self,
        data_dir,
        augmentations,
        fold_id,
        split_strategy: SplitStrategy,
        batch_size=1,
        num_workers=5,
        weighted_random_sampler=False,
        train_num_samples_multiplier=1,
        save_path=None,
        weight_group_lvl=None,
        data_subset_regex=None,
    ):
        # super().__init__()
        self.split_strategy = split_strategy
        self.root_dir = data_dir
        self.augmentations = augmentations
        self.fold_id = fold_id
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = RandomAugmentSelector(augmentations, original_probablilty=0.5)
        self.weighted_random_sampler = weighted_random_sampler
        self.train_num_samples_multiplier = train_num_samples_multiplier
        self.save_path = save_path
        self.weight_group_lvl = weight_group_lvl
        self.data_subset_regex = data_subset_regex

    def setup(self, stage=None):
        data = WSIEmbeddingDataset.load_data(self.root_dir, self.data_subset_regex)
        splits = self.split_strategy.split(data)
        self.label_to_int = {
            label: idx for idx, label in enumerate(sorted(set([d[0] for d in data])))
        }
        self.int_to_label = {idx: label for label, idx in self.label_to_int.items()}
        print(self.int_to_label)

        train_indices, val_indices, test_indices = splits[self.fold_id]
        self.train_data = [data[i] for i in train_indices]
        self.val_data = [data[i] for i in val_indices]
        self.test_data = [data[i] for i in test_indices]

        self.train_dataset = WSIEmbeddingDataset(
            self.train_data,
            root_dir=self.root_dir,
            transform=self.transform,
            label_to_int=self.label_to_int,
        )
        self.val_dataset = WSIEmbeddingDataset(
            self.val_data,
            root_dir=self.root_dir,
            label_to_int=self.label_to_int,
            pass_groups=True,
        )
        self.test_dataset = (
            WSIEmbeddingDataset(
                self.test_data,
                root_dir=self.root_dir,
                label_to_int=self.label_to_int,
                pass_groups=True,
            )
            if self.test_data
            else None
        )
        if self.save_path:
            self.train_dataset.save(self.save_path, "train.csv")
            self.val_dataset.save(self.save_path, "val.csv")
            _ = self.test_dataset and self.test_dataset.save(self.save_path, "test.csv")

    def train_dataloader(self):
        if self.weighted_random_sampler:
            label_counts = self.train_dataset.compute_label_counts()
            if self.weight_group_lvl is not None:
                group_counts = self.train_dataset.compute_group_counts(
                    group_lvl=self.weight_group_lvl
                )
                weights = [
                    1
                    / (
                        label_counts[label]
                        * group_counts[groups.split("/")[self.weight_group_lvl]]
                    )
                    for label, groups, *_ in self.train_data
                ]
            else:
                weights = [1 / (label_counts[label]) for label, *_ in self.train_data]

            sampler = WeightedRandomSampler(
                weights=weights,
                num_samples=int(
                    len(self.train_dataset) * self.train_num_samples_multiplier
                ),
                replacement=True,
            )
            ret = DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                sampler=sampler,
            )
        else:
            ret = DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=True,
            )

        return ret

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        if not self.test_dataset:
            raise ValueError("Testing not properly configured)")
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )
