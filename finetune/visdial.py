from __future__ import annotations
import sys
import random

import os
from typing import Any, Callable, Optional, Tuple

from PIL import Image

from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import download_and_extract_archive, verify_str_arg
import torch

from time import time
import io
import numpy as np
import json

class VisDial(VisionDataset):
    def __init__(
        self,
        root: str,
        split: str = "train",
        dialog_len: int = -1,
        transform: Optional[Callable] = None,
    ) -> None:
        super(VisDial, self).__init__(root, transform=transform)
        assert dialog_len <= 10 and dialog_len >= -1 #-1: random, 0: caption, 1: caption + QA1, ...
        self.dialog_len = dialog_len
        self._split = verify_str_arg(split, "split", ("train", "val"))

        self._data_path = os.path.join(self.root)

        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")

        with open(os.path.join(self._data_path, self._split, f'visdial_1.0_{self._split}.json'), "r") as f:
            self._data = json.load(f)

        self._questions = self._data['data']['questions']
        self._answers = self._data['data']['answers']
        self.transform = transform

        self.image_id_to_file_name = self._image_id_to_file_name()

    def _image_id_to_file_name(self):
        image_id_to_file_name = {}
        if self._split == 'val':
            prefix = 'VisualDialog_val2018_'
        else:
            prefix = 'COCO_train2014_'
        for dialog in self._data['data']['dialogs']:
            image_id = dialog['image_id']
            file_name = prefix + "%012d.jpg"%image_id
            image_id_to_file_name[image_id] = file_name
        return image_id_to_file_name
        
    def __len__(self) -> int:
        return len(self._data['data']['dialogs'])

    def __getitem__(self, idx) -> Tuple[Any, Any]:
        data = self._data['data']['dialogs'][idx]
        file_name = self.image_id_to_file_name[data['image_id']]
        image_file = os.path.join(self._data_path, self._split, 'images', file_name)
        image = Image.open(image_file)
        if self.dialog_len == -1:
            dialog_len = random.randint(0, 10)
        else:
            dialog_len = self.dialog_len
        if self._split == "train" and dialog_len > 0 and random.random() < 0.2:
            caption = []
        else:
            caption = [data['caption']]
        text = caption + [self._questions[data['dialog'][k]['question']] + '? ' + self._answers[data['dialog'][k]['answer']] for k in range(dialog_len)]
        text = ', '.join(text)

        image = self.transform(image)['pixel_values'][0]

        return image, text

    def _check_exists(self) -> bool:
        return os.path.exists(self._data_path) and os.path.isdir(self._data_path)
