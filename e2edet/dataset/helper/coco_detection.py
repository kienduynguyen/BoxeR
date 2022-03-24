import os
import math
from io import BytesIO

import torch
from torchvision.datasets.vision import VisionDataset
from PIL import Image

from e2edet.utils.distributed import get_rank, get_world_size


class CocoDetection(VisionDataset):
    def __init__(
        self,
        root,
        annFile,
        num_replicas=None,
        rank=None,
        transform=None,
        target_transform=None,
        transforms=None,
        cache_mode=False,
    ):
        super(CocoDetection, self).__init__(
            root, transforms, transform, target_transform
        )
        from pycocotools.coco import COCO

        if num_replicas is None:
            num_replicas = get_world_size()
        if rank is None:
            rank = get_rank()

        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.cache_mode = cache_mode
        self.rank = rank
        self.num_replicas = num_replicas
        self.num_samples = int(math.ceil(len(self.ids) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        if cache_mode:
            self.cache = {}
            self.cache_images()

    def cache_images(self):
        indices = torch.arange(len(self.ids)).tolist()
        indices += indices[: (self.total_size - len(indices))]
        assert len(indices) == self.total_size

        offset = self.num_samples * self.rank
        indices = set(indices[offset : offset + self.num_samples])

        self.cache = {}
        for index, img_id in enumerate(self.ids):
            if index not in indices:
                continue

            path = self.coco.loadImgs(img_id)[0]["file_name"]
            with open(os.path.join(self.root, path), "rb") as f:
                self.cache[path] = f.read()

    def get_image(self, path):
        if self.cache_mode:
            if path not in self.cache.keys():
                print("Not found image in the cache")
                with open(os.path.join(self.root, path), "rb") as f:
                    self.cache[path] = f.read()

            return Image.open(BytesIO(self.cache[path])).convert("RGB")

        return Image.open(os.path.join(self.root, path)).convert("RGB")

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target). target is the object returned ``coco.loadAnns``,
        """
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)

        path = coco.loadImgs(img_id)[0]["file_name"]

        img = self.get_image(path)
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.ids)
