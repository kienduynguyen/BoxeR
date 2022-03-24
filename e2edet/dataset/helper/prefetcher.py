# ------------------------------------------------------------------------
# BoxeR
# Copyright (c) 2022. All Rights Reserved.
# Licensed under the MIT License [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from Deformable-DETR (https://github.com/fundamentalvision/Deformable-DETR)
# ------------------------------------------------------------------------
import torch


class Prefetcher:
    def __init__(self, loader, dataset, prefetch=True):
        self.loader = iter(loader)
        self.prefetch = prefetch
        self.dataset = dataset
        if prefetch:
            self.stream = torch.cuda.Stream()
            self.preload()

    def preload(self):
        try:
            self.next_batch = next(self.loader)
        except StopIteration:
            self.next_batch = None
            return

        with torch.cuda.stream(self.stream):
            self.next_batch = self.dataset.prepare_batch(
                self.next_batch, non_blocking=True
            )

    def _record_batch(self, batch):
        samples, targets = batch
        if samples is not None:
            for k, v in samples.items():
                if isinstance(v, torch.Tensor):
                    v.record_stream(torch.cuda.current_stream())
        if targets is not None:
            for t in targets:
                for k, v in t.items():
                    if isinstance(v, torch.Tensor):
                        v.record_stream(torch.cuda.current_stream())

    def get_next_sample(self):
        if self.prefetch:
            torch.cuda.current_stream().wait_stream(self.stream)
            batch = self.next_batch

            if self.dataset.iter_per_update > 1:
                for split in batch:
                    self._record_batch(split)
            else:
                self._record_batch(batch)
            self.preload()
        else:
            try:
                batch = next(self.loader)
                batch = self.dataset.prepare_batch(batch)
            except StopIteration:
                batch = None

        return batch
