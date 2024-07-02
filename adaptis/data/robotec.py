from pathlib import Path
import cv2
import numpy as np

from .base import BaseDataset


class RobotecDataset(BaseDataset):
    def __init__(self, dataset_path, split='train', **kwargs):
        super().__init__(**kwargs)

        self.dataset_path = Path(dataset_path)
        self.dataset_split = split

        self.dataset_samples = []
        images_path = sorted((self.dataset_path / split).rglob('*rgb.png'))
        for image_path in images_path:
            image_path = str(image_path)
            mask_path = image_path.replace('rgb.png', 'im.png')
            self.dataset_samples.append((image_path, mask_path))

    def get_sample(self, index):
        image_path, mask_path = self.dataset_samples[index]

        sample = {}

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        sample['image'] = image

        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED).astype(np.int32)
        # mask decoding: intensity / 4
        mask = (mask / 4).astype(np.int32)

        if self.with_segmentation:
            # left digit: class id
            segmentation_mask = (mask // 10).astype(np.int32)
            sample['semantic_segmentation'] = segmentation_mask

        # right digit: instance
        instance_mask = mask % 10

        instances_ids = self.get_unique_labels(instance_mask, exclude_zero=True)
        instances_info = {
            x: {'class_id': 1, 'ignore': False}
            for x in instances_ids
        }
        sample.update({
            'instances_mask': instance_mask,
            'instances_info': instances_info,
        })

        return sample
    
    @property
    def stuff_labels(self):
        return [0]

    @property
    def things_labels(self):
        return [1, 2, 3, 4, 5]