import json
import os
import random
import sys

import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
from segmentation_models_pytorch.utils.meter import AverageValueMeter
from segmentation_models_pytorch.utils.train import TrainEpoch, ValidEpoch
from skimage.draw import polygon
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
from tqdm import tqdm

from myseg.utils import create_slices_from_label, resize_batch_image


class Dataset(BaseDataset):
    """Read images, apply augmentation and preprocessing transformations.

    Args:
        augmentation (albumentations.Compose): data transfromation pipeline
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing
            (e.g. noralization, shape manipulation, etc.)

    """

    def __init__(
        self,
        select_id_set: set,
        base_path="SEG_Train_Datasets",
        augmentation=None,
        preprocessing=None,
    ):
        self.image_dir_path = os.path.join(base_path, "Train_Images")
        self.label_dir_path = os.path.join(base_path, "Annotations_Arr")
        all_images = set(os.listdir(self.image_dir_path))
        all_labels = set(os.listdir(self.label_dir_path))
        self.images_fps = []
        self.label_fps = []
        for id_ in select_id_set:
            label_fname = ".".join([id_, "npy"])
            image_fname = ".".join([id_, "jpg"])
            assert label_fname in all_labels
            assert image_fname in all_images
            self.images_fps.append(os.path.join(self.image_dir_path, image_fname))
            self.label_fps.append(os.path.join(self.label_dir_path, label_fname))

        self.augmentation = augmentation
        self.preprocessing = preprocessing
        print(f"got {len(self.images_fps)} samples")

    def __getitem__(self, i):

        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # mask = cv2.imread(self.masks_fps[i], 0)
        label = np.load(self.label_fps[i])
        mask = np.expand_dims(label, -1)
        # extract certain classes from mask (e.g. cars)
        # masks = [(mask == v) for v in self.class_values]
        # mask = np.stack(masks, axis=-1).astype('float')

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample["image"], sample["mask"]

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample["image"], sample["mask"]

        return image, mask

    def __len__(self):
        return len(self.images_fps)


class SliceDataset(Dataset):
    def __init__(
        self,
        select_id_set: set,
        base_path="SEG_Train_Datasets",
        augmentation=None,
        preprocessing=None,
        grid_num=(6, 6),
        min_h=384,
        min_w=384,
        pad=20,
    ):
        super().__init__(select_id_set, base_path, augmentation, preprocessing)
        self.grid_num = grid_num
        self.min_h = min_h
        self.min_w = min_w
        self.pad = pad
        self._get_slices()

    def _get_slices(self):
        image_slices = []
        for i in range(len(self.label_fps)):
            label = np.load(self.label_fps[i])
            cur = create_slices_from_label(label, self.grid_num, self.min_h, self.min_w, self.pad)
            image_slices.append(cur)
        self.image_slices = image_slices

    def __getitem__(self, i):

        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # mask = cv2.imread(self.masks_fps[i], 0)
        label = np.load(self.label_fps[i])
        slices = self.image_slices[i]
        if slices:
            r_slice, c_slice = random.choice(slices)
            image = image[r_slice, c_slice, :]
            label = label[r_slice, c_slice]

        mask = np.expand_dims(label, -1)
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample["image"], sample["mask"]

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample["image"], sample["mask"]

        return image, mask

    def __len__(self):
        return len(self.images_fps)


class ValidationDataset(Dataset):
    """validation 時應該在原始尺寸的圖片上做驗證，所以只對 X 做尺寸調整， mask 應保持一致
    並在模型輸出 predict 時， resize 到原始尺寸與 mask 算 loss

    """

    def __getitem__(self, i):
        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = np.load(self.label_fps[i])
        mask = np.expand_dims(label, -1)

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, _ = sample["image"], sample["mask"]

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample["image"], sample["mask"]

        return image, mask


class SamTrainEpoch(TrainEpoch):
    def batch_update(self, x, y):

        prediction = self.model.forward(x)
        loss = self.loss(prediction, y)
        loss.backward()
        self.optimizer.first_step(zero_grad=True)

        # second step
        self.loss(self.model.forward(x), y).backward()  # make sure to do a full forward pass
        self.optimizer.second_step(zero_grad=True)

        return loss, prediction

        # first forward-backward pass


class CustomValidEpoch(ValidEpoch):
    """
        在小尺寸的 X 上 predict, 接著 resize 到原始尺寸與 mask 算 loss

    resize_interpolation:
        * cv2.INTER_NEAREST:   0
        * cv2.INTER_LINEAR:    1
        * cv2.INTER_CUBIC:     2
    """

    def __init__(
        self,
        model,
        loss,
        metrics,
        device="cpu",
        verbose=True,
        resize_interpolation=0,
        pad_r=None,
        pad_l=None,
        pad_u=None,
        pad_d=None,
    ):
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            device=device,
            verbose=verbose,
        )
        self.resize_interpolation = resize_interpolation
        self.pad_r = pad_r
        self.pad_l = pad_l
        self.pad_u = pad_u
        self.pad_d = pad_d

    def batch_update(self, x, y):
        _, _, height, width = y.shape
        with torch.no_grad():
            prediction = self.model.forward(x)
            if y.device.type != "cpu":
                y = y.to("cpu")
            prediction = prediction.to("cpu")
            prediction = resize_batch_image(
                prediction,
                height,
                width,
                self.resize_interpolation,
                pad_r=self.pad_r,
                pad_l=self.pad_l,
                pad_u=self.pad_u,
                pad_d=self.pad_d,
            )

            loss = self.loss(prediction, y)
        return loss, prediction

    def run(self, dataloader):

        self.on_epoch_start()

        logs = {}
        loss_meter = AverageValueMeter()
        metrics_meters = {metric.__name__: AverageValueMeter() for metric in self.metrics}

        with tqdm(
            dataloader,
            desc=self.stage_name,
            file=sys.stdout,
            disable=not (self.verbose),
        ) as iterator:
            for x, y in iterator:
                x = x.to(self.device)

                loss, y_pred = self.batch_update(x, y)

                # update loss logs
                loss_value = loss.cpu().detach().numpy()
                loss_meter.add(loss_value)
                loss_logs = {self.loss.__name__: loss_meter.mean}
                logs.update(loss_logs)

                # update metrics logs
                for metric_fn in self.metrics:
                    metric_value = metric_fn(y_pred, y).cpu().detach().numpy()
                    metrics_meters[metric_fn.__name__].add(metric_value)
                metrics_logs = {k: v.mean for k, v in metrics_meters.items()}
                logs.update(metrics_logs)

                if self.verbose:
                    s = self._format_logs(logs)
                    iterator.set_postfix_str(s)

        return logs
