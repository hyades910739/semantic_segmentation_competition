import os

import albumentations as albu
import cv2
import numpy as np
import torch
import ttach
from tqdm import tqdm

from myseg.augmentations import to_tensor
from myseg.configs import ModelConfig, PredicterConfig, TrainerConfig
from myseg.utils import calculate_padding_size_by_augmentation, resize_batch_image


class Predicter:
    """Entry point for predict"""

    def __init__(self, config: PredicterConfig):
        self.predict_config = config

    @staticmethod
    def load_model_from_trainer_config(train_config: TrainerConfig) -> torch.nn.Module:
        path = train_config.MODELE_PATH
        model = torch.load(path)
        model.device_ids = [i for i in range(torch.cuda.device_count())]
        return model

    @staticmethod
    def build_test_time_preprocess(pred_width: int, pred_height: int, model_config: ModelConfig):
        preprocessing_fn = model_config.get_model_preprocess_funciton()
        test_image_preprocess = [
            albu.LongestMaxSize(max_size=pred_width, always_apply=True),
            albu.PadIfNeeded(
                min_height=pred_height,
                min_width=pred_width,
                always_apply=True,
                border_mode=cv2.BORDER_CONSTANT,
                value=[255] * 3,
            ),
            albu.Lambda(image=preprocessing_fn, always_apply=True),
            albu.Lambda(image=to_tensor, mask=to_tensor, always_apply=True),
        ]
        return albu.Compose(test_image_preprocess)

    def _wrap_model_if_use_tta(self, model: torch.nn.Module):
        if self.predict_config.USE_TTACH:
            model = ttach.wrappers.SegmentationTTAWrapper(model, self.predict_config.test_time_augmentation_compose)

        return model

    def predict(
        self,
        X,
        model: torch.nn.Module,
        model_config: ModelConfig,
    ) -> np.ndarray:
        """
        preprocess, and predict for a array of original sized image.
        X: array of original sized image. 4d array with shape [num_images, height, width, channel_size]
        """
        _, original_height, original_width, _ = X.shape
        preprocess = Predicter.build_test_time_preprocess(
            self.predict_config.TARGET_WIDTH, self.predict_config.TARGET_HEIGHT, model_config
        )
        images = [preprocess(image=img)["image"] for img in X]
        preds = []

        padding_dict = calculate_padding_size_by_augmentation(
            origin_h=original_height,
            origin_w=original_width,
            target_h=self.predict_config.TARGET_HEIGHT,
            target_w=self.predict_config.TARGET_WIDTH,
        )
        self._wrap_model_if_use_tta(model)
        model.eval()
        for i in tqdm(range(0, len(images), self.predict_config.BATCH_SIZE)):
            tensor = torch.from_numpy(np.stack(images[i : i + self.predict_config.BATCH_SIZE]))
            # _, _, origin_height, origin_width = tensor.shape
            tensor = tensor.to(self.predict_config.DEVICE)

            with torch.no_grad():
                pred = model.forward(tensor)

            original_size_pred = resize_batch_image(
                tensor=pred.to("cpu"),
                height=original_height,
                width=original_width,
                resize_interpolation=self.predict_config.RESIZE_INTERPOLATION,
                pad_r=padding_dict["pad_r"],
                pad_l=padding_dict["pad_l"],
                pad_u=padding_dict["pad_u"],
                pad_d=padding_dict["pad_d"],
            )
            preds.append(original_size_pred)

        return np.concatenate(preds)

    def predict_from_folder(
        self,
        folder_path: str,
        model: torch.nn.Module,
        model_config: ModelConfig,
    ) -> np.ndarray:
        names = os.listdir(folder_path)
        names = [os.path.join(folder_path, name) for name in names]
        images = [cv2.imread(name) for name in names]
        images = np.array(images)
        assert len(images) > 0, f"get 0 images from given folder: {folder_path}"
        return self.predict(images, model, model_config)
