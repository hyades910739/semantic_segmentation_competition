import datetime
import os
from typing import Callable, Dict, List, Optional

import segmentation_models_pytorch as smp
import torch
import ttach
from pydantic import BaseModel

from myseg.utils import calculate_padding_size_by_augmentation


def get_timestamp() -> str:
    return (datetime.datetime.now() + datetime.timedelta(hours=8)).strftime("%y%m%d_%H%M")


class TrainConfig(BaseModel):

    LOSS = "dice"
    DEVICE = "cuda"
    USE_SAM = True
    USE_SLICE_DATASET = True
    TRAIN_BATCH_SIZE: int = 8
    VALID_BATCH_SIZE: int = 32
    DATALOADER_NUM_WORKER = 3
    NUM_EPOCH: int = 80
    DICE_BETA: int = 2
    LR = 0.0001 * (TRAIN_BATCH_SIZE / 8)
    NUM_GPUS: int = 8
    OPTIMIZER: str = "adam"
    DECAY_LR_AT_EPOCH: int = 25
    LR_DECAY_RATE: float = 0.1

    def get_loss_function(self) -> Callable:
        if self.LOSS.lower() == "dice":
            loss = smp.utils.losses.DiceLoss(beta=self.DICE_BETA)
        elif self.LOSS.lower() == "cross_entropy":
            loss = smp.utils.losses.BCELoss()
        else:
            raise ValueError("LOSS only support dice and cross_entropy.")
        return loss

    def get_optimizer(self) -> Callable:
        DIC = {
            "adam": torch.optim.Adam,
            "sgd": torch.optim.SGD,
        }
        return DIC[self.OPTIMIZER.lower()]


class MetricsConfig(BaseModel):
    # first metric in the list will be used to choose best model in training process.
    METRICS: List[str] = ["iou_score", "fscore", "recall"]
    EVAL_ACTIVATION: Optional[str] = None
    THRESHOLD: float = 0.5
    METRIC_SAVE_PATH = "records"
    MODEL_SAVE_PATH = "models"

    def check_path(self) -> None:
        if not os.path.exists(self.METRIC_SAVE_PATH):
            os.mkdir(self.METRIC_SAVE_PATH)
        if not os.path.exists(self.MODEL_SAVE_PATH):
            os.mkdir(self.MODEL_SAVE_PATH)

    def get_metrics(self) -> List[Callable]:
        METRIC_MAPS = {
            "iou_score": smp.utils.metrics.IoU,
            "fscore": smp.utils.metrics.Fscore,
            "recall": smp.utils.metrics.Recall,
            "accuracy": smp.utils.metrics.Accuracy,
            "precision": smp.utils.metrics.Precision,
        }
        metrics = []
        for metric in self.METRICS:
            cur = METRIC_MAPS[metric](threshold=self.THRESHOLD, activation=self.EVAL_ACTIVATION)
            metrics.append(cur)
        return metrics


class ModelConfig(BaseModel):
    ENCODER_WEIGHTS = "imagenet"
    # encoder: 'resnet50' #'efficientnet-b2' # 'efficientnetv2_s' #'tu-efficientnetv2_s' #'timm-resnest50d' # tf_efficientnetv2_m_in21ft1k
    ENCODER = "tu-tf_efficientnetv2_l_in21ft1k"
    DECODER = "unetpp"
    ACTIVATION = "sigmoid"
    DECODER_ATTENTION_TYPE = "scse"  # None
    CHANNEL_FACTOR: float = 1.0

    def get_model(self) -> torch.nn.Module:
        if self.DECODER.lower() == "unetpp":
            model = smp.UnetPlusPlus(
                encoder_name=self.ENCODER,
                encoder_weights=self.ENCODER_WEIGHTS,
                classes=1,
                activation=self.ACTIVATION,
                decoder_attention_type=self.DECODER_ATTENTION_TYPE,
                decoder_channels=[int(self.CHANNEL_FACTOR * i) for i in (256, 128, 64, 32, 16)],
            )
        else:
            raise ValueError(f"unknown DECODER NAME: {self.DECODER}")
        return model

    def get_model_preprocess_funciton(self) -> Callable:
        try:
            preprocessing_fn = smp.encoders.get_preprocessing_fn(self.ENCODER, self.ENCODER_WEIGHTS)
        except KeyError:
            # some encoder dont have corresponding preprocessing_fn, use following instead:
            preprocessing_fn = smp.encoders.get_preprocessing_fn("timm-efficientnet-b2", self.ENCODER_WEIGHTS)
        return preprocessing_fn


class DataConfig(BaseModel, extra="allow"):
    ORIGINAL_HEIGHT: int = 942
    ORIGINAL_WIDTH: int = 1716
    TRAIN_WIDTH: int = 320
    TRAIN_HEIGHT: int = 320
    VALIDATE_WIDTH: int = 384
    VALIDATE_HEIGHT: int = 384

    FOLDER_PATH: str = "SEG_Train_Datasets/"
    TEST_RATIO: float = 0.2
    TRAIN_TEST_SPLIT_SEED: int = 123

    @property
    def train_padding_dict(self) -> Dict[str, int]:
        if not hasattr(self, "_train_padding_dict"):
            self._train_padding_dict = calculate_padding_size_by_augmentation(
                origin_h=self.ORIGINAL_HEIGHT,
                origin_w=self.ORIGINAL_WIDTH,
                target_h=self.TRAIN_HEIGHT,
                target_w=self.TRAIN_WIDTH,
            )
        return self._train_padding_dict

    @property
    def test_padding_dict(self) -> Dict[str, int]:
        if not hasattr(self, "_test_padding_dict"):
            self._test_padding_dict = calculate_padding_size_by_augmentation(
                origin_h=self.ORIGINAL_HEIGHT,
                origin_w=self.ORIGINAL_WIDTH,
                target_h=self.VALIDATE_HEIGHT,
                target_w=self.VALIDATE_WIDTH,
            )
        return self._test_padding_dict


class TrainerConfig(BaseModel):
    train_config: TrainConfig = TrainConfig()
    data_config: DataConfig = DataConfig()
    model_config: ModelConfig = ModelConfig()
    metric_config: MetricsConfig = MetricsConfig()
    MODEL_NAME = f"{get_timestamp()}_{model_config.DECODER}_{model_config.ENCODER}_width{data_config.TRAIN_WIDTH}"

    @property
    def MODELE_PATH(self):
        "where you can get model file"
        return os.path.join(self.metric_config.MODEL_SAVE_PATH, f"{self.MODEL_NAME}.pth")

    @classmethod
    def load_from_seperate_files(cls, data_config_fname=None, model_config_fname=None, train_config_name=None):
        configs = [TrainConfig, DataConfig, ModelConfig]
        fnames = [train_config_name, data_config_fname, model_config_fname]
        param_names = ["train_config", "data_config", "model_config"]
        params = dict()
        for config, fname, pname in zip(configs, fnames, param_names):
            # if specific fname is specified, use it, otherwise use `config_fname` as default.
            # if fname is None:
            #     # use config_fname:
            #     if config_fname is None:
            #         raise ValueError(f'{pname}: You should set `config_fname` as default if other fname is not set.')
            #     else:
            #         fname = config_fname
            params[pname] = config.parse_file(fname)
        return TrainerConfig(**params[pname])


class PredicterConfig(BaseModel):

    TARGET_HEIGHT: int = 384
    TARGET_WIDTH: int = 384
    BATCH_SIZE: int = 64
    USE_TTACH = True
    RESIZE_INTERPOLATION: int = 1
    DEVICE: str = "cuda"

    @property
    def test_time_augmentation_compose(self) -> ttach.Compose:
        """currently only support one tta"""
        return ttach.aliases.d4_transform()
