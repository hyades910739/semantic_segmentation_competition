import json
import os
from typing import Any, Dict

import segmentation_models_pytorch as smp
import torch
from torch.utils.data import DataLoader

from myseg.augmentations import (
    get_albu_preprocessing,
    get_training_augmentation,
    get_training_slice_dataset_augmentation,
    get_validation_augmentation,
)
from myseg.configs import TrainConfig, TrainerConfig
from myseg.sam import SAM
from myseg.smp_train_utils import CustomValidEpoch, Dataset, SamTrainEpoch, SliceDataset, ValidationDataset
from myseg.utils import get_train_test_id_set


def set_up_cuda_env(train_config: TrainConfig) -> int:
    device_count = torch.cuda.device_count()
    if train_config.DEVICE == "cpu":
        print("USE CPU BY CONFIG.")
        visible_device = "-1"
        gpu_setting = -1
    else:
        assert train_config.DEVICE == "cuda"
        assert device_count > 0, "No gpu available. Change config to use cpu only."

        if train_config.NUM_GPUS > device_count:
            print(f"there's only {train_config.NUM_GPUS} gpus available.")
            visible_device = ",".join(str(i) for i in range(device_count))
            gpu_setting = device_count
        else:
            print(f"Use {train_config.NUM_GPUS} gpus.")
            visible_device = ",".join(str(i) for i in range(train_config.NUM_GPUS))
            gpu_setting = train_config.NUM_GPUS

    os.environ["CUDA_VISIBLE_DEVICES"] = visible_device
    return int(gpu_setting)


def build_model(config: TrainerConfig, gpu_setting: int):
    model = config.model_config.get_model()
    if gpu_setting > 1:
        model = torch.nn.DataParallel(model)
    return model


def build_dataset(config: TrainerConfig, preprocessing_fn) -> Dict[str, Any]:
    base_path = config.data_config.FOLDER_PATH
    albu_preprocessing = get_albu_preprocessing(preprocessing_fn)
    train_set, test_set = get_train_test_id_set(
        base_path=base_path,
        test_rate=config.data_config.TEST_RATIO,
        seed=config.data_config.TRAIN_TEST_SPLIT_SEED,
    )
    train_dataset = Dataset(
        select_id_set=train_set,
        base_path=base_path,
        augmentation=get_training_augmentation(config.data_config.TRAIN_WIDTH, config.data_config.TRAIN_HEIGHT),
        preprocessing=albu_preprocessing,
    )
    valid_dataset = ValidationDataset(
        select_id_set=test_set,
        base_path=base_path,
        augmentation=get_validation_augmentation(config.data_config.VALIDATE_WIDTH, config.data_config.VALIDATE_HEIGHT),
        preprocessing=albu_preprocessing,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train_config.TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=config.train_config.DATALOADER_NUM_WORKER,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config.train_config.VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=config.train_config.DATALOADER_NUM_WORKER,
    )
    if config.train_config.USE_SLICE_DATASET:
        slice_dataset = SliceDataset(
            select_id_set=train_set,
            base_path=base_path,
            augmentation=get_training_slice_dataset_augmentation(
                config.data_config.TRAIN_WIDTH, config.data_config.TRAIN_HEIGHT
            ),
            preprocessing=albu_preprocessing,
            pad=50,
            min_h=config.data_config.TRAIN_HEIGHT,
            min_w=config.data_config.TRAIN_WIDTH,
        )
        slice_loader = DataLoader(
            slice_dataset,
            batch_size=config.train_config.TRAIN_BATCH_SIZE,
            shuffle=True,
            num_workers=config.train_config.DATALOADER_NUM_WORKER,
        )
    else:
        slice_dataset = None
        slice_loader = None

    return {
        "train_loader": train_loader,
        "valid_loader": valid_loader,
        "train_dataset": train_dataset,
        "valid_dataset": valid_dataset,
        "slice_dataset": slice_dataset,
        "slice_loader": slice_loader,
    }


def build_train_iter(config: TrainerConfig, model: torch.nn.Module, dataset_dic) -> Dict[str, Any]:
    loss = config.train_config.get_loss_function()
    metrics = config.metric_config.get_metrics()
    optim = config.train_config.get_optimizer()
    device = config.train_config.DEVICE
    vaid_pads = config.data_config.test_padding_dict
    _ = config.data_config.train_padding_dict  # just make sure train_padding is correct.

    # build optimizer and train_epoch
    if config.train_config.USE_SAM:
        optimizer = SAM(model.parameters(), optim, lr=config.train_config.LR)
        get_train_epoch = SamTrainEpoch
    else:
        optimizer = optim(model.parameters(), lr=config.train_config.LR)
        get_train_epoch = smp.utils.train.TrainEpoch

    train_epoch = get_train_epoch(
        model,
        loss=loss,
        metrics=metrics,
        optimizer=optimizer,
        device=device,
        verbose=True,
    )
    # build slice dataset:
    if config.train_config.USE_SLICE_DATASET:
        train_slice_epoch = get_train_epoch(
            model,
            loss=loss,
            metrics=metrics,
            optimizer=optimizer,
            device=device,
            verbose=True,
        )
        train_slice_epoch.stage_name = "slice"
    else:
        train_slice_epoch = None
    # build valid epoch:
    valid_epoch = CustomValidEpoch(
        model,
        loss=loss,
        metrics=metrics,
        device=device,
        verbose=True,
        pad_r=vaid_pads["pad_r"],
        pad_l=vaid_pads["pad_l"],
        pad_u=vaid_pads["pad_u"],
        pad_d=vaid_pads["pad_d"],
    )
    return {
        "train_epoch": train_epoch,
        "train_slice_epoch": train_slice_epoch,
        "valid_epoch": valid_epoch,
        "optimizer": optimizer,
    }


def train(
    config: TrainerConfig,
    model,
    optimizer,
    train_epoch,
    valid_epoch,
    train_slice_epoch,
    train_loader,
    valid_loader,
    slice_loader,
):

    history = dict()
    max_score = -1e6

    for i in range(1, config.train_config.NUM_EPOCH + 1):

        print("\nEpoch: {}".format(i))
        train_logs = train_epoch.run(train_loader)
        if config.train_config.USE_SLICE_DATASET:
            train_slice_epoch.run(slice_loader)

        valid_logs = valid_epoch.run(valid_loader)

        # do something (save model, change lr, etc.)
        metric = config.metric_config.METRICS[0]
        if max_score < valid_logs[metric]:
            max_score = valid_logs[metric]
            save_path = os.path.join(config.metric_config.MODEL_SAVE_PATH, f"{config.MODEL_NAME}.pth")
            torch.save(model, save_path)
            print("Model saved!")

        if i == config.train_config.DECAY_LR_AT_EPOCH:
            # optimizer.param_groups[0]['lr'] = LR * 0.1
            optimizer.param_groups[0]["lr"] = config.train_config.LR * config.train_config.LR_DECAY_RATE
            print(f"Decrease decoder learning rate to {optimizer.param_groups[0]['lr']}!")
        # log history
        history[i] = dict()
        history[i]["train"] = train_logs
        history[i]["validation"] = valid_logs
        save_path = os.path.join(config.metric_config.METRIC_SAVE_PATH, f"{config.MODEL_NAME}_logs.json")
        with open(save_path, "wt") as f:
            json.dump(history, f, indent=2)

    return history


class Trainer:
    "entry point of training"

    def _save_config(self, config: TrainerConfig):
        save_path = os.path.join(config.metric_config.METRIC_SAVE_PATH, f"{config.MODEL_NAME}_config.json")
        with open(save_path, "wt") as f:
            json.dump(config.dict(), f, indent=2)

    def train(self, config: TrainerConfig):
        config.metric_config.check_path()
        self._save_config(config)
        gpu_setting = set_up_cuda_env(config.train_config)
        model = build_model(config, gpu_setting)
        preprocessing_fn = config.model_config.get_model_preprocess_funciton()
        dataset_dic = build_dataset(config, preprocessing_fn)
        epochs = build_train_iter(config, model, dataset_dic)
        history = train(
            config=config,
            model=model,
            optimizer=epochs["optimizer"],
            train_epoch=epochs["train_epoch"],
            valid_epoch=epochs["valid_epoch"],
            train_slice_epoch=epochs["train_slice_epoch"],
            train_loader=dataset_dic["train_loader"],
            valid_loader=dataset_dic["valid_loader"],
            slice_loader=dataset_dic["slice_loader"],
        )

        self.history = history
        self.dataset_dic = dataset_dic
        self.epochs = epochs
        self.model = model

        print("Training Complete!")
