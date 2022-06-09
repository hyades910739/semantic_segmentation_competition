import math
import os
from typing import Callable, List, Tuple

import albumentations as albu
import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
from tqdm import tqdm

from .augmentations import to_tensor
from .smp_train_utils import CustomValidEpoch
from .utils import get_pad_r_and_pad_l, resize_batch_image


def load_test_image_paths(folder_path) -> List[str]:
    folder_path = "Public_Image/"
    names = os.listdir(folder_path)
    names = [os.path.join(folder_path, name) for name in names]
    return names


# def get_padding_size(pred_height, pred_width, image) -> Tuple[int, int]:
#     resize = albu.SmallestMaxSize(max_size=pred_height, always_apply=True)
#     resized_img = resize(image=image)['image']
#     re_height, re_width, _ = resized_img.shape
#     total_pad = (pred_width - re_width)/2
#     pad_r,pad_l = math.ceil(total_pad), math.floor(total_pad)
#     return pad_l, pad_r


def build_test_time_preprocess(
    pred_height, pred_width, ENCODER="timm-efficientnet-b2", ENCODER_WEIGHTS="imagenet"
) -> Callable:
    try:
        preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
    except KeyError:
        preprocessing_fn = smp.encoders.get_preprocessing_fn("timm-efficientnet-b2", ENCODER_WEIGHTS)
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


def test_predict(
    model,
    pred_height,
    pred_width,
    origin_height,
    origin_width,
    test_image_folder_path="Public_Image/",
    batch_size=32,
    device="cuda",
    resize_interpolation=0,
    ENCODER="timm-efficientnet-b2",
    ENCODER_WEIGHTS="imagenet",
):

    names = load_test_image_paths(folder_path=test_image_folder_path)
    images = [cv2.imread(name) for name in names]

    test_image_preprocess = build_test_time_preprocess(pred_height, pred_width, ENCODER, ENCODER_WEIGHTS)
    # find padding size:
    # pad_l, pad_r = get_padding_size( pred_height, pred_width, images[0])
    pad = get_pad_r_and_pad_l(
        origin_h=origin_height,
        origin_w=origin_width,
        train_h=pred_height,
        train_w=pred_width,
    )
    pad_r, pad_l, pad_u, pad_d = pad["pad_r"], pad["pad_l"], pad["pad_u"], pad["pad_d"]
    for v in (pad_r, pad_l, pad_u, pad_d):
        assert v >= 0
    print(
        f"pred_height: {pred_height}, pred_width: {pred_width}, pad_right: {pad_r}, pad_left: {pad_l}, pad_up: {pad_u}, pad_down: {pad_d}"
    )

    images = [test_image_preprocess(image=img)["image"] for img in images]
    print(images[0].shape)
    preds = []

    sigmoid = torch.nn.Sigmoid()

    model.eval()
    for i in tqdm(range(0, len(images), batch_size)):
        tensor = torch.from_numpy(np.stack(images[i : i + batch_size]))
        # _, _, origin_height, origin_width = tensor.shape
        tensor = tensor.to(device)

        with torch.no_grad():
            pred = model.forward(tensor)

        original_size_pred = resize_batch_image(
            pred.to("cpu"),
            origin_height,
            origin_width,
            resize_interpolation,
            pad_r,
            pad_l,
            pad_u,
            pad_d,
        )
        preds.append(original_size_pred)

    return np.concatenate(preds)


def find_threshold_for_testing(
    model,
    epoch,
    data_loader,
    device,
    origin_height,
    origin_width,
    pred_height,
    pred_width,
    use_sigmoid=True,
    **fscore_kwargs,
):
    X_interpolation = dict()
    Y_interpolation = dict()

    pad = get_pad_r_and_pad_l(
        origin_h=origin_height,
        origin_w=origin_width,
        train_h=pred_height,
        train_w=pred_width,
    )
    pad_r, pad_l, pad_u, pad_d = pad["pad_r"], pad["pad_l"], pad["pad_u"], pad["pad_d"]
    for v in (pad_r, pad_l, pad_u, pad_d):
        assert v >= 0

    sigmoid = torch.nn.Sigmoid()
    if use_sigmoid:
        print("find threshold: use sigmoid!")
    # get predict data
    for interpolation in tqdm(range(6)):
        valid_epoch = CustomValidEpoch(
            model,
            loss=smp.utils.losses.DiceLoss(),
            metrics=[],
            device=device,
            verbose=True,
            pad_r=pad_r,
            pad_l=pad_l,
            pad_u=pad_u,
            pad_d=pad_d,
            resize_interpolation=interpolation,
        )
        X = []
        Y = []
        for x, y in data_loader:
            _, prediction = valid_epoch.batch_update(x, y)
            if use_sigmoid:
                prediction = sigmoid(prediction)
            X.append(prediction)
            Y.append(y)
        X_interpolation[interpolation] = torch.concat(X)
        Y_interpolation[interpolation] = torch.concat(Y)

    # find best score for different interpolation
    interpolation_fscores = {}
    for interpolation in tqdm(range(6)):
        preds = X_interpolation[interpolation]
        gt = Y_interpolation[interpolation]
        fscore = dict()
        for i in range(0, 10):
            fscore[i / 10] = (
                smp.utils.metrics.Fscore(threshold=i / 10, **fscore_kwargs).forward(preds, gt).numpy().tolist()
            )
            fscore[i / 10 + 0.05] = (
                smp.utils.metrics.Fscore(threshold=i / 10 + 0.05, **fscore_kwargs).forward(preds, gt).numpy().tolist()
            )
        for i in range(1, 10):
            fscore[i / 10] = (
                smp.utils.metrics.Fscore(threshold=i / 10, **fscore_kwargs).forward(preds, gt).numpy().tolist()
            )
            fscore[i / 10 + 0.005] = (
                smp.utils.metrics.Fscore(threshold=i / 10 + 0.05, **fscore_kwargs).forward(preds, gt).numpy().tolist()
            )

        interpolation_fscores[interpolation] = max(fscore.items(), key=lambda x: x[1])
    return interpolation_fscores
