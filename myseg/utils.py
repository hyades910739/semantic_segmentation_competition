import math
import os
from typing import Dict

import albumentations as albu
import matplotlib.pyplot as plt
import numpy as np
import torch


def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(" ".join(name.split("_")).title())
        plt.imshow(image)
    plt.show()


def get_train_test_id_set(base_path, test_rate=0.2, seed=123):
    np.random.seed(seed)
    ids = os.listdir(os.path.join(base_path, "Train_Images"))
    ids = [id_.split(".")[0] for id_ in ids]
    ids = [i for i in ids if len(i) > 0]
    test_size = int(len(ids) * test_rate)
    test_set = set(np.random.choice(ids, size=test_size, replace=False))
    train_set = set(i for i in ids if i not in test_set)
    return train_set, test_set


def resize_batch_image(
    tensor,
    height,
    width,
    resize_interpolation=0,
    pad_r=None,
    pad_l=None,
    pad_u=None,
    pad_d=None,
):
    """tensor: a 4d tensor [batch, 1, height, width]"""
    # if pad_r is not None:
    #     assert pad_r >= 0
    #     if pad_r == 0:
    #         pad_r = None
    #     else:
    #         pad_r = -pad_r
    for value in [pad_r, pad_l, pad_u, pad_d]:
        if value is not None:
            assert value >= 0

    if pad_d == 0:
        pad_d = None
    elif pad_d is None:
        pass
    elif pad_d > 0:
        pad_d = -pad_d

    height_slice = slice(pad_u, pad_d)

    if tensor.device.type != "cpu":
        device = tensor.device.type
        array = tensor.to("cpu").numpy()
    else:
        array = tensor.numpy()
    array = array.squeeze(1)
    resized_array = [
        albu.resize(image[height_slice, :], height, width, resize_interpolation)
        for image in array
    ]
    resized_array = [np.expand_dims(i, 0) for i in resized_array]
    resized_array = np.concatenate(resized_array)
    resized_array = np.expand_dims(resized_array, 1)

    new_tensor = torch.from_numpy(resized_array)
    if tensor.device.type != "cpu":
        new_tensor = new_tensor.to(device)
    return new_tensor


# def get_pad_r_and_pad_l(origin_h, origin_w, train_h, train_w):
#     resized = albu.smallest_max_size(np.random.normal(size=(origin_h, origin_w)), train_h, 0)
#     *_, resize_h, resize_w = resized.shape
#     val = (train_w - resize_w)/2
#     r,l = math.ceil(val), math.floor(val)
#     return {'pad_r':r,'pad_l':l}


# def get_pad_r_and_pad_l(origin_h, origin_w, train_h, train_w):
#     max_size = max(train_w, train_h)
#     resized = albu.longest_max_size(np.random.normal(size=(origin_h, origin_w)), max_size, 0)
#     print(resized.shape)
#     *_, resize_h, resize_w = resized.shape
#     width_pad = (train_w - resize_w)/2
#     r,l = math.ceil(width_pad), math.floor(width_pad)
#     height_pad = (train_h - resize_h)/2
#     d,u = math.ceil(height_pad), math.floor(height_pad)

#     return {'pad_l':l,'pad_r':r, 'pad_u':u, 'pad_d':d}


def create_slices_from_label(
    label, grid_num=(6, 6), min_h=384, min_w=384, pad=20
):  # -> List[List[slice, slice]]
    assert len(label.shape) == 2
    # assert len(label.shape) == 3

    nrow, ncol = label.shape
    row_step = int(nrow / grid_num[0])
    col_step = int(ncol / grid_num[1])

    # calculate grid mean
    vals = {}
    for row_idx in range(0, nrow, row_step):
        for col_idx in range(0, ncol, col_step):
            vals[int(row_idx / row_step), int(col_idx / col_step)] = label[
                row_idx : row_idx + row_step, col_idx : col_idx + col_step
            ].mean()
    not_empty = [
        (r, c, val)
        for (r, c), val in sorted(vals.items(), key=lambda x: x[0])
        if val > 0
    ]
    if not not_empty:
        return []
    # merge groups
    groups = []
    x, y, _ = not_empty.pop()
    groups.append([x, y, x, y])
    while not_empty:
        cur_x, cur_y, _ = not_empty.pop()
        is_updated = False
        for no, (minx, miny, maxx, maxy) in enumerate(groups):
            x_near = abs(cur_x - minx) <= 1 or abs(cur_x - maxx) <= 1
            y_near = abs(cur_y - miny) <= 1 or abs(cur_y - maxy) <= 1
            if x_near and y_near:
                groups[no] = [
                    min(minx, cur_x),
                    min(miny, cur_y),
                    max(maxx, cur_x),
                    max(maxy, cur_y),
                ]
                is_updated = True
                break
        if is_updated:
            continue
        else:
            groups.append([cur_x, cur_y, cur_x, cur_y])
    slices = []
    for min_x, min_y, max_x, max_y in groups:
        x_start = int(min_x * row_step)
        x_end = min(int(max_x * row_step + 1 * row_step), int(x_start + min_w))

        y_start = int(min_y * col_step)
        y_end = min(int(max_y * col_step + 1 * col_step), y_start + min_h)

        if pad > 0:
            x_start, y_start = min(0, x_start - pad), min(0, y_start - pad)
            x_end, y_end = max(nrow, x_start + pad), max(ncol, y_start + pad)
        slice_x = slice(x_start, x_end)
        slice_y = slice(y_start, y_end)
        slices.append([slice_x, slice_y])

    return slices


def calculate_padding_size_by_augmentation(
    origin_h, origin_w, target_h, target_w
) -> Dict[str, int]:
    max_size = max(target_w, target_h)
    resized = albu.longest_max_size(
        np.random.normal(size=(origin_h, origin_w)), max_size, 0
    )
    *_, resize_h, resize_w = resized.shape
    width_pad = (target_w - resize_w) / 2
    r, l = math.ceil(width_pad), math.floor(width_pad)
    height_pad = (target_h - resize_h) / 2
    d, u = math.ceil(height_pad), math.floor(height_pad)
    result = {"pad_l": l, "pad_r": r, "pad_u": u, "pad_d": d}
    if any([i < 0 for i in result.values()]):
        raise ValueError(
            f"Got negative padding value `{result}`, change your train/test width or height in config."
        )
    return result
