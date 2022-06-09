import albumentations as albu
import cv2


def get_training_augmentation(WIDTH, HEIGHT):
    train_transform = [
        albu.SmallestMaxSize(max_size=int(max(WIDTH, HEIGHT) * 1.25), always_apply=True),
        albu.HorizontalFlip(p=0.5),
        albu.ShiftScaleRotate(
            scale_limit=(-0.2, 0.2),
            rotate_limit=0.3,
            shift_limit=0.3,
            p=1,
            border_mode=0,
            value=[255] * 3,
        ),
        # albu.PadIfNeeded(min_height=HEIGHT, min_width=WIDTH, always_apply=True, border_mode=cv2.BORDER_CONSTANT, value=[255]*3),
        albu.RandomCrop(height=HEIGHT, width=WIDTH, always_apply=True),
        # albu.GaussNoise(p=0.4),
        # albu.Perspective(p=0.4),
        # albu.OneOf(
        #     [
        #         albu.CLAHE(p=0.8),
        #         albu.RandomBrightness(p=0.8),
        #         albu.RandomGamma(p=0.8),
        #     ],
        #     p=0.9,
        # ),
        # albu.OneOf(
        #     [
        #         albu.Sharpen(p=1),
        #         albu.Blur(blur_limit=3, p=1),
        #         albu.MotionBlur(blur_limit=3, p=1),
        #     ],
        #     p=0.3,
        # ),
        #         albu.OneOf(
        #             [
        #                 albu.RandomContrast(p=1),
        #                 albu.HueSaturationValue(p=1),
        #             ],
        #             p=0.9,
        #         ),
    ]
    return albu.Compose(train_transform)


def get_validation_augmentation(WIDTH, HEIGHT):
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        albu.LongestMaxSize(max_size=WIDTH, always_apply=True),
        albu.PadIfNeeded(
            min_height=HEIGHT,
            min_width=WIDTH,
            always_apply=True,
            border_mode=cv2.BORDER_CONSTANT,
            value=[255] * 3,
        ),
    ]
    return albu.Compose(test_transform)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype("float32")


def get_albu_preprocessing(preprocessing_fn):
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose

    """

    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)


def get_training_slice_dataset_augmentation(WIDTH, HEIGHT):
    train_transform = [
        albu.SmallestMaxSize(max_size=HEIGHT, always_apply=True),
        albu.HorizontalFlip(p=0.5),
        albu.ShiftScaleRotate(
            scale_limit=(-0.01, 0.01),
            rotate_limit=0.1,
            shift_limit=0.05,
            p=0.5,
            border_mode=0,
            value=[255] * 3,
        ),
        # albu.PadIfNeeded(min_height=HEIGHT, min_width=WIDTH, always_apply=True, border_mode=cv2.BORDER_CONSTANT, value=[255]*3),
        albu.RandomCrop(height=HEIGHT, width=WIDTH, always_apply=True),
        albu.GaussNoise(p=0.4),
        albu.Perspective(p=0.4),
        albu.OneOf(
            [
                albu.CLAHE(p=0.8),
                albu.RandomBrightness(p=0.8),
                albu.RandomGamma(p=0.8),
            ],
            p=0.9,
        ),
        albu.OneOf(
            [
                albu.Sharpen(p=1),
                albu.Blur(blur_limit=3, p=1),
                albu.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.3,
        ),
        #         albu.OneOf(
        #             [
        #                 albu.RandomContrast(p=1),
        #                 albu.HueSaturationValue(p=1),
        #             ],
        #             p=0.9,
        #         ),
    ]
    return albu.Compose(train_transform)
