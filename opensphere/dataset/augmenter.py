import cv2
import numpy as np

from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as F

class Augmenter():
    ''' Augmenter class for data augmentation
        This is taken from the official implementation of AdaFace
        paper: https://arxiv.org/abs/2204.00964
        github: https://github.com/mk-minchul/AdaFace
    '''
    def __init__(
            self,
            horizontal_flip=True, crop_prob=0.2,
            lowres_prob=0.2, photo_prob=0.2,
        ):
        #
        self.horizontal_flip = horizontal_flip
        self.crop_prob = crop_prob
        self.lowres_prob = lowres_prob
        self.photo_prob = photo_prob

        self.T_crop = transforms.RandomResizedCrop(
            size=(112, 112),
            scale=(0.2, 1.0),
            ratio=(0.75, 1.33),
        )
        self.T_photo = transforms.ColorJitter(
            brightness=0.5, contrast=0.5,
            saturation=0.5, hue=0,
        )

    def augment(self, image):
        # horizontally augmentation
        if np.random.random() < 0.5 and self.horizontal_flip:
            image = np.flip(image, axis=1)

        # crop with zero padding augmentation
        if np.random.random() < self.crop_prob:
            image = self.crop_aug(image)

        # low resolution augmentation
        if np.random.random() < self.lowres_prob:
            image = self.lowres_aug(image)

        # photometric augmentation
        if np.random.random() < self.photo_prob:
            image = self.photo_aug(image)

        return image

    def crop_aug(self, image):
        #
        new_image = np.zeros_like(image)
        image = Image.fromarray(image.astype(np.uint8))
        i, j, h, w = self.T_crop.get_params(
            image,
            self.T_crop.scale,
            self.T_crop.ratio,
        )
        cropped_image = F.crop(image, i, j, h, w)
        new_image[i:i+h, j:j+w, :] = np.array(cropped_image)

        return new_image

    def lowres_aug(self, image):
        # resize the image to a small size and enlarge it back
        image_shape = image.shape
        side_ratio = np.random.uniform(0.2, 1.0)
        small_side = int(side_ratio * image_shape[0])
        interp_types = [
            cv2.INTER_NEAREST,
            cv2.INTER_LINEAR,
            cv2.INTER_AREA,
            cv2.INTER_CUBIC,
            cv2.INTER_LANCZOS4,
        ]
        interpolation = np.random.choice(interp_types)
        small_image = cv2.resize(
            image, (small_side, small_side),
            interpolation=interpolation,
        )
        interpolation = np.random.choice(interp_types)
        new_image = cv2.resize(
            small_image, (image_shape[1], image_shape[0]),
            interpolation=interpolation,
        )
        return new_image

    def photo_aug(self, image):
        fn_idx, brightness, contrast, saturation, hue = \
            self.T_photo.get_params(
                self.T_photo.brightness, self.T_photo.contrast,
                self.T_photo.saturation, self.T_photo.hue,
            )

        image = Image.fromarray(image.astype(np.uint8))
        for fn_id in fn_idx:
            if fn_id == 0 and brightness is not None:
                image = F.adjust_brightness(image, brightness)
            elif fn_id == 1 and contrast is not None:
                image = F.adjust_contrast(image, contrast)
            elif fn_id == 2 and saturation is not None:
                image = F.adjust_saturation(image, saturation)
        return np.array(image)
