# Modified from Latte: https://github.com/Vchitect/Latte/blob/main/datasets/video_transforms.py

import io
import random
import numbers
import math

import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T


def _is_tensor_video_clip(clip):
    if not torch.is_tensor(clip):
        raise TypeError("Clip should be Tensor, but it is %s" % type(clip))

    if not clip.ndimension() == 4:
        raise ValueError("Clip should be 4D, but it is %dD" % clip.dim())

    return True


def to_tensor(clip):
    """
    Convert tensor data type from uint8 to float, divide value by 255.0 and
    permute the dimensions of clip tensor
    Args:
        clip (torch.tensor, dtype=torch.uint8): Size is (T, C, H, W)
    Return:
        clip (torch.tensor, dtype=torch.float): Size is (T, C, H, W)
    """
    try:
        _is_tensor_video_clip(clip)
    except Exception as e:
        print(f"An error occurred: {e}")
    if not clip.dtype == torch.uint8:
        raise TypeError(
            "Clip tensor should have data type uint8, but it is %s" % str(clip.dtype)
        )

    return clip.float() / 255.0


def to_tensor_after_resize(clip):
    """
    Convert resized tensor to [0, 1]
    Args:
        clip (torch.tensor, dtype=torch.float): Size is (T, C, H, W)
    Return:
        clip (torch.tensor, dtype=torch.float): Size is (T, C, H, W), but in [0, 1]
    """
    try:
        _is_tensor_video_clip(clip)
    except Exception as e:
        print(f"An error occurred: {e}")
    return clip.float() / 255.0


def hflip(clip):
    """
    Args:
        clip (torch.tensor): Video clip to be normalized. Size is (T, C, H, W)
    Returns:
        flipped clip (torch.tensor): Size is (T, C, H, W)
    """
    if not _is_tensor_video_clip(clip):
        raise ValueError("clip should be a 4D torch.tensor")
    return clip.flip(-1)


def crop(clip, i, j, h, w):
    """
    Args:
        clip (torch.tensor): Video clip to be cropped. Size is (T, C, H, W)
    """
    if len(clip.size()) != 4:
        raise ValueError("clip should be a 4D tensor")
    return clip[..., i: i + h, j: j + w]


def resize(clip, target_size, interpolation_mode, align_corners=False, antialias=False):
    if len(target_size) != 2:
        raise ValueError(
            f"target size should be tuple (height, width), instead got {target_size}"
        )
    return torch.nn.functional.interpolate(
        clip,
        size=target_size,
        mode=interpolation_mode,
        align_corners=align_corners,
        antialias=antialias,
    )


def resize_scale(clip, target_size, interpolation_mode):
    if len(target_size) != 2:
        raise ValueError(
            f"target size should be tuple (height, width), instead got {target_size}"
        )
    H, W = clip.size(-2), clip.size(-1)
    scale = target_size[0] / min(H, W)
    return torch.nn.functional.interpolate(
        clip, scale_factor=scale, mode=interpolation_mode, align_corners=False
    )


def center_crop(clip, crop_size):
    if not _is_tensor_video_clip(clip):
        raise ValueError("clip should be a 4D torch.tensor")
    h, w = clip.size(-2), clip.size(-1)
    th, tw = crop_size
    if h < th or w < tw:
        raise ValueError(f"height and width must be no smaller than crop_size, height: {h}, width: {w}, crop_size: {crop_size}")

    i = int(round((h - th) / 2.0))
    j = int(round((w - tw) / 2.0))
    return crop(clip, i, j, th, tw)


def center_crop_using_short_edge(clip):
    if not _is_tensor_video_clip(clip):
        raise ValueError("clip should be a 4D torch.tensor")
    h, w = clip.size(-2), clip.size(-1)
    if h < w:
        th, tw = h, h
        i = 0
        j = int(round((w - tw) / 2.0))
    else:
        th, tw = w, w
        i = int(round((h - th) / 2.0))
        j = 0
    return crop(clip, i, j, th, tw)


def center_crop_th_tw(clip, th, tw, top_crop):
    if not _is_tensor_video_clip(clip):
        raise ValueError("clip should be a 4D torch.tensor")

    h, w = clip.size(-2), clip.size(-1)
    tr = th / tw
    if h / w > tr:
        new_h = int(w * tr)
        new_w = w
    else:
        new_h = h
        new_w = int(h / tr)

    i = 0 if top_crop else int(round((h - new_h) / 2.0))
    j = int(round((w - new_w) / 2.0))
    return crop(clip, i, j, new_h, new_w)


def resize_crop_to_fill(clip, target_size):
    if not _is_tensor_video_clip(clip):
        raise ValueError("clip should be a 4D torch.tensor")
    h, w = clip.size(-2), clip.size(-1)
    th, tw = target_size[0], target_size[1]
    rh, rw = th / h, tw / w
    if rh > rw:
        sh, sw = th, round(w * rh)
        clip = resize(clip, (sh, sw), "bilinear")
        i = 0
        j = int(round(sw - tw) / 2.0)
    else:
        sh, sw = round(h * rw), tw
        clip = resize(clip, (sh, sw), "bilinear")
        i = int(round(sh - th) / 2.0)
        j = 0
    if i + th > clip.size(-2) or j + tw > clip.size(-1):
        raise AssertionError("size mismatch.")
    return crop(clip, i, j, th, tw)


def longsideresize(h, w, size, skip_low_resolution):
    if h <= size[0] and w <= size[1] and skip_low_resolution:
        return h, w
    
    if h / w > size[0] / size[1]:
        # hxw 720x1280  size 320x640  hw_raito 9/16 > size_ratio 8/16  neww=320/720*1280=568  newh=320
        w = math.ceil(size[0] / h * w)
        h = size[0]
    else:
        # hxw 720x1280  size 480x640  hw_raito 9/16 < size_ratio 12/16   newh=640/1280*720=360 neww=640
        # hxw 1080x1920  size 720x1280  hw_raito 9/16 = size_ratio 9/16   newh=1280/1920*1080=720 neww=1280
        h = math.ceil(size[1] / w * h)
        w = size[1]
    return h, w


def shortsideresize(h, w, size, skip_low_resolution):
    if h <= size[0] and w <= size[1] and skip_low_resolution:
        return h, w
    
    if h / w < size[0] / size[1]:
        w = math.ceil(size[0] / h * w)
        h = size[0]
    else:
        h = math.ceil(size[1] / w * h)
        w = size[1]
    return h, w


def calculate_statistics(data):
    if len(data) == 0:
        return None
    data = np.array(data)
    mean = np.mean(data)
    variance = np.var(data)
    std_dev = np.std(data)
    minimum = np.min(data)
    maximum = np.max(data)

    return {
        'mean': mean,
        'variance': variance,
        'std_dev': std_dev,
        'min': minimum,
        'max': maximum
    }


def get_params(h, w, stride):
    th, tw = h // stride * stride, w // stride * stride

    i = (h - th) // 2
    j = (w - tw) // 2

    return (i, j, th, tw)


def maxhwresize(ori_height, ori_width, max_hxw):
    if ori_height * ori_width > max_hxw:
        scale_factor = np.sqrt(max_hxw / (ori_height * ori_width))
        new_height = int(ori_height * scale_factor)
        new_width = int(ori_width * scale_factor)
    else:
        new_height = ori_height
        new_width = ori_width
    return new_height, new_width

def filter_resolution(h, w, max_h_div_w_ratio=17 / 16, min_h_div_w_ratio=8 / 16):
    if h / w <= max_h_div_w_ratio and h / w >= min_h_div_w_ratio:
        return True
    return False

class AENorm:
    """
    Apply an ae_norm to a PIL image or video.
    """

    def __init__(self):
        pass

    @staticmethod
    def __call__(clip):
        """
        Apply the center crop to the input video.

        Args:
            video (clip): The input video.

        Returns:
            video: The ae_norm video.
        """

        clip = 2.0 * clip - 1.0
        return clip

    def __repr__(self) -> str:
        return self.__class__.__name__


class ResizeCropToFill:
    """
    Apply a resize crop to a PIL image.
    """

    def __init__(self, size=256):
        self.size = size

    def __call__(self, pil_image):
        """
        Apply the resize crop to the input PIL image.

        Args:
            pil_image (PIL.Image): The input PIL image.

        Returns:
            PIL.Image: The resize-cropped image.
        """
        return resize_crop_to_fill(pil_image, self.size)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size})"


class BaseCrop:
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size})"


class ResizeCrop(BaseCrop):
    def __call__(self, clip):
        clip = resize_crop_to_fill(clip, self.size)
        return clip


class RandomSizedCrop(BaseCrop):
    def __call__(self, clip):
        i, j, h, w = self.get_params(clip)
        return crop(clip, i, j, h, w)

    def get_params(self, clip, multiples_of=8):
        h, w = clip.shape[-2:]

        # get random target h w
        th = random.randint(self.size[0], self.size[1])
        tw = random.randint(self.size[0], self.size[1])
        # ensure that h w are factors of 8
        th = th - th % multiples_of
        tw = tw - tw % multiples_of

        if h < th:
            th = h - h % multiples_of
        if w < tw:
            tw = w - w % multiples_of

        if w == tw and h == th:
            return 0, 0, h, w

        # get random start pos
        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw


class ToTensorVideo:
    """
    Convert tensor data type from uint8 to float, divide value by 255.0 and
    permute the dimensions of clip tensor
    """

    def __init__(self):
        pass

    @staticmethod
    def __call__(clip):
        """
        Args:
            clip (torch.tensor, dtype=torch.uint8): Size is (T, C, H, W)
        Return:
            clip (torch.tensor, dtype=torch.float): Size is (T, C, H, W)
        """
        return to_tensor(clip)

    def __repr__(self) -> str:
        return self.__class__.__name__


class ToTensorAfterResize:
    """
    Convert tensor data type from uint8 to float, divide value by 255.0 and
    permute the dimensions of clip tensor
    """

    def __init__(self):
        pass

    @staticmethod
    def __call__(clip):
        """
        Args:
            clip (torch.tensor, dtype=torch.float): Size is (T, C, H, W)
        Return:
            clip (torch.tensor, dtype=torch.float): Size is (T, C, H, W), but in [0, 1]
        """
        return to_tensor_after_resize(clip)

    def __repr__(self) -> str:
        return self.__class__.__name__


class RandomHorizontalFlipVideo:
    """
    Flip the video clip along the horizontal direction with a given probability
    Args:
        p (float): probability of the clip being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor): Size is (T, C, H, W)
        Return:
            clip (torch.tensor): Size is (T, C, H, W)
        """
        if random.random() < self.p:
            clip = hflip(clip)
        return clip

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p})"


class SpatialStrideCropVideo:
    def __init__(self, stride):
        self.stride = stride

    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor): Video clip to be cropped. Size is (T, C, H, W)
        Returns:
            torch.tensor: cropped video clip by stride.
                size is (T, C, OH, OW)
        """
        h, w = clip.shape[-2:] 
        i, j, h, w = get_params(h, w, self.stride)
        return crop(clip, i, j, h, w)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(stride={self.stride})"


class LongSideResizeVideo:
    """
    First use the long side,
    then resize to the specified size
    """

    def __init__(
            self,
            size,
            skip_low_resolution=False,
            interpolation_mode="bilinear",
            align_corners=False, 
            antialias=False
    ):
        self.size = size
        self.skip_low_resolution = skip_low_resolution
        self.interpolation_mode = interpolation_mode
        self.align_corners = align_corners
        self.antialias = antialias

    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor): Video clip to be cropped. Size is (T, C, H, W)
        Returns:
            torch.tensor: scale resized video clip.
        """
        _, _, h, w = clip.shape
        tr_h, tr_w = longsideresize(h, w, self.size, self.skip_low_resolution)
        if h == tr_h and w == tr_w:
            return clip
        resize_clip = resize(
            clip, 
            target_size=(tr_h, tr_w),
            interpolation_mode=self.interpolation_mode,
            align_corners=self.align_corners,
            antialias=self.antialias
        )
        return resize_clip

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size}, interpolation_mode={self.interpolation_mode}"


class MaxHWResizeVideo:
    '''
    First use the h*w,
    then resize to the specified size
    '''

    def __init__(
            self,
            transform_size=None,
            interpolation_mode="bilinear",
            align_corners=False, 
            antialias=False
    ):
        if transform_size is None or "max_hxw" not in transform_size:
            raise ValueError("Missing required param: max_hxw in data transform.")
        self.max_hxw = transform_size["max_hxw"]
        self.interpolation_mode = interpolation_mode
        self.align_corners = align_corners
        self.antialias = antialias

    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor): Video clip to be cropped. Size is (T, C, H, W)
        Returns:
            torch.tensor: scale resized video clip.
        """
        _, _, h, w = clip.shape
        tr_h, tr_w = maxhwresize(h, w, self.max_hxw)
        if h == tr_h and w == tr_w:
            return clip
        resize_clip = resize(
            clip, 
            target_size=(tr_h, tr_w),
            interpolation_mode=self.interpolation_mode,
            align_corners=self.align_corners,
            antialias=self.antialias
        )
        return resize_clip

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.max_hxw}, interpolation_mode={self.interpolation_mode}"


class CenterCropResizeVideo:
    """
    First use the short side for cropping length,
    center crop video, then resize to the specified size
    """

    def __init__(
            self,
            transform_size=None,
            interpolation_mode="bilinear",
            align_corners=False,
            antialias=False,
    ):
        self.size = transform_size
        self.interpolation_mode = interpolation_mode
        self.align_corners = align_corners
        self.antialias = antialias

    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor): Video clip to be cropped. Size is (T, C, H, W)
        Returns:
            torch.tensor: scale resized / center cropped video clip.
                size is (T, C, crop_size, crop_size)
        """
        _, _, h, w = clip.shape
        tr_h, tr_w = shortsideresize(
            h, 
            w, 
            self.size, 
            skip_low_resolution=False
        )
        clip = resize(
            clip,
            target_size=(tr_h, tr_w),
            interpolation_mode=self.interpolation_mode,
            align_corners=self.align_corners,
            antialias=self.antialias,
        )
        clip_center_crop = center_crop(clip, self.size)
        return clip_center_crop

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size}, interpolation_mode={self.interpolation_mode}, align_corners={self.align_corners}, antialias={self.antialias}"
    

class ResizeVideo:
    def __init__(
        self,
        transform_size="auto",
        interpolation_mode="bilinear",
        skip_low_resolution=False,
        align_corners=False, 
        antialias=False,
        mode="resize" # resize / longside / shortside / hxw
    ):  
        self.mode = mode
        if mode == 'hxw':
            self.transform_size = transform_size["max_hxw"] if isinstance(transform_size, dict) else transform_size
        elif mode in ["resize", "longside", "shortside"]:
            self.transform_size = (transform_size["max_height"], transform_size["max_width"]) if isinstance(transform_size, dict) else transform_size
        else:
            raise NotImplementedError(f"ResizeVideo only support mode `resize` / `longside` / `shortside` / `hxw`, {mode} is not implemented.")
        
        self.interpolation_mode = interpolation_mode
        self.align_corners = align_corners
        self.antialias = antialias
        self.skip_low_resolution = skip_low_resolution

    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor): Video clip to be cropped. Size is (T, C, H, W)
        Returns:
            torch.tensor: scale resized video clip.
        """
        h, w = clip.shape[-2:]
        if self.mode == "hxw":
            tr_h, tr_w = maxhwresize(h, w, self.transform_size)
        elif self.mode == "resize":
            tr_h, tr_w = self.transform_size
        elif self.mode == "longside":
            tr_h, tr_w = longsideresize(h, w, self.transform_size, skip_low_resolution=self.skip_low_resolution)
        elif self.mode == "shortside":
            tr_h, tr_w = shortsideresize(h, w, self.transform_size, skip_low_resolution=self.skip_low_resolution)
        
        if h == tr_h and w == tr_w:
            return clip
        resize_clip = resize(
            clip, 
            target_size=(tr_h, tr_w),
            interpolation_mode=self.interpolation_mode,
            align_corners=self.align_corners,
            antialias=self.antialias
        )
        return resize_clip

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.transform_size}, interpolation_mode={self.interpolation_mode})"
        

class UCFCenterCropVideo:
    """
    First scale to the specified size in equal proportion to the short edge,
    then center cropping
    """

    def __init__(
            self,
            size,
            interpolation_mode="bilinear",
    ):
        if isinstance(size, list):
            size = tuple(size)
        if isinstance(size, tuple):
            if len(size) != 2:
                raise ValueError(
                    f"size should be tuple (height, width), instead got {size}"
                )
            self.size = size
        else:
            self.size = (size, size)

        self.interpolation_mode = interpolation_mode

    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor): Video clip to be cropped. Size is (T, C, H, W)
        Returns:
            torch.tensor: scale resized / center cropped video clip.
                size is (T, C, crop_size, crop_size)
        """
        clip_resize = resize_scale(
            clip=clip, target_size=self.size, interpolation_mode=self.interpolation_mode
        )
        clip_center_crop = center_crop(clip_resize, self.size)
        return clip_center_crop

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size}, interpolation_mode={self.interpolation_mode}"


class TemporalRandomCrop:
    """Temporally crop the given frame indices at a random location.
    Args:
            size (int): Desired length of frames will be seen in the model.
    """

    def __init__(self, size, force_cut_video_from_start=False):
        self.size = size
        self.force_cut_video_from_start = force_cut_video_from_start

    def __call__(self, total_frames):
        if self.force_cut_video_from_start:
            begin_index = 0
        else:
            rand_end = max(0, total_frames - self.size - 1)
            begin_index = random.randint(0, rand_end)
        end_index = min(begin_index + self.size, total_frames)
        return begin_index, end_index


class Expand2Square:
    """
    Expand the given PIL image to a square by padding it with a background color.
    Args:
        mean (sequence): Sequence of means for each channel.
    """

    def __init__(self, mean):
        self.background_color = tuple(int(x * 255) for x in mean)

    def __call__(self, pil_img):
        width, height = pil_img.size
        if width == height:
            return pil_img
        elif width > height:
            result = Image.new(pil_img.mode, (width, width), self.background_color)
            result.paste(pil_img, (0, (width - height) // 2))
            return result
        else:
            result = Image.new(pil_img.mode, (height, height), self.background_color)
            result.paste(pil_img, ((height - width) // 2, 0))
            return result
