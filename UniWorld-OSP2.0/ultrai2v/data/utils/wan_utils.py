import torch
import math
import numpy as np
from torchvision.transforms import Compose
from tqdm import tqdm

from .transforms import CenterCropResizeVideo, TemporalRandomCrop, ToTensorAfterResize, AENorm, filter_resolution
from .utils import AbstractDataProcessor, AbstractDataFilter, TextProcessor, read_ann_txt
from .video_reader import VideoReader, video_reader_contextmanager
from .image_reader import ImageReader

class StartFrameNoiseAdder:
    def __init__(self, mean=-3.0, std=0.5):
        self.mean = mean
        self.std = std

    def __call__(self, start_frame): # C T H W
        if start_frame.ndim == 5:
            batch_size = start_frame.shape[0]
        else:
            batch_size = 1
        noise_sigma = torch.normal(
            mean=self.mean, std=self.std, size=(batch_size,), device=start_frame.device
        )
        noise_sigma = torch.exp(noise_sigma)
        while noise_sigma.ndim < start_frame.ndim:
            noise_sigma = noise_sigma.unsqueeze(-1)
        start_frame = start_frame + torch.randn_like(start_frame) * noise_sigma
        return start_frame

class WanTextProcessor(TextProcessor):

    def __init__(
        self,
        tokenizer,
        model_max_length=512,
        use_clean_caption=True,
        enable_text_preprocessing=True,
        padding_type="max_length",
        support_chinese=False,
        return_prompt_mask=True,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.model_max_length = model_max_length
        self.use_clean_caption = use_clean_caption
        self.enable_text_preprocessing = enable_text_preprocessing
        self.padding_type = padding_type
        self.support_chinese = support_chinese
        self.return_prompt_mask = return_prompt_mask

    def __call__(self, text):
        if self.enable_text_preprocessing:
            text = self.text_preprocessing(text, use_clean_caption=self.use_clean_caption, support_chinese=self.support_chinese)

        prompt_mask = None
        prompt_input_ids_and_mask = self.tokenizer(
            text,
            max_length=self.model_max_length,
            padding=self.padding_type,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        prompt_input_ids = prompt_input_ids_and_mask.input_ids

        if self.return_prompt_mask:
            prompt_mask = prompt_input_ids_and_mask.attention_mask

        return prompt_input_ids, prompt_mask


class WanVideoFilter(AbstractDataFilter):
    def __init__(
        self,
        ann_txt_path,
        save_absolute_path=True,
        sample_height=480,
        sample_width=832,
        sample_num_frames=49,
        train_fps=16,
        sample_stride=None,
        min_hxw=480*832,
        too_long_factor=5.0,
        max_h_div_w_ratio=2.0,
        min_h_div_w_ratio=0.5,
        max_motion_value=1.0,
        min_motion_value=0.0,
        **kwargs,
    ):
        super().__init__()
        self.data_samples = read_ann_txt(ann_txt_path, use_absolute_path=save_absolute_path)
        print(f'data length: {len(self.data_samples)}')
        
        self.sample_num_frames = sample_num_frames
        self.sample_height = sample_height
        self.sample_width = sample_width

        self.sample_stride = sample_stride
        self.train_fps = train_fps

        self.sample_mode = None
        if self.train_fps is not None:
            print(f"Using train_fps mode, train_fps: {self.train_fps}")
            self.sample_mode = "train_fps"
        elif self.sample_stride is not None:
            print(f"Using sample_stride mode, sample_stride: {self.sample_stride}")
            self.sample_mode = "sample_stride"
        else:
            raise ValueError("Must specify either train_fps or sample_stride")
        
        self.max_h_div_w_ratio = max_h_div_w_ratio
        self.min_h_div_w_ratio = min_h_div_w_ratio
        self.max_motion_value = max_motion_value
        self.min_motion_value = min_motion_value
        self.min_hxw = min_hxw
        self.too_long_factor = too_long_factor

    def filter_data_samples_force_shape(self, data_samples=None):
        if data_samples is None:
            data_samples = self.data_samples
        cnt = len(data_samples)
        cnt_too_long = 0
        cnt_too_short = 0
        cnt_too_fast = 0
        cnt_too_slow = 0
        cnt_no_cap = 0
        cnt_no_resolution = 0
        cnt_aspect_mismatch = 0
        cnt_res_too_small = 0
        cnt_fps_too_low = 0

        cnt_after_filter = 0
        filtered_data_samples = []

        for i in tqdm(data_samples, desc=f"flitering samples"):
            path = i["path"]

            cap = i.get("cap", None)
            # ======no caption=====
            if cap is None:
                cnt_no_cap += 1
                continue

            # ======resolution mismatch=====
            if i.get("resolution", None) is None:
                cnt_no_resolution += 1
                continue
            else:
                if i["resolution"].get("height", None) is None or i["resolution"].get("width", None) is None:
                    cnt_no_resolution += 1
                    continue
                else:
                    height, width = i["resolution"]["height"], i["resolution"]["width"]
                    if height <= 0 or width <= 0:
                        cnt_no_resolution += 1
                        continue

                    # filter aspect
                    is_pick = filter_resolution(
                        height, 
                        width, 
                        max_h_div_w_ratio=self.max_h_div_w_ratio, 
                        min_h_div_w_ratio=self.min_h_div_w_ratio
                    )

                    if not is_pick:
                        cnt_aspect_mismatch += 1
                        continue

                    target_h_div_w = self.sample_height / self.sample_width
                    current_h_div_w = height / width
                    min_hxw_scale = max(current_h_div_w / target_h_div_w, target_h_div_w / current_h_div_w)
                    min_hxw = math.ceil(self.min_hxw * min_hxw_scale)
                    # filter min_hxw
                    if height * width < min_hxw:
                        cnt_res_too_small += 1
                        continue


                fps = i.get('fps', 24)
                # max 5.0 and min 1.0 are just thresholds to filter some videos which have suitable duration. 
                if i['num_frames'] > self.too_long_factor * (self.sample_num_frames * fps / self.train_fps):  # too long video is not suitable for this training stage (self.num_frames)
                    cnt_too_long += 1
                    continue

                if self.sample_mode == "train_fps":
                    # resample in case high fps, such as 50/60/90/144 -> train_fps(e.g, 24)
                    frame_interval = (
                        1.0 if abs(fps - self.train_fps) < 0.1 else fps / self.train_fps
                    )
                elif self.sample_mode == "sample_stride":
                    frame_interval = self.sample_stride

                if self.sample_mode == "train_fps" and frame_interval < 1.0:
                    cnt_fps_too_low += 1
                    continue

                motion_value = i.get("motion", 0.0)
                if motion_value < self.min_motion_value:
                    cnt_too_fast += 1
                    continue
                elif motion_value > self.max_motion_value:
                    cnt_too_slow += 1
                    continue
                

                start_frame_idx = i.get("cut", [0])[0]
                i["start_frame_idx"] = start_frame_idx
                frame_indices = np.arange(start_frame_idx, start_frame_idx + i["num_frames"], frame_interval).astype(int)

                #  too long video will be temporal-crop randomly
                if len(frame_indices) < self.sample_num_frames:
                    cnt_too_short += 1
                    continue

                cnt_after_filter += 1
                filtered_data_samples.append(i)

        print('-----------------------filtering result------------------------')
        print(f"filtering config: sample_height: {self.sample_height}, sample_width: {self.sample_width}, sample_num_frames: {self.sample_num_frames}")
        print(f"before filtering, total: {cnt}")
        print(f"after filtering, total: {cnt_after_filter}")
        print(f"filtered out data:")
        print(f"cnt_too_long: {cnt_too_long}, cnt_too_short: {cnt_too_short}")
        print(f"cnt_too_fast: {cnt_too_fast}, cnt_too_slow: {cnt_too_slow}")
        print(f"cnt_no_cap: {cnt_no_cap}, cnt_no_resolution: {cnt_no_resolution}")
        print(f"cnt_aspect_mismatch: {cnt_aspect_mismatch}, cnt_res_too_small: {cnt_res_too_small}, cnt_fps_too_low: {cnt_fps_too_low}")
        print('-----------------------filtering result------------------------')

        return filtered_data_samples

    def filter_data_samples_dynamic_shape(self, data_samples=None):
        
        raise NotImplementedError

    def filter_data_samples(self, data_samples=None):
        filter_mode = "force_shape"
        if filter_mode == "force_shape":
            data_samples_after_filtering = self.filter_data_samples_force_shape(data_samples)
        elif filter_mode == "dynamic_shape":
            data_samples_after_filtering = self.filter_data_samples_dynamic_shape(data_samples)
        else:
            raise ValueError("Must specify either force_shape or dynamic_shape")

        return data_samples_after_filtering


class WanVideoProcessor(AbstractDataProcessor):
    def __init__(
        self, 
        video_layout_type='TCHW',
        sample_height=480,
        sample_width=832,
        sample_num_frames=49,
        train_fps=None,
        sample_stride=None,
        force_cut_video_from_start=False,
    ):
        super().__init__()
        self.video_reader_factory = VideoReader('decord')
        self.video_layout_type = video_layout_type

        self.sample_height = sample_height
        self.sample_width = sample_width
        self.sample_num_frames = sample_num_frames
        
        self.train_fps = train_fps
        self.sample_stride = sample_stride

        self.sample_mode = None
        if self.train_fps is not None:
            print(f"Using train_fps mode, train_fps: {self.train_fps}")
            self.sample_mode = "train_fps"
        elif self.sample_stride is not None:
            print(f"Using sample_stride mode, sample_stride: {self.sample_stride}")
            self.sample_mode = "sample_stride"
        else:
            raise ValueError("Must specify either train_fps or sample_stride")

        
        self.video_transforms = Compose(
            [
                CenterCropResizeVideo((self.sample_height, self.sample_width), interpolation_mode='bicubic', align_corners=False, antialias=True),
                ToTensorAfterResize(),
                AENorm()
            ]
        )

        print(f'video_transforms: \n {self.video_transforms}')

        self.temporal_sample = TemporalRandomCrop(self.sample_num_frames, force_cut_video_from_start)


    def read_one_sample(self, path, meta_info=None):
        with video_reader_contextmanager(self.video_reader_factory, path, self.video_layout_type, num_threads=1) as video_reader:
            fps = video_reader.get_video_fps() if video_reader.get_video_fps() > 0 else 30.0
            if abs(fps - meta_info.get('fps', fps)) > 0.1:
                raise ValueError(f"fps is not correct, fps: {fps}, path: {path}")
            start_frame_idx = meta_info.get('start_frame_idx', 0)
            orig_total_frames = video_reader.get_num_frames()
            if orig_total_frames < self.sample_num_frames:
                raise ValueError(f"num_frames of video is not enough, orig_total_frames: {orig_total_frames}, sample_total_frames: {sample_total_frames}, path: {path}")
            sample_total_frames = meta_info.get('num_frames', orig_total_frames)
            s_x, e_x, s_y, e_y = meta_info.get('crop', [None, None, None, None])

            if self.sample_mode == "train_fps":
                # resample in case high fps, such as 50/60/90/144 -> train_fps(e.g, 24)
                frame_interval = (
                    1.0 if abs(fps - self.train_fps) < 0.1 else fps / self.train_fps
                )
            elif self.sample_mode == "sample_stride":
                frame_interval = self.sample_stride

            # some special video should be set to a different number
            frame_indices = np.arange(start_frame_idx, start_frame_idx + sample_total_frames, frame_interval).astype(int)
            #  too long video will be temporal-crop randomly
            if len(frame_indices) >= self.sample_num_frames:
                begin_index, end_index = self.temporal_sample(len(frame_indices))
                frame_indices = frame_indices[begin_index:end_index]
            else:
                raise ValueError(f"we need to sample {self.sample_num_frames} frames, but got {len(frame_indices)} frames, path: {path}")

            sample = video_reader.get_batch(frame_indices) # (T, C, H, W)
            if s_y is not None:
                sample = sample[:, :, s_y: e_y, s_x: e_x]
        return sample
        
    def process_one_sample(self, sample):
        # import ipdb; ipdb.set_trace()
        sample = self.video_transforms(sample)
        sample = sample.permute(1, 0, 2, 3) # (T, C, H, W) -> (C, T, H, W)
        if sample.shape[1:] != (self.sample_num_frames, self.sample_height, self.sample_width):
            raise ValueError(f"sample shape is not correct, sample.shape: {sample.shape}")
        return sample 

    def __call__(self, video_path, meta_info={}, need_processing=True):
        sample = self.read_one_sample(video_path, meta_info)
        first_frame = sample[0]
        if need_processing:
            sample = self.process_one_sample(sample)
        return first_frame, sample


class WanImageProcessor(AbstractDataProcessor):
    def __init__(
        self,
        image_layout_type='CHW',
        sample_height=480,
        sample_width=832,
    ):
        super().__init__()
        self.image_reader = ImageReader
        self.image_layout_type = image_layout_type
        
        self.sample_height = sample_height
        self.sample_width = sample_width

        self.image_transforms = Compose(
            [
                CenterCropResizeVideo((self.sample_height, self.sample_width), interpolation_mode='bicubic', align_corners=False, antialias=True),
                ToTensorAfterResize(),
                AENorm()
            ]
        )
        
        print(f'image_transforms: \n {self.image_transforms}')

    def read_one_sample(self, path, meta_info=None):
        sample = self.image_reader(path, self.image_layout_type).load_image()
        return sample

    def process_one_sample(self, sample):
        if self.image_layout_type == "CHW":
            # add T dimension
            sample = sample.unsqueeze(1)
            sample = self.image_transforms(sample)
        elif self.image_layout_type == "HWC":
            sample = sample.unsqueeze(0)
            sample = sample.permute(0, 3, 1, 2) # (1 H W C) -> (1 C H W)
            sample = self.image_transforms(sample)
            sample = sample.permute(0, 2, 3, 1) # (1 C H W) -> (1 H W C)
        return sample

    def __call__(self, image_path, meta_info=None, need_processing=True):
        sample = self.read_one_sample(image_path, meta_info)
        if need_processing:
            sample = self.process_one_sample(sample)
        return sample
        