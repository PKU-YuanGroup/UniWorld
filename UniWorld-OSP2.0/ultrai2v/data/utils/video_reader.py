import os
import re
import gc
import html
import math
import copy
import random
import urllib.parse as ul
from fractions import Fraction
from collections import Counter
from typing import Any, Dict, Optional, Tuple, Union, Sequence, Literal, Type, List
from abc import ABC, abstractmethod
from pathlib import Path
import typing

try:
    import decord
except ImportError:
    print("Failed to import decord module.")

import av
import torch
import torchvision
import numpy as np
from torchvision import get_video_backend
from torchvision.io.video import (
    _align_audio_frames,
    _check_av_available,
    _read_from_stream,
    _video_opt,
)
from contextlib import contextmanager
import ctypes


VideoReaderBackends = Literal["decord", "torchvision", "av"]
VideoLayoutType = Literal["THWC", "TCHW"]
VideoArrayType = Literal["numpy", "torch"]
VideoMaxFrames = 1024

def is_video_file(file_path):
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm', '.mpeg', '.mpg', '.3gp'}
    file_extension = os.path.splitext(file_path)[1].lower()
    return file_extension in video_extensions

@contextmanager
def video_reader_contextmanager(reader_cls, *args, **kwargs):
    """
    A context manager wrapper for safe use of a video reader class instance.

    Args:
        reader_cls: Video reader class (e.g., decord.VideoReader or custom class)
        *args, **kwargs: Arguments to initialize the reader

    Yields:
        An instance of the video reader.
    """
    vr = None
    try:
        vr = reader_cls(*args, **kwargs)
        yield vr
    finally:
        # Explicit close/release if available
        if hasattr(vr, 'close'):
            try:
                vr.close()
            except Exception:
                pass  # Suppress close error
        del vr
        gc.collect()

class VideoReader:
    def __init__(self, backend: VideoReaderBackends = "decord"):
        backend = backend.lower()
        if backend == "decord":
            self.reader_cls = DecordVideo
        elif backend == "torchvision":
            self.reader_cls = TorchvisionVideo
        elif backend == "av":
            self.reader_cls = AvVideo
        else:
            raise ValueError(f"Unsupported video reader backend: {backend}")
        
    def __call__(self, *args, **kwargs):
        return self.reader_cls(*args, **kwargs)

class Video(ABC):
    """
    Abstract base class defining the common video processing interface
    """
    def __init__(self, video_path: str, layout: VideoLayoutType = "TCHW", array_type: VideoArrayType = "torch", **kwargs):
        """
        Initialize video source
        
        Args:
            video_path: String path to video file
            layout (VideoLayoutType): 
                Desired tensor layout format. Options:
                - "TCHW": Time, Channel, Height, Width (default)
                - "THWC": Time, Height, Width, Channel
            array_type (VideoArrayType):
                Target array container type. Options:
                - "torch": PyTorch tensors (default)
                - "numpy": NumPy ndarrays
        """
        self.video_path = Path(video_path)
        self.layout = layout
        self.array_type = array_type
        self.vframes = None
        self._validate_params()
        self._load_data(**kwargs)

    def _validate_params(self):
        """param validation"""
        if not is_video_file(str(self.video_path)):
            raise ValueError(f"Invalid video type: {self.video_path}")
        if self.layout not in typing.get_args(VideoLayoutType):
            raise ValueError(f"Invalid video layout type: {self.layout}")
        if self.array_type not in typing.get_args(VideoArrayType):
            raise ValueError(f"Invalid video array type: {self.array_type}")

    def _validate_load_params(self, frame_indices: Union[np.ndarray, torch.Tensor]):
        if len(frame_indices) > VideoMaxFrames:
            raise ValueError(f"Frames has to be less than or equal to {VideoMaxFrames}")

    @abstractmethod
    def _load_data(self, **kwargs):
        """
        Abstract method for implementation-specific data loading
        
        Raises:
            VideoLoadError: If video file cannot be processed
        """

    @abstractmethod
    def _get_batch(self, frame_indices: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        Retrieve video data batch in implementation-specific format
        """

    def get_batch(self, frame_indices: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        self._validate_load_params(frame_indices)
        return self._get_batch(frame_indices)

    @abstractmethod
    def get_video_fps(self) -> float:
        """
        Retrieve frames per second (FPS) information
        
        Returns:
            Frame rate as floating point value
        """
    
    @abstractmethod
    def get_num_frames(self) -> int:
        """
        Get the number of frames
        """

    def close(self):
        if self.vframes is not None:
            del self.vframes
            self.vframes = None
            gc.collect()

            try:
                ctypes.CDLL("libc.so.6").malloc_trim(0)
            except:
                pass
    
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

class DecordVideo(Video):
    """
    Decord-based video decoder implementation with shared decoder instance
    """

    def _load_data(self, **kwargs):
        decord.bridge.set_bridge("torch")
        self.vframes = decord.VideoReader(str(self.video_path), **kwargs)

    def _get_batch(self, frame_indices: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        video_data = self.vframes.get_batch(frame_indices)
        
        if self.layout == "TCHW":
            # THWC -> TCHW,  [T: temporal, C: channel, H: height, W: width]
            video_data = video_data.permute(0, 3, 1, 2)
        
        return video_data
    
    def get_video_fps(self) -> float:
        return self.vframes.get_avg_fps() if self.vframes.get_avg_fps() > 0 else 30.0
    
    def get_num_frames(self) -> int:
        return len(self.vframes)


    
class TorchvisionVideo(Video):
    """Torchvision-based video reader implementation"""
    def _load_data(self, **kwargs):
        self.vframes, _, self.metadata = torchvision.io.read_video(
            str(self.video_path), pts_unit="sec", output_format=self.layout
        )

    def _get_batch(self, frame_indices: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        video_data = self.vframes[frame_indices]
        
        if self.array_type == "numpy":
            video_data = video_data.numpy()
        return video_data

    def get_video_fps(self) -> float:
        return self.metadata.get("video_fps")

    def get_num_frames(self) -> int:
        return len(self.vframes)


class AvVideo(Video):
    """AV-based video reader implementation"""
    def _load_data(self, **kwargs):
        self.vframes, _, self.metadata = read_video_av(
            str(self.video_path), pts_unit="sec", output_format=self.layout
        )
    
    def _get_batch(self, frame_indices: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        video_data = self.vframes[frame_indices]
        
        if self.array_type == "numpy":
            video_data = video_data.numpy()
        return video_data

    def get_video_fps(self) -> float:
        return self.metadata.get("video_fps")
    
    def get_num_frames(self) -> int:
        return len(self.vframes)


def read_video_av(
        filename: str,
        start_pts: Union[float, Fraction] = 0,
        end_pts: Optional[Union[float, Fraction]] = None,
        pts_unit: str = "pts",
        output_format: str = "THWC",
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """
    Reads a video from a file, returning both the video frames and the audio frames

    Args:
        filename (str): path to the video file
        start_pts (int if pts_unit = "pts", float / Fraction if pts_unit = "sec", optional):
            The start presentation time of the video
        end_pts (int if pts_unit = "pts", float / Fraction if pts_unit = "sec", optional):
            The end presentation time
        pts_unit (str, optional): unit in which start_pts and end_pts values will be interpreted,
            either "pts" or "sec". Defaults to "pts".
        output_format (str, optional): The format of the output video tensors. Can be either "THWC" (default) or "TCHW".

    Returns:
        vframes (Tensor[T, H, W, C] or Tensor[T, C, H, W]): the `T` video frames
        aframes (Tensor[K, L]): the audio frames, where `K` is the number of channels and `L` is the number of points
        info (Dict): metadata for the video and audio. Can contain the fields video_fps (float) and audio_fps (int)
    """
    output_format = output_format.upper()
    if output_format not in ("THWC", "TCHW"):
        raise ValueError(f"output_format should be either 'THWC' or 'TCHW', got {output_format}.")

    if not os.path.exists(filename):
        raise RuntimeError(f"File not found: {filename}")

    if get_video_backend() != "pyav":
        vframes, aframes, info = _video_opt._read_video(filename, start_pts, end_pts, pts_unit)
    else:
        _check_av_available()

        if end_pts is None:
            end_pts = float("inf")

        if end_pts < start_pts:
            raise ValueError(
                f"end_pts should be larger than start_pts, got start_pts={start_pts} and end_pts={end_pts}"
            )

        info = {}
        video_frames = []
        audio_frames = []
        audio_timebase = _video_opt.default_timebase

        container = av.open(filename, metadata_errors="ignore")
        try:
            if container.streams.audio:
                audio_timebase = container.streams.audio[0].time_base
            if container.streams.video:
                video_frames = _read_from_stream(
                    container,
                    start_pts,
                    end_pts,
                    pts_unit,
                    container.streams.video[0],
                    {"video": 0},
                )
                video_fps = container.streams.video[0].average_rate
                # guard against potentially corrupted files
                if video_fps is not None:
                    info["video_fps"] = float(video_fps)

            if container.streams.audio:
                audio_frames = _read_from_stream(
                    container,
                    start_pts,
                    end_pts,
                    pts_unit,
                    container.streams.audio[0],
                    {"audio": 0},
                )
                info["audio_fps"] = container.streams.audio[0].rate
        except av.AVError as ex:
            raise ex
        finally:
            container.close()
            del container
            # NOTE: manually garbage collect to close pyav threads
            gc.collect()

        vframes_list = [frame.to_rgb().to_ndarray() for frame in video_frames]
        aframes_list = [frame.to_ndarray() for frame in audio_frames]

        if vframes_list:
            vframes = torch.as_tensor(np.stack(vframes_list))
        else:
            vframes = torch.empty((0, 1, 1, 3), dtype=torch.uint8)

        if aframes_list:
            aframes = np.concatenate(aframes_list, 1)
            aframes = torch.as_tensor(aframes)
            if pts_unit == "sec":
                start_pts = int(math.floor(start_pts * (1 / audio_timebase)))
                if end_pts != float("inf"):
                    end_pts = int(math.ceil(end_pts * (1 / audio_timebase)))
            aframes = _align_audio_frames(aframes, audio_frames, start_pts, end_pts)
        else:
            aframes = torch.empty((1, 0), dtype=torch.float32)

    if output_format == "TCHW":
        # [T,H,W,C] --> [T,C,H,W]
        vframes = vframes.permute(0, 3, 1, 2)

    return vframes, aframes, info