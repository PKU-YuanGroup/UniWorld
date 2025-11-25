import os
import math
import torch
import imageio

from ultrai2v.data.utils.image_reader import ImageReader, is_image_file

def save_videos(videos, start_index, save_path, fps):
    os.makedirs(save_path, exist_ok=True)
    if isinstance(videos, (list, tuple)) or videos.ndim == 5:  # [b, t, h, w, c]
        for i, video in enumerate(videos):
            save_path_i = os.path.join(save_path, f"video_{start_index + i}.mp4")
            imageio.mimwrite(save_path_i, video, fps=fps, codec='libx264', quality=8)
    elif videos.ndim == 4:
        save_path = os.path.join(save_path, f"video_{start_index}.mp4")
        imageio.mimwrite(save_path, videos, fps=fps, codec='libx264', quality=8)
    else:
        raise ValueError("The video must be in either [b, t, h, w, c] or [t, h, w, c] format.")
    
def save_video_with_name(video, name, save_path, fps):
    os.makedirs(save_path, exist_ok=True)
    save_path = os.path.join(save_path, f"{name}.mp4")
    imageio.mimwrite(save_path, video, fps=fps, codec='libx264', quality=8)

def save_video_grid(videos, save_path, fps, nrow=None):
    b, t, h, w, c = videos.shape
    if nrow is None:
        nrow = math.ceil(math.sqrt(b))
    ncol = math.ceil(b / nrow)
    padding = 1
    video_grid = torch.zeros(
        (
            t,
            (padding + h) * nrow + padding,
            (padding + w) * ncol + padding,
            c
        ),
        dtype=torch.uint8
    )

    for i in range(b):
        r = i // ncol
        c = i % ncol
        start_r = (padding + h) * r
        start_c = (padding + w) * c
        video_grid[:, start_r: start_r + h, start_c: start_c + w] = videos[i]

    imageio.mimwrite(os.path.join(save_path, "video_grid.mp4"), video_grid, fps=fps, codec='libx264', quality=8)

def load_prompts(prompt):
    if os.path.exists(prompt):
        with open(prompt, "r") as f:
            lines = f.readlines()
            if len(lines) > 100:
                print("The file has more than 100 lines of prompts, we can only proceed the first 100")
                lines = lines[:100]
            prompts = [line.strip() for line in lines]
        return prompts
    else:
        return [prompt]


def load_images(image=None, dual_image=False, layout="CHW", array_type="torch"):
    if image is None:
        print("The input image is None, execute text to video task")
        return None

    if os.path.exists(image):
        if is_image_file(image) and not dual_image:
            return [ImageReader(image, layout=layout, array_type=array_type).load_image()]
        else:
            with open(image, "r") as f:
                lines = f.readlines()
                if len(lines) > 100:
                    print("The file has more than 100 lines of images, we can only process the first 100")
                    lines = lines[:100]
                if dual_image:
                    images = []
                    for line in lines:
                        paths = line.strip().split(',')
                        if len(paths) != 2:
                            raise ValueError(f"Each line must contain two paths separated by commas (,). Current line:{line}")
                        image1 = ImageReader(paths[0], layout=layout, array_type=array_type).load_image()
                        image2 = ImageReader(paths[1], layout=layout, array_type=array_type).load_image()
                        images.append([image1, image2])
                else:
                    images = [ImageReader(line.strip(), layout=layout, array_type=array_type).load_image() for line in lines]
            return images
    else:
        raise FileNotFoundError(f"The image path {image} does not exist")