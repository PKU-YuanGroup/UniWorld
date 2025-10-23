from typing import List
import numpy as np
import cv2
from PIL import Image, ImageChops
import torch
import torch.nn.functional as F

def concat_images_row(images: List[Image.Image], bg_color=(0, 0, 0)) -> Image.Image:
    """
    将多张 PIL 图像按行（横向）拼接在一起。
    
    Args:
        images: 要拼接的图像列表。
        bg_color: 背景颜色，默认黑色；如果图像有透明通道，可以用 (0,0,0,0)。
    
    Returns:
        一张横向拼接后的新图。
    """
    if not images:
        raise ValueError("images 列表不能为空")
    
    # 统一 mode，如果有透明通道，则用 RGBA，否则用 RGB
    modes = {img.mode for img in images}
    mode = "RGBA" if any(m.endswith("A") for m in modes) else "RGB"

    # 计算拼接后画布的尺寸
    widths, heights = zip(*(img.size for img in images))
    total_width = sum(widths)
    max_height = max(heights)

    # 新建画布
    canvas = Image.new(mode, (total_width, max_height), bg_color)

    # 依次粘贴
    x_offset = 0
    for img in images:
        # 如果 img 的 mode 不同，先转换
        if img.mode != mode:
            img = img.convert(mode)
        canvas.paste(img, (x_offset, 0), img if mode=="RGBA" else None)
        x_offset += img.width

    return canvas


def downsample_mask_pytorch(pil_mask: Image.Image, factor: int) -> Image.Image:
    """
    用 PyTorch 的 max_pool2d 对二值 mask 进行下采样，保留块内任何白色。
    
    Args:
        pil_mask: mode='1' 或 'L' 的二值 PIL 图（0/255）。
        factor: 下采样倍数（stride 和 kernel 大小都设为这个值）。
    
    Returns:
        下采样后的二值 PIL Image（mode='1'）。
    """
    # 转成 0/1 float tensor，形状 [1,1,H,W]
    arr = np.array(pil_mask.convert('L'), dtype=np.uint8)
    tensor = torch.from_numpy(arr).float().div_(255.0).unsqueeze(0).unsqueeze(0)
    
    # 用 max_pool2d，下采样
    pooled = F.max_pool2d(tensor, kernel_size=factor, stride=factor)
    
    # 恢复成 0/255，并转回 PIL
    out = (pooled.squeeze(0).squeeze(0) > 0).to(torch.uint8).mul_(255).cpu().numpy()
    return Image.fromarray(out, mode='L').convert('1')

def create_all_white_like(pil_img: Image.Image) -> Image.Image:
    """
    给定一个 PIL 图像，返回一张同样大小的全白二值图（mode='1'）。
    """
    w, h = pil_img.size
    white_array = np.ones((h, w), dtype=np.uint8) * 255  # 注意 shape 是 (H, W)
    return Image.fromarray(white_array, mode='L').convert('1')

def union_masks_np(masks: List[Image.Image]) -> Image.Image:
    """
    接受一个 PIL.Image 列表（mode='1' 或 'L' 的二值图），
    返回它们的并集。
    """
    if not masks:
        raise ValueError("输入的 masks 列表不能为空")

    # 把每张图都转成 0/1 numpy 数组
    bin_arrays = []
    for m in masks:
        arr = np.array(m.convert('L'), dtype=np.uint8)
        bin_arr = (arr > 127).astype(np.bool_)
        bin_arrays.append(bin_arr)

    # 做逐像素逻辑或
    union_bool = np.logical_or.reduce(bin_arrays)

    # 恢复成 0/255 uint8
    union_arr = union_bool.astype(np.uint8) * 255

    # 转回 PIL（二值）
    return Image.fromarray(union_arr, mode='L').convert('1')

def intersect_masks_np(masks: List[Image.Image]) -> Image.Image:
    """
    接受一个 PIL.Image 列表（mode='1' 或 'L' 的二值图），
    返回它们的交集。
    """
    if not masks:
        raise ValueError("输入的 masks 列表不能为空")

    # 把每张图都转成 0/1 numpy 数组
    bin_arrays = []
    for m in masks:
        arr = np.array(m.convert('L'), dtype=np.uint8)
        bin_arr = (arr > 127).astype(np.bool_)
        bin_arrays.append(bin_arr)

    # 做逐像素逻辑或
    intersect_bool = np.logical_and.reduce(bin_arrays)

    # 恢复成 0/255 uint8
    intersect_arr = intersect_bool.astype(np.uint8) * 255

    # 转回 PIL（二值）
    return Image.fromarray(intersect_arr, mode='L').convert('1')

def close_small_holes(pil_mask, kernel_size=5):
    """
    用闭运算填平小的黑点。
    kernel_size: 结构元尺寸，越大能填的洞越大，通常取奇数。
    """
    # 1. 转成 0/255 二值
    mask = np.array(pil_mask.convert('L'))
    _, bin_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
    # 2. 定义结构元
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    # 3. 闭运算
    closed = cv2.morphologyEx(bin_mask, cv2.MORPH_CLOSE, kernel)
    
    return Image.fromarray(closed)


def get_mask(src_image, tgt_image, threshold=1):
    """
    差异大的地方（差值大）当成前景（白），否则当背景（黑）
    """
    diff = ImageChops.difference(src_image, tgt_image)
    diff_gray = diff.convert("L")
    mask = diff_gray.point(lambda x: 255 if x >= threshold else 0).convert("1")
    return mask

def filter_small_components(pil_mask, area_threshold=0.10):
    """
    删除小于 area_threshold （默认 10%）的连通白色区域。
    pil_mask: PIL.Image，mode='L' 或 '1'（0/255 二值图）
    area_threshold: 阈值，相对于整张图面积的比例
    返回: 处理后的 PIL.Image
    """
    # 1. 转为二值 NumPy 数组（0,255）
    mask = np.array(pil_mask.convert('L'))
    # 确保是 0/255
    _, bin_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
    # 2. 连通组件标记（4 或 8 邻域都可以）
    num_labels, labels = cv2.connectedComponents(bin_mask, connectivity=8)
    
    h, w = bin_mask.shape
    total_area = h * w
    total_area = np.count_nonzero(bin_mask)
    
    # 3. 遍历各个连通块
    output = np.zeros_like(bin_mask)
    for lbl in range(1, num_labels):  # 0 是背景
        # 取出该连通块
        comp_mask = (labels == lbl)
        comp_area = comp_mask.sum()
        # 面积比
        if comp_area >= area_threshold * total_area:
            # 保留
            output[comp_mask] = 255
    
    # 4. 转回 PIL
    return Image.fromarray(output)



def is_binary_255(t: torch.Tensor) -> bool:
    """
    判断给定的 tensor 是否只包含 0 和 255 两种值。
    """
    unique_vals = torch.unique(t)
    return torch.equal(unique_vals, torch.tensor([0], dtype=t.dtype)) or \
           torch.equal(unique_vals, torch.tensor([255], dtype=t.dtype)) or \
           torch.equal(unique_vals, torch.tensor([0, 255], dtype=t.dtype))

def get_weight(mask_u_ds, weight_type='log'):
    mask_u_ds_tensor = torch.from_numpy(np.array(mask_u_ds)).float()
    assert is_binary_255(mask_u_ds_tensor), "is_binary_255(mask_u_ds_tensor)"
    mask_u_ds_tensor_bool = mask_u_ds_tensor.bool()
    x = mask_u_ds_tensor_bool.numel() / mask_u_ds_tensor_bool.sum()
    if weight_type == 'log':
        weight = torch.log2(x) + 1
    elif weight_type == 'exp':
        weight = 2 ** (x**0.5 - 1)
    else:
        raise NotImplementedError(f'Support log | exp, but found {weight_type}')
    weight = torch.round(weight, decimals=6)
    assert weight >= 1, \
        f"weight >= 1 but {weight}, {mask_u_ds_tensor_bool.shape}, mask_u_ds_tensor_bool.numel(): {mask_u_ds_tensor_bool.numel()}, mask_u_ds_tensor_bool.sum(): {mask_u_ds_tensor_bool.sum()}"
    mask_u_ds_tensor[mask_u_ds_tensor==255] = weight
    mask_u_ds_tensor[mask_u_ds_tensor==0] = 1.0
    return mask_u_ds_tensor.unsqueeze(0)  # h w -> 1 h w

def get_weight_mask(pil_pixel_values, prompt=None, weight_type='log', need_weight='true'):
    # area_threshold = 1/64
    area_threshold = 0.001
    # base_kernel_size_factor = (5 / 448) ** 2
    # if len(pil_pixel_values) > 0:
    #     w, h = pil_pixel_values[-1].size
    #     kernel_size = max(int((base_kernel_size_factor * h * w) ** 0.5), 3)
    # else:
    kernel_size = 5

    if need_weight.lower() == 'false':
        mask_intersect = create_all_white_like(pil_pixel_values[-1])
        mask_intersect_ds = downsample_mask_pytorch(mask_intersect, factor=8)  # factor is downsample ratio of vae
        mask_intersect_ds = close_small_holes(mask_intersect_ds, kernel_size=kernel_size)
        weight = get_weight(mask_intersect_ds, weight_type)
        return mask_intersect_ds, weight

    filtered_masks = []
    for ii, j in enumerate(pil_pixel_values[:-1]):
        # each reference image will compare with target image to get mask
        mask = get_mask(j, pil_pixel_values[-1], threshold=18)
        # fill small holes
        fill_mask = close_small_holes(mask, kernel_size=kernel_size)
        # del small components
        filtered_mask = filter_small_components(fill_mask, area_threshold=0.3)
        # filtered_mask = fill_mask
        filtered_masks.append(filtered_mask)
    if len(filtered_masks) == 0:
        # t2i task do not have reference image
        assert len(pil_pixel_values) == 1, "len(pil_pixel_values) == 1"
        mask_intersect = create_all_white_like(pil_pixel_values[-1])
    else:
        mask_intersect = intersect_masks_np(filtered_masks)
    # while area / total area muse greater than 1/16 (just a threshold)
    mask_intersect_area_ratio = np.array(mask_intersect).astype(np.float32).sum() / np.prod(np.array(mask_intersect).shape)
    # print(mask_intersect_area_ratio)
    if mask_intersect_area_ratio < area_threshold:
        if mask_intersect_area_ratio == 0.0:
            # mask_intersect_area_ratio == 0 mean reconstruct data in stage 1
            assert len(pil_pixel_values) == 2, "len(pil_pixel_values) == 2"
            mask_intersect = create_all_white_like(pil_pixel_values[-1])
        else:
            # concat_images_row(pil_pixel_values + [mask_intersect], bg_color=(255,255,255)).show()
            raise ValueError(f'TOO SMALL mask_intersect_area_ratio: {mask_intersect_area_ratio}, prompt: {prompt}')
    mask_intersect_ds = downsample_mask_pytorch(mask_intersect, factor=8)  # factor is downsample ratio of vae
    mask_intersect_ds = close_small_holes(mask_intersect_ds, kernel_size=kernel_size)
    weight = get_weight(mask_intersect_ds, weight_type)
    return mask_intersect_ds, weight



def get_weight_mask_test(pil_pixel_values, prompt=None, weight_type='log'):
    area_threshold = 1/64
    base_kernel_size_factor = (5 / 448) ** 2
    if len(pil_pixel_values) > 0:
        w, h = pil_pixel_values[-1].size
        kernel_size = max(int((base_kernel_size_factor * h * w) ** 0.5), 3)
    else:
        kernel_size = 5

    filtered_masks = []
    for ii, j in enumerate(pil_pixel_values[:-1]):
        # each reference image will compare with target image to get mask
        mask = get_mask(j, pil_pixel_values[-1], threshold=18)
        # fill small holes
        fill_mask = close_small_holes(mask, kernel_size=kernel_size)
        # del small components
        filtered_mask = filter_small_components(fill_mask, area_threshold=1/64)
        # filtered_mask = fill_mask
        filtered_masks.append(filtered_mask)
    if len(filtered_masks) == 0:
        # t2i task do not have reference image
        assert len(pil_pixel_values) == 1, "len(pil_pixel_values) == 1"
        mask_intersect = create_all_white_like(pil_pixel_values[-1])
    else:
        mask_intersect = intersect_masks_np(filtered_masks)
    # while area / total area muse greater than 1/16 (just a threshold)
    mask_intersect_area_ratio = np.array(mask_intersect).astype(np.float32).sum() / np.prod(np.array(mask_intersect).shape)
    return mask_intersect_area_ratio