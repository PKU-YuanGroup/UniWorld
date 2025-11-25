import json
import glob
import shutil
import os
from tqdm import tqdm
from einops import rearrange

from ultrai2v.data.utils.utils import LMDBReader, LMDBWriter

sample_height = 480
sample_width = 832
vbench_image_dir = "/work/share/projects/gyy/resi2v/ResI2V_MM/vbench_26-15"
vbench_json_path = "/work/share/projects/gyy/resi2v/ResI2V_MM/vbench2_i2v_full_info_refine_2.json"
with open(vbench_json_path, 'r') as f:
    org_meta_info_list = json.load(f)
name2idx = {}
for index, item in enumerate(org_meta_info_list):
    name2idx[item['image_name']] = index

vbench_eval_data_save_dir = "/work/share/projects/gyy/resi2v/ResI2V_MM/vbench_eval_data"
os.makedirs(vbench_eval_data_save_dir, exist_ok=True)
vbench_images = glob.glob(f'{vbench_image_dir}/*.jpg')
vbench_images = sorted(vbench_images)

image_num = len(vbench_images)
new_meta_info_list = []
for index, image in enumerate(vbench_images):
    meta_info = {"sample_height": sample_height, "sample_width": sample_width}
    base_name = os.path.basename(image)
    org_meta_info = org_meta_info_list[name2idx[base_name]]
    meta_info["cap"] = org_meta_info["refined_caption"]
    meta_info["old_name"] = org_meta_info["image_name"]
    new_path = os.path.join(vbench_eval_data_save_dir, f'{index:06d}.jpg')
    meta_info["path"] = new_path
    shutil.copy(image, new_path)
    new_meta_info_list.append(meta_info)

with open(os.path.join(vbench_eval_data_save_dir, 'meta_info.json'), 'w') as f:
    json.dump(new_meta_info_list, f, indent=4)

writer = LMDBWriter()
writer.save_filtered_data_samples(new_meta_info_list, vbench_eval_data_save_dir)
test_reader = LMDBReader(vbench_eval_data_save_dir) 
print(test_reader.getitem(0))