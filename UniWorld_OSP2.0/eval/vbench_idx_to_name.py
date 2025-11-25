import os
import glob
import shutil

videos = glob.glob("/work/share/projects/gyy/UltraI2V/samples/flashi2v_14b_vbench_49x480x832/*.mp4")

idx_to_name_txt = "/work/share/projects/gyy/UltraI2V/eval/vbench_idx_to_name.txt"
save_dir = "/work/share/projects/gyy/UltraI2V/samples/flashi2v_14b_vbench_49x480x832_renamed5"
os.makedirs(save_dir, exist_ok=True)
with open(idx_to_name_txt, "r") as f:
    idx_to_name = {int(line.split('\t')[0]): line.split('\t')[1].strip().removesuffix('.jpg') for line in f}

for video in videos:
    video_name = os.path.basename(video)
    idx = int(video_name.split("_")[1])
    local_idx = int(video_name.split("_")[-1].removesuffix(".mp4"))
    if idx in idx_to_name:
        new_name = idx_to_name[idx] + f"-{local_idx}.mp4"
        new_path = shutil.copy(video, os.path.join(save_dir, new_name))
        print(f"Renaming {video} to {new_path}")
        # Uncomment the next line to actually rename the files
        # os.rename(video, new_path)
    else:
        print(f"Index {idx} not found in idx_to_name mapping.")