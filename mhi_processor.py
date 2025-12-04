import os
from pathlib import Path
from typing import List, Dict, Tuple

import cv2
import numpy as np

VALID_EXT = (".avi", ".mp4", ".mpeg", ".mpg", ".mov")

# Optional: for debugging / printing
ACTION_ID_TO_NAME = {
    0: "walking",
    1: "jogging",
    2: "running",
    3: "boxing",
    4: "waving",
    5: "clapping"
}
ACTION_NAME_TO_ID: Dict[str, int] = {v: k for k, v in ACTION_ID_TO_NAME.items()}


def is_video_file(path):
    lower=path.lower()
    for ext in VALID_EXT:
        if lower.endswith(ext):
            return True
    return False



def collect_kth_videos(root_dir):
    video_paths=[]
    for dirpath,_,filenames in os.walk(root_dir):
        for fname in filenames:
            full=os.path.join(dirpath,fname)
            if is_video_file(full):
                video_paths.append(full)
    video_paths.sort()
    return video_paths

def infer_action_label_from_path(path):
    filename=os.path.basename(path)
    name_no_ext,_=os.path.splitext(filename)
    parts=name_no_ext.split("_")

    if len(parts)<2:
        raise ValueError(f"Unexpected filename format: {filename}")

    try:
        action_id=int(parts[1])
    except ValueError:
        raise ValueError(f"Cannot parse action id from: {filename}")

    if action_id not in ACTION_ID_TO_NAME:
        raise ValueError(f"Unknown action id {action_id} in {filename}")

    return ACTION_ID_TO_NAME[action_id]


def build_label_map(action_names):
    print("Inferring action IDs from file paths ...")
    unique_actions=sorted(set(action_names))

    label_map={}
    for name in unique_actions:
        if name not in ACTION_NAME_TO_ID:
            raise ValueError(f"Unknown action name: {name}")
        label_map[name]=ACTION_NAME_TO_ID[name]

    print("Example mapping (id -> name):")
    for action_id,name in sorted(ACTION_ID_TO_NAME.items()):
        print(f"  {action_id} -> {name}")

    return label_map


def load_video_frames(video_path,cfg):
    cap=cv2.VideoCapture(video_path)
    frames=[]
    idx=0

    while True:
        ret,frame=cap.read()
        if not ret:
            break

        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

        if cfg.resize_width is not None:
            h,w=gray.shape
            if w>cfg.resize_width:
                scale=cfg.resize_width/float(w)
                new_size=(cfg.resize_width,int(h*scale))
                gray=cv2.resize(gray,new_size,interpolation=cv2.INTER_AREA)

        if cfg.blur_sigma is not None and cfg.blur_sigma>0:
            ksize=int(2*round(3*cfg.blur_sigma)+1)
            gray=cv2.GaussianBlur(gray,(ksize,ksize),cfg.blur_sigma)

        frames.append(gray.astype(np.float32))
        idx+=1

    cap.release()
    return frames


