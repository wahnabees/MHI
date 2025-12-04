import cv2
import os
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from collections import Counter

import joblib

from mhi_core import (
    MotionFeatureConfig,
    get_mhi_features_from_video
)
from mhi_metrics import compute_and_plot_confusion_matrix


def load_motion_pipeline(load_path):
    package=joblib.load(load_path)
    return package




# Building the Dataset

def build_dataset(video_paths,labels,cfg,max_per_video=80):
    if len(video_paths)!=len(labels):
        raise ValueError("video_paths and labels must have the same length")

    X_list=[]
    y_list=[]

    for path,label in zip(video_paths,labels):
        print(f"Processing video for dataset: {path}")
        feats=get_mhi_features_from_video(path,cfg)
        if feats.shape[0]==0:
            print(f"  [warning] No features extracted for {path}, skipping.")
            continue

        n=feats.shape[0]
        if n>max_per_video:
            idx=np.random.choice(n,size=max_per_video,replace=False)
            feats=feats[idx]

        X_list.append(feats)
        y_list.extend([label]*feats.shape[0])

    if not X_list:
        raise RuntimeError("No features extracted from any video; please check dataset paths and config.")

    X=np.vstack(X_list).astype(np.float32)
    y=np.asarray(y_list,dtype=np.int32)
    return X,y



# 3. Test-time evaluation

def evaluate_on_test_set(model,cfg,test_paths,test_labels):
    if len(test_paths)==0:
        return 0.0,np.array([]),np.array([])

    y_true=[]
    y_pred=[]

    for path,label in zip(test_paths,test_labels):
        print(f"Testing video: {path}")
        feats=get_mhi_features_from_video(path,cfg)
        if feats.shape[0]==0:
            print(f"  [warning] No features for {path}; skipping from eval.")
            continue

        frame_preds=model.predict(feats)
        counts=Counter(frame_preds.tolist())
        maj_label=counts.most_common(1)[0][0]

        y_true.append(label)
        y_pred.append(maj_label)

    if not y_true:
        print("[warning] No test predictions made; returning 0 accuracy.")
        return 0.0,np.array([]),np.array([])

    y_true_arr=np.asarray(y_true,dtype=np.int32)
    y_pred_arr=np.asarray(y_pred,dtype=np.int32)
    acc=float((y_true_arr==y_pred_arr).mean())

    return acc,y_true_arr,y_pred_arr




# 7. Saving / Loading the Pipeline

def save_motion_pipeline(grid,cfg,save_path):
    models_dir="models"
    os.makedirs(models_dir,exist_ok=True)

    final_save_path=os.path.join(models_dir,os.path.basename(save_path))

    package={"config":cfg,"model":grid.best_estimator_}

    joblib.dump(package,final_save_path)
    print(f"Saved motion pipeline to {final_save_path}")
