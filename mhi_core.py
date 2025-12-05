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
from sklearn.neural_network import MLPClassifier
from skimage.feature import hog
from collections import Counter,deque

import joblib



@dataclass
class MotionFeatureConfig:
    tau:int=40
    thresh:int=20
    resize_width:Optional[int]=220
    blur_sigma:Optional[float]=1.0
    k_size:int=9
    sigma:float=1.0
    frames_reset_freq:int=40
    add_features:bool=False



# MHI / MEI Computation

def compute_mhi_sequence(frames:np.ndarray,cfg:MotionFeatureConfig)->np.ndarray:
    T,H,W=frames.shape
    mhi=np.zeros((H,W),dtype=np.float32)
    mhi_seq=np.zeros((T,H,W),dtype=np.float32)

    prev=cv2.GaussianBlur(frames[0].astype(np.float32),(cfg.k_size,cfg.k_size),cfg.sigma)
    frames_counter=0

    for t in range(1,T):
        curr=cv2.GaussianBlur(frames[t].astype(np.float32),(cfg.k_size,cfg.k_size),cfg.sigma)
        diff=cv2.absdiff(curr,prev)
        bin_mask=(diff>cfg.thresh).astype(np.uint8)

        kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
        bin_mask=cv2.morphologyEx(bin_mask,cv2.MORPH_GRADIENT,kernel).astype(bool)

        mhi[mhi==0]=1
        mhi-=1
        mhi[mhi<0]=0

        if np.count_nonzero(bin_mask)!=0:
            mhi[bin_mask]=cfg.tau

        mhi_seq[t]=mhi
        prev=curr

        frames_counter+=1
        if frames_counter>=cfg.frames_reset_freq:
            mhi[:]=0
            frames_counter=0

    return mhi_seq




# 4. Feature Extraction

def _extract_hog_features_from_mhi(mhi_img:np.ndarray)->np.ndarray:
    m=cv2.normalize(mhi_img,None,0,255,cv2.NORM_MINMAX)
    m=m.astype(np.uint8)

    target_width=128
    h,w=m.shape[:2]
    if w!=target_width:
        target_height=int(h*target_width/float(w))
        m=cv2.resize(m,(target_width,target_height))

    hog_vec=hog(
        m,
        orientations=9,
        pixels_per_cell=(16,16),
        cells_per_block=(2,2),
        block_norm="L2-Hys",
        feature_vector=True,
    )

    return hog_vec.astype(np.float32)




def _extract_hu_features_from_mhi(mhi_img:np.ndarray)->np.ndarray:
    m=cv2.normalize(mhi_img,None,0,255,cv2.NORM_MINMAX)
    m=m.astype(np.float32)

    if np.all(m==0):
        return np.zeros(7,dtype=np.float32)

    h,w=m.shape
    ys,xs=np.mgrid[0:h,0:w]

    m00=np.sum(m)
    if m00==0:
        return np.zeros(7,dtype=np.float32)

    m10=np.sum(xs*m)
    m01=np.sum(ys*m)

    x_bar=m10/m00
    y_bar=m01/m00

    dx=xs-x_bar
    dy=ys-y_bar

    def central_moment(p:int,q:int)->float:
        return np.sum((dx**p)*(dy**q)*m)

    mu20=central_moment(2,0)
    mu02=central_moment(0,2)
    mu11=central_moment(1,1)
    mu30=central_moment(3,0)
    mu03=central_moment(0,3)
    mu21=central_moment(2,1)
    mu12=central_moment(1,2)

    mu00=m00

    def eta(mu_pq:float,p:int,q:int)->float:
        gamma=1.0+(p+q)/2.0
        return mu_pq/(mu00**gamma+1e-12)

    eta20=eta(mu20,2,0)
    eta02=eta(mu02,0,2)
    eta11=eta(mu11,1,1)
    eta30=eta(mu30,3,0)
    eta03=eta(mu03,0,3)
    eta21=eta(mu21,2,1)
    eta12=eta(mu12,1,2)

    phi1=eta20+eta02
    phi2=(eta20-eta02)**2+4.0*(eta11**2)
    phi3=(eta30-3.0*eta12)**2+(3.0*eta21-eta03)**2
    phi4=(eta30+eta12)**2+(eta21+eta03)**2
    phi5=((eta30-3.0*eta12)*(eta30+eta12)*((eta30+eta12)**2-3.0*(eta21+eta03)**2)+
          (3.0*eta21-eta03)*(eta21+eta03)*(3.0*(eta30+eta12)**2-(eta21+eta03)**2))
    phi6=((eta20-eta02)*((eta30+eta12)**2-(eta21+eta03)**2)+
          4.0*eta11*(eta30+eta12)*(eta21+eta03))
    phi7=((3.0*eta21-eta03)*(eta30+eta12)*((eta30+eta12)**2-3.0*(eta21+eta03)**2)-
          (eta30-3.0*eta12)*(eta21+eta03)*(3.0*(eta30+eta12)**2-(eta21+eta03)**2))

    hu=np.array([phi1,phi2,phi3,phi4,phi5,phi6,phi7],dtype=np.float64)

    eps=1e-7
    hu_log=-np.sign(hu)*np.log10(np.abs(hu)+eps)

    return hu_log.astype(np.float32)



def get_mhi_features_from_video(video_path:str,cfg:MotionFeatureConfig)->np.ndarray:
    cap=cv2.VideoCapture(video_path)
    ok,frame=cap.read()
    if not ok:
        cap.release()
        print(f"[get_mhi_features_from_video] Could not read first frame from {video_path}")
        return np.empty((0,0),dtype=np.float32)

    def preprocess(f):
        if f.ndim==3:
            f=cv2.cvtColor(f,cv2.COLOR_BGR2GRAY)
        f=f.astype(np.float32)
        k=int(getattr(cfg,"k_size",7))
        sigma=float(getattr(cfg,"sigma",0.0))
        f=cv2.GaussianBlur(f,(k,k),sigma)
        return f

    prev=preprocess(frame)
    h,w=prev.shape
    mhi=np.zeros((h,w),dtype=np.float32)
    mei=np.zeros((h,w),dtype=np.uint8)

    frames_reset_freq=int(getattr(cfg,"frames_reset_freq",20))
    tau=float(getattr(cfg,"tau",240))
    theta=float(getattr(cfg,"thresh",15))

    frames_counter=0
    features=[]
    frame_idx=0
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))

    use_extra=bool(getattr(cfg,"add_features",False))
    prev_mhi_for_change=None
    motion_history=deque(maxlen=10)

    while True:
        ok,frame=cap.read()
        if not ok:
            break

        curr=preprocess(frame)
        diff=cv2.absdiff(curr,prev)

        bin_mask=(diff>theta).astype(np.uint8)
        bin_mask=cv2.morphologyEx(bin_mask,cv2.MORPH_GRADIENT,kernel)
        motion_present=np.count_nonzero(bin_mask)>0

        mei=np.logical_or(mei,bin_mask).astype(np.uint8)

        mhi[mhi==0]=1
        mhi-=1
        mhi[mhi<0]=0
        if motion_present:
            mhi[bin_mask.astype(bool)]=tau

        if np.count_nonzero(mhi)>0:
            feature_type=getattr(cfg,"mhi_feature_type","hu")

            if feature_type=="hu":
                feats_mhi=_extract_hu_features_from_mhi(mhi)
                feats_mei=_extract_hu_features_from_mhi(mei.astype(np.float32))
            elif feature_type=="hog":
                feats_mhi=_extract_hog_features_from_mhi(mhi)
                feats_mei=_extract_hog_features_from_mhi(mei.astype(np.float32))
            else:
                raise ValueError(f"Unknown mhi_feature_type:{feature_type}")

            base_feats=np.concatenate([feats_mhi,feats_mei],axis=0)

            if use_extra:
                H,W=bin_mask.shape

                motion_pixels=float(np.count_nonzero(bin_mask))
                motion_history.append(motion_pixels)
                motion_std=float(np.std(motion_history)) if len(motion_history)>1 else 0.0

                if prev_mhi_for_change is not None:
                    mhi_diff=np.abs(mhi-prev_mhi_for_change)
                    avg_temporal_change=float(mhi_diff.mean())
                else:
                    avg_temporal_change=0.0
                prev_mhi_for_change=mhi.copy()

                lower=bin_mask[H//2:,:]
                motion_pixels_lower=float(np.count_nonzero(lower))

                x1,x2=W//3,2*W//3
                central=bin_mask[:,x1:x2]
                motion_pixels_central=float(np.count_nonzero(central))

                left=bin_mask[:,:W//3]
                right=bin_mask[:,2*W//3:]
                motion_left=float(np.count_nonzero(left))
                motion_right=float(np.count_nonzero(right))
                side_motion=motion_left+motion_right
                central_ratio=motion_pixels_central/(side_motion+1e-6)

                extra_feats=np.array(
                    [
                        motion_pixels,
                        avg_temporal_change,
                        motion_pixels_lower,
                        motion_std,
                        motion_pixels_central,
                        central_ratio,
                    ],
                    dtype=np.float32,
                )

                feats=np.concatenate([base_feats,extra_feats],axis=0)
            else:
                feats=base_feats

            features.append(feats)

        frame_idx+=1
        prev=curr
        frames_counter+=1

        if frames_counter>=frames_reset_freq:
            mhi.fill(0.0)
            frames_counter=0
            prev_mhi_for_change=None
            motion_history.clear()

    cap.release()

    if not features:
        return np.empty((0,0),dtype=np.float32)

    return np.stack(features,axis=0).astype(np.float32)




# 6. Training (CV for k or C)

def train_motion_classifier(X,y,classifier_type="svm",random_state=0):
    if classifier_type=="svm":
        clf=SVC(kernel="rbf",random_state=random_state)
        param_grid={
            "clf__C":[1,10,100,1000],
            "clf__gamma":["scale",0.01,0.001],
            # "clf__C":[0.1, 1, 10, 100, 1000],
            # "clf__gamma":["scale",0.01,0.001],
        }
    elif classifier_type=="knn":
        clf=KNeighborsClassifier()
        param_grid={
            "clf__n_neighbors":[1,3,5],
            "clf__weights":["uniform","distance"]
        }
    elif classifier_type=="mlp":
        clf=MLPClassifier(
            hidden_layer_sizes=(128,),            
            max_iter=400,
            random_state=random_state,
            early_stopping=True,
            n_iter_no_change=10,            
        )
        param_grid={
            "clf__hidden_layer_sizes":[
                (32,),
                (64,),
                (128,),
                (64,32),
                (32,32,32),
                (64,64,32),
            ],
            "clf__alpha":[1e-5,1e-4,1e-3,1e-2],
            "clf__activation":["relu","tanh"],
        }
    else:
        raise ValueError("classifier_type must be 'svm' or 'knn'")

    pipe=Pipeline([
        ("scaler",StandardScaler()),
        ("clf",clf),
    ])

    cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=random_state)

    grid=GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1,
        verbose=1
    )

    grid.fit(X,y)
    return grid



























