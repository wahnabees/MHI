import cv2
import os
import numpy as np
from collections import Counter
import argparse

from mhi_pipeline import load_motion_pipeline
from mhi_processor import load_video_frames
from mhi_main import ACTION_ID_TO_NAME

from mhi_core import (
    compute_mhi_sequence,
    get_mhi_features_from_video

)


def normalize_int(frame):
    f=frame.astype(np.float32)
    min_v,max_v=f.min(),f.max()
    if max_v>min_v:
        f=(f-min_v)/(max_v-min_v)
    else:
        f=np.zeros_like(f)
    return (f*255.0).astype(np.uint8)


def label_video_action(video_path:str,pipeline_path:str,output_path:str=None)->tuple[str,str]:
    # 1) Load trained pipeline
    package=load_motion_pipeline(pipeline_path)
    cfg=package["config"]
    model=package["model"]

    # 2) Extract per-frame features & predictions
    feats=get_mhi_features_from_video(video_path,cfg)  
    frame_preds=model.predict(feats)                   

    counts=Counter(frame_preds.tolist())
    pred_id=counts.most_common(1)[0][0]
    video_action_name=ACTION_ID_TO_NAME.get(int(pred_id),f"class {pred_id}")
    print(f"[label_video_action] Predicted action (majority): {video_action_name} (id={pred_id})")

    frame_labels=[ACTION_ID_TO_NAME.get(int(pid),f"class {pid}") for pid in frame_preds]

    # 3) Build and save MHI video
    base_name=os.path.splitext(os.path.basename(video_path))[0]
    labeled_out_path=os.path.join(output_dir,f"{base_name}_labeled.mp4")
    mhi_out_path=os.path.join(output_dir,f"{base_name}_mhi.mp4")

    frames_list=load_video_frames(video_path,cfg)
    frames=np.stack(frames_list,axis=0)  # (T,H,W)
    mhi_seq=compute_mhi_sequence(frames,cfg)

    cap_tmp=cv2.VideoCapture(video_path)
    fps=cap_tmp.get(cv2.CAP_PROP_FPS)
    if fps is None or fps<=1.0:
        fps=25.0
    cap_tmp.release()

    h,w=mhi_seq.shape[1],mhi_seq.shape[2]
    fourcc_mhi=cv2.VideoWriter_fourcc(*"mp4v")
    mhi_writer=cv2.VideoWriter(mhi_out_path,fourcc_mhi,fps,(w,h),True)

    for t in range(mhi_seq.shape[0]):
        gray_u8=normalize_int(mhi_seq[t])
        bgr=cv2.cvtColor(gray_u8,cv2.COLOR_GRAY2BGR)
        mhi_writer.write(bgr)
    mhi_writer.release()
    print(f"[label_video_action] MHI video saved to: {mhi_out_path}")

    # 4) Show & save labeled RGB video
    cap=cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open input video: {video_path}")

    fps=cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps<=1.0:
        fps=25.0

    orig_w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc=cv2.VideoWriter_fourcc(*"mp4v")
    writer=cv2.VideoWriter(labeled_out_path,fourcc,fps,(orig_w,orig_h),True)

    print(f"[label_video_action] Writing labeled video to {labeled_out_path} ({orig_w}x{orig_h}, {fps:.1f} FPS)")

    frame_idx=0
    frames_written=0
    window_name="Labeled Action Video"

    while True:
        ret,frame=cap.read()
        if not ret:
            break

        if frame_idx<len(frame_labels):
            label_text=f"Action: {frame_labels[frame_idx]}"
        else:
            label_text=f"Action: {frame_labels[-1]}"

        cv2.putText(frame,label_text,(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),1,cv2.LINE_AA)

        cv2.imshow(window_name,frame)

        writer.write(frame)
        frames_written+=1
        frame_idx+=1

        if cv2.waitKey(30)&0xFF==27:
            print("[label_video_action] ESC pressed, stopping early.")
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    
    print(f"[label_video_action] Labeled video saved to: {labeled_out_path}")

    return labeled_out_path,mhi_out_path



def save_actual_binary_mhi_mei(video_path,cfg,frame_indices,out_dir):
    frame_indices=set(frame_indices)

    cap=cv2.VideoCapture(video_path)
    ok,frame=cap.read()
    if not ok:
        print(f"[error] Could not read from video: {video_path}")
        cap.release()
        return

    def preprocess(f):
        if f.ndim==3:
            f=cv2.cvtColor(f,cv2.COLOR_BGR2GRAY)
        f=f.astype(np.float32)
        k=int(getattr(cfg,"k_size",7))
        sigma=float(getattr(cfg,"sigma",0.0))
        f=cv2.GaussianBlur(f,(k,k),sigma)
        return f

    prev_orig=frame.copy()
    prev=preprocess(frame)

    h,w=prev.shape
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    theta=float(getattr(cfg,"thresh",15))
    tau=float(getattr(cfg,"tau",40))

    os.makedirs(out_dir,exist_ok=True)

    mhi=np.zeros((h,w),dtype=np.float32)
    mei=np.zeros((h,w),dtype=np.uint8)

    stacked_list=[]

    frame_idx=1
    while True:
        ok,frame=cap.read()
        if not ok:
            break

        frame_idx+=1
        curr_orig=frame.copy()
        curr=preprocess(frame)

        diff=cv2.absdiff(curr,prev)

        bin_mask=(diff>theta).astype(np.uint8)
        bin_mask=cv2.morphologyEx(bin_mask,cv2.MORPH_OPEN,kernel)
        bin_mask=cv2.morphologyEx(bin_mask,cv2.MORPH_CLOSE,kernel)

        mhi=np.where(bin_mask>0,tau,np.maximum(mhi-1.0,0.0))
        mei=np.logical_or(mei>0,bin_mask>0).astype(np.uint8)

        if frame_idx in frame_indices:
            actual_resized=cv2.resize(curr_orig,(w,h))
            bin_vis=(bin_mask>0).astype(np.uint8)*255
            bin_vis_color=cv2.cvtColor(bin_vis,cv2.COLOR_GRAY2BGR)
            stacked=np.vstack([actual_resized,bin_vis_color])
            stacked_list.append(stacked)

        prev=curr
        prev_orig=curr_orig

    cap.release()

    if stacked_list:
        big_image=np.hstack(stacked_list)
        base=os.path.splitext(os.path.basename(video_path))[0]
        sorted_indices=sorted(frame_indices)
        idx_str="_".join(f"{i:04d}" for i in sorted_indices)
        out_name=f"{base}_frames_{idx_str}_actual_binary_all.png"
        out_path=os.path.join(out_dir,out_name)
        cv2.imwrite(out_path,big_image)
        print(f"[info] Saved combined image: {out_path}")
    else:
        print("[info] No selected frames were found in the video; nothing to save.")




def save_cumulative_mhi_mei(video_path,cfg,out_dir):
    os.makedirs(out_dir,exist_ok=True)

    frames_list=load_video_frames(video_path,cfg)
    if len(frames_list)==0:
        print(f"[error] No frames loaded from {video_path}")
        return

    frames=np.stack(frames_list,axis=0)
    mhi_seq=compute_mhi_sequence(frames,cfg)

    mhi_last=mhi_seq[-1].astype(np.float32)
    if mhi_last.max()>0:
        mhi_vis=(mhi_last/mhi_last.max()*255.0).astype(np.uint8)
    else:
        mhi_vis=mhi_last.astype(np.uint8)

    mhi_vis_bgr=cv2.cvtColor(mhi_vis,cv2.COLOR_GRAY2BGR)

    base=os.path.splitext(os.path.basename(video_path))[0]
    mhi_path=os.path.join(out_dir,f"{base}_MHI.png")

    cv2.imwrite(mhi_path,mhi_vis_bgr)
    print(f"[info] Saved cumulative MHI to {mhi_path}")

    mei_mask=(mhi_seq>0).any(axis=0).astype(np.uint8)
    mei_vis=mei_mask*255
    mei_vis_bgr=cv2.cvtColor(mei_vis,cv2.COLOR_GRAY2BGR)

    mei_path=os.path.join(out_dir,f"{base}_MEI.png")
    cv2.imwrite(mei_path,mei_vis_bgr)
    print(f"[info] Saved cumulative MEI to {mei_path}")



def main():
    parser=argparse.ArgumentParser(
        description="Classify a video using trained MHI model and generate:\n"
                    "1) labeled RGB video\n"
                    "2) MHI video"
    )

    parser.add_argument(
        "--video",
        required=True,
        type=str,
        help="Path to the input video to classify."
    )

    parser.add_argument(
        "--model",
        required=True,
        type=str,
        help="Path to the trained motion model (.joblib file)."
    )

    parser.add_argument(
        "--out",
        type=str,        
        help="Optional output path for labeled RGB video. Default=<video>_labeled.mp4"
    )

    parser.add_argument(
        "--frames",
        nargs="+",
        type=int,
        default=[30,60,90],   # override → keep
        help="Frame indices to visualize MHIs for."
    )

    args=parser.parse_args()

    output_dir="demo_outputs"
    os.makedirs(output_dir,exist_ok=True)

    labeled_path,mhi_path=label_video_action(
        video_path=args.video,
        pipeline_path=args.model,
        output_path=os.path.join(output_dir,"labeled.mp4")
    )

    package=load_motion_pipeline(args.model)
    cfg=package["config"]
    model=package["model"]

    frames_to_visualize=args.frames

    save_actual_binary_mhi_mei(
        video_path=args.video,
        cfg=cfg,
        frame_indices=frames_to_visualize,
        out_dir=output_dir,
    )

    save_cumulative_mhi_mei(
        video_path=args.video,
        cfg=cfg,
        out_dir=output_dir,
    )

    print("\n=== Output Files ===")
    print("Labeled RGB Video :",labeled_path)
    print("MHI Video         :",mhi_path)
    print("Saved MHI/MEI frames:",frames_to_visualize)



if __name__ == "__main__":
    output_dir = "demo_outputs"
    os.makedirs(output_dir, exist_ok=True)
    main()



