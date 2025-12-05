import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from typing import List, Tuple, Dict, Any, Optional
from sklearn.metrics import classification_report



from mhi_pipeline import (
    MotionFeatureConfig,
    build_dataset,
    evaluate_on_test_set,
    save_motion_pipeline      
)

from mhi_core import (
    MotionFeatureConfig,
    train_motion_classifier
)

from mhi_processor import (    
    collect_kth_videos,
    infer_action_label_from_path,    
    build_label_map,    
)

from mhi_metrics import (
    plot_split_histogram_overall,
    compute_and_plot_confusion_matrix
)



ACTION_ID_TO_NAME = {
    0: "walking",
    1: "jogging",
    2: "running",
    3: "boxing",
    4: "waving",
    5: "clapping"
}
ACTION_NAME_TO_ID: Dict[str, int] = {v: k for k, v in ACTION_ID_TO_NAME.items()}



def main():
    parser=argparse.ArgumentParser(description="Train MHI/MEI-based action classifier on KTH dataset.")
    parser.add_argument("--root",type=str,required=True,help="Root directory of KTH dataset (containing videos).")
    parser.add_argument("--output",type=str,default="svm-model",help="Path to save trained pipeline (joblib file).")
    parser.add_argument("--classifier",type=str,choices=["svm","knn","mlp"],default="svm",help="Classifier type: 'svm', 'knn', or 'mlp' (default: svm).")

    parser.add_argument("--tau",type=int,default=40,help="MHI duration in frames (tau). Default: 40.")
    parser.add_argument("--thresh",type=int,default=20,help="Frame difference threshold for motion. Default: 20.")
    parser.add_argument("--resize_width",type=int,default=220,help="Max width for resizing frames. Use <=0 to disable. Default: 220.")
    parser.add_argument("--blur_sigma",type=float,default=1.0,help="Gaussian blur sigma. Use <=0 to disable. Default: 1.0.")
    parser.add_argument("--add_features",type=bool,default=False,help="Additional Features apart from 14.")


    parser.add_argument("--mhi_feature_type",type=str,choices=["hu","hog"],default="hu",help="Type of features to extract from MHI/MEI: 'hu' for Hu moments, 'hog' for HOG descriptors.")
    args=parser.parse_args()

    
    all_video_paths=collect_kth_videos(args.root)
   
    action_names=[]
    for p in all_video_paths:
        action_names.append(infer_action_label_from_path(p))
    label_map=build_label_map(action_names)
    all_labels=[label_map[a] for a in action_names]
    

    
    train_paths,temp_paths,train_labels,temp_labels=train_test_split(all_video_paths,all_labels,test_size=0.30,random_state=0,stratify=all_labels)

    val_paths,test_paths,val_labels,test_labels=train_test_split(temp_paths,temp_labels,test_size=0.50,random_state=0,stratify=temp_labels)

    print("Train videos: {}, Val videos: {}, Test videos: {}".format(len(train_paths),len(val_paths),len(test_paths)))

    plot_split_histogram_overall(train_count=len(train_paths),val_count=len(val_paths),test_count=len(test_paths),save_path="analysis/trainingVSprediction.png")

    resize_width=args.resize_width if args.resize_width>0 else None
    blur_sigma=args.blur_sigma if args.blur_sigma>0 else None
    add_features=args.add_features
    cfg=MotionFeatureConfig(tau=args.tau,thresh=args.thresh,resize_width=resize_width,blur_sigma=blur_sigma,add_features=add_features)
    cfg.mhi_feature_type=args.mhi_feature_type

    print("Building training dataset")
    X_train,y_train=build_dataset(train_paths,train_labels,cfg)
    print("Training feature matrix shape:",X_train.shape)

    print("Training classifier")
    grid=train_motion_classifier(X_train,y_train,classifier_type=args.classifier)
    
    print("Best CV accuracy (train):",grid.best_score_)
    print("Best params:",grid.best_params_)

    print("Saving trained pipeline to {} ...".format(args.output))
    save_motion_pipeline(grid,cfg,args.output)

    best_model=grid.best_estimator_

    print("Evaluating on Training set ")
    train_frame_acc=best_model.score(X_train,y_train)
    print("Training frame-level accuracy:",train_frame_acc)

    if len(val_paths)>0:
        print("Building validation dataset ")
        X_val,y_val=build_dataset(val_paths,val_labels,cfg)
        print("Validation feature matrix shape:",X_val.shape)

        val_frame_acc=best_model.score(X_val,y_val)
        print("Validation frame-level accuracy:",val_frame_acc)

        print("Evaluating on Validation set")
        val_video_acc,y_true_v,y_pred_v=evaluate_on_test_set(best_model,cfg,val_paths,val_labels)
        print(classification_report(y_true_v,y_pred_v))
        print("Validation video-level accuracy : {:.4f}".format(val_video_acc))

        class_names=[ACTION_ID_TO_NAME[i] for i in sorted(ACTION_ID_TO_NAME.keys())]

        compute_and_plot_confusion_matrix(
            y_true=y_true_v,
            y_pred=y_pred_v,
            class_names=class_names,
            save_path=f"analysis/cm_val_{os.path.splitext(os.path.basename(args.classifier))[0]}_{args.mhi_feature_type}.png",
            title="Video-level Confusion Matrix (Validation Set)",
        )
    else:
        print("No validation samples; skipping val evaluation.")

    if len(test_paths)>0:
        print("Building Test Dataset")
        X_test,y_test=build_dataset(test_paths,test_labels,cfg)
        print("Test feature matrix shape:",X_test.shape)

        test_frame_acc=best_model.score(X_test,y_test)
        print("Test frame-level accuracy :",test_frame_acc)

        print("Evaluating on test set ")
        test_video_acc,y_true_t,y_pred_t=evaluate_on_test_set(best_model,cfg,test_paths,test_labels)
        print("Test video-level accuracy: {:.4f}".format(test_video_acc))

        compute_and_plot_confusion_matrix(
            y_true=y_true_t,
            y_pred=y_pred_t,
            class_names=class_names,
            save_path=f"analysis/cm_test_{os.path.splitext(os.path.basename(args.classifier))[0]}_{args.mhi_feature_type}.png",
            title="Video-level Confusion Matrix (Test Set)",
        )
        print(classification_report(y_true_t,y_pred_t))
    else:
        print("No test samples; skipping test evaluation.")


    

if __name__ == "__main__":
    main()
