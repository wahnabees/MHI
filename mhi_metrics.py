from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix

def plot_split_histogram_overall(train_count,val_count,test_count,save_path):
    os.makedirs(os.path.dirname(save_path),exist_ok=True)

    labels=["Train","Val","Test"]
    counts=[train_count,val_count,test_count]

    plt.figure(figsize=(6,4))
    bars=plt.bar(labels,counts)

    plt.ylabel("Number of videos")
    plt.title("Dataset Split: Train / Val / Test (Counts) [75%/15%/15%]")

    for bar,count in zip(bars,counts):
        height=bar.get_height()
        plt.text(bar.get_x()+bar.get_width()/2.0,height,str(count),ha="center",va="bottom")

    plt.tight_layout()
    plt.savefig(save_path,dpi=150)
    plt.close()





def compute_and_plot_confusion_matrix(y_true,y_pred,class_names=None,save_path=None,title="Confusion Matrix (%)"):
    cm=confusion_matrix(y_true,y_pred)
    cm=cm.astype(float)

    row_sums=cm.sum(axis=1,keepdims=True)
    cm_percent=np.divide(cm,row_sums,out=np.zeros_like(cm),where=row_sums!=0)*100

    if class_names is None:
        class_names=[str(i) for i in range(cm.shape[0])]

    plt.figure(figsize=(8,6))
    plt.imshow(cm_percent,interpolation="nearest")
    plt.title(title)
    plt.colorbar()

    tick_marks=np.arange(len(class_names))
    plt.xticks(tick_marks,class_names,rotation=45)
    plt.yticks(tick_marks,class_names)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            text_val=f"{cm_percent[i,j]:.1f}%"
            plt.text(j,i,text_val,horizontalalignment="center",color="white" if cm_percent[i,j]>50 else "black")

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path),exist_ok=True)
        plt.savefig(save_path)
        print(f"Percentage confusion matrix saved to: {save_path}")

    plt.close()
    return cm_percent


