# Instructions for Running the MHI/MEI Activity Classification Pipeline

This guide provides clear, step-by-step instructions for generating models and running activity classification demos using the provided Python scripts. 

---

## Codebase Overview

The table below summarizes the purpose of each major file in the project:

| File Name        | Description |
|------------------|-------------|
| **mhi_core.py**  | **Implements the core logic for computing MHI, MEI, extracting Hu moment features, applying thresholding, and executing classifier-related operations.** |
| **mhi_main.py**  | **The main entry point of the pipeline. Handles argument parsing, orchestrates the workflow, performs the 75/15/15 train/validation/test split, and runs model training + evaluation.** |
| **mhi_demo.py**  | **Uses a trained model to generate predicted action labels, as well as MHI/MEI images and videos for visualization or demonstration.** |
| **mhi_import.py** | Handles downloading and organizing the KTH human action dataset from online sources. |
| **mhi_metrics.py** | Provides utilities for generating histograms, confusion matrices, and other evaluation plots for classifier comparison. |
| **mhi_pipeline.py** | Defines the pipeline steps: preprocessing, dataset loading, feature extraction, model training, evaluation, and saving/loading trained models. |
| **mhi_processor.py** | Performs preprocessing operations such as video validation, frame extraction, resizing, gray-scaling, and preparing data for MHI/MEI computation. |



## Environment Setup

All required dependencies can be installed using the provided **Conda environment file** `cv_proj.yml`.

### **Create and Activate the Environment**

```bash
conda env create -f cv_proj.yml
conda activate cv_proj
```

### **Included Dependencies (via cv_proj.yml)**

The environment installs:

* Python 3.8.8
* NumPy 1.21.2
* SciPy 1.7.1
* h5py 3.5.0
* setuptools 58.0.4
* PyTorch 1.10
* TorchVision 0.11.1
* TensorFlow-GPU 2.5.0
* Cython 0.29.24
* scikit-learn 1.0.1
* OpenCV 4.5.4.58
* scikit-image
* skorch 0.11.0
* matplotlib
* joblib


---

## Demo Videos Structure

The demo videos are available in the below folder. Execute the python commands given in the next section to visualize the output.  

```
demo_inputs/
    clipXXXXX_Y_Z.avi
```


---

# Running Classification Demos

Use the **mhi_demo.py** script to visualize MHI/MEI templates and run classification on sample action videos.

You may optionally specify `--frames` to visualize intermediate MHI states but by default 30,60,90 frames are taken in the codebase.

---

## Waving Example

```bash
python3 mhi_demo.py --video demo_inputs/clip00494_4_2.avi --model models/svm-model 
```

---

## Running Example

```bash
python3 mhi_demo.py --video demo_inputs/clip00240_2_0.avi --model models/svm-model --frames 5 10 15 
```

---
---

## Jogging Example

```bash
python3 mhi_demo.py --video demo_inputs/clip00100_1_0.avi --model models/svm-model --frames 30 35 40  
```

---

## Boxing Example

```bash
python3 mhi_demo.py --video demo_inputs/clip00308_3_0.avi --model models/svm-model 
```

---

## Handclapping Example

```bash
python3 mhi_demo.py --video demo_inputs/clip00508_5_0.avi --model models/svm-model --frames 7 14 21 
```


## Walking Example

```bash
python3 mhi_demo.py --video demo_inputs/clip00006_0_2.avi --model models/svm-model 
```

---

# Notes

* If you want any other model to be used please replace it with mlp-model, knn-model or svn-model-basic to test the same videos. 
* The `--frames` argument is optional and shows intermediate MHI snapshots.
* The report covers above videos and its frames. 
* Once you run the above commands you should be able to see all the generated videos/images as below 

```
demo_outputs/
    clipXXXXX_Y_Z_labeled.mp4  
    clipXXXXX_Y_Z_mhi.mp4
    clipXXXXX_Y_Z__frames_<FRAME>0001_<FRAME>0010_<FRAME>0015_actual_binary_all.png
    clipXXXXX_Y_Z_mhi.png
    clipXXXXX_Y_Z_mei.png

```

| **Video Name** | **Description** |
|----------|-----------|
| clipXXXXX_Y_Z_labeled.mp4    | **Video with Action Label**   |
| clipXXXXX_Y_Z_mhi.mp4   | **MHI Video**  |
| clipXXXXX_Y_Z__frames_<FRAME>0001_<FRAME>0010_<FRAME>0015_actual_binary_all.png    | **MHI Screenshot Captured in a Frame**   |
| clipXXXXX_Y_Z_mhi.png    |  **MHI Image**  |
| clipXXXXX_Y_Z_mei.png   | **Cumulative MEI**   |

---


##  Model Training Dataset Download

Inorder to download the entire dataset from activeloop website. Please execute the below command. It will take a minimum of **25 mins** to completely download the videos from the website. Please use the below python script as the **video filename** structure should be in a particular format 

```
python3 mhi_import.py
```

The expected dataset folder structure is given below which should be in your folder structure:

```
kth_local_avi/
    clipXXXXX_Y_Z.avi
```

This folder must contain all KTH action video `.avi` files.



# Model Generation

Models can be trained for SVM, KNN, and MLP classifiers using **Hu moment features** extracted from MHI/MEI templates.

All generated models are saved in the `models/` directory.

Generated model files appear under:

```
models/
    svm-model
    knn-model
    mlp-model-hu
```

---

## 1a. Train SVM Model with Additional Features(Hu Features)

```bash
python3 mhi_main.py --root kth_local_avi --output svm-model --classifier svm --mhi_feature_type hu --add_features True
```
## 1b. Train SVM Model without Additional Features(Hu Features)

```bash
python3 mhi_main.py --root kth_local_avi --output svm-model --classifier svm --mhi_feature_type hu
```


## 2. Train KNN Model (Hu Features)

```bash
python3 mhi_main.py --root kth_local_avi --output knn-model --classifier knn --mhi_feature_type hu
```

## 3. Train MLP Model (Hu Features)

```bash
python3 mhi_main.py --root kth_local_avi --output mlp-model-hu --classifier mlp --mhi_feature_type hu
```



---

