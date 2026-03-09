#!/usr/bin/env python
# coding: utf-8

# In[1]:


# !pip install spectral
# !pip install vit-keras
# !pip install tensorflow-addons
# !pip install keras_cv_attention_models
# # Loading Drive for Colab
# from google.colab import drive
# from google.colab import files
import os, argparse
import time
import warnings
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as sio
import seaborn as sns
import spectral
import spectral.io.envi as envi
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras
from keras.layers import (
    Conv3D,
    Conv2D,
    Conv1D,
    MaxPooling1D,
    Dense,
    Dropout,
    Flatten,
    Input,
    Reshape,
    MaxPooling2D,
    SeparableConv2D,
    MaxPooling3D,
    MaxPooling2D,
    GlobalAveragePooling2D,
    BatchNormalization,
    GlobalAveragePooling3D,
    concatenate,
    Reshape,
)
from keras_cv_attention_models import attention_layers
from tensorflow.keras import regularizers
from tensorflow.keras import layers
from keras.losses import categorical_crossentropy
from keras.models import Model, Sequential

# from keras.utils import np_utils
from tensorflow.keras.layers import Activation, Lambda, multiply
from tensorflow.keras.optimizers import Adam
from keras.optimizers import Adam
from tensorflow.keras.optimizers import legacy
from operator import truediv
from vit_keras import utils

# from keras.utils.vis_utils import plot_model
from keras.utils import to_categorical
import warnings

# Settings the warnings to be ignored
warnings.filterwarnings("ignore")


# Create an ArgumentParser object
parser = argparse.ArgumentParser(description="Script Description")

# Add arguments for each parameter
parser.add_argument(
    "--dataset",
    "-d",
    type=str,
    # choices=["HC", "HH", "Pingan", "Tangdaowan"],
    default="PU",
    help="Dataset identifier",
)
parser.add_argument("--patch_size", "-ps", type=int, default=14, help="Patch size")
parser.add_argument(
    "--dl_method", "-dlm", type=str, default="PCA", help="Dimension Reduction Method"
)
parser.add_argument("--train_ratio", "-tr", type=float, default=0.5, help="Train ratio")
parser.add_argument("--test_ratio", "-te", type=float, default=0.9, help="Test ratio")
parser.add_argument(
    "--validation_ratio", "-vr", type=float, default=0.5, help="Validation ratio"
)
parser.add_argument(
    "--number_of_bands",
    "-k",
    type=int,
    default=15,
    help="Number of bands principal components for dimensionality reduction",
)
parser.add_argument(
    "--epochs", "-e", type=int, default=50, help="Number of epochs for training"
)
parser.add_argument(
    "--batch_size", "-bs", type=int, default=56, help="Batch size for training"
)

# Parse the command-line arguments
args = parser.parse_args()


## Loading Hyperspectral Datasets
def LoadHSIData(method):
    data_path = os.path.join(os.getcwd(), "../datasets/")
    if method == "HH":
        ## http://rsidea.whu.edu.cn/resource_WHUHi_sharing.htm
        HSI = sio.loadmat(os.path.join(data_path, "WHU_Hi_HongHu.mat"))["WHU_Hi_HongHu"]
        GT = sio.loadmat(os.path.join(data_path, "WHU_Hi_HongHu_gt"))[
            "WHU_Hi_HongHu_gt"
        ]
        Num_Classes = 22
        target_names = [
            "Red roof",
            "Road",
            "Bare soil",
            "Cotton",
            "Cotton firewood",
            "Rape",
            "Chinese cabbage",
            "Pakchoi",
            "Cabbage",
            "Tuber mustard",
            "Brassica parachinensis",
            "Brassica chinensis",
            "Small Brassica chinensis",
            "Lactuca sativa",
            "Celtuce",
            "Film covered lettuce",
            "Romaine lettuce",
            "Carrot",
            "White radish",
            "Garlic sprout",
            "Broad bean",
            "Tree",
        ]
    elif method == "HC":
        ## http://rsidea.whu.edu.cn/resource_WHUHi_sharing.htm
        HSI = sio.loadmat(os.path.join(data_path, "WHU_Hi_HanChuan.mat"))[
            "WHU_Hi_HanChuan"
        ]
        GT = sio.loadmat(os.path.join(data_path, "WHU_Hi_HanChuan_gt"))[
            "WHU_Hi_HanChuan_gt"
        ]
        Num_Classes = 16
        target_names = [
            "Strawberry",
            "Cowpea",
            "Soybean",
            "Sorghum",
            "Water spinach",
            "Watermelon",
            "Greens",
            "Trees",
            "Grass",
            "Red roof",
            "Gray roof",
            "Plastic",
            "Bare soil",
            "Road",
            "Bright object",
            "Water",
        ]

    elif method == "Pingan":
        HSI = sio.loadmat(os.path.join(data_path, "QUH-Pingan.mat"))["Haigang"]
        GT = sio.loadmat(os.path.join(data_path, "QUH-Pingan_GT.mat"))["HaigangGT"]
        Num_Classes = 10
        target_names = [
            "Ship",
            "Seawater",
            "Trees",
            " Concrete structure building",
            "Floating pier",
            "Brick houses",
            "Steel houses",
            "Wharf construction land",
            "Car",
            "Road",
        ]
    elif method == "Tangdaowan":
        HSI = sio.loadmat(os.path.join(data_path, "QUH-Tangdaowan.mat"))["Tangdaowan"]
        GT = sio.loadmat(os.path.join(data_path, "QUH-Tangdaowan_GT.mat"))[
            "TangdaowanGT"
        ]
        Num_Classes = 18
        target_names = [
            "Rubber track",
            "Flaggingv",
            "Sandy",
            "Asphalt",
            "Boardwalk",
            "Rocky shallows",
            "Grassland",
            "Bulrush",
            "Gravel road",
            "Ligustrum vicaryi",
            "Coniferous pine",
            "Spiraea",
            "Bare soil",
            "Buxus sinica",
            "Photinia serrulata",
            "Populus",
            "Ulmus pumila L",
            "Seawater",
        ]
    elif method == 'Qingyun':
        HSI = sio.loadmat(os.path.join(data_path, 'QUH-Qingyun.mat'))['Chengqu']
        GT = sio.loadmat(os.path.join(data_path, 'QUH-Qingyun_GT.mat'))['ChengquGT']
        Num_Classes = 6     
        target_names = ["Trees", "Concrete building", "Car", "Ironhide building",
                      "Plastic playground", "Asphalt road"]
    elif method == 'SA':
        HSI = sio.loadmat(os.path.join(data_path, 'Salinas_corrected.mat'))['salinas_corrected']
        GT = sio.loadmat(os.path.join(data_path, 'Salinas_gt.mat'))['salinas_gt']
        Num_Classes = 16
        target_names = ['Weeds_1','Weeds_2','Fallow',
                        'Fallow_rough_plow','Fallow_smooth', 'Stubble','Celery',
                        'Grapes_untrained','Soil_vinyard_develop','Corn_Weeds',
                        'Lettuce_4wk','Lettuce_5wk','Lettuce_6wk',
                        'Lettuce_7wk', 'Vinyard_untrained','Vinyard_trellis']
        
    elif method == 'PU':
        HSI = sio.loadmat(os.path.join(data_path, 'PaviaU.mat'))['paviaU']
        GT = sio.loadmat(os.path.join(data_path, 'PaviaU_gt.mat'))['paviaU_gt']
        Num_Classes = 9       
        target_names = ['Asphalt','Meadows','Gravel','Trees', 'Painted','Soil','Bitumen',
                        'Bricks','Shadows']
        
    return HSI, GT, Num_Classes, target_names


# In[3]:


## Prediction Model
def PreModel(Actual, model):
    ## Validation Prediction Model
    prediction = model.predict(Actual)
    argmax_prediction = (
        np.argmax(prediction, axis=1) + 1
    )  # Add 1 to the argmax result to shift the class labels
    non_zero_classes = np.unique(
        argmax_prediction
    )  # Get the unique non-zero class labels
    # Remove the 0 class label if it exists in the non_zero_classes array
    if 0 in non_zero_classes:
        non_zero_classes = non_zero_classes[non_zero_classes != 0]
    # Filter out the non-zero classes from the argmax_prediction array
    Pre = argmax_prediction[np.isin(argmax_prediction, non_zero_classes)]
    return Pre


## Transform the Predicted Labels into Grouth Truths Shape
def Tranform_Labels(Sample_Matrix, Ind, Predicted):
    # Create a blank array to store the class labels for Validation Set
    labels = np.zeros_like(Sample_Matrix)
    # Replace the non-zero values in class_labels with filtered_prediction values
    for index, value in zip(Ind, Predicted):
        row = index // Sample_Matrix.shape[1]
        col = index % Sample_Matrix.shape[1]
        if row < Sample_Matrix.shape[0] and col < Sample_Matrix.shape[1]:
            labels[row, col] = value
    return labels


## Computing the Accuracies and Confusion Matrix for Disjoint Samples
def ClassificationReports(TeC, HSID, Te_Pre, target_names):
    classification = classification_report(
        np.argmax(TeC, axis=1) + 1, Te_Pre, target_names=target_names
    )
    oa = accuracy_score(np.argmax(TeC, axis=1) + 1, Te_Pre)
    oa = oa
    confusion = confusion_matrix(np.argmax(TeC, axis=1) + 1, Te_Pre)
    list_diag = np.diag(confusion)
    list_raw_sum = np.sum(confusion, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    aa = np.mean(each_acc)
    aa = aa
    kappa = cohen_kappa_score(np.argmax(TeC, axis=1) + 1, Te_Pre)
    kappa = kappa
    return classification, confusion, oa * 100, each_acc * 100, aa * 100, kappa * 100


## Computing the Accuracies and Confusion Matrix for Complete HSI
def ClassificationReports_HSI(TeC, HSID, Te_Pre, target_names):
    T_classification = classification_report(
        np.argmax(TeC, axis=1), np.argmax(Te_Pre, axis=1), target_names=target_names
    )
    T_oa = accuracy_score(np.argmax(TeC, axis=1), np.argmax(Te_Pre, axis=1))
    oa = T_oa
    T_confusion = confusion_matrix(np.argmax(TeC, axis=1), np.argmax(Te_Pre, axis=1))
    list_diag = np.diag(T_confusion)
    list_raw_sum = np.sum(T_confusion, axis=1)
    T_each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    T_aa = np.mean(T_each_acc)
    T_aa = T_aa
    T_kappa = cohen_kappa_score(np.argmax(TeC, axis=1), np.argmax(Te_Pre, axis=1))
    T_kappa = T_kappa
    return (
        T_classification,
        T_confusion,
        T_oa * 100,
        T_each_acc * 100,
        T_aa * 100,
        T_kappa * 100,
    )


## Plot Ground Truths for Complete Dataset as Trivial Cases
def GT_Plot(RDHSI, GT, model, WS, k, batch_size=256):
    # RDHSI = RDHSI.reshape(-1, WS, WS, k, 1)
    height, width = GT.shape
    ## Calculate the predicted Ground Truths
    outputs = np.zeros((height, width))
    count = 0
    ## Batch-wise prediction
    for i in range(0, len(RDHSI), batch_size):
        batch_RDHSI = RDHSI[i : i + batch_size]
        batch_pred = model.predict(batch_RDHSI)
        batch_pred_argmax = np.argmax(batch_pred, axis=1)
        batch_pred_argmax = batch_pred_argmax.reshape(-1)
        for j in range(len(batch_pred_argmax)):
            if count >= height * width:
                break
            target = int(GT[count // width, count % width])
            if target != 0:
                outputs[count // width, count % width] = (
                    batch_pred_argmax[j] + 1
                )  # Increment the value by 1
            count += 1
    return outputs


## Convert GT and Predicted HSI GTs into one-Hot encoding
def Convert(GT, flattened):
    height, width = GT.shape
    unique_values = np.unique(flattened)
    unique_values = np.delete(unique_values, np.where(unique_values == 0))
    AB = len(unique_values)
    if len(unique_values) > AB:
        unique_values = unique_values[:AB]
    GTA = np.zeros((height * width, len(unique_values)), dtype=int)
    T_labels_flat = flattened.flatten()
    for idx, cls in enumerate(unique_values):
        GTA[:, idx] = T_labels_flat == cls
    return GTA


## Writing Results in CSV files
def CSVResults_Complete(
    file_name,
    Va_classification,
    Va_Confusion,
    Tr_Time,
    Va_Time,
    Te_Time,
    T_Time,
    DL_Time,
    Va_Kappa,
    Va_OA,
    Va_AA,
    Va_Per_Class,
    Te_classification,
    Te_Confusion,
    Te_Kappa,
    Te_OA,
    Te_AA,
    Te_Per_Class,
    T_classification,
    T_Confusion,
    T_Kappa,
    T_OA,
    T_AA,
    T_Per_Class,
):
    Va_classification = str(Va_classification)
    Va_Confusion = str(Va_Confusion)
    Te_classification = str(Te_classification)
    Te_Confusion = str(Te_Confusion)
    with open(file_name, "w") as CSV_file:
        CSV_file.write("{} Tr_Time".format(Tr_Time))
        CSV_file.write("\n")
        CSV_file.write("{} Va_Time".format(Va_Time))
        CSV_file.write("\n")
        CSV_file.write("{} Te_Time".format(Te_Time))
        CSV_file.write("\n")
        CSV_file.write("{} T_Time".format(T_Time))
        CSV_file.write("\n")
        CSV_file.write("\n")
        CSV_file.write("{} DL_Time".format(DL_Time))
        CSV_file.write("\n")
        CSV_file.write("{} Va Kappa (%)".format(Va_Kappa))
        CSV_file.write("\n")
        CSV_file.write("{} Va Overall (%)".format(Va_OA))
        CSV_file.write("\n")
        CSV_file.write("{} VA Average (%)".format(Va_AA))
        CSV_file.write("\n")
        CSV_file.write("{} Te Kappa (%)".format(Te_Kappa))
        CSV_file.write("\n")
        CSV_file.write("{} Te Overall (%)".format(Te_OA))
        CSV_file.write("\n")
        CSV_file.write("{} Te Average (%)".format(Te_AA))
        CSV_file.write("\n")
        CSV_file.write("{} VA Classification".format(Va_classification))
        CSV_file.write("\n")
        CSV_file.write("{} VA Per Class".format(Va_Per_Class))
        CSV_file.write("\n")
        CSV_file.write("{} VA Confussion".format(Va_Confusion))
        CSV_file.write("\n")
        CSV_file.write("{} Te Classification".format(Te_classification))
        CSV_file.write("\n")
        CSV_file.write("{} Te Per Class".format(Te_Per_Class))
        CSV_file.write("\n")
        CSV_file.write("{} Te Confussion".format(Te_Confusion))
        CSV_file.write("\n")
        CSV_file.write("{} Te Kappa".format(Te_Kappa))
        CSV_file.write("\n")
        CSV_file.write("\n")
        CSV_file.write("{} Te OA".format(Te_OA))
        CSV_file.write("\n")
        CSV_file.write("\n")
        CSV_file.write("{} Te AA".format(Te_AA))
        CSV_file.write("\n")
        CSV_file.write("\n")
        CSV_file.write("{} Te Per Class".format(Te_Per_Class))
        CSV_file.write("\n")
        CSV_file.write("\n")
        CSV_file.write("{} T Classification".format(T_classification))
        CSV_file.write("\n")
        CSV_file.write("\n")
        CSV_file.write("{} T Confussion".format(T_Confusion))
        CSV_file.write("\n")
        CSV_file.write("{} T Kappa".format(T_Kappa))
        CSV_file.write("\n")
        CSV_file.write("\n")
        CSV_file.write("{} T OA".format(T_OA))
        CSV_file.write("\n")
        CSV_file.write("\n")
        CSV_file.write("{} T AA".format(T_AA))
        CSV_file.write("\n")
        CSV_file.write("\n")
        CSV_file.write("{} T Per Class".format(T_Per_Class))
        CSV_file.write("\n")
        print("CSV writed Successfully!")
    return CSV_file


# In[4]:


## Different Dimensional Reduction Methods
def DLMethod(HSI, NC=75):
    RHSI = np.reshape(HSI, (-1, HSI.shape[2]))
    pca = PCA(n_components=NC, whiten=True)
    RHSI = pca.fit_transform(RHSI)
    RHSI = np.reshape(RHSI, (HSI.shape[0], HSI.shape[1], NC))
    return RHSI


# In[5]:
def calculate_training_percentage(teRatio, trRatio):
    """
    Calculate and return the percentage of the total dataset used for training.

    Parameters:
    teRatio (float): Ratio of the data used for testing.
    trRatio (float): Ratio of the remaining data used for training.

    Returns:
    float: Percentage of the total dataset used for training.
    """
    # Calculate the remaining data ratio
    remaining_data_ratio = 1 - teRatio

    # Calculate the effective training ratio
    effective_training_ratio = trRatio

    # Calculate the training percentage
    training_percentage = effective_training_ratio * remaining_data_ratio * 100
    print("Train Percentage: ", training_percentage)
    return training_percentage


HSID = (
    args.dataset
)  ## "SLA", "IP", "PU", "PC", "SA", "KSC", "BS", "LK", "HH" (difficult to compile), "HC"
DLM = args.dl_method  ## "PCA", "iPCA"
WS = args.patch_size
# teRatio = 0.50
# vrRatio = 0.50
# Using the calculated values
teRatio = args.test_ratio
vrRatio = args.validation_ratio
trRatio = args.train_ratio
adam = tf.keras.optimizers.legacy.Adam(lr=0.0001, decay=1e-06)
# adam = tf.keras.optimizers.Adam(learning_rate=0.001, decay = 1e-06)
k = args.number_of_bands
epochs = args.epochs
batch_size = args.batch_size
output_dir = os.path.join(f"gt_viz/{HSID}/")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


# In[6]:


## Creat Patches for 3D (Spatial-Spectral) Models
def ImageCubes(HSI, GT, WS=WS, removeZeroLabels=True):
    num_rows, num_cols, num_bands = HSI.shape
    margin = int(WS / 2)
    padded_data = np.pad(
        HSI, ((margin, margin), (margin, margin), (0, 0)), mode="constant"
    )
    image_cubes = np.zeros((num_rows * num_cols, WS, WS, num_bands))
    patchesLabels = np.zeros((num_rows * num_cols))
    patchIndex = 0
    for r in range(margin, num_rows + margin):
        for c in range(margin, num_cols + margin):
            cube = padded_data[r - margin : r + margin, c - margin : c + margin, :]
            image_cubes[patchIndex, :, :, :] = cube
            patchesLabels[patchIndex] = GT[r - margin, c - margin]
            patchIndex = patchIndex + 1
    return image_cubes, patchesLabels


# In[7]:


## Main Function to load Datasets, Dimensional Reduction and Creating Patchs for CNN
HSI, GT, Num_Classes, target_names = LoadHSIData(HSID)
## Reduce the Dimensionality
start = time.time()
RDHSI = DLMethod(HSI, NC=k)
end = time.time()
DL_Time = end - start
## Create Image Cubes for Model Building
CRDHSI, CGT = ImageCubes(RDHSI, GT, WS=WS)


# In[ ]:


## Set seed for reproducibility
np.random.seed(42)
## Calculate the number of rows for each sample
num_rows, num_cols = GT.shape
## Flatten the matrix into a 1D array
flattened = GT.flatten()
## Get the unique values and their counts, excluding the 0 class
unique_values, value_counts = np.unique(flattened, return_counts=True)
nonzero_indices = np.where(unique_values != 0)[0]
unique_values = unique_values[nonzero_indices]
## Create a DataFrame to store sample counts
Samples = pd.DataFrame(columns=["Training", "Validation", "Test"])
## Create lists to store the indices
TrInd = []
VaInd = []
TeInd = []
## Split the data for each class
for value in unique_values:
    class_indices = np.where(flattened == value)[0]
    train_indices, test_indices = train_test_split(class_indices, test_size=teRatio)
    train_indices, val_indices = train_test_split(train_indices, test_size=vrRatio)
    ## Save the sample counts
    Samples.loc[value] = [len(train_indices), len(val_indices), len(test_indices)]
    ## Store the indices
    TrInd.extend(train_indices)
    VaInd.extend(val_indices)
    TeInd.extend(test_indices)

## Convert the DataFrame to a CSV file
file_name_sample = f"{HSID}_{teRatio}_{vrRatio}_{k}_{WS}_Samples.csv"
Samples.to_csv(os.path.join(output_dir, file_name_sample), index_label="Class")
# files.download(file_name)
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import os

# Prepare binary sample masks
train_matrix = np.zeros_like(GT)
val_matrix = np.zeros_like(GT)
test_matrix = np.zeros_like(GT)

train_matrix.flat[TrInd] = 1
val_matrix.flat[VaInd] = 1
test_matrix.flat[TeInd] = 1

TRC_labels = Tranform_Labels(train_matrix, TrInd, CGT[TrInd])
VAC_labels = Tranform_Labels(val_matrix, VaInd, CGT[VaInd])
TEC_labels = Tranform_Labels(test_matrix, TeInd, CGT[TeInd])

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import os
from matplotlib.lines import Line2D  # Import Line2D for circular markers

# Colormap and normalization
cmap = "nipy_spectral"
cmap_obj = plt.get_cmap(cmap)
unique_classes = np.unique(GT)
unique_classes = unique_classes[unique_classes != 0]  # Removing background (class 0)
norm = mcolors.Normalize(vmin=1, vmax=len(target_names))

# Create figure with TrueMap and legend on the right
fig, axs = plt.subplots(1, 1, figsize=(2, 1.8))  # Only one subplot for TrueMap

# Plot TrueMap
axs.imshow(GT, cmap=cmap_obj, norm=norm)

# Hide everything: no title, no axis labels, no ticks
axs.set_xticks([]), axs.set_yticks([])

# # Legend using circular markers instead of rectangles
# legend_patches = [
#     Line2D(
#         [0],
#         [0],
#         marker="o",
#         color="w",
#         markerfacecolor=cmap_obj(norm(i)),
#         markersize=6,
#         label=target_names[i - 1],
#     )
#     for i in unique_classes
# ]

# # Compact legend on the right side of the TrueMap
# fig.legend(
#     handles=legend_patches,
#     loc="center right",
#     ncol=1,
#     fontsize=4,
#     frameon=False,
#     handlelength=3,
#     # columnspacing=0.8,
# )

# Tighten spacing and adjust layout to remove white space
# plt.subplots_adjust(left=0.01, right=0.85, top=0.90, bottom=0.25)

# Save the figure
file_name = f"{HSID}_{teRatio}_{vrRatio}_{k}_{WS}_GT.png"
plt.savefig(
    os.path.join(output_dir, file_name),
    dpi=500,
    format="png",
    bbox_inches="tight",
    pad_inches=0,
)
plt.close()