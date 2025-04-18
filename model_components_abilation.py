# %%
import matplotlib.pyplot as plt
import time, json
import numpy as np
import os
import scipy.io as sio

# import seaborn as sns
import pandas as pd
from operator import truediv

# import spectral
from sklearn.manifold import TSNE
from sklearn.decomposition import IncrementalPCA, PCA
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split

# import keras
import tensorflow as tf

# from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    Activation,
    Reshape,
    Conv2D,
    BatchNormalization,
)  # , Layer, DepthwiseConv2D,Concatenate, LayerNormalization
from tensorflow.keras.models import Sequential  # , Model
from keras.losses import categorical_crossentropy
from keras.utils import to_categorical
from keras.optimizers import legacy

# from tensorflow.keras.optimizers import Adam
# from keras.optimizers import Adam
import warnings

warnings.filterwarnings("ignore")

# %%
HSID = "HC"
DLM = "PCA"
WS = 14
teRatio = 0.90
vrRatio = 0.50
trRatio = 0.50
k = 15
# lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
#     initial_learning_rate=1e-4,
#     decay_steps=10000,
#     end_learning_rate=1e-6
# )
# adam = tf.keras.optimizers.AdamW(learning_rate=lr_schedule, weight_decay=1e-4)
adam = tf.keras.optimizers.legacy.Adam(lr=0.001, decay=1e-06, clipnorm=1.0)

epochs = 50
batch_size = 56

output_dir = os.path.join(f"model_components_abilation/{HSID}/")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


# %%
## Loading Hyperspectral Datasets
def LoadHSIData(method):
    data_path = os.path.join(os.getcwd(), "../datasets/")
    if method == "HH":
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
    return HSI, GT, Num_Classes, target_names


# %%
def DLMethod(method, HSI, NC=75):
    RHSI = np.reshape(HSI, (-1, HSI.shape[2]))
    if method == "PCA":  ## PCA
        pca = PCA(n_components=NC, whiten=True)
        RHSI = pca.fit_transform(RHSI)
        RHSI = np.reshape(RHSI, (HSI.shape[0], HSI.shape[1], NC))
    elif method == "iPCA":  ## Incremental PCA
        n_batches = 256
        inc_pca = IncrementalPCA(n_components=NC)
        for X_batch in np.array_split(RHSI, n_batches):
            inc_pca.partial_fit(X_batch)
        X_ipca = inc_pca.transform(RHSI)
        RHSI = np.reshape(X_ipca, (HSI.shape[0], HSI.shape[1], NC))
    return RHSI


def TrTeSplit(HSI, GT, trRatio, vrRatio, teRatio, randomState=345):
    Tr, Te, TrC, TeC = train_test_split(
        HSI, GT, test_size=teRatio, random_state=randomState, stratify=GT
    )
    totalTrRatio = trRatio + vrRatio
    new_vrRatio = vrRatio / totalTrRatio
    Tr, Va, TrC, VaC = train_test_split(
        Tr, TrC, test_size=new_vrRatio, random_state=randomState, stratify=TrC
    )
    return Tr, Va, Te, TrC, VaC, TeC


# %%
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
    if removeZeroLabels:
        image_cubes = image_cubes[patchesLabels > 0, :, :, :]
        patchesLabels = patchesLabels[patchesLabels > 0]
        patchesLabels -= 1
    return image_cubes, patchesLabels


def ClassificationReports(TeC, Te_Pred, target_names):
    classification = classification_report(
        np.argmax(TeC, axis=1), np.argmax(Te_Pred, axis=1), target_names=target_names
    )
    oa = accuracy_score(np.argmax(TeC, axis=1), np.argmax(Te_Pred, axis=1))
    confusion = confusion_matrix(np.argmax(TeC, axis=1), np.argmax(Te_Pred, axis=1))
    list_diag = np.diag(confusion)
    list_raw_sum = np.sum(confusion, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    aa = np.mean(each_acc)
    kappa = cohen_kappa_score(np.argmax(TeC, axis=1), np.argmax(Te_Pred, axis=1))
    return classification, confusion, oa * 100, each_acc * 100, aa * 100, kappa * 100


def CSVResults(
    file_name,
    classification,
    confusion,
    Tr_Time,
    Te_Time,
    DL_Time,
    kappa,
    oa,
    aa,
    each_acc,
    total_params,
):
    classification = str(classification)
    confusion = str(confusion)
    with open(file_name, "w") as CSV_file:
        CSV_file.write("{} Tr_Time".format(Tr_Time))
        CSV_file.write("\n")
        CSV_file.write("{} Te_Time".format(Te_Time))
        CSV_file.write("\n")
        CSV_file.write("{} Parameters".format(total_params))
        CSV_file.write("\n")
        CSV_file.write("{} Kappa accuracy (%)".format(kappa))
        CSV_file.write("\n")
        CSV_file.write("{} Overall accuracy (%)".format(oa))
        CSV_file.write("\n")
        CSV_file.write("{} Average accuracy (%)".format(aa))
        CSV_file.write("\n")
        CSV_file.write("{}".format(classification))
        CSV_file.write("\n")
        CSV_file.write("{}".format(each_acc))
        CSV_file.write("\n")
        CSV_file.write("{}".format(confusion))
    return CSV_file


# %%
## Plot Ground Truths
def GT_Plot(CRDHSI, GT, model, WS, k):
    Predicted = model.predict(CRDHSI)
    Predicted = np.argmax(Predicted, axis=1)
    height, width = np.shape(GT)
    ## Calculate the predicted Ground Truths
    outputs = np.zeros((height, width))
    count = 0
    for AA in range(height):
        for BB in range(width):
            target = int(GT[AA, BB])
            if target == 0:
                continue
            else:
                outputs[AA][BB] = Predicted[count]
                count = count + 1
    return outputs


# %%
HSI, GT, Num_Classes, target_names = LoadHSIData(HSID)
start = time.time()
RDHSI = DLMethod(DLM, HSI, NC=k)
end = time.time()
DL_Time = end - start
CRDHSI, CGT = ImageCubes(RDHSI, GT, WS=WS)
Tr, Va, Te, TrC, VaC, TeC = TrTeSplit(CRDHSI, CGT, trRatio, vrRatio, teRatio)
TrC = to_categorical(TrC)
VaC = to_categorical(VaC)
TeC = to_categorical(TeC)


# =============================================================================
# Model Components (DiMAMamba)
# =============================================================================


class SpectralSpatialTokenGeneration(tf.keras.layers.Layer):
    def __init__(self, out_channels, **kwargs):
        super(SpectralSpatialTokenGeneration, self).__init__(**kwargs)
        self.spatial_tokens = Dense(out_channels)
        self.spectral_tokens = Dense(out_channels)

    def call(self, x):
        B, H, W, C = x.shape
        spatial_tokens = self.spatial_tokens(
            tf.reshape(tf.transpose(x, [0, 2, 3, 1]), [tf.shape(x)[0], H * W, C])
        )
        spectral_tokens = self.spectral_tokens(
            tf.reshape(tf.transpose(x, [0, 1, 2, 3]), [tf.shape(x)[0], H * W, C])
        )
        return spatial_tokens, spectral_tokens


class DiMA(tf.keras.layers.Layer):
    def __init__(self, num_heads, d_model, memory_dim, dropout_rate=0.1, **kwargs):
        super(DiMA, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.d_model = d_model
        self.memory_dim = memory_dim
        self.depth = d_model // num_heads

        self.wq = Dense(d_model)
        self.wk = Dense(d_model)
        self.wv = Dense(d_model)

        self.k_mem = Dense(self.depth)
        self.v_mem = Dense(self.depth)

        self.concat_project = Dense(d_model)
        self.dropout = Dropout(dropout_rate)

        # Learnable memory matrix
        self.memory = self.add_weight(
            shape=(1, num_heads, memory_dim, self.depth),
            initializer="glorot_uniform",
            trainable=True,
            name="memory_matrix",
        )

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def differential_attention(self, scores):
        diff_scores = scores[:, :, 1:] - scores[:, :, :-1]
        diff_scores = tf.pad(diff_scores, [[0, 0], [0, 0], [1, 0], [0, 0]])
        return diff_scores

    def call(self, q, k, v, training=False):
        batch_size = tf.shape(q)[0]

        q_proj = self.split_heads(self.wq(q), batch_size)
        k_proj = self.split_heads(self.wk(k), batch_size)
        v_proj = self.split_heads(self.wv(v), batch_size)

        # Standard attention
        scores = tf.matmul(q_proj, k_proj, transpose_b=True) / tf.math.sqrt(
            tf.cast(self.depth, tf.float32)
        )
        diff_scores = self.differential_attention(scores)
        scores += diff_scores
        attn_weights = tf.nn.softmax(scores, axis=-1)
        attn_output = tf.matmul(self.dropout(attn_weights, training=training), v_proj)

        # Memory-based attention
        k_mem_proj = self.k_mem(self.memory)  # [1, num_heads, memory_dim, depth]
        v_mem_proj = self.v_mem(self.memory)
        mem_scores = tf.matmul(q_proj, k_mem_proj, transpose_b=True) / tf.math.sqrt(
            tf.cast(self.depth, tf.float32)
        )
        mem_weights = tf.nn.softmax(mem_scores, axis=-1)
        mem_output = tf.matmul(mem_weights, v_mem_proj)

        # Concatenate both outputs
        combined = tf.concat([attn_output, mem_output], axis=-1)
        combined = tf.transpose(
            combined, perm=[0, 2, 1, 3]
        )  # [B, seq, num_heads, 2*depth]
        combined = tf.reshape(
            combined, [batch_size, -1, 2 * self.depth * self.num_heads]
        )
        output = self.concat_project(combined)
        return output


class SpectralSpatialFeatureEnhancement(tf.keras.layers.Layer):
    def __init__(self, out_channels, **kwargs):
        super(SpectralSpatialFeatureEnhancement, self).__init__(**kwargs)
        self.spatial_gate = Sequential(
            [Dense(out_channels), Activation("sigmoid"), Reshape((1, out_channels))]
        )
        self.spectral_gate = Sequential(
            [Dense(out_channels), Activation("sigmoid"), Reshape((1, out_channels))]
        )

    def call(self, spatial_tokens, spectral_tokens, center_tokens):
        spatial_enhanced = spatial_tokens * self.spatial_gate(center_tokens)
        spectral_enhanced = spectral_tokens * self.spectral_gate(center_tokens)
        return spatial_enhanced, spectral_enhanced


class StateSpaceModel(tf.keras.layers.Layer):
    def __init__(self, state_dim, **kwargs):
        super(StateSpaceModel, self).__init__(**kwargs)
        self.state_dim = state_dim
        self.state_transition = Dense(state_dim)
        self.state_update = Dense(state_dim)

    def call(self, x):
        state = tf.zeros([tf.shape(x)[0], self.state_dim])
        for t in range(tf.shape(x)[1]):
            state = self.state_transition(state) + self.state_update(x[:, t, :])
        return state


class SSMambaModel(tf.keras.Model):
    def __init__(self, out_channels, num_heads, state_dim, dropout=0.1, **kwargs):
        super(SSMambaModel, self).__init__(**kwargs)
        self.pre_feature = tf.keras.Sequential(
            [
                Conv2D(16, kernel_size=3, padding="same", activation="relu"),
                BatchNormalization(),
                Conv2D(32, kernel_size=3, padding="same", activation="relu"),
                BatchNormalization(),
            ]
        )
        self.token_generation = SpectralSpatialTokenGeneration(out_channels)
        self.multi_head_attention = DiMA(
            num_heads=num_heads,
            d_model=out_channels,
            memory_dim=16,
            dropout_rate=dropout,
        )
        self.feature_enhancement = SpectralSpatialFeatureEnhancement(out_channels)
        self.state_space_model = StateSpaceModel(state_dim)
        self.dense = Dense(
            units=128,
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.l2(0.01),
        )
        self.dropout = Dropout(0.4)
        # Note: Num_Classes should be globally defined or passed in appropriately.
        self.classifier = Dense(Num_Classes, activation="softmax")

        # The following flags will be set in the ablation experiments.
        self.use_tokenization = True
        self.use_token_enhancement = True

    def call(self, x):
        x = self.pre_feature(x)
        # Conditional tokenization:
        if self.use_tokenization:
            spatial_tokens, spectral_tokens = self.token_generation(x)
        else:
            # If not using tokenization, flatten the features into tokens.
            B = tf.shape(x)[0]
            H = tf.shape(x)[1]
            W = tf.shape(x)[2]
            C = tf.shape(x)[3]
            tokens = tf.reshape(x, [B, H * W, C])
            spatial_tokens, spectral_tokens = tokens, tokens

        # Define center tokens as the mean of the spatial tokens.
        center_tokens = tf.reduce_mean(spatial_tokens, axis=1, keepdims=True)
        # Conditional token enhancement:
        if self.use_token_enhancement:
            spatial_enhanced, spectral_enhanced = self.feature_enhancement(
                spatial_tokens, spectral_tokens, center_tokens
            )
        else:
            spatial_enhanced, spectral_enhanced = spatial_tokens, spectral_tokens

        attention_output = self.multi_head_attention(
            spatial_enhanced, spectral_enhanced, spectral_enhanced
        )
        state_output = self.state_space_model(attention_output)
        output = self.classifier(state_output)
        return output


# =============================================================================
# Ablation Experiment Pipeline
# =============================================================================

# Define the ablation experiment configurations.
# Each dictionary sets flags for tokenization and token enhancement.
experiments = [
    {"tokenization": True, "token_enhancement": True, "name": "Token+Enhancement"},
    {"tokenization": False, "token_enhancement": True, "name": "No Tokenization"},
    {"tokenization": True, "token_enhancement": False, "name": "No Enhancement"},
    {
        "tokenization": False,
        "token_enhancement": False,
        "name": "No Token+No Enhancement",
    },
]


# Model configuration - to be adjusted to your specific environment
def build_model(
    config, out_channels=64, num_heads=4, state_dim=128, num_classes=Num_Classes
):
    model = SSMambaModel(
        out_channels=out_channels,
        num_heads=num_heads,
        state_dim=state_dim,
    )
    # Force an initial call to build the weights.
    _ = model(Tr[:batch_size])
    # Set the ablation flags based on configuration.
    model.use_tokenization = config["tokenization"]
    model.use_token_enhancement = config["token_enhancement"]
    return model


# Define the training function
def train_and_evaluate_model(
    config, Tr, TrC, Va, VaC, Te, TeC, epochs=50, batch_size=56
):
    model = build_model(config)

    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    # Train the model
    history = model.fit(
        Tr, TrC, epochs=epochs, batch_size=batch_size, validation_data=(Va, VaC)
    )

    # Evaluate the model
    test_loss, test_acc = model.evaluate(Te, TeC)

    # Return the results
    return test_acc, history.history


results = []
for exp in experiments:
    print(f"Running experiment: {exp['name']}")
    test_acc, hist = train_and_evaluate_model(
        exp, Tr, TrC, Va, VaC, Te, TeC, epochs, batch_size
    )
    results.append(
        {
            "experiment": exp["name"],
            "test_accuracy": test_acc,
            "history": hist,
            # "training_time": t_time,
        }
    )

# Save results to a JSON file
with open(os.path.join(output_dir, "ablation_results.json"), "w") as f:
    json.dump(results, f, indent=4)

results_df = pd.DataFrame(
    [
        {
            "experiment": r["experiment"],
            "test_accuracy": r["test_accuracy"],
            # "training_time": r["training_time"],
        }
        for r in results
    ]
)
results_df.to_csv(os.path.join(output_dir, "ablation_results.csv"), index=False)

# Step 5: Plotting the results

# Plot Test Accuracy for the experiments.
experiment_names = [r["experiment"] for r in results]
test_accuracies = [r["test_accuracy"] for r in results]

plt.figure(figsize=(10, 6))
bars = plt.barh(experiment_names, test_accuracies, color="skyblue")
plt.xlabel("Test Accuracy")
plt.ylabel("Experiment Configuration")
plt.title("Ablation Study: Impact of Tokenization and Enhancement on DiMAMamba")
for bar, acc in zip(bars, test_accuracies):
    plt.text(
        bar.get_width() + 0.005,
        bar.get_y() + bar.get_height() / 2,
        f"{acc:.3f}",
        va="center",
        fontsize=10,
    )
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "ablation_test_accuracy.png"))
# plt.show()
