# Set IS_KAGGLE_KERNEL= False for use on any other platform and update directories
IS_KAGGLE_KERNEL = True


import datetime
import os

import tensorflow as tf
from tensorflow import keras as keras
from tensorflow.keras import layers, optimizers, activations, losses, backend
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
import tensorflow_addons as tfa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.utils import resample, shuffle
from sklearn.model_selection import KFold
from sklearn.metrics import recall_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import log_loss
# !pip install iterative_stratification
# if IS_KAGGLE_KERNEL:
#     ! pip install "/kaggle/input/moa-env/joblib-0.17.0-py3-none-any.whl"
#     ! pip install "/kaggle/input/moa-env/iterative_stratification-0.1.6-py3-none-any.whl"
# !pip install ../input/iterstrat
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

def with_input_path(s):
    src = "./kaggle/input/lish-moa/"
    if IS_KAGGLE_KERNEL:
        src = src[1:]
    return os.path.join(src, s)

# Prepare train data
df_train = pd.read_csv(with_input_path("train_features.csv"))
df_test = pd.read_csv(with_input_path("test_features.csv"))

# id is meaningess signifier
df_train = df_train.drop("sig_id", axis=1)
df_test = df_test.drop("sig_id", axis=1)

# cp_type indicates control (just vehicle) vs. drug. For now, we'll set all control experiments
# to have zero MoA's before submission. We will therefore ignore this feature in training
train = df_train.copy()
df_train = df_train[train["cp_type"] != 'ctl_vehicle'].reset_index(drop=True)
# train_control_locs = df_train.loc[df_train["cp_type"] == 'ctl_vehicle'].index
df_train = df_train.drop("cp_type", axis=1)

# Save these to set control exp MoA's to zero after training
test_control_locs = df_test.loc[df_test["cp_type"] == 'ctl_vehicle'].index
df_test = df_test.drop("cp_type", axis=1)

# Dosages are strings right now. I don't exactly know the dosages used but we can pretend it was either a single
# dose or a double dose
df_train['cp_dose'].replace('D1', 1, inplace=True)
df_train['cp_dose'].replace('D2', 2, inplace=True)
df_test['cp_dose'].replace('D1', 1, inplace=True)
df_test['cp_dose'].replace('D2', 2, inplace=True)

# Normalize train data and test data simultaneously
scaler = MinMaxScaler(feature_range=(-1, 1))
X_total = np.vstack((df_train, df_test))
scaler.fit(X_total)
X_train = scaler.transform(df_train)
X_test = scaler.transform(df_test)
# # Compress cell viabilities with PCA since they're highly correlated
n, _ = X_train.shape
pca = PCA(0.97) # Cutoff at 97% cum. explained variance
cell_v_pca = pca.fit_transform(X_total[:,-100:])

X_train = np.hstack((X_train[:,:-100], cell_v_pca[:n,:]))
X_test = np.hstack((X_test[:,:-100], cell_v_pca[n:,:]))

# Prepare train labels
df_targets = pd.read_csv(with_input_path("train_targets_scored.csv"))
df_targets = df_targets[train["cp_type"] != 'ctl_vehicle'].reset_index(drop=True)
y_train = df_targets.drop("sig_id", axis=1).to_numpy()
n, input_dim = X_train.shape
print(input_dim)
n, num_labels = y_train.shape
n_test, _ = X_test.shape

# Prediction Clipping Thresholds

p_min = 0.001
p_max = 0.999


def logloss(y_true, y_pred):
    y_pred = tf.clip_by_value(y_pred, p_min, p_max)
    return -backend.mean(y_true * backend.log(y_pred) + (1 - y_true) * backend.log(1 - y_pred))


def make_model(input_dim):
    # 4 Layer feedforward NN
    model = keras.Sequential()

    model.add(layers.Input(input_dim))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.2))

    model.add(tfa.layers.WeightNormalization(
        layers.Dense(2048, activation="relu", name="layer1")))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(.4))

    model.add(tfa.layers.WeightNormalization(
        layers.Dense(1024, activation="relu", name="layer2")))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(.4))

    model.add(tfa.layers.WeightNormalization(
        layers.Dense(512, activation="sigmoid", name="layer3")))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(.4))

    model.add(tfa.layers.WeightNormalization(
        layers.Dense(256, activation="sigmoid", name="layer4")))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(.4))

    model.add(tfa.layers.WeightNormalization(
        layers.Dense(256, activation="relu", name="layer5")))
    model.add(layers.Dense(num_labels, activation="sigmoid", name="output"))

    optimizer = optimizers.Adam()
    loss = losses.BinaryCrossentropy(label_smoothing=0.005)

    # Early stopping if model converges
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_logloss', min_delta=1e-5, patience=5, verbose=0,
                                                      mode='min', restore_best_weights=True)
    model.compile(optimizer=optimizer, loss=loss, metrics=logloss)

    return model


epochs = 300
batch_size = 2300

# CV splits (different seeds)
n_splits = 7
seeds = [394, 388, 2772, 105]
n_seeds = len(seeds)

# Rolling averages for validation scores and test predictions
avg_score = 0
test_preds = np.zeros((n_test, num_labels))

histories = []

df_targets = pd.read_csv(with_input_path("train_targets_scored.csv"))
df_targets = df_targets[train["cp_type"] != 'ctl_vehicle'].reset_index(drop=True)
df_id = pd.read_csv(with_input_path("train_drug.csv"))
df_id = df_id[train["cp_type"] != 'ctl_vehicle'].reset_index(drop=True)
pkg = (df_id, df_targets)

for i, seed in enumerate(seeds):
    for j, (train_locs, val_locs) in enumerate(
            MultilabelStratifiedKFold(n_splits=n_splits, random_state=seed, shuffle=True).split(X_train, y_train)):
        model = make_model(input_dim=input_dim)
        X_train_bal = X_train[train_locs]
        y_train_bal = y_train[train_locs]
        Xval = X_train[val_locs]
        yval = y_train[val_locs]
        reduce_lr_loss = ReduceLROnPlateau(
            monitor='val_logloss', factor=0.1, patience=5, verbose=1, min_delta=1e-4, mode='min')

        history_model = model.fit(x=X_train_bal,
                                  y=y_train_bal,
                                  epochs=epochs,
                                  batch_size=batch_size,
                                  validation_data=(Xval, yval),
                                  callbacks=[reduce_lr_loss])
        histories.append(history_model)
        y_preds = model.predict(Xval)
        fold_score = logloss(yval, y_preds)
        print("\t seed {}, fold {} validation score: {}".format(i, j, fold_score))
        avg_score += fold_score / (n_splits * n_seeds)

        # Update test score from this fold/cv
        test_preds += model.predict(X_test) / (n_splits * n_seeds)

trg_loss_dnn_orig = history_model.history['loss']
val_loss_dnn_orig = history_model.history['val_loss']
epochs = range(1, 301)
fig = plt.figure(figsize=(20, 8))
ax = fig.add_subplot(1, 2, 1)
plt.plot(epochs, trg_loss_dnn_orig, 'r', linewidth=3, label='Training Loss')
plt.plot(epochs, val_loss_dnn_orig, 'g', linewidth=3, label='Validation Loss')
plt.title("Training / Validation Loss")
ax.set_ylabel("Loss")
ax.set_xlabel("Epochs")
ax.set_facecolor("white")
plt.legend(loc='best')

plt.tight_layout()
plt.show()

