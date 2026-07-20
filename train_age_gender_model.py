import os
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split

from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import (
    Dense,
    GlobalAveragePooling2D,
    Dropout
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint
)

from tensorflow.keras.applications.efficientnet import preprocess_input

from tensorflow.keras.regularizers import l2

# =====================================================
# SETTINGS
# =====================================================

DATASET_PATH = "UTKFace/"

IMG_SIZE = 224
BATCH_SIZE = 16

# =====================================================
# LOAD DATASET
# =====================================================

data = []

for file in os.listdir(DATASET_PATH):

    try:
        age = int(file.split("_")[0])
        gender = int(file.split("_")[1])

        filepath = os.path.join(
            DATASET_PATH,
            file
        )

        data.append([
            filepath,
            age,
            gender
        ])

    except:
        continue

df = pd.DataFrame(
    data,
    columns=[
        "image",
        "age",
        "gender"
    ]
)

print("Total Images:", len(df))

# =====================================================
# TRAIN / VALIDATION SPLIT
# =====================================================

train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["gender"]
)

# =====================================================
# IMAGE LOADER
# =====================================================

def load_image(path, age, gender):

    image = tf.io.read_file(path)

    image = tf.image.decode_jpeg(
        image,
        channels=3
    )

    image = tf.image.resize(
        image,
        (IMG_SIZE, IMG_SIZE)
    )

    image = tf.cast(
        image,
        tf.float32
    )

    image = preprocess_input(image)

    return image, {
        "gender": tf.one_hot(gender, 2),
        "age": tf.cast(age, tf.float32)
    }

# =====================================================
# TRAIN DATASET
# =====================================================

train_ds = tf.data.Dataset.from_tensor_slices(
    (
        train_df["image"].values,
        train_df["age"].values,
        train_df["gender"].values
    )
)

train_ds = train_ds.map(
    load_image,
    num_parallel_calls=tf.data.AUTOTUNE
)

train_ds = train_ds.shuffle(
    buffer_size=len(train_df)
)

train_ds = train_ds.batch(BATCH_SIZE)

train_ds = train_ds.prefetch(
    tf.data.AUTOTUNE
)

# =====================================================
# VALIDATION DATASET
# =====================================================

val_ds = tf.data.Dataset.from_tensor_slices(
    (
        val_df["image"].values,
        val_df["age"].values,
        val_df["gender"].values
    )
)

val_ds = val_ds.map(
    load_image,
    num_parallel_calls=tf.data.AUTOTUNE
)

val_ds = val_ds.batch(BATCH_SIZE)

val_ds = val_ds.prefetch(
    tf.data.AUTOTUNE
)

# =====================================================
# DATA AUGMENTATION
# =====================================================

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
])

# =====================================================
# EFFICIENTNETB0
# =====================================================

base_model = EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(224, 224, 3)
)

# Freeze most layers
base_model.trainable = True

for layer in base_model.layers[:-100]:
    layer.trainable = False

# =====================================================
# MODEL
# =====================================================

inputs = tf.keras.Input(
    shape=(224,224,3)
)

x = data_augmentation(inputs)

x = base_model(x, training=False)

x = GlobalAveragePooling2D()(x)

x = tf.keras.layers.BatchNormalization()(x)

x = Dense(
    256,
    activation="relu",
    kernel_regularizer=l2(0.001)
)(x)

x = Dropout(0.4)(x)

# =====================================================
# GENDER HEAD
# =====================================================

gender_branch = Dense(
    128,
    activation="relu"
)(x)

gender_branch = Dropout(0.3)(
    gender_branch
)

gender_output = Dense(
    2,
    activation="softmax",
    name="gender"
)(gender_branch)

# =====================================================
# AGE HEAD
# =====================================================

age_branch = Dense(
    128,
    activation="relu"
)(x)

age_branch = Dropout(0.3)(
    age_branch
)

age_output = Dense(
    1,
    activation="linear",
    name="age"
)(age_branch)

# =====================================================
# BUILD MODEL
# =====================================================

model = Model(
    inputs=inputs,
    outputs=[
        gender_output,
        age_output
    ]
)

# =====================================================
# COMPILE
# =====================================================

optimizer = tf.keras.optimizers.Adam(
    learning_rate=1e-4
)

model.compile(
    optimizer=optimizer,

    loss={
        "gender": tf.keras.losses.CategoricalCrossentropy(
            label_smoothing=0.1
        ),
        "age": tf.keras.losses.Huber()
    },

    loss_weights={
        "gender": 5.0,
        "age": 0.1
    },

    metrics={
        "gender": ["accuracy"],
        "age": ["mae"]
    }
)

# =====================================================
# CALLBACKS
# =====================================================

early_stop = EarlyStopping(
    monitor="val_gender_accuracy",
    mode="max",
    patience=8,
    restore_best_weights=True,
    verbose=1
)

checkpoint = ModelCheckpoint(
    "age_gender_model.keras",
    monitor="val_gender_accuracy",
    mode="max",
    save_best_only=True,
    verbose=1
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_gender_accuracy",
    factor=0.5,
    patience=3,
    mode="max",
    verbose=1
)

# =====================================================
# TRAIN
# =====================================================

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=50,
    callbacks=[
        early_stop,
        checkpoint,
        reduce_lr
    ]
)

print("Training Complete!")

print(df.head())
print(df["gender"].value_counts())

print(
    "Best Validation Gender Accuracy:",
    max(history.history["val_gender_accuracy"])
)

print(
    "Best Validation Age MAE:",
    min(history.history["val_age_mae"])
)