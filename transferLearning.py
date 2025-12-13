# %% LIBRARY IMPORTS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

from keras.src.layers import Resizing
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras import mixed_precision
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint,ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model
from tensorflow.keras.applications import ResNet50

from pathlib import Path
import os.path
from sklearn.model_selection import train_test_split


# %% DATA LOADING AND PREPARATION
dataset = "Dataset"
image_dir = Path(dataset)

filepaths = list(image_dir.glob(r"**/*.jpg")) + list(image_dir.glob(r"**/*.png"))

labels = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], filepaths))

filepaths = pd.Series(filepaths, name="filepath").astype("str")

labels = pd.Series(labels, name="labels").astype("str")

image_df = pd.concat([filepaths, labels], axis=1)

print(f"Total number of images: {len(image_df)}")
print(f"Classes: {image_df.labels.unique()}")
print(f"First few rows:\n{image_df.head()}")

# %% DATA VISUALIZATION
if len(image_df) > 0:

    random_index = np.random.randint(0, len(image_df), min(25, len(image_df)))

    fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(15, 15))

    for i, ax in enumerate(axes.flat):
        if i < len(random_index):
            try:

                img = plt.imread(image_df.filepath.iloc[random_index[i]])

                ax.imshow(img)

                ax.set_title(image_df.labels.iloc[random_index[i]], fontsize=10)

                ax.axis('off')

            except Exception as e:

                ax.text(0.5, 0.5, f'Error:\n{str(e)[:30]}',
                        ha='center', va='center', fontsize=8)
                ax.axis('off')
        else:
            ax.axis('off')

    plt.tight_layout()
    plt.show()
else:
    print("WARNING: No images found!")

# %% DATA SPLITTING (TRAIN-TEST SPLIT)

train_df, test_df = train_test_split(image_df, test_size=0.2, random_state=42, shuffle=True)

# %% DATA AUGMENTATION AND GENERATOR CREATION

train_generator = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.resnet50.preprocess_input,
    validation_split=0.2,
    rotation_range = 20,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    shear_range = 0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_generator = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.resnet50.preprocess_input
)

train_images = train_generator.flow_from_dataframe(
    dataframe=train_df,
    x_col="filepath",
    y_col="labels",
    target_size=(224, 224),
    color_mode="rgb",
    class_mode="categorical",
    batch_size=64,
    shuffle=True,
    seed=42,
    subset="training"
)

val_images = train_generator.flow_from_dataframe(
    dataframe=train_df,
    x_col="filepath",
    y_col="labels",
    target_size=(224, 224),
    color_mode="rgb",
    class_mode="categorical",
    batch_size=64,
    shuffle=True,
    seed=42,
    subset="validation"
)

test_images = train_generator.flow_from_dataframe(
    dataframe=test_df,
    x_col="filepath",
    y_col="labels",
    target_size=(224, 224),
    color_mode="rgb",
    class_mode="categorical",
    batch_size=64,
)

# %% IMAGE PREPROCESSING PIPELINE

resize_and_rescale = tf.keras.Sequential(
    [
        Resizing(224, 224),
    ]
)
mixed_precision.set_global_policy('mixed_float16')

# %% TRANSFER LEARNING MODEL: resnet50

pretrained_model = ResNet50(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet",
    pooling="avg"
)

pretrained_model.trainable = False

# %% CALLBACK FUNCTIONS

checkpoint_path = "pharmaceutical_drugs_and_vitamins_classification_model_checkpoint.weights.h5"

checkpoint_callback = ModelCheckpoint(
    checkpoint_path,
    save_weights_only=True,
    monitor="val_accuracy",
    save_best_only=True,
)

early_stopping = EarlyStopping(
    monitor="val_accuracy",
    patience=3,
    restore_best_weights=True,
    verbose=1
)
reduce_lr = ReduceLROnPlateau(
    monitor='val_accuracy',
    factor=0.2,
    patience=3,
    min_lr=0.00001,
    verbose=1
)

# %% MODEL ARCHITECTURE CREATION (FUNCTIONAL API)

inputs = pretrained_model.input

x = resize_and_rescale(inputs)

x = Dense(256, activation="relu")(pretrained_model.output)

x = Dropout(0.2)(x)

x = Dense(256, activation="relu")(x)

x = Dropout(0.2)(x)

outputs = Dense(10, activation="softmax",dtype='float32')(x)

model = Model(inputs=inputs, outputs=outputs)

model.summary()

# %% MODEL COMPILE
model.compile(
    optimizer=Adam(1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# %% MODEL TRAINING
history = model.fit(
    train_images,
    steps_per_epoch=len(train_images),
    validation_data=val_images,
    validation_steps=len(val_images),
    epochs=10,
    callbacks=[early_stopping, checkpoint_callback, reduce_lr]
)

# %% MODEL EVALUATION AND VISUALIZATION

loss, accuracy = model.evaluate(test_images, verbose=1)
print(f"Loss: {loss:.3}, accuracy: {accuracy:.3}")

plt.figure()

plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], marker="o", label="Training Accuracy")
plt.plot(history.history["val_accuracy"], marker="o", label="Validation Accuracy")
plt.title("Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], marker="o", label="Training Loss")
plt.plot(history.history["val_loss"], marker="o", label="Validation Loss")
plt.title("Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

# %% PREDICTION AND RESULT VISUALIZATION

pred = model.predict(test_images)

pred = np.argmax(pred, axis=1)

labels = (train_images.class_indices)
labels = dict((v, k) for k, v in labels.items())
pred = [labels[k] for k in pred]

random_index = np.random.randint(0, len(test_df) - 1, 15)

fig, axes = plt.subplots(nrows=5, ncols=3, figsize=(10, 10))

for i, ax in enumerate(axes.flat):

    ax.imshow(plt.imread(image_df.filepath[random_index[i]]))

    if test_df.labels.iloc[random_index[i]] == pred[random_index[i]]:
        color = "green"
    else:
        color = "red"

    ax.set_title(f"True: {test_df.labels.iloc[random_index[i]]} \n predict: {pred[random_index[i]]}",
                 color=color)

plt.tight_layout()
plt.show()


