#pip install tensorflow matplotlib tensorflow-datasets

from tensorflow_datasets import load
from tensorflow.data import AUTOTUNE
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Dense, Conv2D, MaxPooling2D, Flatten, Dropout)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint
)

import tensorflow as tf
import matplotlib.pyplot as plt

(ds_train, ds_val), ds_info = load("tf_flowers", split=["train[:80%]", "train[80%:]"], as_supervised=True, with_info=True)

print(ds_info.features)
print("Number of classes: ", ds_info.features["label"].num_classes)

fig = plt.figure(figsize=(10, 5))

for i, (image, label) in enumerate(ds_train.take(3)):
    ax = fig.add_subplot(1, 3, i+1)
    ax.imshow(image.numpy().astype("uint8"))
    ax.set_title(f"Etiket: {label.numpy()}")
    ax.axis("off")

plt.tight_layout()
plt.show()

IMG_SIZE = (180, 180)

def preprocess_train(image, label):
    """
    resize, random flip, brightness, contrast, crop
    normalize
    """
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.9, upper=1.2)
    image = tf.image.random_crop(image, size=(160, 160, 3))
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32) / 255.0
    return image, label  # HATA 1: return eksikti


def preprocess_val(image, label):  # HATA 2: label parametresi eksikti
    """
    resize, normalize
    """
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32) / 255.0
    return image, label


ds_train = (
    ds_train.map(preprocess_train, num_parallel_calls=AUTOTUNE)
    .shuffle(1000)
    .batch(32)
    .prefetch(AUTOTUNE)
)

ds_val = (
    ds_val
    .map(preprocess_val, num_parallel_calls=AUTOTUNE)
    .batch(32)
    .prefetch(AUTOTUNE)
)

model = Sequential()
model.add(Conv2D(32, (3, 3), activation="relu", input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)))  # HATA 3: inputshape -> input_shape
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation="relu"))
model.add(MaxPooling2D((2, 2)))

model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(ds_info.features["label"].num_classes, activation="softmax"))

# callbacks
callbacks = [
    EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=2, verbose=1, min_lr=1e-9),
    ModelCheckpoint("best_model.h5", save_best_only=True)
]

model.compile(
    optimizer=Adam(learning_rate=0.001),  # HATA 4: lerning_rate -> learning_rate
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print(model.summary())

history = model.fit(
    ds_train,
    validation_data=ds_val,  # HATA 5: validation -> validation_data
    epochs=10,
    callbacks=callbacks,
    verbose=1
)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Eğitim Doğruluğu")
plt.plot(history.history["val_accuracy"], label="Validation Doğruluğu")
plt.xlabel("Epochs")
plt.ylabel("Doğruluk")
plt.title("Model Accuracy")
plt.legend()
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Eğitim Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.title("Model Loss")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()