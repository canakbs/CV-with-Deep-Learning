import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model 
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os
import kagglehub 

path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")

print("Path to dataset files:", path)

# Dizin yapısını otomatik bul
def find_chest_xray_path(base_path):
    # Olası yolları kontrol et
    for root, dirs, files in os.walk(base_path):
        if 'train' in dirs and 'test' in dirs:
            return root
    return base_path

chest_xray_path = find_chest_xray_path(path)
print(f"Chest X-ray data path: {chest_xray_path}")

train_dir = os.path.join(chest_xray_path, "train")
test_dir = os.path.join(chest_xray_path, "test")
val_dir = os.path.join(chest_xray_path, "val")

train_datagen = ImageDataGenerator(
    rescale=1/255.0,
    rotation_range=10,
    brightness_range=[0.8, 1.2],
    horizontal_flip=True,
    validation_split=0.1
)

test_datagen = ImageDataGenerator(rescale=1/255.0)
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
CLASS_MODE = "binary"


train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode=CLASS_MODE,
    shuffle=True,
    subset="training"
)

val_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode=CLASS_MODE,
    shuffle=False,
    subset="validation"
)

test_gen = test_datagen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode=CLASS_MODE,
    shuffle=False,
)


class_names = list(train_gen.class_indices.keys())
images, labels = next(train_gen)

plt.figure(figsize=(10, 4))
for i in range(3):
    ax = plt.subplot(1, 3, i + 1)
    ax.imshow(images[i])
    ax.set_title(class_names[int(labels[i])])
    ax.axis("off")

plt.tight_layout()

plt.show()


base_model = DenseNet121(
    weights="imagenet",
    include_top=False,
    input_shape=(*IMG_SIZE, 3)
)

base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
pred = Dense(1, activation="sigmoid")(x)

model = Model(inputs=base_model.input, outputs=pred)

callbacks = [
    EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", patience=2, min_lr=1e-6),
    ModelCheckpoint("best_model.h5", monitor="val_loss", save_best_only=True)
]

model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

print(model.summary())

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=2,
    callbacks=callbacks,
    verbose=1
)


pred_probs = model.predict(test_gen, verbose=1)
pred_labels = (pred_probs > 0.5).astype(int).ravel()
true_labels = test_gen.classes

cm = confusion_matrix(true_labels, pred_labels)
disp = ConfusionMatrixDisplay(cm, display_labels=class_names)

plt.figure(figsize=(8, 8))
disp.plot(cmap="Blues", colorbar=False)
plt.title("Test Seti Confusion Matrix")
plt.show()