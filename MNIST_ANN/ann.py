"""
MNIST veri seti : 0-9 rakamlarının el yazısı görüntüleri

Image preprocessing: 
histogram eşitleme: kontrast iyileştirme
gaussian blur: gürültü azaltma
canny edge detection: kenar tespiti

ANN ile sınıflandırma(10 grup)
"""

# pip install tensorflow matplotlib opencv-python

import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(f"X shape: {x_train.shape}")
print(f"Y_train: {y_train.shape}")

img = x_train[0]

stages = {"original": img}

eq = cv2.equalizeHist(img)
stages["histogram eşitleme"] = eq

blur = cv2.GaussianBlur(eq, (5, 5), 0)
stages["gaussian blur"] = blur

edges = cv2.Canny(blur, 50, 150)
stages["canny edges"] = edges

fig, axes = plt.subplots(2, 2)
axes = axes.flat

for ax, (title, im) in zip(axes, stages.items()):
    ax.imshow(im, cmap="gray")
    ax.set_title(title)
    ax.axis("off")

plt.suptitle("MNIST Image Processing Stage")
plt.tight_layout()
plt.show()


def preprocess_image(img):
    """
    histogram eşitleme
    gaussian blur
    canny ile kenar belirleme
    flattening
    normalizasyon
    """
    img = cv2.equalizeHist(img)
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    
    features = edges.flatten() / 255.0
    return features


num_train = 10000
num_test = 2000

X_train = np.array([preprocess_image(img) for img in x_train[:num_train]])
y_train_sub = y_train[:num_train]

X_test = np.array([preprocess_image(img) for img in x_test[:num_test]])
y_test_sub = y_test[:num_test]

model = Sequential([
    Dense(128, activation="relu", input_shape=(784,)),
    Dropout(0.5),
    Dense(64, activation="relu"),
    Dense(10, activation="softmax")
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print(model.summary())

history = model.fit(
    X_train, y_train_sub,
    validation_data=(X_test, y_test_sub),
    epochs=50,
    batch_size=32,
    verbose=2
)



test_loss,test_acc=model.evaluate(X_test,y_test_sub)
print(f"Test loss: {test_loss}, Test Acc: {test_acc}")


plt.figure(figsize=(10,6))
plt.subplot(1,2,1)
plt.plot(history.history["loss"],label="Training loss")
plt .plot(history.history["val_loss"],label="Validation Loss")
plt.title("Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history["accuracy"],label="Training Accuracy")
plt .plot(history.history["val_accuracy"],label="Validation Accuracy")
plt.title("Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()


plt.tight_layout()
plt.show()

