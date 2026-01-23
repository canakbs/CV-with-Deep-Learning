"""
MNIST veri seti : 0-9 rakamlarının el yazısı görüntüleri

CNN ile sınıflandırma
Convolutional katmanlar görüntü özelliklerini otomatik öğrenir
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

# Veri yükleme
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(f"X shape: {x_train.shape}")
print(f"Y_train: {y_train.shape}")

# Görselleştirme için preprocessing aşamaları
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

plt.suptitle("MNIST Image Processing Stages")
plt.tight_layout()
plt.show()


def preprocess_for_cnn(images):
    """
    CNN için veri hazırlama:
    - Histogram eşitleme
    - Gaussian blur
    - Normalizasyon
    - Reshape (CNN için 4D tensor gerekir)
    """
    processed = []
    for img in images:
        # Histogram eşitleme
        img_eq = cv2.equalizeHist(img)
        # Gaussian blur
        img_blur = cv2.GaussianBlur(img_eq, (5, 5), 0)
        processed.append(img_blur)
    
    # Numpy array'e çevir ve normalize et
    processed = np.array(processed)
    processed = processed.reshape(-1, 28, 28, 1) / 255.0
    
    return processed


# Veri hazırlama
num_train = 10000
num_test = 2000

X_train = preprocess_for_cnn(x_train[:num_train])
y_train_sub = to_categorical(y_train[:num_train], 10)  # One-hot encoding

X_test = preprocess_for_cnn(x_test[:num_test])
y_test_sub = to_categorical(y_test[:num_test], 10)

print(f"X_train shape: {X_train.shape}")
print(f"y_train_sub shape: {y_train_sub.shape}")

# CNN Modeli
model = Sequential([
    # İlk Convolutional Blok
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    
    # İkinci Convolutional Blok
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    
    # Üçüncü Convolutional Blok
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    
    # Flatten ve Dense Katmanlar
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print(model.summary())

# Model eğitimi
history = model.fit(
    X_train, y_train_sub,
    validation_data=(X_test, y_test_sub),
    epochs=50,
    batch_size=32,
    verbose=2
)

# Test değerlendirmesi
test_loss, test_acc = model.evaluate(X_test, y_test_sub)
print(f"\nTest Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

# Sonuçların görselleştirilmesi
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Örnek tahminler
predictions = model.predict(X_test[:10])
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(y_test_sub[:10], axis=1)

print("\nİlk 10 test görüntüsü için tahminler:")
for i in range(10):
    print(f"Gerçek: {true_classes[i]}, Tahmin: {predicted_classes[i]}")