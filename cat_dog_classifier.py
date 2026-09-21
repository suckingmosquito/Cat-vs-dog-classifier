import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# 1. Load dataset dari folder
dataset_path = r"C:\Users\Advan Aigen\Documents\Dataset\PetImages"
img_size = (128, 128)
batch_size = 32

train_data = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset='training',
    seed=123,
    image_size=img_size,
    batch_size=batch_size
)

val_data = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset='validation',
    seed=123,
    image_size=img_size,
    batch_size=batch_size
)

class_names = train_data.class_names
print("Kategori yang ketemu:", class_names)

# 2. Normalisasi pixel
normalization_layer = layers.Rescaling(1./255)
train_data = train_data.map(lambda x, y: (normalization_layer(x), y))
val_data = val_data.map(lambda x, y: (normalization_layer(x), y))

# Skip file corrupt
train_data = train_data.apply(tf.data.experimental.ignore_errors())
val_data = val_data.apply(tf.data.experimental.ignore_errors())

# 3. Data Augmentation — bikin variasi foto biar model gak hafal foto asli
data_augmentation = models.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

# 4. Bikin model CNN + Dropout
model = models.Sequential([
    data_augmentation,  # augmentation jalan otomatis cuma pas training
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dropout(0.5),  # <- matiin 50% neuron random pas training
    layers.Dense(64, activation='relu'),
    layers.Dense(len(class_names), activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 5. Early Stopping — otomatis berhenti kalau udah gak improve
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=5,           # kasih toleransi 5 epoch dulu sebelum nyerah
    restore_best_weights=True  # balik ke kondisi terbaik, bukan yang terakhir
)

# 6. Training (epoch dikasih tinggi, tapi early stopping bakal berhenti sendiri)
print("\nMulai training...")
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=30,
    callbacks=[early_stop]
)

# 7. Simpan model
model.save('cat_dog_classifier.keras')
print("\nModel tersimpan sebagai 'cat_dog_classifier.keras'")

# 8. Plot grafik
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Progress Training: Cat vs Dog Classifier (v2)')
plt.show()