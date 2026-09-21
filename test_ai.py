import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model('cat_dog_classifier.keras')
class_names = ['Cat', 'Dog']

img_path = r"C:\Users\Advan Aigen\Pictures\kucing3.jpg"  # ganti ini

img = tf.keras.utils.load_img(img_path, target_size=(128, 128))
img_array = tf.keras.utils.img_to_array(img)
img_array = img_array / 255.0
img_array = np.expand_dims(img_array, axis=0)

prediction = model.predict(img_array)
predicted_class = class_names[np.argmax(prediction)]
confidence = np.max(prediction) * 100

print(f"\nModel nebak: {predicted_class}")
print(f"Tingkat keyakinan: {confidence:.2f}%")