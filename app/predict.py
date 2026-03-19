import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import sys
import os

MODEL_PATH = os.path.join("..", "model", "dr_model_72.h5")

# 🔥 Rebuild model architecture manually
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(224, 224, 3)),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(5, activation='softmax')
])

# 🔥 Load weights only
model.load_weights(MODEL_PATH)

classes = ['Mild', 'Moderate', 'No_DR', 'Proliferate_DR', 'Severe']

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)

    print("\nRaw prediction:", prediction)
    print("Predicted index:", np.argmax(prediction))

    predicted_class = classes[np.argmax(prediction)]
    confidence = np.max(prediction)

    print(f"\n🧠 Prediction: {predicted_class}")
    print(f"📊 Confidence: {confidence*100:.2f}%")
if __name__ == "__main__":
    predict_image(sys.argv[1])