import tensorflow as tf
import numpy as np
from PIL import Image
import io
import cv2
import base64
# model = tf.keras.models.load_model('models/my_ktp_model.h5')
path = r'/home/rnd/Development/Desk/models'
model = tf.saved_model.load(path)


def classify(file):
    try:
        imageBytes = base64.b64decode(file)
        imageArray = np.frombuffer(imageBytes, np.uint8)
        img_data = cv2.imdecode(imageArray, cv2.IMREAD_COLOR)
        # Simpan gambar ke file
        cv2.imwrite("images/received_img.png", img_data)
        # Baca gambar yang telah disimpan
        img = cv2.imread("images/received_img.png")
        
        # Preprocess gambar sebelum memprediksi
        img = cv2.resize(img, (150, 150))
        input_data = np.array(img) /255.0
        input_data = input_data.reshape((1, 150, 150, 3))
        input_data = input_data.astype(np.float32)
        predictions = model(input_data)
        threshold = 0.5

        # Ambil nilai prediksi
        predict_class = 0 if predictions[0] > threshold else 1
        print(predict_class)
        return predict_class
    except Exception as e:
        print(e)
    
