import torch
from PIL import Image
from torchvision import models, transforms
import base64
import numpy as np
import cv2
import os
import time

model = models.resnet18(pretrained=True)
# Cek ketersediaan GPU pada device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.fc = torch.nn.Linear(model.fc.in_features, 1)
# Muat model dari file (pastikan model Anda telah didefinisikan sebelumnya)  # Gantilah 'YourModel' dengan model yang sesuai
model.load_state_dict(torch.load('models/model.pth'))
model.to(device)
model.eval()

# Praproses gambar
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def classify(file):
    try:
        # start_time = time.time()
        imageBytes = base64.b64decode(file)
        imageArray = np.frombuffer(imageBytes, np.uint8)
        img_data = cv2.imdecode(imageArray, cv2.IMREAD_COLOR)
        # Simpan gambar ke file
        cv2.imwrite("images/received_img.png", img_data)
        # Baca gambar yang telah disimpan
        img = cv2.imread("images/received_img.png")
        img = transform(Image.fromarray(img))  # Terapkan transformasi
        # Prediksi
        with torch.no_grad():
            img = img.unsqueeze(0)  # Tambahkan dimensi batch (size 1)
            img = img.to(device)  # Pindahkan ke perangkat yang tersedia
            predict = model(img)  # Prediksi
            
        predict_class = predict.argmax().item()
        threshold = 0.5
        if torch.sigmoid(predict) < threshold:
            hasil = 0
            # print("Prediction : back")
        else:
            hasil = 1
            # print("Prediction : front")
        # end_time = time.time()
        # inference_time = (end_time - start_time) * 1000
        # print(inference_time)
    except Exception as e:
        print(e)
        hasil = None
    finally:
        if os.path.exists("images/received_img.png"):
            os.remove("images/received_img.png")

    return hasil


