import cv2
import json
import numpy as np
import ocr
import time
import base64
import os
import datetime
# Note: Uncomment for YOLO feature
# import yolo_detect
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

def get_time():
    current_datetime = datetime.datetime.now()
    timeformat = current_datetime.strftime("%H%M%S")
    return timeformat

# menyimpan data image terlebih dahulu
def write_image(data):
    pipepath = 'receive_image/'
    os.makedirs(pipepath, exist_ok=True)
    time = get_time()
    filename = os.path.join(pipepath, f'image_{time}.png')
    data = data.replace('"', '')
    decoded_data=base64.b64decode(data)
    
    # with open(filename , 'wb') as img_file:
    #     img_file.write(decoded_data)
    #     img_file.close()

    return decoded_data, filename

@app.route('/', methods = ['POST'])
def receive_json():
    request_data = request.get_json()
    if 'image' not in request_data or request_data['image'] == "":
        json_content ={
            'Message' : "image is empty",
        }
    else:   
        try:
            json_data=request.get_json()
            binary, filename = write_image(json_data['image'])
            npimg = np.frombuffer(binary, np.uint8)
            image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
            # imagefile = request.files['image'].read()
            # npimg = np.frombuffer(imagefile, np.uint8)
            # image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
            # nik, nama, tempat_lahir, tgl_lahir, agama = ocr.main(image)
            try:
                data = ocr.main(image)
                json_content = {
                    'PROVINSI': data['PROVINSI'],
                    'KOTA/KAB': data['KOTA/KAB'],
                    'NIK': data['NIK'],
                    'Nama': data['Nama'],
                    'Tempat/Tgl Lahir': data['Tempat/Tgl Lahir'],
                    'Jenis Kelamin': data['Jenis Kelamin'],
                    'Gol. Darah' : data['Gol. Darah'],
                    'Alamat' : data['Alamat'],
                    'RT/RW' : data['RT/RW'],
                    'Kel/Desa' : data['Kel/Desa'],
                    'Kecamatan' : data['Kecamatan'],
                    'Agama': data['Agama'],
                    'Status Perkawinan' : data['Status Perkawinan'],
                    'Pekerjaan' : data['Pekerjaan'],
                    'Kewarganegaraan': data['Kewarganegaraan'],
                    'Berlaku Hingga': data['Berlaku Hingga']
                }
            except Exception as e:
                json_content = 'Image KTP not Found!'
                return jsonify(json_content, ), 400
        except Exception as e:
            return jsonify({'error': str(e)}), 400
    return jsonify(json_content, ), 200
    # return render_template('result.html', data=json_content)
    
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
