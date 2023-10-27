import socket
import cv2
import base64
import json
import tensorflow as tf
import threading

# Inisialisasi socket server
tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
SERVER_ADDRESS = ("192.168.10.120", 5555)
tcp_socket.bind(SERVER_ADDRESS)
tcp_socket.listen(1)

# Inisialisasi kamera
cap = cv2.VideoCapture(0)
camera_on = False
camera_state = "OFF"
base64data = None
model = tf.keras.models.load_model("your_model.h5")  # Gantilah "your_model.h5" dengan model Anda.

response = {
    "camera": "OFF",
    "status": "-",
    "message": {
        "data_ktp": {
            'PROVINSI': "-",
            'KOTA/KAB': "-",
            'NIK': "-",
            'Nama': "-",
            'Tempat/Tgl Lahir': "-",
            'Jenis Kelamin': "-",
            'Gol. Darah': "-",
            'Alamat': "-",
            'RT/RW': "-",
            'Kel/Desa': "-",
            'Kecamatan': "-",
            'Agama': "-",
            'Status Perkawinan': "-",
            'Pekerjaan': "-",
            'Kewarganegaraan': "-",
            'Berlaku Hingga': "-"
        },
        "data_image": {
            "long_data": "-",
            "image64": "-"
        }
    }
}

# Fungsi untuk terus-menerus mendapatkan frame kamera
def camera_thread():
    global camera_on
    while True:
        if camera_on:
            ret, frame = cap.read()
            if not ret:
                print('Error: Could not capture a frame!')
                camera_on = False
            else:
                base64data = convert(frame)

# Mulai thread kamera
camera_thread = threading.Thread(target=camera_thread)
camera_thread.start()

def reset_response(camera_state):
    global response
    response = {
        "camera": camera_state,
        "status": "-",
        "message": {
            "data_ktp": {
                'PROVINSI': "-",
                'KOTA/KAB': "-",
                'NIK': "-",
                'Nama': "-",
                'Tempat/Tgl Lahir': "-",
                'Jenis Kelamin': "-",
                'Gol. Darah': "-",
                'Alamat': "-",
                'RT/RW': "-",
                'Kel/Desa': "-",
                'Kecamatan': "-",
                'Agama': "-",
                'Status Perkawinan': "-",
                'Pekerjaan': "-",
                'Kewarganegaraan': "-",
                'Berlaku Hingga': "-"
            },
            "data_image": {
                "long_data": "-",
                "image64": "-"
            }
        }
    }

def handle_client(connection, response):
    global camera_state
    while True:
        receive = connection.recv(100)
        decode_data = receive.decode()
        print("Received data: {}".format(decode_data))
        try:
            if receive == b"close" or not receive:
                print('Connection Closed by Client!')
                response["status"] = "400"
                break

            if receive == b'opencamera':
                camera_on = True
                camera_state = "ON"
                response["status"] = "200"
                response["camera"] = camera_state
                print("Camera On.")

            if receive == b'capture':
                if camera_on:
                    try:
                        data_length_bytes = len(base64data)
                        data_length_kilobytes = data_length_bytes / 1024
                        print(f"bytes size:{data_length_bytes}, kilobyte size:{data_length_kilobytes}!")
                        response["message"]["data_ktp"]["PROVINSI"] = "ACEH"
                        response["message"]["data_image"]["long_data"] = str(data_length_bytes)
                        response["message"]["data_image"]["image64"] = base64data
                    except Exception as e:
                        print(e)
                        response["status"] = "400"
                else:
                    response["camera"] = "OFF"

            if receive == b'closecamera':
                camera_on = False
                camera_state = "OFF"
                response["status"] = "200"
                response["camera"] = camera_state
                print("Camera Off.")

            if receive == b'breakup':
                print('Received breakup command. Closing connection and stopping the program.')
                response["status"] = "200"
                connection.send(json.dumps(response).encode())
                connection.close()
                tcp_socket.close()
                camera_thread.join()
                exit()

            response_status = json.dumps(response)
            connection.send(response_status.encode())
            reset_response(camera_state)

        except Exception as e:
            print(e)
            response["status"] = "400"
            response_status = json.dumps(response)
            connection.send(response_status.encode())

    connection.close()

def connection():
    while True:
        print("Waiting for Connection")
        connect, client = tcp_socket.accept()
        try:
            print('Connected to client IP: {}'.format(client))
            handle_client(connect, response)
        except socket.timeout:
            print('No incoming connection')
            break
    tcp_socket.close()

if __name__ == '__main__':
    connection()
