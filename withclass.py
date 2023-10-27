import socket
import cv2
import base64
import json
import tensorflow as tf
from classification import classify

# Initialize socket server
tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
SERVER_ADDRESS = ("192.168.10.120", 5555)
tcp_socket.bind(SERVER_ADDRESS)
tcp_socket.listen(1)

# Initialize the camera
cap = cv2.VideoCapture(0)
camera_on = False
camera_state = "OFF"
base64data = None

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


def camera():
    global camera_on
    # Check if the camera is open
    if not cap.isOpened():
        print("Error: Could not open the camera.")
        exit()

    # Capture a frame from the camera
    ret, frame = cap.read()

    if not ret:
        print('Error: Could not capture a frame!')
        exit()

    return frame


def convert(frame):
    w = 480
    h = 320
    try:
        # Resize the image
        resize_image = cv2.resize(frame, (w, h))
        _, buffer = cv2.imencode(".jpg", resize_image)
        # Convert the frame to Base64
        image_base64 = base64.b64encode(buffer).decode()
        return image_base64
    except Exception as e:
        print("Error: Could not encode the image as Base64.")


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
    global base64data
    global camera_state
    global camera_on
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
                        Image = camera()
                        base64data = convert(Image)
                        classfction = classify(base64data)
                        print(classfction)
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
                exit()

            response_status = json.dumps(response)
            connection.send(response_status.encode())
            # print(response)
            reset_response(camera_state)

        except Exception as e:
            print(e)
            response["status"] = "400"
            response_status = json.dumps(response)
            connection.send(response_status.encode())

    connection.close()
        

class camera():
    def __init__(self) -> None:
        self.output = []
        self.frame = []
    
    def main(self):
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            else:
                return frame
            
class connection():
    def __init__(self) -> None:
        self.connect = None
        self.client = None
        
    def main(self):
        while True:
            print("Waiting for Connection")
            self.connect, self.client = tcp_socket.accept()
            try:
                print('Connected to client IP: {}'.format(client))
                handle_client(self.connect, response)
            except socket.timeout:
                print('No incoming connection')
                break
        tcp_socket.close()
            
            
if __name__ == '__main__':
    connection()
