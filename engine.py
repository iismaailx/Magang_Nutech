import socket
import cv2
import ocr
import base64
import json
from pyclassification import classify
from gst_camera import camera


# Initialize socket server
tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
SERVER_ADDRESS = ("192.168.10.120", 5555)
tcp_socket.bind(SERVER_ADDRESS)
tcp_socket.listen(1)

# Initialize the camera
#cap = cv2.VideoCapture(camera(0, 480, 360))
# cap = cv2.VideoCapture(0)
camera_on = False
camera_state = "OFF"
base64data = None

response = {
    "status":{
        "kode_respon" : "",
        "message" : ""
    },
    "message": {
        "data_ktp": {
            'PROVINSI': "-",
            'KOTA/KAB': "-",
            'NIK': "-",
            'Nama': "-",
            'Tempat/Tgl_Lahir': "-",
            'Jenis_Kelamin': "-",
            'Gol. Darah': "-",
            'Alamat': "-",
            'RT/RW': "-",
            'Kel/Desa': "-",
            'Kecamatan': "-",
            'Agama': "-",
            'Status_Perkawinan': "-",
            'Pekerjaan': "-",
            'Kewarganegaraan': "-",
            'Berlaku_Hingga': "-"
        },
        "data_image": {
            "long_data": "-",
            "image64": "-"
        }
    }
}


def cameracap():
    cap = cv2.VideoCapture(camera(0, 480, 360))
    # cap = cv2.VideoCapture(0)
    # Check if the camera is open
    if not cap.isOpened():
        print("Error: Could not open the camera.")
        exit()

    # Capture a frame from the camera
    
    ret, frame = cap.read()
    # image = frame
    if not ret:
        print('Error: Could not capture a frame!')
        exit()
        
    cap.release()
    # return image
    return frame


def convert(frame):
    w = 480
    h = 360
    try:
        # Resize the image
        resize_image = cv2.resize(frame, (w, h))
        _, buffer = cv2.imencode(".jpg", resize_image)
        # Convert the frame to Base64
        image_base64 = base64.b64encode(buffer).decode()
        return image_base64
    except Exception as e:
        print("Error: Could not encode the image as Base64.")

def reset_response():
    global response
    response = {
    "status":{
        "kode_respon" : "",
        "message" : ""
    },
    "message": {
        "data_ktp": {
            'PROVINSI': "-",
            'KOTA/KAB': "-",
            'NIK': "-",
            'Nama': "-",
            'Tempat/Tgl_Lahir': "-",
            'Jenis_Kelamin': "-",
            'Gol. Darah': "-",
            'Alamat': "-",
            'RT/RW': "-",
            'Kel/Desa': "-",
            'Kecamatan': "-",
            'Agama': "-",
            'Status_Perkawinan': "-",
            'Pekerjaan': "-",
            'Kewarganegaraan': "-",
            'Berlaku_Hingga': "-"
        },
        "data_image": {
            "long_data": "-",
            "image64": "-"
        }
    }
    }

def handle_client(connection, response):
    global base64data
    while True:
        receive = connection.recv(100)
        decode_data = receive.decode()
        print("Received data: {}".format(decode_data))
        try:
            if receive == b"close" or not receive:
                print('Connection Closed by Client!')
                response["status"] = "400"
                break

            if receive == b'capture':
                try:
                    Image = cameracap()
                    base64data = convert(Image)
                    classfction = classify(base64data)
                    if classfction == 0:
                        try:
                            data = ocr.main(Image)
                            response["message"]["data_ktp"]["PROVINSI"]=data['PROVINSI']
                            response["message"]["data_ktp"]["KOTA/KAB"]=data['KOTA/KAB']
                            response["message"]["data_ktp"]["NIK"]=data['NIK']
                            response["message"]["data_ktp"]["Nama"]=data['Nama']
                            response["message"]["data_ktp"]["Tempat/Tgl_Lahir"]=data['Tempat/Tgl_Lahir']
                            response["message"]["data_ktp"]["Jenis_Kelamin"]=data['Jenis_Kelamin']
                            response["message"]["data_ktp"]["Gol_Darah"]=data['Gol_Darah']
                            response["message"]["data_ktp"]["Alamat"]=data['Alamat']
                            response["message"]["data_ktp"]["RT/RW"]=data['RT/RW']
                            response["message"]["data_ktp"]["Kel/Desa"]=data['Kel/Desa']
                            response["message"]["data_ktp"]["Kecamatan"]=data['Kecamatan']
                            response["message"]["data_ktp"]["Agama"]=data['Agama']
                            response["message"]["data_ktp"]["Status_Perkawinan"]=data['Status_Perkawinan']
                            response["message"]["data_ktp"]["Pekerjaan"]=data['Pekerjaan']
                            response["message"]["data_ktp"]["Kewarganegaraan"]=data['Kewarganegaraan']
                            response["message"]["data_ktp"]["Berlaku_Hingga"]=data['Berlaku_Hingga']
                            response["status"]["kode_respon"] = "200"
                        except Exception as e:
                            response["status"]["kode_respon"] = "400"
                            response["status"]["message"] = str(e)
                    else:
                        response["status"]["kode_respon"] = "300" # gambar yang tidak terdeteksi bukan KTP!
                        response["status"]["message"] = "KTP not recognized"
                        
                    response["message"]["data_image"]["long_data"] = str(len(base64data))
                    response["message"]["data_image"]["image64"] = base64data
                    Image = None
                    base64data = None
                   
                except Exception as e:
                    response["status"]["kode_respon"] = "500" 
                    response["status"]["message"] = str(e)
         
            response_status = json.dumps(response)
            connection.send(response_status.encode())
            reset_response()
            print(receive)

        except Exception as e:
            response["status"]["kode_respon"] = "400" 
            response["status"]["message"] = str(e)
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
