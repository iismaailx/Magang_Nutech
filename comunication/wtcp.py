import socket
import json
import cv2
import base64
import numpy as np



# Create a connection to the server application on port 5555
tcp_socket = socket.create_connection(('192.168.10.120', 5555))

def saveImage(img):
    imageBytes = base64.b64decode(img)
    imageArray = np.frombuffer(imageBytes, np.uint8)
    img = cv2.imdecode(imageArray, cv2.IMREAD_COLOR)
    cv2.imwrite("receive_img.png", img)

try:
    while True:
        datain = input("Command: ")
        data = str.encode(datain)
        tcp_socket.sendall(data)
        if datain == 'close':
            break

        received_data = tcp_socket.recv(int(1e6))  # Adjust the buffer size as needed
        if received_data:
            try:
                data = received_data.decode()
                data_json = json.loads(data)

                # Handle JSON data here
                camera_status = data_json.get("camera")
                status = data_json.get("status")
                data_ktp = data_json.get("message", {}).get("data_ktp")
                data_image = data_json.get("message", {}).get("data_image", {})
                image64 = data_image.get("image64")
                if image64 != "-":
                    saveImage(image64)
                
                long_data = data_image.get("long_data")

                print("Camera Status:", camera_status)
                print("Status:", status)
                print("Data KTP:", data_ktp)
                print("Image Length:", long_data)
                print("Image Data Length:", len(image64))

            except json.JSONDecodeError as json_error:
                print("JSON Decode Error:", json_error)
            except Exception as e:
                print("Error:", e)

finally:
    print("Closing socket")
    tcp_socket.close()
