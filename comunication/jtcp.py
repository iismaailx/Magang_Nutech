import socket

tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
SERVER_ADRESS = ("192.168.10.120", 500)
tcp_socket.bind(SERVER_ADRESS)
tcp_socket.listen(1)

while True:
    print("Waiting Connection")
    connection, client = tcp_socket.accept()
    try:
        print('Conneted to client IP: {}'.format(client))
        
        while True:
            data = connection.recv(64)
            if data == "close" or not data:
                print('Connection Closed by Client!')
                break
            decode_data = data.decode()
            print("Received data: {}".format(decode_data))

    except socket.timeout:
        print('No incoming connection')
        break
    finally:
        connection.close()

tcp_socket.close()
        
    