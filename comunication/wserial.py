import serial

# Konfigurasi port serial di Windows
serial_port = serial.Serial(
    port='COM3',  # Ganti 'COMx' dengan nomor port serial yang digunakan di Windows (misalnya, 'COM1')
    baudrate=115200,
    bytesize=serial.EIGHTBITS,
    parity=serial.PARITY_NONE,
    stopbits=serial.STOPBITS_ONE,
)

try:
    # Kirim data dari Windows ke Jetson
    data_to_send = "Hello from Windows\r\n"
    serial_port.write(data_to_send.encode())

    while True:
        user_input = input("Masukkan data yang ingin Anda kirim (tekan Enter untuk mengirim): ")
        serial_port.write(user_input.encode() + b'\r\n')  # Mengirim data yang dimasukkan ke port serial


except KeyboardInterrupt:
    print("Exiting Program")

except Exception as exception_error:
    print("Error occurred. Exiting Program")
    print("Error: " + str(exception_error))

finally:
    serial_port.close()
