import cv2

# Inisialisasi kamera
cap = cv2.VideoCapture(1)
camera_on = True

# Fungsi untuk menghidupkan kembali kamera
def turn_on_camera():
    global cap, camera_on
    if not camera_on:
        cap = cv2.VideoCapture(0)  # Hidupkan kamera
        camera_on = True

# Fungsi untuk mematikan kamera
def turn_off_camera():
    global cap, camera_on
    if camera_on:
        cap.release()  # Matikan kamera
        camera_on = False

# Menggunakan kamera
while True:
    # Mengecek perintah untuk menghidupkan/memadamkan kamera
    # Misalnya, perintah "start" untuk menghidupkan kamera
    command = input("Enter command (start/stop): ")
    if command == "start":
        turn_on_camera()
    elif command == "stop":
        turn_off_camera()

    if camera_on:
        ret, frame = cap.read()
        if not ret:
            print('Error: Could not capture a frame!')
            break

        # Lakukan pemrosesan frame di sini
        cv2.imshow('Frame', frame)

    # Keluar dari loop dengan menekan tombol 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Setelah selesai, pastikan kamera dimatikan
turn_off_camera()

# Tutup jendela tampilan frame
cv2.destroyAllWindows()
