import os
import cv2

DATA_DIR = './dataset' # Bisa masukkan nama path yang diinginkan
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 46 # bisa diadjust sesuai kelas yang ingin dibuat, 46 adalah kelas 26 abjad dan 20 kata
dataset_size = 500

# Buka kamera (indeks 0 biasanya webcam internal)
cap = cv2.VideoCapture(2)

for j in range(number_of_classes):
    class_dir = os.path.join(DATA_DIR, str(j))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print('Mengumpulkan data untuk kelas: {}'.format(j))

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Gagal membaca frame dari kamera.")
            break
        
        cv2.putText(frame, 'Siap? Tekan "Q" untuk memulai!', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('q'):
            break

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            break
        
        cv2.imshow('frame', frame)
        cv2.waitKey(25) 
        
        cv2.imwrite(os.path.join(class_dir, '{}.jpg'.format(counter)), frame)

        counter += 1
        print(f'Gambar ke-{counter} untuk kelas {j} disimpan')


        if counter % 100 == 0 and counter < dataset_size:
            print(f"Batch ke-{counter // 100} dari {dataset_size // 100} selesai. Jeda...")
            
            while True:
                ret, pause_frame = cap.read()
                if not ret:
                    break
                
                line1 = f"Batch {counter // 100} Selesai. Atur pose Anda."
                line2 = "Tekan 'Q' untuk Lanjut..."

                cv2.putText(pause_frame, line1, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.putText(pause_frame, line2, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                
                cv2.imshow('frame', pause_frame)
                
                if cv2.waitKey(25) == ord('q'):
                    break

print('Pengumpulan data selesai.')
cap.release()
cv2.destroyAllWindows()