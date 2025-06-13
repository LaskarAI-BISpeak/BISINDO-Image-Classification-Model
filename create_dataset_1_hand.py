import os
import pickle
import mediapipe as mp
import cv2

# Inisialisasi MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.3)

DATA_DIRS = ['./dataset_1_tangan']

data = []
labels = []

print("Memulai pemrosesan dataset...")

for DATA_DIR in DATA_DIRS:
    for dir_name in os.listdir(DATA_DIR):
        dir_path = os.path.join(DATA_DIR, dir_name)
        
        if not os.path.isdir(dir_path):
            continue

        print(f'Memproses direktori: {dir_name}')
        
        for img_name in os.listdir(dir_path):
            data_aux = []
            x_ = []
            y_ = []

            img_path = os.path.join(dir_path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                print(f"Gagal membaca gambar: {img_path}")
                continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                
                for landmark in hand_landmarks.landmark:
                    x_.append(landmark.x)
                    y_.append(landmark.y)

                for landmark in hand_landmarks.landmark:
                    data_aux.append(landmark.x - min(x_))
                    data_aux.append(landmark.y - min(y_))
                
                data.append(data_aux)
                labels.append(dir_name)

print("Pemrosesan selesai. Menyimpan data...")
with open('data_1_tangan.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print("Dataset berhasil disimpan sebagai data.pickle")