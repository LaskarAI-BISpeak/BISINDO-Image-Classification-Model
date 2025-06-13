import os
import pickle
import mediapipe as mp
import cv2

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.3)

DATA_DIRS = ['./dataset_2_tangan']

data = []
labels = []

print("Memulai pemrosesan dataset untuk DUA TANGAN...")


for DATA_DIR in DATA_DIRS:
    for dir_name in os.listdir(DATA_DIR):
        dir_path = os.path.join(DATA_DIR, dir_name)
        
        if not os.path.isdir(dir_path):
            continue

        print(f'Memproses direktori: {dir_name}')
        
        # Loop melalui setiap gambar dalam direktori
        for img_name in os.listdir(dir_path):
            img_path = os.path.join(dir_path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)

            if results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 2:
                data_aux = []
                x_ = []
                y_ = []
                
                all_landmarks = []
                handedness_list = [item.classification[0].label for item in results.multi_handedness]
                
                if 'Left' in handedness_list and 'Right' in handedness_list:
                    left_index = handedness_list.index('Left')
                    right_index = handedness_list.index('Right')
                    
                    all_landmarks.extend(results.multi_hand_landmarks[left_index].landmark)
                    all_landmarks.extend(results.multi_hand_landmarks[right_index].landmark)
                else:
                    all_landmarks.extend(results.multi_hand_landmarks[0].landmark)
                    all_landmarks.extend(results.multi_hand_landmarks[1].landmark)

                for landmark in all_landmarks:
                    x_.append(landmark.x)
                    y_.append(landmark.y)

                min_x = min(x_)
                min_y = min(y_)
                for landmark in all_landmarks:
                    data_aux.append(landmark.x - min_x)
                    data_aux.append(landmark.y - min_y)

                data.append(data_aux)
                labels.append(dir_name)

print("Pemrosesan selesai. Menyimpan data...")
with open('data_2_tangan.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print("Dataset dua tangan berhasil disimpan sebagai data.pickle")