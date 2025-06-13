import pickle
import cv2
import mediapipe as mp
import numpy as np


try:
    with open('model_1_hand.p', 'rb') as f:
        model_1_hand = pickle.load(f)['model']
    with open('model_2_hand.p', 'rb') as f:
        model_2_hands = pickle.load(f)['model']
except FileNotFoundError as e:
    print(f"Error: Pastikan file model .p sudah ada. {e}")
    exit()

cap = cv2.VideoCapture(2)

# Inisialisasi MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Atur untuk bisa mendeteksi hingga 2 tangan
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

# Gabungkan semua label dalam satu kamus
labels_dict = {
    '0': 'C', '1': 'E', '2': 'I', '3': 'J', '4': 'L',
    '5': 'O', '6': 'R', '7': 'U', '8': 'V', '9': 'Z',
    '10': 'A', '11': 'B', '12': 'D', '13': 'F', '14': 'G',
    '15': 'H', '16': 'K', '17': 'M', '18': 'N', '19': 'P',
    '20': 'Q', '21': 'S', '22': 'T', '23': 'W', '24': 'X',
    '25': 'Y',

    'AYAH': 'Ayah',
    'BERJALAN': 'Berjalan',
    'BERMAIN': 'Bermain',
    'BICARA': 'Bicara',
    'DUDUK': 'Duduk',
    'KAMU': 'Kamu',
    'MAAF': 'Maaf',
    'MAKAN': 'Makan',
    'MEMBACA': 'Membaca',
    'MELIHAT': 'Melihat',
    'MINUM': 'Minum',
    'SAYA': 'Saya',
    'SIAPA': 'Siapa',
    'HALO': 'Halo',
    'IBU': 'Ibu',
    'SUKA': 'Suka',
    'TERIMA KASIH': 'Terima Kasih',
    'NAMA': 'Nama',
    'RUMAH': 'Rumah',
    'SEHAT': 'Sehat'
}

while True:
    ret, frame = cap.read()
    if not ret:
        print("Gagal membaca frame. Keluar...")
        break

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        num_hands = len(results.multi_hand_landmarks)

        # --- KASUS 1: SATU TANGAN TERDETEKSI ---
        if num_hands == 1:
            hand_landmarks = results.multi_hand_landmarks[0]
            
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                    mp_drawing_styles.get_default_hand_landmarks_style(),
                                    mp_drawing_styles.get_default_hand_connections_style())
            
            data_aux, x_, y_ = [], [], []
            for lm in hand_landmarks.landmark:
                x_.append(lm.x)
                y_.append(lm.y)
            for lm in hand_landmarks.landmark:
                data_aux.append(lm.x - min(x_))
                data_aux.append(lm.y - min(y_))

            prediction = model_1_hand.predict([np.asarray(data_aux)])
            predicted_character = labels_dict[str(prediction[0])]

            x1, y1 = int(min(x_) * W) - 10, int(min(y_) * H) - 10
            x2, y2 = int(max(x_) * W) + 10, int(max(y_) * H) + 10
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
            cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 3, cv2.LINE_AA)

        # --- KASUS 2: DUA TANGAN TERDETEKSI ---
        elif num_hands == 2:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                        mp_drawing_styles.get_default_hand_landmarks_style(),
                                        mp_drawing_styles.get_default_hand_connections_style())

            data_aux, x_, y_ = [], [], []
            all_landmarks = []
            handedness_list = [item.classification[0].label for item in results.multi_handedness]
            
            if 'Left' in handedness_list and 'Right' in handedness_list:
                left_index = handedness_list.index('Left')
                right_index = handedness_list.index('Right')
                
                all_landmarks.extend(results.multi_hand_landmarks[left_index].landmark)
                all_landmarks.extend(results.multi_hand_landmarks[right_index].landmark)
            else:
                continue 

            for lm in all_landmarks:
                x_.append(lm.x)
                y_.append(lm.y)
            
            for lm in all_landmarks:
                data_aux.append(lm.x - min(x_))
                data_aux.append(lm.y - min(y_))
            
            prediction = model_2_hands.predict([np.asarray(data_aux)])
            predicted_character = labels_dict[str(prediction[0])]
            
            x1, y1 = int(min(x_) * W) - 10, int(min(y_) * H) - 10
            x2, y2 = int(max(x_) * W) + 10, int(max(y_) * H) + 10
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 3, cv2.LINE_AA)

    cv2.imshow('frame', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print("Menutup aplikasi...")
cap.release()
cv2.destroyAllWindows()