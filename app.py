# File: app.py
import streamlit as st
import cv2
import pickle
import mediapipe as mp
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av

st.set_page_config(page_title="Deteksi BISINDO", layout="wide")
st.title("🤟 Deteksi Bahasa Isyarat Indonesia (BISINDO)")

@st.cache_resource
def load_models():
    try:
        with open('model_1_hand.p', 'rb') as f:
            model_1_hand = pickle.load(f)['model']
        with open('model_2_hand.p', 'rb') as f:
            model_2_hands = pickle.load(f)['model']
        return model_1_hand, model_2_hands
    except FileNotFoundError as e:
        st.error(f"File model tidak ditemukan. Pastikan file 'model_1_hand_xgb_tuned.p' dan 'model_2_hand.p' tersedia. Detail: {e}")
        return None, None

model_1_hand, model_2_hands = load_models()

# Gabungkan semua label dalam satu kamus
labels_dict = {
    '0': 'C', '1': 'E', '2': 'I', '3': 'J', '4': 'L',
    '5': 'O', '6': 'R', '7': 'U', '8': 'V', '9': 'Z',
    '10': 'A', '11': 'B', '12': 'D', '13': 'F', '14': 'G',
    '15': 'H', '16': 'K', '17': 'M', '18': 'N', '19': 'P',
    '20': 'Q', '21': 'S', '22': 'T', '23': 'W', '24': 'X',
    '25': 'Y',

    # Label teks dengan huruf pertama kapital
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

# --- SIDEBAR ---
st.sidebar.header("Tentang Proyek")
st.sidebar.info("Aplikasi ini menggunakan MediaPipe dan Scikit-learn untuk mengenali isyarat satu tangan dan dua tangan dalam sistem BISINDO.")

st.sidebar.header("Pilih Mode Input")
app_mode = st.sidebar.radio(
    "Pilih antara deteksi real-time atau unggah gambar:",
    ('Kamera Langsung', 'Unggah Gambar')
)

# Inisialisasi MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# =================================================================================
# MODE 1: KAMERA LANGSUNG
# =================================================================================
if app_mode == 'Kamera Langsung':
    st.header("Deteksi Real-Time")

    start_camera = st.checkbox("Aktifkan Kamera", value=True)

    if start_camera:
        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        class SignLanguageProcessor(VideoProcessorBase):
            def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
                img = frame.to_ndarray(format="bgr24")
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = hands.process(img_rgb)

                if results.multi_hand_landmarks and model_1_hand and model_2_hands:
                    num_hands = len(results.multi_hand_landmarks)
                    H, W, _ = img.shape

                    # --- KASUS 1: SATU TANGAN TERDETEKSI ---
                    if num_hands == 1:
                        hand_landmarks = results.multi_hand_landmarks[0]
                        mp_drawing.draw_landmarks(
                            img, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style())

                        data_aux, x_, y_ = [], [], []
                        for lm in hand_landmarks.landmark:
                            x_.append(lm.x); y_.append(lm.y)
                        for lm in hand_landmarks.landmark:
                            data_aux.append(lm.x - min(x_)); data_aux.append(lm.y - min(y_))

                        prediction = model_1_hand.predict([np.asarray(data_aux)])
                        # Menyesuaikan key lookup dengan inference_final.py
                        predicted_character = labels_dict.get(str(prediction[0]), '?')

                        x1, y1 = int(min(x_) * W) - 10, int(min(y_) * H) - 10
                        x2, y2 = int(max(x_) * W) + 10, int(max(y_) * H) + 10
                        
                        # Menyesuaikan gaya visualisasi dengan inference_final.py
                        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 2) # Kotak putih
                        cv2.putText(img, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 3, cv2.LINE_AA) # Teks merah

                    # --- KASUS 2: DUA TANGAN TERDETEKSI ---
                    elif num_hands == 2:
                        for hand_landmarks in results.multi_hand_landmarks:
                           mp_drawing.draw_landmarks(
                               img, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                               mp_drawing_styles.get_default_hand_landmarks_style(),
                               mp_drawing_styles.get_default_hand_connections_style())

                        data_aux, x_, y_, all_landmarks = [], [], [], []
                        handedness_list = [h.classification[0].label for h in results.multi_handedness]
                        
                        if 'Left' in handedness_list and 'Right' in handedness_list:
                            left_idx = handedness_list.index('Left')
                            right_idx = handedness_list.index('Right')
                            
                            # Mengurutkan landmark: Kiri dulu, baru Kanan
                            all_landmarks.extend(results.multi_hand_landmarks[left_idx].landmark)
                            all_landmarks.extend(results.multi_hand_landmarks[right_idx].landmark)

                            for lm in all_landmarks:
                                x_.append(lm.x); y_.append(lm.y)
                            for lm in all_landmarks:
                                data_aux.append(lm.x - min(x_)); data_aux.append(lm.y - min(y_))

                            prediction = model_2_hands.predict([np.asarray(data_aux)])
                            # Menyesuaikan key lookup
                            predicted_character = labels_dict.get(str(prediction[0]), '?')

                            x1, y1 = int(min(x_) * W) - 10, int(min(y_) * H) - 10
                            x2, y2 = int(max(x_) * W) + 10, int(max(y_) * H) + 10
                            
                            # Menyesuaikan gaya visualisasi
                            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2) # Kotak merah
                            cv2.putText(img, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 3, cv2.LINE_AA) # Teks merah

                return av.VideoFrame.from_ndarray(img, format="bgr24")

        RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
        webrtc_streamer(
            key="BISINDO-Detector",
            video_processor_factory=SignLanguageProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
            rtc_configuration=RTC_CONFIGURATION
        )
    else:
        st.info("Kamera tidak aktif. Aktifkan dengan mencentang kotak di atas.")

# =================================================================================
# MODE 2: UNGGAH GAMBAR
# =================================================================================
elif app_mode == 'Unggah Gambar':
    st.header("Prediksi dari Gambar")
    uploaded_file = st.file_uploader("Pilih sebuah gambar...", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)

        # Salinan gambar untuk ditampilkan tanpa anotasi hasil
        img_display = img.copy()

        # Resize agar tidak terlalu besar
        max_width = 640
        if img.shape[1] > max_width:
            scale = max_width / img.shape[1]
            img = cv2.resize(img, (max_width, int(img.shape[0] * scale)))
            img_display = cv2.resize(img_display, (max_width, int(img_display.shape[0] * scale)))

        st.image(img_display, channels="BGR", caption="Gambar yang Diunggah")

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks and model_1_hand and model_2_hands:
            num_hands = len(results.multi_hand_landmarks)
            
            # --- KASUS 1: SATU TANGAN TERDETEKSI ---
            if num_hands == 1:
                hand_landmarks = results.multi_hand_landmarks[0]
                data_aux, x_, y_ = [], [], []
                for lm in hand_landmarks.landmark:
                    x_.append(lm.x); y_.append(lm.y)
                for lm in hand_landmarks.landmark:
                    data_aux.append(lm.x - min(x_)); data_aux.append(lm.y - min(y_))
                
                prediction = model_1_hand.predict([np.asarray(data_aux)])
                predicted_character = labels_dict.get(str(prediction[0]), '?')
                st.success(f"Hasil Prediksi (1 Tangan): {predicted_character}")

            # --- KASUS 2: DUA TANGAN TERDETEKSI ---
            elif num_hands == 2:
                handedness_list = [h.classification[0].label for h in results.multi_handedness]
                if 'Left' in handedness_list and 'Right' in handedness_list:
                    left_idx = handedness_list.index('Left')
                    right_idx = handedness_list.index('Right')
                    
                    all_landmarks = list(results.multi_hand_landmarks[left_idx].landmark)
                    all_landmarks.extend(results.multi_hand_landmarks[right_idx].landmark)

                    data_aux, x_, y_ = [], [], []
                    for lm in all_landmarks:
                        x_.append(lm.x); y_.append(lm.y)
                    for lm in all_landmarks:
                        data_aux.append(lm.x - min(x_)); data_aux.append(lm.y - min(y_))
                        
                    prediction = model_2_hands.predict([np.asarray(data_aux)])
                    # Menyesuaikan key lookup
                    predicted_character = labels_dict.get(str(prediction[0]), '?')
                    st.success(f"Hasil Prediksi (2 Tangan): {predicted_character}")
                else:
                    st.warning("Terdeteksi dua tangan, tapi bukan pasangan Kiri-Kanan yang jelas. Prediksi mungkin tidak akurat.")

            # Menggambar landmark pada gambar untuk ditampilkan
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    img, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
            st.image(img, channels="BGR", caption="Gambar dengan Hasil Deteksi")
        else:
            st.warning("Tidak ada tangan yang terdeteksi pada gambar.")

        hands.close()