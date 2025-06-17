# File: app.py
import streamlit as st
import cv2
import pickle
import mediapipe as mp
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av
import time

# Konfigurasi halaman Streamlit
st.set_page_config(page_title="Deteksi BISINDO", layout="wide")
st.title("🤟 Deteksi Bahasa Isyarat Indonesia (BISINDO)")

# Fungsi untuk memuat model dan resource lainnya dengan caching
@st.cache_resource
def load_resources():
    try:
        with open('model_1_hand.p', 'rb') as f:
            model_1_hand = pickle.load(f)['model']
        with open('model_2_hand.p', 'rb') as f:
            model_2_hands = pickle.load(f)['model']
        hands = mp.solutions.hands.Hands(
            static_image_mode=False, max_num_hands=2,
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        )
        drawing_utils = mp.solutions.drawing_utils
        drawing_styles = mp.solutions.drawing_styles
        st.success("Resource (model dan MediaPipe) berhasil dimuat dari cache.")
        return model_1_hand, model_2_hands, hands, drawing_utils, drawing_styles
    except FileNotFoundError:
        st.error("File model tidak ditemukan. Pastikan 'model_1_hand.p' dan 'model_2_hand.p' ada di repository Anda.")
        return None, None, None, None, None
    except Exception as e:
        st.error(f"Terjadi error saat memuat resource: {e}")
        return None, None, None, None, None

# Memuat semua resource yang dibutuhkan
model_1_hand, model_2_hands, mp_hands_instance, mp_drawing, mp_drawing_styles = load_resources()

# Kamus label untuk hasil prediksi
labels_dict = {
    '0': 'C', '1': 'E', '2': 'I', '3': 'J', '4': 'L', '5': 'O', '6': 'R', '7': 'U', '8': 'V',
    '9': 'Z', '10': 'A', '11': 'B', '12': 'D', '13': 'F', '14': 'G', '15': 'H', '16': 'K',
    '17': 'M', '18': 'N', '19': 'P', '20': 'Q', '21': 'S', '22': 'T', '23': 'W', '24': 'X',
    '25': 'Y', 'AYAH': 'Ayah', 'BERJALAN': 'Berjalan', 'BERMAIN': 'Bermain', 'BICARA': 'Bicara',
    'DUDUK': 'Duduk', 'KAMU': 'Kamu', 'MAAF': 'Maaf', 'MAKAN': 'Makan', 'MEMBACA': 'Membaca',
    'MELIHAT': 'Melihat', 'MINUM': 'Minum', 'SAYA': 'Saya', 'SIAPA': 'Siapa', 'HALO': 'Halo',
    'IBU': 'Ibu', 'SUKA': 'Suka', 'TERIMA KASIH': 'Terima Kasih', 'NAMA': 'Nama',
    'RUMAH': 'Rumah', 'SEHAT': 'Sehat'
}

# Sidebar untuk navigasi dan informasi
st.sidebar.header("Tentang Proyek")
st.sidebar.info("Aplikasi ini menggunakan MediaPipe dan Scikit-learn untuk mengenali isyarat BISINDO. Dioptimalkan untuk berjalan di Streamlit Community Cloud.")
st.sidebar.header("Pilih Mode Input")
app_mode = st.sidebar.radio("Pilih mode:", ('Kamera Langsung', 'Unggah Gambar'))

# Memastikan resource berhasil dimuat sebelum menjalankan aplikasi
if not all((model_1_hand, model_2_hands, mp_hands_instance, mp_drawing, mp_drawing_styles)):
    st.warning("Aplikasi tidak dapat berjalan karena resource gagal dimuat.")
else:
    # Mode Kamera Langsung
    if app_mode == 'Kamera Langsung':
        st.header("Deteksi Real-Time")
        st.info("💡 Hasil prediksi akan muncul di dalam kotak di pojok kanan bawah video.")

        class SignLanguageProcessor(VideoProcessorBase):
            def __init__(self):
                self.frame_counter = 0
                self.process_every_n_frame = 2
                self.last_predicted_char = ""
                self.last_prediction_time = 0

            def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
                img = frame.to_ndarray(format="bgr24")
                target_width = 480
                scale = target_width / img.shape[1]
                dim = (target_width, int(img.shape[0] * scale))
                img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
                
                H, W, _ = img.shape
                self.frame_counter += 1

                if self.frame_counter % self.process_every_n_frame == 0:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    results = mp_hands_instance.process(img_rgb)
                    current_prediction = ""
                    if results.multi_hand_landmarks:
                        num_hands = len(results.multi_hand_landmarks)
                        data_aux, x_, y_ = [], [], []
                        if num_hands == 1:
                            hand_landmarks = results.multi_hand_landmarks[0]
                            for lm in hand_landmarks.landmark: x_.append(lm.x); y_.append(lm.y)
                            for lm in hand_landmarks.landmark: data_aux.append(lm.x - min(x_)); data_aux.append(lm.y - min(y_))
                            prediction = model_1_hand.predict([np.asarray(data_aux)])
                            current_prediction = labels_dict.get(str(prediction[0]), '?')
                        elif num_hands == 2:
                            handedness_list = [h.classification[0].label for h in results.multi_handedness]
                            if 'Left' in handedness_list and 'Right' in handedness_list:
                                left_idx = handedness_list.index('Left'); right_idx = handedness_list.index('Right')
                                all_landmarks = []
                                all_landmarks.extend(results.multi_hand_landmarks[left_idx].landmark)
                                all_landmarks.extend(results.multi_hand_landmarks[right_idx].landmark)
                                for lm in all_landmarks: x_.append(lm.x); y_.append(lm.y)
                                for lm in all_landmarks: data_aux.append(lm.x - min(x_)); data_aux.append(lm.y - min(y_))
                                prediction = model_2_hands.predict([np.asarray(data_aux)])
                                current_prediction = labels_dict.get(str(prediction[0]), '?')
                        if current_prediction:
                            self.last_predicted_char = current_prediction
                            self.last_prediction_time = time.time()
                
                if time.time() - self.last_prediction_time < 2.0:
                    if self.last_predicted_char:
                        text = self.last_predicted_char
                        font_face = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 2.5
                        thickness = 4
                        margin = 10
                        
                        text_size, _ = cv2.getTextSize(text, font_face, font_scale, thickness)
                        text_w, text_h = text_size

                        box_x1 = W - text_w - 2 * margin
                        box_y1 = H - text_h - 2 * margin
                        box_x2 = W - margin
                        box_y2 = H - margin

                        overlay = img.copy()
                        cv2.rectangle(overlay, (box_x1, box_y1), (box_x2, box_y2), (0, 255, 0), -1)
                        alpha = 0.4
                        img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
                        
                        text_x = W - text_w - margin - 5
                        text_y = H - margin - 10
                        cv2.putText(img, text, (text_x, text_y), font_face, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

                return av.VideoFrame.from_ndarray(img, format="bgr24")

        col1, col2 = st.columns([2, 1])
        with col1:
            RTC_CONFIGURATION = RTCConfiguration({
                "iceServers": [
                    {"urls": ["stun:stun.l.google.com:19302"]},
                    {
                        "urls": ["turn:openrelay.metered.ca:80"],
                        "username": "openrelayproject",
                        "credential": "openrelayproject",
                    },
                    {
                        "urls": ["turn:openrelay.metered.ca:443"],
                        "username": "openrelayproject",
                        "credential": "openrelayproject",
                    }
                ]
            })
            
            webrtc_streamer(
                key="BISINDO-Detector", 
                video_processor_factory=SignLanguageProcessor,
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True, 
                rtc_configuration=RTC_CONFIGURATION
            )
        with col2:
            st.subheader("Informasi")
            st.write("Aplikasi akan mendeteksi isyarat tangan dari video di sebelah kiri.")
            st.write("Hasil deteksi akan muncul di dalam kotak di pojok kanan bawah.")

    # Mode Unggah Gambar
    elif app_mode == 'Unggah Gambar':
        st.header("Prediksi dari Gambar")
        uploaded_file = st.file_uploader("Pilih sebuah gambar...", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, 1)
            st.image(img, channels="BGR", caption="Gambar yang Diunggah")
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = mp_hands_instance.process(img_rgb)
            if results.multi_hand_landmarks:
                num_hands = len(results.multi_hand_landmarks)
                if num_hands == 1:
                    hand_landmarks = results.multi_hand_landmarks[0]
                    data_aux, x_, y_ = [], [], []
                    for lm in hand_landmarks.landmark: x_.append(lm.x); y_.append(lm.y)
                    for lm in hand_landmarks.landmark: data_aux.append(lm.x - min(x_)); data_aux.append(lm.y - min(y_))
                    prediction = model_1_hand.predict([np.asarray(data_aux)])
                    predicted_character = labels_dict.get(str(prediction[0]), '?')
                    st.success(f"Hasil Prediksi (1 Tangan): {predicted_character}")
                elif num_hands == 2:
                    handedness_list = [h.classification[0].label for h in results.multi_handedness]
                    if 'Left' in handedness_list and 'Right' in handedness_list:
                        left_idx = handedness_list.index('Left'); right_idx = handedness_list.index('Right')
                        all_landmarks = list(results.multi_hand_landmarks[left_idx].landmark)
                        all_landmarks.extend(results.multi_hand_landmarks[right_idx].landmark)
                        data_aux, x_, y_ = [], [], []
                        for lm in all_landmarks: x_.append(lm.x); y_.append(lm.y)
                        for lm in all_landmarks: data_aux.append(lm.x - min(x_)); data_aux.append(lm.y - min(y_))
                        prediction = model_2_hands.predict([np.asarray(data_aux)])
                        predicted_character = labels_dict.get(str(prediction[0]), '?')
                        st.success(f"Hasil Prediksi (2 Tangan): {predicted_character}")
                    else:
                        st.warning("Terdeteksi dua tangan, tapi bukan pasangan Kiri-Kanan.")
            else:
                st.warning("Tidak ada tangan yang terdeteksi pada gambar.")