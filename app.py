import streamlit as st
import cv2
import pickle
import mediapipe as mp
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av
import time

st.set_page_config(page_title="Deteksi BISINDO", layout="wide")
st.title("🤟 Deteksi Bahasa Isyarat Indonesia (BISINDO)")

@st.cache_resource
def load_resources():
    try:
        with open('model_1_hand.p', 'rb') as f:
            model_1_hand = pickle.load(f)['model']
        with open('model_2_hand.p', 'rb') as f:
            model_2_hands = pickle.load(f)['model']
        
        hands = mp.solutions.hands.Hands(
            static_image_mode=False, max_num_hands=2,
            min_detection_confidence=0.5, min_tracking_confidence=0.5)
        
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        
        return model_1_hand, model_2_hands, hands, mp_drawing, mp_drawing_styles
    except Exception as e:
        st.error(f"Gagal memuat resource: {e}")
        return None, None, None, None, None

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

st.sidebar.header("Tentang Proyek")
st.sidebar.info("Aplikasi ini menggunakan MediaPipe dan Scikit-learn untuk mengenali isyarat BISINDO.")
st.sidebar.header("Pilih Mode Input")
app_mode = st.sidebar.radio("Pilih mode:", ('Kamera Langsung', 'Unggah Gambar'))

if app_mode == 'Kamera Langsung':
    st.header("Deteksi Real-Time")
    
    # --- PERUBAHAN UTAMA DIMULAI DI SINI ---
    # Buat dua kolom, kolom kiri 2x lebih besar dari kolom kanan
    col1, col2 = st.columns([2, 1])

    with col1:
        # Pindahkan webrtc_streamer ke dalam kolom 1
        class SignLanguageProcessor(VideoProcessorBase):
            def __init__(self):
                self.model_1_hand = None
                self.model_2_hands = None
                self.mp_hands_instance = None
                self.mp_drawing = None
                self.mp_drawing_styles = None
                self.frame_counter = 0
                self.process_every_n_frame = 2
                self.last_predicted_char = ""
                self.last_prediction_time = 0

            def _load_resources_if_needed(self):
                if self.model_1_hand is None:
                    (self.model_1_hand, self.model_2_hands, self.mp_hands_instance, 
                     self.mp_drawing, self.mp_drawing_styles) = load_resources()

            def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
                self._load_resources_if_needed()
                
                if not self.mp_hands_instance:
                    return frame

                img = frame.to_ndarray(format="bgr24")
                
                predicted_character = ""
                
                self.frame_counter += 1
                if self.frame_counter % self.process_every_n_frame == 0:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    results = self.mp_hands_instance.process(img_rgb)

                    if results.multi_hand_landmarks:
                        num_hands = len(results.multi_hand_landmarks)
                        
                        if num_hands == 1:
                            hand_landmarks = results.multi_hand_landmarks[0]
                            data_aux, x_, y_ = [], [], []
                            for lm in hand_landmarks.landmark: x_.append(lm.x); y_.append(lm.y)
                            for lm in hand_landmarks.landmark: data_aux.append(lm.x - min(x_)); data_aux.append(lm.y - min(y_))
                            prediction = self.model_1_hand.predict([np.asarray(data_aux)])
                            predicted_character = labels_dict.get(str(prediction[0]), '?')

                        elif num_hands == 2:
                            handedness_list = [h.classification[0].label for h in results.multi_handedness]
                            if 'Left' in handedness_list and 'Right' in handedness_list:
                                data_aux, x_, y_, all_landmarks = [], [], [], []
                                left_idx = handedness_list.index('Left')
                                right_idx = handedness_list.index('Right')
                                all_landmarks.extend(results.multi_hand_landmarks[left_idx].landmark)
                                all_landmarks.extend(results.multi_hand_landmarks[right_idx].landmark)
                                for lm in all_landmarks: x_.append(lm.x); y_.append(lm.y)
                                for lm in all_landmarks: data_aux.append(lm.x - min(x_)); data_aux.append(lm.y - min(y_))
                                prediction = self.model_2_hands.predict([np.asarray(data_aux)])
                                predicted_character = labels_dict.get(str(prediction[0]), '?')
                    
                    if predicted_character:
                        self.last_predicted_char = predicted_character
                        self.last_prediction_time = time.time()
                
                if time.time() - self.last_prediction_time < 2.0 and self.last_predicted_char:
                    text = self.last_predicted_char
                    font_face = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 2.0
                    thickness = 3
                    margin = 10
                    H, W, _ = img.shape
                    
                    text_size, _ = cv2.getTextSize(text, font_face, font_scale, thickness)
                    text_w, text_h = text_size

                    box_x1 = W - text_w - 2 * margin
                    box_y1 = H - text_h - 2 * margin
                    box_x2 = W - margin
                    box_y2 = H - margin

                    overlay = img.copy()
                    cv2.rectangle(overlay, (box_x1, box_y1), (box_x2, box_y2), (0, 0, 0), -1)
                    alpha = 0.5 
                    img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
                    
                    text_x = W - text_w - margin - 5
                    text_y = H - margin - 10
                    cv2.putText(img, text, (text_x, text_y), font_face, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

                return av.VideoFrame.from_ndarray(img, format="bgr24")
        
        RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
        webrtc_streamer(
            key="BISINDO-Detector",
            video_processor_factory=SignLanguageProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
            rtc_configuration=RTC_CONFIGURATION
            )

    with col2:
        # Tambahkan semua teks informasi di dalam kolom 2
        st.subheader("Informasi")
        st.write("Aplikasi akan mendeteksi isyarat tangan dari video di sebelah kiri.")
        st.write("Hasil deteksi akan muncul di dalam kotak di pojok kanan bawah.")

elif app_mode == 'Unggah Gambar':
    st.header("Prediksi dari Gambar")
    model_1_hand, model_2_hands, mp_hands_instance, _, _ = load_resources()

    if not mp_hands_instance:
        st.warning("Gagal memuat resource yang dibutuhkan untuk prediksi gambar.")
    else:
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
            else:
                st.warning("Tidak ada tangan yang terdeteksi pada gambar.")