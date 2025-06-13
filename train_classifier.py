import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

print("Memuat dataset...")
with open('data_1_tangan.pickle', 'rb') as f: #jika 2_tangan maka ganti ini
    data_dict = pickle.load(f)

data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, shuffle=True, stratify=labels
)

model = RandomForestClassifier()

print("Melatih model...")
model.fit(x_train, y_train)

y_predict = model.predict(x_test)

score = accuracy_score(y_predict, y_test)
print('{}% sampel berhasil diklasifikasikan!'.format(score * 100))

print("Menyimpan model...")
with open('model_1_hand_RF.p', 'wb') as f: #jika 2_tangan maka ganti ini
    pickle.dump({'model': model}, f)

print("Model berhasil disimpan sebagai model_1_hand.p") 