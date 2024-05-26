# Sekcja 1: Import modułów i pakietów
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# Sekcja 2: Wczytywanie danych z pliku csv
# Wczytanie zestawów danych
df_drewno = pd.read_csv(r"C:\Users\mateu\PycharmProjects\PwOI\Ćwiczenie4\features_drewno.csv")
df_tynk = pd.read_csv(r"C:\Users\mateu\PycharmProjects\PwOI\Ćwiczenie3\features_tynk.csv")
df_gres = pd.read_csv(r"C:\Users\mateu\PycharmProjects\PwOI\Ćwiczenie3\features_gres.csv")
df_kamien = pd.read_csv(r"C:\Users\mateu\PycharmProjects\PwOI\Ćwiczenie3\features_kamien.csv")

# Dodanie kategorii etykiet na podstawie początku etykiety
df_drewno['category'] = 'drewno'
df_tynk['category'] = 'tynk'
df_gres['category'] = 'gres'
df_kamien['category'] = 'kamien'

# Łączenie wszystkich zestawów danych
df_combined = pd.concat([df_drewno, df_tynk, df_gres, df_kamien])
data = df_combined.to_numpy()

# Wyodrębnienie cech (X) i etykiet (y)
X = data[:, :-2].astype('float')  # Wyłączamy kolumnę z kategorią
y = data[:, -1]  # Ostatnia kolumna to kategoria

# Sekcja 3: Wstępne przetwarzanie danych
# Kodowanie etykiet jako liczby całkowite
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(y)

# Kodowanie gorącojedynkowe
onehot_encoder = OneHotEncoder(sparse_output=False, categories='auto')
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
y_onehot = onehot_encoder.fit_transform(integer_encoded)

# Podział danych na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.3)

# Wyświetlenie kształtu danych treningowych i testowych
print("Kształt danych treningowych X_train:", X_train.shape)
print("Kształt danych testowych X_test:", X_test.shape)
print("Kształt etykiet treningowych y_train:", y_train.shape)
print("Kształt etykiet testowych y_test:", y_test.shape)
unique_labels = np.unique(y)
print("Unikalne etykiety:", unique_labels)
print("Liczba unikalnych etykiet:", len(unique_labels))

# Sekcja 4: Tworzenie modelu sieci neuronowej
# Tworzenie modelu sieci neuronowej
model = Sequential()
#model.add(Dense(10, input_dim=6, activation='sigmoid'))
model.add(Dense(64, input_dim=6, activation='relu'))
model.add(Dense(4, activation='softmax'))

# Kompilacja modelu
model.compile(loss='categorical_crossentropy', optimizer="sgd", metrics=['accuracy'])
model.summary()

# Sekcja 5: Trenowanie sieci
# Trenowanie modelu
history = model.fit(X_train, y_train, epochs=100, batch_size=10, shuffle=True)

# Sekcja 6: Testowanie sieci
# Predykcje na zbiorze testowym
y_pred = model.predict(X_test)

# Konwersja wyników do postaci etykiet całkowitoliczbowych
y_pred_int = np.argmax(y_pred, axis=1)
y_test_int = np.argmax(y_test, axis=1)

# Wyznaczenie macierzy pomyłek
cm = confusion_matrix(y_test_int, y_pred_int)

# Opisy kategorii
categories = ['D', 'T', 'G', 'K']

# Tworzenie ramki danych z opisami kategorii
confusion_df = pd.DataFrame(cm, index=categories, columns=categories)

# Wyświetlenie czytelniejszej macierzy pomyłek
print("Macierz pomyłek:")
print(confusion_df)


