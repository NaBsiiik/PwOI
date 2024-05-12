import os
from PIL import Image
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from skimage import io, color
import pandas as pd
from colorama import Fore, Style
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


"""2. Wczytywanie obrazów i wycinanie próbek"""
def crop_textures(input_dir, output_dir, crop_size):
    # Sprawdzenie istnienia katalogu wyjściowego, jeśli nie istnieje, to utworzenie go
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Pętla po plikach w katalogu wejściowym
    for filename in os.listdir(input_dir):
        # Sprawdzenie, czy plik jest obrazem
        if filename.endswith(".jpg") or filename.endswith(".jpeg"):
            # Wczytanie obrazu przy użyciu PIL
            image = Image.open(os.path.join(input_dir, filename))
            # Pobranie rozmiaru obrazu
            width, height = image.size

            # Pętla po obrazie w krokach zależnych od rozmiaru wycinanego fragmentu
            for y in range(0, height - crop_size[1] + 1, crop_size[1]):
                for x in range(0, width - crop_size[0] + 1, crop_size[0]):
                    # Wycinanie fragmentu
                    crop = image.crop((x, y, x + crop_size[0], y + crop_size[1]))
                    # Utworzenie nazwy pliku wyjściowego
                    output_filename = f"{os.path.splitext(filename)[0]}_{y}_{x}.jpg"
                    # Zapisanie fragmentu do odpowiedniego katalogu
                    crop.save(os.path.join(output_dir, output_filename))

"""3. Wczytywania próbek tekstury i wyznaczania dla nich cech tekstury na podstawie modelu macierzy zdarzeń"""

def compute_texture_features(input_dir, distances, angles):
    # Lista cech tekstur dla wszystkich próbek
    texture_features_all = []

    # Pobieramy listę plików obrazów
    image_files = os.listdir(input_dir)

    # Iterujemy po wszystkich plikach obrazów
    for filename in image_files:
        # Sprawdzamy, czy plik jest obrazem
        if not (filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png")):
            continue  # Pomijamy pliki, które nie są obrazami

        # Wczytujemy obraz
        image_path = os.path.join(input_dir, filename)
        image = io.imread(image_path)

        # Przekształcamy obraz do skali szarości
        gray_image = color.rgb2gray(image)

        # Zmniejszamy głębię jasności do 5 bitów (64 poziomy)
        gray_image = (gray_image * 63).astype(np.uint8)

        # Obliczamy macierz GLCM dla każdej kombinacji odległości i kątów
        glcm = graycomatrix(gray_image, distances=distances, angles=angles)

        # Obliczamy cechy tekstury dla każdej macierzy GLCM
        texture_features = []
        for prop in ['dissimilarity', 'correlation', 'contrast', 'energy', 'homogeneity', 'ASM']:
            # Obliczamy cechę
            feature = graycoprops(glcm, prop)
            # Dodajemy cechę do listy cech tekstury dla aktualnej próbki
            texture_features.append(feature[0, 0])  # Zapisujemy tylko pierwszą wartość z macierzy cech
        # Dodajemy nazwę pliku jako kategorię tekstury do wektora cech
        texture_features.append(filename.split('.')[0])
        # Dodajemy wektor cech do listy cech tekstur dla wszystkich próbek
        texture_features_all.append(texture_features)
        # Wyświetlamy wektor cech dla aktualnej próbki
        print(texture_features)

    return texture_features_all

# Definicja odległości pikseli i kątów
distances = [1, 3, 5]
angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]

# Definicja ścieżki do katalogu z próbkami tekstury "Drewno"
input_dir_drewno = r"C:\Users\mateu\PycharmProjects\PwOI\Ćwiczenie3\Drewno"
output_dir_drewno = r"C:\Users\mateu\PycharmProjects\PwOI\Ćwiczenie3\Drewno_probki"
crop_size = (100, 100)

# Wywołanie funkcji dla folderu "Drewno"
crop_textures(input_dir_drewno, output_dir_drewno, crop_size)

print(Fore.BLUE + Style.BRIGHT + "DREWNO" + Style.RESET_ALL)

# Obliczenie cech tekstury dla próbek "Drewno"
features_drewno = compute_texture_features(output_dir_drewno, distances, angles)

# Definicja ścieżki do katalogu z próbkami tekstury "Tynk"
input_dir_tynk = r"C:\Users\mateu\PycharmProjects\PwOI\Ćwiczenie3\Tynk"
output_dir_tynk = r"C:\Users\mateu\PycharmProjects\PwOI\Ćwiczenie3\Tynk_probki"

# Wywołanie funkcji dla folderu "Tynk"
crop_textures(input_dir_tynk, output_dir_tynk, crop_size)

print(Fore.BLUE + Style.BRIGHT + "TYNK" + Style.RESET_ALL)

# Obliczenie cech tekstury dla próbek "Tynk"
features_tynk = compute_texture_features(output_dir_tynk, distances, angles)

# Definicja ścieżki do katalogu z próbkami tekstury "Gres"
input_dir_gres = r"C:\Users\mateu\PycharmProjects\PwOI\Ćwiczenie3\Gres"
output_dir_gres = r"C:\Users\mateu\PycharmProjects\PwOI\Ćwiczenie3\Gres_probki"
crop_size = (100, 100)

# Wywołanie funkcji dla folderu "Gres"
crop_textures(input_dir_gres, output_dir_gres, crop_size)

print(Fore.BLUE + Style.BRIGHT + "GRES" + Style.RESET_ALL)

# Obliczenie cech tekstury dla próbek "Gres"
features_gres = compute_texture_features(output_dir_gres, distances, angles)

print(Fore.BLUE + Style.BRIGHT + "KAMIEN" + Style.RESET_ALL)

# Definicja ścieżki do katalogu z próbkami tekstury "Kamien"
input_dir_kamien = r"C:\Users\mateu\PycharmProjects\PwOI\Ćwiczenie3\Kamien"
output_dir_kamien = r"C:\Users\mateu\PycharmProjects\PwOI\Ćwiczenie3\Kamien_probki"

# Wywołanie funkcji dla folderu "Kamien"
crop_textures(input_dir_kamien, output_dir_kamien, crop_size)

# Obliczenie cech tekstury dla próbek "Kamien"
features_kamien = compute_texture_features(output_dir_kamien, distances, angles)

"""4. Zapisać zbiór wektorów danych do pliku csv """

print(Fore.BLUE + Style.BRIGHT + "ZAPIS" + Style.RESET_ALL)

# Tworzymy ramkę danych (DataFrame) na podstawie listy cech dla każdej tekstury
df_drewno = pd.DataFrame(features_drewno, columns=['Dissimilarity', 'Correlation', 'Contrast', 'Energy', 'Homogeneity', 'ASM', 'Texture'])
df_tynk = pd.DataFrame(features_tynk, columns=['Dissimilarity', 'Correlation', 'Contrast', 'Energy', 'Homogeneity', 'ASM', 'Texture'])
df_gres = pd.DataFrame(features_gres, columns=['Dissimilarity', 'Correlation', 'Contrast', 'Energy', 'Homogeneity', 'ASM', 'Texture'])
df_kamien = pd.DataFrame(features_kamien, columns=['Dissimilarity', 'Correlation', 'Contrast', 'Energy', 'Homogeneity', 'ASM', 'Texture'])

# Zapisujemy ramki danych do plików CSV
output_csv_drewno = r"C:\Users\mateu\PycharmProjects\PwOI\Ćwiczenie3\features_drewno.csv"
output_csv_tynk = r"C:\Users\mateu\PycharmProjects\PwOI\Ćwiczenie3\features_tynk.csv"
output_csv_gres = r"C:\Users\mateu\PycharmProjects\PwOI\Ćwiczenie3\features_gres.csv"
output_csv_kamien = r"C:\Users\mateu\PycharmProjects\PwOI\Ćwiczenie3\features_kamien.csv"

df_drewno.to_csv(output_csv_drewno, index=False)
df_tynk.to_csv(output_csv_tynk, index=False)
df_gres.to_csv(output_csv_gres, index=False)
df_kamien.to_csv(output_csv_kamien, index=False)

"""5. Klasyfikacji wektorów cech"""

# Łączymy ramki danych dla tekstur "Drewno" i "Tynk"
df_combined = pd.concat([df_drewno, df_tynk, df_gres, df_kamien])

# Dzielimy dane na zbiór treningowy i testowy
X = df_combined.drop('Texture', axis=1)  # Wektory cech (wszystko poza kolumną 'Texture')
y = df_combined['Texture']  # Kolumna 'Texture' jako etykiety klas
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=42)

# Tworzymy instancję klasyfikatora SVM
svm_classifier = SVC()

# Trenujemy klasyfikator na danych treningowych
svm_classifier.fit(X_train, y_train)

# Testujemy klasyfikator na danych testowych
y_predict = svm_classifier.predict(X_test)

# Przetwarzanie nazw plików w y_test, aby uzyskać rzeczywiste etykiety klas
y_test_labels = y_test.apply(lambda x: x.split('_')[0])

# Przetwarzanie nazw plików w y_pred, aby uzyskać przewidywane etykiety klas
y_pred_labels = [label.split('_')[0] for label in y_predict]

# Obliczanie dokładności klasyfikacji
accuracy = accuracy_score(y_test_labels, y_pred_labels)

# Wyświetlanie wyników
print("Dokładność klasyfikacji SVM:", accuracy)
