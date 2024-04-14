import numpy as np
from sklearn.cluster import KMeans

"""
Zadanie 1: Implementacja algorytmu RANSAC do dopasowania płaszczyzny do chmury punktów.
"""
def fit_plane_ransac(points, n_iterations=500, threshold_distance=1):

    best_plane = None
    best_inliers = []
    best_num_inliers = 0

    if len(points) < 3:
        return None, []

    for _ in range(n_iterations):
        # Losowo wybieramy trzy punkty, aby utworzyć próbną płaszczyznę
        sample_indices = np.random.choice(len(points), 3, replace=False)
        sample_points = [points[i] for i in sample_indices]

        # Wyznaczamy parametry płaszczyzny na podstawie wybranych punktów
        plane = calculate_plane_parameters(sample_points)

        # Liczymy odległość punktów od płaszczyzny i określamy, które punkty są wlierami
        inliers = []
        for p in points:
            distance = calculate_distance_to_plane(plane, p)
            if distance < threshold_distance:
                inliers.append(p)

        num_inliers = len(inliers)

        # Aktualizujemy najlepsze dopasowanie
        if num_inliers > best_num_inliers:
            best_plane = plane
            best_inliers = inliers
            best_num_inliers = num_inliers

    return best_plane, best_inliers


def calculate_plane_parameters(points):
    # Wyznaczamy parametry płaszczyzny na podstawie trzech punktów
    p1, p2, p3 = points
    v1 = np.array(p2) - np.array(p1)
    v2 = np.array(p3) - np.array(p1)
    normal = np.cross(v1, v2)
    d = -np.dot(normal, p1)
    return normal, d


def calculate_distance_to_plane(plane, point):
    # Liczymy odległość punktu od płaszczyzny
    normal, d = plane
    return abs(np.dot(normal, point) + d) / np.linalg.norm(normal)

"""
Zadanie 2: Wczytanie pliku XYZ zawierającego chmurę punktów w przestrzeni 3D.
"""
def load_xyz_file(filename):

    points = []
    with open(filename, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) != 3:
                parts = line.strip().split(',')
            if len(parts) == 3:
                try:
                    x, y, z = map(float, parts)
                    points.append((x, y, z))
                except ValueError:
                    print("Skipping line - invalid format:", line)
            else:
                print("Skipping line - invalid format:", line)
    return points

"""
Zadanie 3: Znalezienie rozłącznych chmur punktów za pomocą algorytmu k-średnich (dla k=3).
"""
def find_clusters(points, k=3):

    kmeans = KMeans(n_clusters=k)
    kmeans.fit(points)
    cluster_centers = kmeans.cluster_centers_
    labels = kmeans.labels_
    return cluster_centers, labels

"""
Zadanie 4: Dopasowanie płaszczyzny za pomocą algorytmu RANSAC dla każdej chmury punktów.
Wypisać na ekranie współrzędne wektora normalnego do znalezionej płaszczyzny i określić, czy dana chmura:
a. jest płaszczyzną (np. na podstawie średniej odległości wszystkich punktów chmury do tej płaszczyzny)
b. jeśli jest płaszczyzną, czy ta płaszczyzna jest pionowa, czy pozioma.
"""
def determine_plane_type(normal_vector):

    # Określenie czy płaszczyzna jest pionowa czy pozioma
    # Załóżmy, że jeśli współrzędna z wektora normalnego jest bliska 0, to płaszczyzna jest pozioma
    # W przeciwnym razie, jest pionowa
    if abs(normal_vector[2]) < 35:
        return "vertical"
    else:
        return "horizontal"


# Wczytanie plików CSV zawierających chmury punktów
filenames = ['CsvData_flat_horizontal.xyz', 'CsvData_flat_vertical.xyz', 'CylindricalData_horizontal.xyz', 'CylindricalData_vertical.xyz']

for filename in filenames:
    print(f"Processing file: {filename}")

    points = load_xyz_file(filename)

    # Dopasowanie płaszczyzny za pomocą algorytmu RANSAC
    plane, inliers = fit_plane_ransac(points, threshold_distance=1)

    if plane is not None:
        print("Parameters of the fitted plane:", plane)
        print("Number of inliers:", len(inliers))
        print("Number of points loaded:", len(points))

        # Znalezienie klastrów
        cluster_centers, labels = find_clusters(points)
        print("Cluster centers:", cluster_centers)
        print("Labels:", labels)

        # Dopasowanie płaszczyzny za pomocą algorytmu RANSAC dla każdego klastra
        for i, label in enumerate(set(labels)):
            cloud_points = [points[j] for j, l in enumerate(labels) if l == label]
            plane, inliers = fit_plane_ransac(cloud_points, threshold_distance=1)
            if plane is not None:
                normal_vector = plane[0]
                plane_type = determine_plane_type(normal_vector)
                print(f"Cloud {i + 1}:")
                print("Normal vector of the fitted plane:", normal_vector)
                print("Plane type:", plane_type)
            else:
                print(f"Could not fit a plane for cloud {i + 1}.")
    else:
        print("Could not fit a plane for the given set of points.")

