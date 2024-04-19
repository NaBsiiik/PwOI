import numpy as np
from sklearn.cluster import KMeans
import pyransac3d as pyrsc
from colorama import Fore, Style

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
        sample_indices = np.random.choice(len(points), 3, replace=False)
        sample_points = [points[i] for i in sample_indices]
        plane = calculate_plane_parameters(sample_points)

        inliers = []
        for p in points:
            distance = calculate_distance_to_plane(plane, p)
            if distance < threshold_distance:
                inliers.append(p)

        num_inliers = len(inliers)

        if num_inliers > best_num_inliers:
            best_plane = plane
            best_inliers = inliers
            best_num_inliers = num_inliers

    return best_plane, best_inliers


def calculate_plane_parameters(points):
    p1, p2, p3 = points
    v1 = np.array(p2) - np.array(p1)
    v2 = np.array(p3) - np.array(p1)
    normal = np.cross(v1, v2)
    d = -np.dot(normal, p1)
    return normal, d


def calculate_distance_to_plane(plane, point):
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

    if abs(normal_vector[2]) < 35:
        return "vertical"
    else:
        return "horizontal"


# Wczytanie plików CSV zawierających chmury punktów
filenames = ['Horizontal_flat.xyz', 'Vertical_flat.xyz', 'Horizontal_cylindrical.xyz', 'Vertical_cylindrical.xyz']

print(Fore.BLUE + Style.BRIGHT + "Część 1 - Wykorzystanie algorytmu RANSAC" + Style.RESET_ALL)

for filename in filenames:
    print(Fore.GREEN + Style.BRIGHT + f"Processing file: {filename}" + Style.RESET_ALL)

    points = load_xyz_file(filename)

    # Dopasowanie płaszczyzny za pomocą algorytmu RANSAC
    plane, inliers = fit_plane_ransac(points, threshold_distance=1)

    if plane is not None:
        # Znalezienie klastrów
        cluster_centers, labels = find_clusters(points)

        # Dopasowanie płaszczyzny za pomocą algorytmu RANSAC dla każdego klastra
        for i, label in enumerate(set(labels)):
            cloud_points = [points[j] for j, l in enumerate(labels) if l == label]
            plane, inliers = fit_plane_ransac(cloud_points, threshold_distance=1)
            if plane is not None:
                normal_vector = plane[0]
                plane_type = determine_plane_type(normal_vector)
                print(Fore.CYAN + f"Cloud {i + 1}:" + Style.RESET_ALL)
                print("Normal vector of the fitted plane:", normal_vector)
                print(Fore.YELLOW + "Plane type:" + Fore.MAGENTA + " " + plane_type + Style.RESET_ALL)
            else:
                print(f"Could not fit a plane for cloud {i + 1}.")
    else:
        print("Could not fit a plane for the given set of points.")
        print("Could not fit a plane for the given set of points.")

# Modyfikacja zadania 3 wykorzystując algorytm DBSCAN

from sklearn.cluster import DBSCAN

def find_clusters2(points, epsilon=400, min_samples=780):
    dbscan = DBSCAN(eps=epsilon, min_samples=min_samples)
    labels = dbscan.fit_predict(points)
    unique_labels = np.unique(labels)
    cluster_centers = []
    for label in unique_labels:
        cluster_points = points[labels == label]
        cluster_center = np.mean(cluster_points, axis=0)
        cluster_centers.append(cluster_center)
    return np.array(cluster_centers), labels

# Modyfikacja zadania 4 wykorzystując dedykowane funkcje z pakiet pyransac3d

def load_xyz_file2(filename):
    points = np.loadtxt(filename, delimiter=',')
    return points

print(Fore.BLUE + Style.BRIGHT + "Część 2 - Wykorzystanie funkcji z pakietu pyransac3d" + Style.RESET_ALL)


for filename in filenames:
    print(Fore.GREEN + Style.BRIGHT + f"Processing file: {filename}" + Style.RESET_ALL)

    points = load_xyz_file2(filename)

    # Dopasowanie płaszczyzny za pomocą algorytmu RANSAC
    plane_model = pyrsc.Plane()
    best_eq, best_inliers = plane_model.fit(points, 0.01)

    if best_eq is not None:
        print("Parameters of the fitted plane:", best_eq)
        print("Number of inliers:", len(best_inliers))
        print("Number of points loaded:", len(points))

        # Określenie orientacji płaszczyzny
        normal_vector = best_eq[:3]
        if abs(normal_vector[2]) < 0.1:
            plane_type = "vertical"
        else:
            plane_type = "horizontal"

        print("Normal vector of the fitted plane:", normal_vector)
        print(Fore.YELLOW + "Plane type:" + Fore.MAGENTA + " " + plane_type + Style.RESET_ALL)
    else:
        print("Could not fit a plane for the given set of points.")