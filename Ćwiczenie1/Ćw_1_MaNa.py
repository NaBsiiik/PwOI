# Kod do Laboratorium 1

from scipy.stats import norm
from csv import writer
import numpy as np

def generate_points_flat_horizontal(num_points: int = 2000, width: float = 100, length: float = 100, height: float = 1):
    distribution_x = norm(loc=0, scale=width)
    distribution_y = norm(loc=0, scale=length)

    x = distribution_x.rvs(size=num_points)
    y = distribution_y.rvs(size=num_points)
    z = np.random.uniform(0, height, num_points)

    points = zip(x, y, z)
    return points

def generate_points_flat_vertical(num_points: int = 2000, width: float = 100, height: float = 100, length: float = 1):
    distribution_x = norm(loc=0, scale=width)
    distribution_z = norm(loc=0, scale=height)

    x = distribution_x.rvs(size=num_points)
    z = distribution_z.rvs(size=num_points)
    y = np.random.uniform(0, length, num_points)

    points = zip(x, y, z)
    return points

def generate_points_cylindrical_vertical(num_points: int = 2000, radius: float = 40, height: float = 200):
    angle = np.random.uniform(0, 2 * np.pi, num_points)
    radius_sample = np.random.uniform(0, radius, num_points)
    height_sample = np.random.uniform(0, height, num_points)

    x = radius_sample * np.cos(angle)
    y = radius_sample * np.sin(angle)
    z = height_sample

    points = zip(x, y, z)
    return points

def generate_points_cylindrical_horizontal(num_points: int = 2000, radius: float = 40, length: float = 200):
    angle = np.random.uniform(0, 2 * np.pi, num_points)
    radius_sample = np.random.uniform(0, radius, num_points)
    length_sample = np.random.uniform(0, length, num_points)

    x = radius_sample * np.cos(angle)
    z = radius_sample * np.sin(angle)
    y = length_sample

    points = zip(x, y, z)
    return points

def save_points_to_csv(points, filename):
    with open(filename, 'w', encoding='utf-8', newline='\n') as csvfile:
        csvwriter = writer(csvfile)
        for p in points:
            csvwriter.writerow(p)

if __name__ == '__main__':
    # a. płaska pozioma powierzchnia o ograniczonej szerokości i długości
    cloud_points = generate_points_flat_horizontal(2000, width=100, length=100, height=1)
    save_points_to_csv(cloud_points, 'CsvData_flat_horizontal.xyz')

    # b. płaska pionowa powierzchnia o ograniczonej szerokości i wysokości
    cloud_points = generate_points_flat_vertical(2000, width=100, height=100, length=1)
    save_points_to_csv(cloud_points, 'CsvData_flat_vertical.xyz')

    # c. powierzchnia cylindryczna - pozioma o zadanym promieniu i ograniczonej wysokości
    cloud_points = generate_points_cylindrical_horizontal(2000, radius=50, length=150)
    save_points_to_csv(cloud_points, 'CylindricalData_horizontal.xyz')

    # d. powierzchnia cylindryczna - pionowa o zadanym promieniu i ograniczonej wysokości
    cloud_points = generate_points_cylindrical_vertical(2000, radius=50, height=150)
    save_points_to_csv(cloud_points, 'CylindricalData_vertical.xyz')


