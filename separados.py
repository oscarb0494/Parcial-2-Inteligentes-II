import os
import random
import shutil

def split_images(source_dir, destination_dir1, destination_dir2, split_ratio=0.8):
    if not os.path.exists(destination_dir1):
        os.makedirs(destination_dir1)
    if not os.path.exists(destination_dir2):
        os.makedirs(destination_dir2)

    # Obtener lista de archivos de imagen en el directorio de origen
    image_files = [f for f in os.listdir(source_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

    # Mezclar aleatoriamente la lista de imágenes
    random.shuffle(image_files)

    # Calcular la cantidad de imágenes para cada destino según el ratio
    num_images_dest1 = int(len(image_files) * split_ratio)
    num_images_dest2 = len(image_files) - num_images_dest1

    # Copiar imágenes al destino 1
    for i in range(num_images_dest1):
        file = image_files[i]
        source_path = os.path.join(source_dir, file)
        destination_path = os.path.join(destination_dir1, file)
        shutil.copyfile(source_path, destination_path)

    # Copiar imágenes al destino 2
    for i in range(num_images_dest1, len(image_files)):
        file = image_files[i]
        source_path = os.path.join(source_dir, file)
        destination_path = os.path.join(destination_dir2, file)
        shutil.copyfile(source_path, destination_path)

# Rutas de directorios
source_directory = 'C:\\Inteligentes II\\Examen 2\\CNN-Digits\\dataset\\train\\Viral Pneumonia'
destination_directory_1 = 'C:\\Inteligentes II\\Examen 2\\CNN-Digits\\dataset\\train\\Viral Pneumonia\\Entrenamiento'
destination_directory_2 = 'C:\\Inteligentes II\\Examen 2\\CNN-Digits\\dataset\\train\\Viral Pneumonia\\Pruebas'

# Llamar a la función para dividir las imágenes
split_images(source_directory, destination_directory_1, destination_directory_2)
