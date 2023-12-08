import os

def enumerar_imagenes(carpeta):
    # Obtener la lista de archivos en la carpeta
    lista_archivos = os.listdir(carpeta)
    
    # Filtrar archivos de imagen (puedes ajustar esto según los tipos de archivos que tengas)
    archivos_imagen = [archivo for archivo in lista_archivos if archivo.endswith(('.jpg', '.jpeg', '.png', '.gif'))]

    # Enumerar y renombrar los archivos
    for i, archivo in enumerate(archivos_imagen):
        # Crear el nuevo nombre de archivo
        nuevo_nombre = "3_"+f"{i}.jpg"  # Puedes cambiar la extensión según el tipo de archivo
        ruta_original = os.path.join(carpeta, archivo)
        ruta_nueva = os.path.join(carpeta, nuevo_nombre)

        # Renombrar el archivo
        os.rename(ruta_original, ruta_nueva)

# Ejemplo de uso:
carpeta_imagenes = "C:\\Users\\JUANDIEGO\\Downloads\\Proyecto\\CNN-Digits\\dataset\\test\\3"
enumerar_imagenes(carpeta_imagenes)