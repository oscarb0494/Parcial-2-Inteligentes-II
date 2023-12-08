import tensorflow as tf
import numpy as np
import cv2
from keras.applications import VGG19
from keras.models import Model
from keras.layers import Flatten, Dense
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def cargarDatos(rutaOrigen, numeroCategorias, limite, ancho, alto):
    imagenesCargadas = []
    valorEsperado = []
    for categoria in range(0, numeroCategorias):
        for idImagen in range(0, limite[categoria]):
            ruta = rutaOrigen + str(categoria) + "/" + str(categoria) + "_" + str(idImagen) + ".jpg"
            imagen = cv2.imread(ruta)
            imagen = cv2.resize(imagen, (ancho, alto))  # Redimensionar la imagen
            ##imagen = imagen / 255.0  # Normalizar los valores de píxeles
            imagenesCargadas.append(imagen)
            probabilidades = np.zeros(numeroCategorias)
            probabilidades[categoria] = 1
            valorEsperado.append(probabilidades)
    imagenesEntrenamiento = np.array(imagenesCargadas)
    valoresEsperados = np.array(valorEsperado)
    return imagenesEntrenamiento, valoresEsperados


#################################
ancho = 224
alto = 224
numeroCanales = 3
formaImagen = (ancho, alto, numeroCanales)
numeroCategorias = 4

#cantidaDatosEntrenamiento=[6995,6138,6570,4291]
#cantidaDatosPruebas=[1748,1534,1642,1072]

cantidaDatosEntrenamiento=[4291,4291,4291,4291]
cantidaDatosPruebas=[1072,1072,1072,1072]

#Cargar las imágenes
# 1: COVID
# 2: Lung opacity
# 3: Normal
# 4: Viral Pneumonia

imagenes, probabilidades = cargarDatos("dataset/train/", numeroCategorias, cantidaDatosEntrenamiento, ancho, alto)
base_model = VGG19(weights='imagenet', include_top=False, input_shape=(ancho, alto, numeroCanales))

# Añadir capas densas personalizadas
x = Flatten()(base_model.output)
x = Dense(128, activation='relu')(x)
output = Dense(numeroCategorias, activation='softmax')(x)

# Crear modelo completo
model = Model(inputs=base_model.input, outputs=output)# Compilar y entrenar el modelo
model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=["accuracy", tf.keras.metrics.Precision(),
                       tf.keras.metrics.Recall(),
                       ])

resultados = model.fit(x=imagenes, y=probabilidades, epochs=1, batch_size=16)
print(resultados)

#Prueba del modelo
imagenesPrueba,probabilidadesPrueba=cargarDatos("dataset/test/",numeroCategorias,cantidaDatosPruebas,ancho,alto)
resultados=model.evaluate(x=imagenesPrueba,y=probabilidadesPrueba)
print("Accuracy=",resultados[1])
print(resultados)

print("F1Score=",((2*resultados[1]*resultados[2])/(resultados[1]+resultados[2])))

#Metricas

y_pred = model.predict(x=imagenesPrueba)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(probabilidadesPrueba, axis=1)

# Matriz de confusión
confusion = confusion_matrix(y_true, y_pred_classes)
print("Matriz de confusión:")
print(confusion)

# Matriz de confusión gráfica

labels = ["COVID", "Lung opacity", "Normal", "Viral Pneumonia"]

# Crear una figura de matplotlib
fig, ax = plt.subplots(figsize=(8, 6))

# Utilizar seaborn para crear un mapa de calor de la matriz de confusión
sns.heatmap(confusion, annot=True, cmap="Blues", fmt="d", cbar=False)
# Configurar etiquetas de los ejes
ax.set_xlabel("Predicciones")
ax.set_ylabel("Valores verdaderos")
ax.set_title("Matriz de Confusión")
# Configurar etiquetas personalizadas en los ejes x e y
ax.set_xticklabels(labels)
ax.set_yticklabels(labels)

plt.show()

# Guardar modelo
ruta="models/modeloVGG.h5"
model.save(ruta)

model.summary()