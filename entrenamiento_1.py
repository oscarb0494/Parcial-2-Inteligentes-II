import tensorflow as tf
import keras
import numpy as np
import cv2
###Importar componentes de la red neuronal
from keras.models import Sequential
from keras.layers import InputLayer,Input,Conv2D, MaxPool2D,Reshape,Dense,Flatten
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def cargarDatos(rutaOrigen,numeroCategorias,limite,ancho,alto):
    imagenesCargadas=[]
    valorEsperado=[]
    for categoria in range(0,numeroCategorias):
        for idImagen in range(0,limite[categoria]):
            ruta=rutaOrigen+str(categoria)+"/"+str(categoria)+"_"+str(idImagen)+".jpg"
            print(ruta)
            imagen = cv2.imread(ruta)
            imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
            imagen = cv2.resize(imagen, (ancho, alto))
            imagen = imagen.flatten()
            imagen = imagen / 255
            imagenesCargadas.append(imagen)
            probabilidades = np.zeros(numeroCategorias)
            probabilidades[categoria] = 1
            valorEsperado.append(probabilidades)
    imagenesEntrenamiento = np.array(imagenesCargadas)
    valoresEsperados = np.array(valorEsperado)
    return imagenesEntrenamiento, valoresEsperados

#################################
ancho=256
alto=256
pixeles=ancho*alto
#Imagen RGB -->3
numeroCanales=1
formaImagen=(ancho,alto,numeroCanales)
numeroCategorias=4

#cantidaDatosEntrenamiento=[6995,6138,6570,4291]
#cantidaDatosPruebas=[1748,1534,1642,1072]

cantidaDatosEntrenamiento=[4291,4291,4291,4291]
cantidaDatosPruebas=[1072,1072,1072,1072]

#Cargar las imágenes
# 1: COVID
# 2: Lung opacity
# 3: Normal
# 4: Viral Pneumonia
imagenes, probabilidades=cargarDatos("dataset/train/",numeroCategorias,cantidaDatosEntrenamiento,ancho,alto)

model=Sequential()
#Capa entrada
model.add(InputLayer(input_shape=(pixeles,)))
model.add(Reshape(formaImagen))

#Capas Ocultas
#Capas convolucionales
model.add(Conv2D(kernel_size=5,strides=2,filters=16,padding="same",activation="relu",name="capa_1"))
model.add(MaxPool2D(pool_size=2,strides=2))

model.add(Conv2D(kernel_size=3,strides=1,filters=36,padding="same",activation="relu",name="capa_2"))
model.add(MaxPool2D(pool_size=2,strides=2))

#Aplanamiento
model.add(Flatten())
model.add(Dense(128,activation="relu"))

#Capa de salida
model.add(Dense(numeroCategorias,activation="softmax"))


#Traducir de keras a tensorflow
model.compile(optimizer="Adam",loss="categorical_crossentropy", 
              metrics=["accuracy",
                       tf.keras.metrics.Precision(),
                       tf.keras.metrics.Recall()
                       ])
#Entrenamiento
model.fit(x=imagenes,y=probabilidades,epochs=10,batch_size=32)

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
ruta="models/modeloA.h5"
model.save(ruta)
# Informe de estructura de la red
model.summary()
