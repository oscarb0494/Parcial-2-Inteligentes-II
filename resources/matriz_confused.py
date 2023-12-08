from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Matriz de confusión
confusion = confusion_matrix(y_true, y_pred_classes)
print("Matriz de confusión:")
print(confusion)

# Matriz de confusión gráfica

labels = ["0", "1", "2", "3"]

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
