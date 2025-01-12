import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
import os

# Relative path
relative_path = os.getcwd()

# Leer la imagen y convertirla a escala de grises
plt.rcParams['figure.figsize'] = [16, 8]
A = imread(relative_path+'/imagen.jpg')  # Asegúrate de tener esta imagen en la misma carpeta
X = np.mean(A, axis=-1)

# Mostrar la imagen original en escala de grises
plt.imshow(X, cmap='gray')
plt.axis('off')
plt.title('Imagen Original')
plt.show()

# Realizar la Descomposición en Valores Singulares (DVS)
U, S, VT = np.linalg.svd(X, full_matrices=False)

# Función para reconstruir una imagen comprimida
def reconstruir_imagen(k):
    """
    Reconstruye la imagen usando los primeros k valores singulares.
    """
    S_k = np.diag(S[:k])  # Matriz diagonal con los primeros k valores singulares
    X_comprimida = U[:, :k] @ S_k @ VT[:k, :]
    return X_comprimida

# Mostrar las imágenes comprimidas para diferentes valores de k
k_values = [5, 20, 100]
for i, k in enumerate(k_values):
    X_comprimida = reconstruir_imagen(k)
    
    plt.figure()
    plt.imshow(X_comprimida, cmap='gray')
    plt.axis('off')
    plt.title(f'k = {k}')
    plt.show()
