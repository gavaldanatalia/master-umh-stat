# Importación de librerías necesarias
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score, roc_curve

# 1. Carga del Dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
columns = ['ID', 'Diagnosis', 'Radius Mean', 'Texture Mean', 'Perimeter Mean', 'Area Mean', 'Smoothness Mean', 
           'Compactness Mean', 'Concavity Mean', 'Concave Points Mean', 'Symmetry Mean', 'Fractal Dimension Mean', 
           'Radius SE', 'Texture SE', 'Perimeter SE', 'Area SE', 'Smoothness SE', 'Compactness SE', 
           'Concavity SE', 'Concave Points SE', 'Symmetry SE', 'Fractal Dimension SE', 'Radius Worst', 
           'Texture Worst', 'Perimeter Worst', 'Area Worst', 'Smoothness Worst', 'Compactness Worst', 
           'Concavity Worst', 'Concave Points Worst', 'Symmetry Worst', 'Fractal Dimension Worst']

# Leer los datos
data = pd.read_csv(url, header=None, names=columns)

# 2. Exploración de los datos
# Información general sobre los datos
print(data.info())
print(data.describe())

# Ver las primeras filas
print(data.head())