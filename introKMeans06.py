# https://scikit-learn.org/stable/modules/generated/
# sklearn.cluster.KMeans.html

#%% Importa e ajusta DataFrame
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb
from sklearn.cluster import KMeans


import cv2  # OpenCV
import skimage
import skimage.exposure
import skimage.transform
import scipy.signal
import scipy


df = pd.read_csv('Painted_Data01.csv')
# df = pd.read_csv('Painted_Data02.csv')
# df = pd.read_csv('Painted_Data03.csv')


X_df = df.to_numpy()


def converter_numero(valor):
    texto = str(valor).strip()

    # Os arquivos Data02/Data03 possuem números como
    # 10.277.283.337.102.500, que representam 0.10277283337102500.
    if texto.count('.') > 1:
        return float('0.' + texto.replace('.', ''))

    return float(texto)


x_numeros = np.array([converter_numero(valor) for valor in X_df[2:, 0]])
y_numeros = np.array([converter_numero(valor) for valor in X_df[2:, 1]])

df_raw = pd.DataFrame({"x": x_numeros,
                       "y": y_numeros})
#plot1
sb.pairplot(df_raw)

df_target = pd.DataFrame({"x": x_numeros,
                          "y": y_numeros,
                          "target": X_df[2:, 2].astype(str)})
#plot2
sb.pairplot(df_target, hue="target")


#%% np 2 pd.dataframe
X_df_raw = df_raw.to_numpy()
X_df_target = df_target.to_numpy()


kmeans = KMeans(n_clusters=5, random_state=0)
kmeans.fit(X_df_raw)
kmeans.labels_

df_raw['K-classes'] = kmeans.labels_

sb.pairplot(df_raw, hue='K-classes')
sb.pairplot(df_target, hue='target')

#%% Fazer Selecional classe especifica
# pandas.DataFrame.lookup
# retrieving rows and some columns by iloc method
df_SelectAll = df_raw.iloc[:, [0, 1, 2]]
print(df_SelectAll)


df_SelectClass_0 = df_SelectAll[df_SelectAll['K-classes'] == 0]
sb.pairplot(df_SelectClass_0, hue='K-classes')

df_SelectClass_1 = df_SelectAll[df_SelectAll['K-classes'] == 1]
sb.pairplot(df_SelectClass_1, hue='K-classes')

df_SelectClass_2 = df_SelectAll[df_SelectAll['K-classes'] == 2]
sb.pairplot(df_SelectClass_2, hue='K-classes')

df_SelectClass_3 = df_SelectAll[df_SelectAll['K-classes'] == 3]
sb.pairplot(df_SelectClass_3, hue='K-classes')

df_SelectClass_4 = df_SelectAll[df_SelectAll['K-classes'] == 4]
sb.pairplot(df_SelectClass_4, hue='K-classes')

plt.show()