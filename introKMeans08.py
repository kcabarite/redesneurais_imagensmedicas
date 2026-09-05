# https://scikit-learn.org/stable/modules/generated/
# sklearn.cluster.KMeans.html

#%% Abrir Imagem
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

# img = cv2.imread('ImagemFuzzyCluster01.pgm', 0)
# img = cv2.imread('ImagemFuzzyCluster01B.pgm', 0)
# img = cv2.imread('ImagemFuzzyCluster02.pgm', 0)
img = cv2.imread('ImagemFuzzyCluster02B.pgm', 0)

img = skimage.img_as_float(img)

(M, N) = np.shape(img)

plt.figure()
plt.title('img_original')
plt.imshow(img, cmap='gray')  # cmap='jet'


#%% Converter para vetor
at_Int = img.flatten()  # atributo intensidade

#%% Constrói Matriz de atributos / DataFrame
X_df = np.column_stack((at_Int, at_Int))
df_raw = pd.DataFrame({"x": X_df[:, 0].astype(float), "y": X_df[:, 1].astype(float)})
#plot1
sb.pairplot(df_raw)


#%% np 2 pd.dataframe
X_df_raw = df_raw.to_numpy()
kmeans = KMeans(n_clusters=4, random_state=0)
kmeans.fit(X_df_raw)
kmeans.labels_

df_raw['K-classes'] = kmeans.labels_
sb.pairplot(df_raw, hue='K-classes')


#%% Fazer Selecional classe especifica
#pandas.DataFrame.lookup
# retrieving all rows and some columns by iloc method
df_SelectAll = df_raw.iloc[:, [0, 1, 2]]
print(df_SelectAll)

df_Classes = df_raw.iloc[:, [2]]
print(df_Classes)

X_df_Classes = df_Classes.to_numpy()
matrixClasses = X_df_Classes.reshape(M, N)

# Binariza classe desejada
matrixClassesXBin = matrixClasses == 1

plt.figure()
plt.title('matrixClassesXBin')
plt.imshow(matrixClassesXBin, cmap='gray')  # cmap='jet'

# pixels das regiões da classe desejada
imFinal = img*matrixClassesXBin
plt.figure()
plt.title('imFinal')
plt.imshow(imFinal, cmap='gray')  # cmap='jet'
plt.show()

# df_SelectClass_0 = df_SelectAll[df_SelectAll["K-classes"] == 0]
# sb.pairplot(df_SelectClass_0, hue='K-classes')

# df_SelectClass_1 = df_SelectAll[df_SelectAll["K-classes"] == 1]
# sb.pairplot(df_SelectClass_1, hue='K-classes')

# df_SelectClass_2 = df_SelectAll[df_SelectAll["K-classes"] == 2]
# sb.pairplot(df_SelectClass_2, hue='K-classes')

# df_SelectClass_3 = df_SelectAll[df_SelectAll["K-classes"] == 3]
# sb.pairplot(df_SelectClass_3, hue='K-classes')

# df_SelectClass_4 = df_SelectAll[df_SelectAll["K-classes"] == 4]
# sb.pairplot(df_SelectClass_4, hue='K-classes')
