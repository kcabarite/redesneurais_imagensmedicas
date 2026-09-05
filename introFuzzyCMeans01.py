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
img = cv2.imread('ImagemFuzzyCluster01B.pgm', 0)
img = skimage.img_as_float(img)

(M, N) = np.shape(img)

plt.figure()
plt.title('img_original')
plt.imshow(img, cmap='gray')  # cmap='jet'


#%% Construir Função de Pertinência
roi = cv2.selectROI(img)
#
cv2.destroyAllWindows()
cmin = roi[0]
lmin = roi[1]
cmax = roi[0] + roi[2]
lmax = roi[1] + roi[3]

seed_lin = np.int64(lmin + np.round(roi[3] / 2))
seed_col = np.int64(cmin + np.round(roi[2] / 2))

#%% Pertinência à Média e Desvio de Intensidade

mediaIntROI = np.mean(img[lmin:lmax,cmin:cmax])
desvioIntROI = np.std(img[lmin:lmax,cmin:cmax])
x_values = np.arange(0, 1, 0.01)

funcPertinenciaInten = np.exp(-0.5*(((x_values - mediaIntROI)/desvioIntROI)**2))

plt.figure()
plt.title('Pertinência a Intensidade')
plt.plot(x_values, funcPertinenciaInten)

#%% Pertinência à Distância
mediaDistROI = np.sqrt((200-seed_lin)**2 + (200-seed_col)**2)
desvioDistROI = np.min([roi[2], roi[3]])  # Considerando desvio como o menor Delta de largura
x_values = np.arange(0, 200, 1)

funcPertinenciaDist = np.exp(-0.5*(((x_values - mediaDistROI)/
desvioDistROI)**2))

plt.figure()
plt.title('Pertinência a Distância')
plt.plot(x_values, funcPertinenciaDist)

#%% Construção das Matrizes de Atributos
img_PertinenciaInten = np.zeros((M,N), dtype = float)
img_PertinenciaDist = np.zeros((M,N), dtype = float)
for l in range(M):
    for c in range(N):

        # Matriz de atributos Pertinência Intensidade
        img_PertinenciaInten[l,c] = np.exp(-0.5*(((img[l,c] - mediaIntROI)/desvioIntROI)**2))

        # Matriz de atributos Pertinência Distância
        distc = c - (N/2)
        distl = l - (M/2)
        dist = np.sqrt(distc**2 + distl**2)
        img_PertinenciaDist[l,c] = np.exp(-0.5*(((dist -mediaDistROI)/desvioDistROI)**2))

plt.figure()
plt.title('img_PertinenciaInten')
plt.imshow(img_PertinenciaInten, cmap='gray')  # cmap='jet

plt.figure()
plt.title('img_PertinenciaDist')
plt.imshow(img_PertinenciaDist, cmap='gray')  # cmap='jet

#%% Converter para vetor

at1_Int = img_PertinenciaInten.flatten()  # atributo intensidade
at2_Dist = img_PertinenciaDist.flatten()  # atributo distância

#%% Constrói Matriz de atributos / DataFrame

X_df = np.column_stack((at1_Int, at2_Dist))

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
#matrixClassesXBin = matrixClasses == 0
#matrixClassesXBin = matrixClasses == 1
#matrixClassesXBin = matrixClasses == 2
matrixClassesXBin = matrixClasses == 3

plt.figure()
plt.title('matrixClassesXBin')
plt.imshow(matrixClassesXBin, cmap='gray')  # cmap='jet'

# pixels das regiões da classe desejada
imFinal = img*matrixClassesXBin
plt.figure()
plt.title('imFinal')
plt.imshow(imFinal, cmap='gray')  # cmap='jet'

# %%
