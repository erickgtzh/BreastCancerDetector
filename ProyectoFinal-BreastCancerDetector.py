# -*- coding: utf-8 -*-
"""
Created on Sun Jun  3 16:54:06 2018

@author: Erick
"""
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report,accuracy_score
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn.cross_validation import cross_val_score,train_test_split
from sklearn.metrics import confusion_matrix

def Graficar(dataset_cancer):
    dataset_cancer.hist(figsize=(10,10))
    plt.show()
    scatter_matrix(dataset_cancer,figsize=(11,11))
    plt.show()
    
#Cada columna = una variable, estos nombres fueron tomados desde la página donde se descargó esta base de datos (ics.uci.edu)
nombre_variables = ['SampleCodeNumber','Clump thickness','Uniformity of Cell size','Uniformity of cell shape', 'Marginal adhesion', 'Single epithelial cell size', 'Bare nuclei','Bland Chromatin','Normal nucleoli','Mitoses','Class'] #Class: 2 benign, 4 malignant
dataset_cancer = pd.read_csv('breast-cancer-wisconsin.csv',header=None,names=nombre_variables)

#Omitimos la información de la tabla de 'SampleCodeNumbers' ya que no nos da información para lo que buscamos
dataset_cancer = dataset_cancer.drop(columns=['SampleCodeNumber'])

#En la base de datos no se encontraron algunos datos, estos aparecen con el símbolo '?'
#para que no nos dañen la información o la modifiquen, reemplazaremos ese símbolo por el
#número -99999, de esta forma python no le hará caso
dataset_cancer = dataset_cancer.replace('?',-99999)

Graficar(dataset_cancer)

#Ahora seleccionamos los modelos que utilizaremos para el machine learning

k = 5
knn = KNeighborsClassifier(n_neighbors=k)
svm = SVC(kernel='linear')
modelos = ['KNN','SVM']
#Tomamos nuestras x_test,x_train,y_test,y_train
auxY = pd.read_csv('breast-cancer-wisconsin.csv',header=None).values
fil,col = auxY.shape
Y = []
for i in range(fil):
    Y.append(auxY[i][-1])

X = dataset_cancer.drop(columns=['Class'])
Y = np.asarray(Y)
X_train,x_test,Y_train,y_test = train_test_split(X,Y,test_size=0.4)
#print('X_train: ',X_train,'x_test: ',x_test,'Y_train: ',Y_train,'y_test: ',y_test)

#KNN
modelo_KNN = []
cv_resultados = cross_val_score(knn,X_train,Y_train,cv=10,scoring='accuracy')
modelo_KNN.append(modelos[0])
modelo_KNN.append(cv_resultados.mean())
modelo_KNN.append(cv_resultados.std())
print(modelo_KNN)

#Predicción
KNN_fit = knn.fit(X_train,Y_train)
prediccion = knn.predict(x_test)
print(accuracy_score(y_test,prediccion))
print(classification_report(y_test,prediccion))
cm = confusion_matrix(y_test,prediccion)
print(pd.DataFrame(data=cm))

print('')

#SVM
modelo_SVM = []
cv_resultados = cross_val_score(svm,X_train,Y_train,cv=10,scoring='accuracy')
modelo_SVM.append(modelos[1])
modelo_SVM.append(cv_resultados.mean())
modelo_SVM.append(cv_resultados.std())
print(modelo_SVM)

#Predicción
SMV_fit = svm.fit(X_train,Y_train)
prediccion = svm.predict(x_test)
print(accuracy_score(y_test,prediccion))
print(classification_report(y_test,prediccion))
cm = confusion_matrix(y_test,prediccion)
print(pd.DataFrame(data=cm))

#Ejemplo
#nombre_variables = ['SampleCodeNumber','Clump thickness',
#'Uniformity of Cell size','Uniformity of cell shape', 'Marginal adhesion', 
#'Single epithelial cell size', 'Bare nuclei','Bland Chromatin',
#'Normal nucleoli','Mitoses','Class'] #Class: 2 benign, 4 malignant
print('')
print('Ejemplo')
ejemplo = np.array([[1,2,3,4,3,2,1,4,5]])
ejemplo= ejemplo.reshape(len(ejemplo),-1)
modelo = KNeighborsClassifier(n_neighbors=5)
entrenamiento = modelo.fit(X_train,Y_train)
error = 1-modelo.score(x_test,y_test)
print('Array de ejemplo: ',ejemplo)
print('error: ',error)
prediccion = knn.predict(ejemplo)
if prediccion==4:
    print('Células cancerígenas malignas')
else:
    print('Células cancerígenas benignas')