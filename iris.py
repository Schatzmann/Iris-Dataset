#coding: utf-8

#Autor:
#Annelyse Schatzmann           GRR20151731

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.decomposition import PCA
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import load_iris
from sklearn import tree
import graphviz

# Gaussian Naive Bayes
from sklearn import datasets
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score
from sklearn.model_selection import ShuffleSplit

iris = datasets.load_iris()


#------------- Gráfico 2D -------------#

# import some data to play with
iris = datasets.load_iris()
X = iris.data[:, :4]  # we only take the first two features.
y = iris.target

x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5

plt.figure(1, figsize=(8, 6))
plt.clf()

plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1, edgecolor='k')
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')


x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 2].min() - .5, X[:, 2].max() + .5

plt.figure(2, figsize=(8, 6))
plt.clf()

plt.scatter(X[:, 0], X[:, 2], c=y, cmap=plt.cm.Set1, edgecolor='k')
plt.xlabel('Sepal length')
plt.ylabel('Petal Length')

x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 3].min() - .5, X[:, 3].max() + .5

plt.figure(3, figsize=(8, 6))
plt.clf()

plt.scatter(X[:, 0], X[:, 3], c=y, cmap=plt.cm.Set1, edgecolor='k')
plt.xlabel('Sepal length')
plt.ylabel('Petal Width')

x_min, x_max = X[:, 1].min() - .5, X[:, 1].max() + .5
y_min, y_max = X[:, 2].min() - .5, X[:, 2].max() + .5

plt.figure(4, figsize=(8, 6))
plt.clf()

plt.scatter(X[:, 1], X[:, 2], c=y, cmap=plt.cm.Set1, edgecolor='k')
plt.xlabel('Sepal Width')
plt.ylabel('Petal Length')

x_min, x_max = X[:, 1].min() - .5, X[:, 1].max() + .5
y_min, y_max = X[:, 3].min() - .5, X[:, 3].max() + .5

plt.figure(5, figsize=(8, 6))
plt.clf()

plt.scatter(X[:, 1], X[:, 3], c=y, cmap=plt.cm.Set1, edgecolor='k')
plt.xlabel('Sepal Width')
plt.ylabel('Petal Width')


x_min, x_max = X[:, 2].min() - .5, X[:, 2].max() + .5
y_min, y_max = X[:, 3].min() - .5, X[:, 3].max() + .5

plt.figure(6, figsize=(8, 6))
plt.clf()

plt.scatter(X[:, 2], X[:, 3], c=y, cmap=plt.cm.Set1, edgecolor='k')
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')


plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())


#------------- Gráfico 3D -------------#

fig = plt.figure(7, figsize=(8, 6))
ax = Axes3D(fig, elev=-150, azim=110)
X_reduced = PCA(n_components=4).fit_transform(iris.data)

ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=y,cmap=plt.cm.Set1, edgecolor='k', s=40)

ax.set_title("First three PCA directions")
ax.set_xlabel("Sepal Length")
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("Sepal Width")
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("Petal Length")
ax.w_zaxis.set_ticklabels([])


fig = plt.figure(8, figsize=(8, 6))
ax = Axes3D(fig, elev=-150, azim=110)
X_reduced = PCA(n_components=4).fit_transform(iris.data)

ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 3], c=y,cmap=plt.cm.Set1, edgecolor='k', s=40)

ax.set_title("First three PCA directions")
ax.set_xlabel("Sepal Length")
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("Sepal Width")
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("Petal Width")
ax.w_zaxis.set_ticklabels([])


fig = plt.figure(9, figsize=(8, 6))
ax = Axes3D(fig, elev=-150, azim=110)
X_reduced = PCA(n_components=4).fit_transform(iris.data)

ax.scatter(X_reduced[:, 0], X_reduced[:, 2], X_reduced[:, 3], c=y,cmap=plt.cm.Set1, edgecolor='k', s=40)

ax.set_title("First three PCA directions")
ax.set_xlabel("Sepal Length")
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("Petal Length")
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("Petal Width")
ax.w_zaxis.set_ticklabels([])

fig = plt.figure(10, figsize=(8, 6))
ax = Axes3D(fig, elev=-150, azim=110)
X_reduced = PCA(n_components=4).fit_transform(iris.data)

ax.scatter(X_reduced[:, 1], X_reduced[:, 2], X_reduced[:, 3], c=y,cmap=plt.cm.Set1, edgecolor='k', s=40)

ax.set_title("First three PCA directions")
ax.set_xlabel("Sepal Width")
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("Petal Length")
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("Petal Width")
ax.w_zaxis.set_ticklabels([])

plt.show()



#------------- Árvore de Decisão -------------#


clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, y)

expectedA = y
predictedA = clf.predict(X)

clf = tree.DecisionTreeClassifier()
clf = clf.fit(iris.data, iris.target)

dot_data = tree.export_graphviz(clf, out_file=None, 
                         feature_names=iris.feature_names,  
                         class_names=iris.target_names,  
                         filled=True, rounded=True,  
                         special_characters=True) 
graph = graphviz.Source(dot_data) 
graph.render("Gini") 


clf_entropy = tree.DecisionTreeClassifier(criterion = "entropy")
clf_entropy.fit(iris.data, iris.target)

dot_data = tree.export_graphviz(clf_entropy, out_file=None, 
                         feature_names=iris.feature_names,  
                         class_names=iris.target_names,  
                         filled=True, rounded=True,  
                         special_characters=True) 
graph = graphviz.Source(dot_data) 
graph.render("Entropy") 


# #------------- Naive Bayes -------------#

dataset = datasets.load_iris()
# Naive Bayes
model = GaussianNB()
model.fit(dataset.data, dataset.target)
#print(model)
# make predictions
expected = dataset.target
predicted = model.predict(dataset.data)
# summarize the fit of the model
print('----------------------------------------------------------------')
print(metrics.classification_report(expected, predicted))

gnb = GaussianNB()
y_pred = gnb.fit(iris.data, iris.target).predict(iris.data)
print("Número de pontos incorretamente rotulados em um total %d é de: %d"% (iris.data.shape[0],(iris.target != y_pred).sum()))
print('----------------------------------------------------------------')
print('\n')

print('Matriz de confusão')
print(metrics.confusion_matrix(expected, predicted))

print('----------------------------------------------------------------')
print('\n')


#------------- Eficácia -------------#

print ("EFICÁCIA:")
print ("Taxa de Precisão, que é calculada em Naive Bayes é: %f" % accuracy_score(expected, predicted))
print ("Taxa de Precisão, que é calculada na Árvore de Decisão é: %f" % accuracy_score(expectedA, predictedA))
print('----------------------------------------------------------------')
print('\n')

#------------- Divisão das bases -------------#


rs = ShuffleSplit(n_splits=5,  test_size=0.4, train_size=0.6, random_state=0)
rs.get_n_splits(X)

print('5 partições:')
print('\n')

rs= ShuffleSplit(n_splits=5, random_state=0, test_size=0.4, train_size=0.6)
for train_index, test_index in rs.split(X):
	#chamar 30x para cada classificador?
	print("TRAIN:", train_index, "TEST:", test_index)

print('\n')
