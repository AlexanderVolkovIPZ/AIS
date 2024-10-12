from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn import preprocessing
import numpy as np

# Вхідний файл, який містить дані
input_file = 'income_data.txt'

# Читання даних
X = []
y = []
count_class1 = 0
count_class2 = 0
max_datapoints = 25000

with open(input_file, 'r') as f:
 for line in f.readlines():
     if count_class1 >= max_datapoints and count_class2 >=max_datapoints:
      break
     if '?' in line:
      continue

     data = line[:-1].split(', ')
     if data[-1] == '<=50K' and count_class1 < max_datapoints:
      X.append(data)
      count_class1 += 1
     if data[-1] == '>50K' and count_class2 < max_datapoints:
      X.append(data)
      count_class2 += 1

# Перетворення на масив numpy
X = np.array(X)

# Перетворення рядкових даних на числові
label_encoder = []
X_encoded = np.empty(X.shape)
for i,item in enumerate(X[0]):
 if item.isdigit():
  X_encoded[:, i] = X[:, i]
 else:
  label_encoder.append(preprocessing.LabelEncoder())
  X_encoded[:, i] = label_encoder[-1].fit_transform(X[:,i])

X = X_encoded[:, :-1].astype(int)
Y = X_encoded[:, -1].astype(int)

#Розділяю данні на тренувальні та тестові
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=5)

# Завантажуємо алгоритми моделей
models = []
models.append(('LR', OneVsRestClassifier(LogisticRegression(solver='liblinear'))))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))
# Оцінюємо модель на кожній ітерації
accuracies = []
percisions = []
recalls = []
names = []

for name, model in models:
    # Навчання моделі
    model.fit(X_train, y_train)

    # Отримання прогнозу
    y_pred = model.predict(X_test)

    # Оцінка аккуратності моделі
    accuracy = round(accuracy_score(y_test, y_pred) * 100, 2)

    # Оцінка точності моделі
    percision = round(precision_score(y_test, y_pred, average='weighted')*100, 2)

    # Оцінка повноти моделі
    recall = round(recall_score(y_test, y_pred, average='weighted') * 100, 2)

    # Збереження результатів
    accuracies.append(accuracy)
    percisions.append(percision)
    recalls.append(recall)
    names.append(name)

    print('Accuracy of %s model -> %s, percision -> %s, recall -> %s' % (name, accuracy, percision, recall))

# Вибір найкращого алгоритму
best_model_index = np.argmax(accuracies)
best_model_name = names[best_model_index]
best_model_accuracy = accuracies[best_model_index]
model_percision = percisions[best_model_index]
model_recall = recalls[best_model_index]

print('\nThe best algoritm is %s with accuracy -> %s, percision -> %s, recall -> %s' % (best_model_name, best_model_accuracy, model_percision, model_recall))