import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import recall_score, precision_score, accuracy_score

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

# Створення SVМ-класифікатора
classifier = OneVsOneClassifier(LinearSVC(random_state=0))

# Навчання класифікатора
classifier.fit(X, Y)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=5)

classifier = OneVsOneClassifier(LinearSVC(random_state=0))
classifier.fit(X_train, y_train)
y_test_pred = classifier.predict(X_test)

# Обчислення F-міри для SVМ-класифікатора
f1 = cross_val_score(classifier, X, Y, scoring='f1_weighted', cv=3)
print("F1 score: " + str(round(100*f1.mean(), 2)) + "%")

# Передбачення результату для тестової точки даних
input_data = ['37', 'Private', '215646', 'HS-grad', '9', 'Never-married', 'Handlers-cleaners', 'Not-in-family', 'White', 'Male',
'0', '0', '40', 'United-States']

# Кодування тестової точки даних
input_data_encoded = [-1] * len(input_data)
count = 0
for i, item in enumerate(input_data):
 if item.isdigit():
  input_data_encoded[i] = int(input_data[i])
 else:
  try:
   input_data_encoded[i] = int(label_encoder[count].transform([input_data[i]])[0])
  except ValueError as e:
   print(f"Значення '{item}' не було знайдено у label_encoder. Перевірте дані.")
   print(e)

  count += 1

input_data_encoded = np.array(input_data_encoded)

# Використання класифікатора для кодованої точки даних
# та виведення результату
predicted_class = classifier.predict(input_data_encoded.reshape(1, -1))
print(label_encoder[-1].inverse_transform(predicted_class)[0])

# Прогнозування значень для тренувальних даних
y_pred = classifier.predict(X_test)

#Акуратність
accuracy_percentage = accuracy_score(y_test, y_pred) * 100;
print("Accuracy = ", round(accuracy_percentage,2), "%")

#Повнота
recall_percentage = recall_score(y_test, y_pred, average='weighted') * 100
print("Recall: ", round(recall_percentage,2), "%")

#Точність
precision_percentage = precision_score(y_test, y_pred, average='weighted') * 100
print("Precision: ", round(precision_percentage,2), "%")