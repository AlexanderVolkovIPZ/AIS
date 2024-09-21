import numpy as np
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Завантаження даних
data = np.loadtxt('data_multivar_nb.txt', delimiter=',')

X = data[:, :-1]  # Ознаки об'єкту
y = data[:, -1]   # Мітки класів

# Розбивка даних на навчальний та тестовий набори
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)

# Ініціалізація та тренування моделі SVM
svm_model = SVC(kernel='linear', random_state=3)
svm_model.fit(X_train, y_train)

# Прогнозування на тестових даних SVM
y_pred_svm = svm_model.predict(X_test)

# Оцінка якості класифікації SVM
svm_accuracy = accuracy_score(y_test, y_pred_svm)
svm_conf_matrix = confusion_matrix(y_test, y_pred_svm)
svm_classification_rep = classification_report(y_test, y_pred_svm)

print('SVM Results:')
print('Accuracy: ', round(svm_accuracy, 4))
print('Confusion Matrix:')
print(svm_conf_matrix)
print('Classification Report:')
print(svm_classification_rep)


# Ініціалізація та тренування моделі Gaussian Naive Bayes
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

# Прогнозування на тестових даних Gaussian Naive Bayes
y_pred_nb = nb_model.predict(X_test)

# Оцінка якості класифікації Gaussian Naive Bayes
nb_accuracy = accuracy_score(y_test, y_pred_nb)
nb_conf_matrix = confusion_matrix(y_test, y_pred_nb)
nb_classification_rep = classification_report(y_test, y_pred_nb)

print('Naive Bayes Results:')
print('Accuracy: ', round(nb_accuracy, 4))
print('Confusion Matrix:')
print(nb_conf_matrix)
print('Classification Report:')
print(nb_classification_rep)

# Порівняння моделей
if nb_accuracy > svm_accuracy:
    print("Модель 'Gaussian Naive Bayes (GNB)' за точністю краща.")
elif nb_accuracy < svm_accuracy:
    print("Модель 'Support Vector Machine (SVM)' за точністю краща.")
else:
    print("Обидві моделі мають однакову точність.")
