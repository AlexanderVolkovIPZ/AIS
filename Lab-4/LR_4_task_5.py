import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model, preprocessing

# Випадкові дані для 7 варіанту
m = 100
X = np.linspace(-3, 3, m)
y = np.sin(X) + np.random.uniform(-0.5, 0.5, m)

# Перетворення в двовимірний масив з одним стовпцем
X = X.reshape(-1, 1)

#Створення моделі лінійної регресії
linear_regressor = linear_model.LinearRegression()
linear_regressor.fit(X, y)

#Створення моделі поліноміальної регресії
polynomial_features = preprocessing.PolynomialFeatures(degree=2, include_bias=False)
X_poly_train  = polynomial_features.fit_transform(X)
polynomial_regressor = linear_model.LinearRegression()
polynomial_regressor.fit(X_poly_train, y)

#Вивід коофіцієнтів лінійгої регресії
print('ЛІНІЙНА РЕГРЕСІЯ')
print('Коефіцієнт "coef_" -> ', linear_regressor.coef_)
print('Коефіцієнт "intercept_" -> ', linear_regressor.intercept_)

#Вивід поліноміальної лінійгої регресії
print('ПОЛІНОМІАЛЬНА РЕГРЕСІЯ')
print('Коефіцієнт "coef_" -> ', polynomial_regressor.coef_)
print('Коефіцієнт "intercept_" -> ', polynomial_regressor.intercept_)

#Передбачення результатів
y_linear = linear_regressor.predict(X)
y_polynomial = polynomial_regressor.predict(X_poly_train)

# Побудова графіка
plt.figure(figsize=(10, 6))
plt.scatter(X, y, label="Дані", color='b')
plt.plot(X, y_linear, label="Лінійна регресія", color='g', linewidth=2)
plt.plot(X, y_polynomial, label="Поліноміальна регресія", color='r', linewidth=2)
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.title("Модель регресії")
plt.grid(True)
plt.show()