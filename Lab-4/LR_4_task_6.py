from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model, preprocessing
from sklearn.pipeline import Pipeline

# Визначення функції, яка будує криві навчання моделі для встановлених навчальних даних
def plot_learning_curves(model, X, y):
    # Розбиття даних на тренувальні і тестові
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

    train_errors, val_errors = [], []
    for m in range(1, len(X_train)):
        model.fit(X_train[:m], y_train[:m])

        # Прогнозування
        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)

        # Обробка помилок
        train_errors.append(mean_squared_error(y_train_predict, y_train[:m]))
        val_errors.append(mean_squared_error(y_val_predict, y_val))

    # Побудова кривих навчання
    plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="Навчальний набір")
    plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="Тестовий набір")
    plt.xlabel("Training set size")
    plt.ylabel("RMSE")
    plt.legend()
    plt.show()

# Випадкові дані для 7 варіанту
m = 100
X = np.linspace(-3, 3, m)
y = np.sin(X) + np.random.uniform(-0.5, 0.5, m)

# Перетворення в двовимірний масив з одним стовпцем
X = X.reshape(-1, 1)

#Створення моделі лінійної регресії
linear_regressor = linear_model.LinearRegression()

#Створення моделі поліноміальної регресії
polynomial_features = preprocessing.PolynomialFeatures(degree=2, include_bias=False)
X_poly_train = polynomial_features.fit_transform(X)
polynomial_regressor = Pipeline([
    ("poly_features", polynomial_features),
    ("lin_reg", linear_model.LinearRegression()),
])

# Заповнення тестовими даними
linear_regressor.fit(X, y)
polynomial_regressor.fit(X_poly_train, y)

#Передбачення результатів
y_linear = linear_regressor.predict(X)
y_polynomial = polynomial_regressor.predict(X_poly_train)

# Криві навчання для лінійної регресії
plot_learning_curves(linear_regressor, X, y)

# Криві навчання для поліноміальної регресії
plot_learning_curves(polynomial_regressor, X_poly_train, y)
