import datetime
import json
import numpy as np
from sklearn import covariance, cluster
import yfinance as yf

# Вхідний файл із символічними позначеннями компаній
input_file = "company_symbol_mapping.json"

# Завантаження мапи символів компаній
with open(input_file, "r") as f:
    company_symbols_map = json.loads(f.read())

symbols, names = np.array(list(company_symbols_map.items())).T

# Визначення діапазону дат для історичних котирувань акцій
start_date = datetime.datetime(2003,7,3)
end_date = datetime.datetime(2007,5,4)

# Завантаження історичних котирувань акцій за допомогою yfinance
quotes = []
valid_symbols = []
for symbol in symbols:
    try:
        data = yf.download(symbol, start=start_date, end=end_date)
        if not data.empty:
            quotes.append(data)
            valid_symbols.append(symbol)
        else:
            print(f"Дані для {symbol} відсутні у вказаний період.")
    except Exception as e:
        print(f"Не вдалося завантажити дані для {symbol}: {e}")

# Перевірка, чи є валідні символи
if not quotes:
    print(
        "Немає валідних даних для жодного символу. Перевірте вашу мапу символів та наявність даних."
    )
else:
    symbols = valid_symbols  # Оновлюємо символи на валідні

    opening_quotes = []
    closing_quotes = []
    # Видобуття котирувань при відкритті та закритті
    try:
        opening_quotes = np.array([quote["Open"].values for quote in quotes]).T
        closing_quotes = np.array([quote["Close"].values for quote in quotes]).T
    except Exception as e:
        print("Помилка при обробці даних:", e)

    # Обчислення різниці між котируваннями при відкритті та закритті
    quotes_diff = closing_quotes - opening_quotes

    # Нормалізація даних
    X = quotes_diff.copy()
    X /= X.std(axis=0)

    # Створення моделі графу
    edge_model = covariance.GraphicalLassoCV()

    # Навчання моделі
    with np.errstate(invalid="ignore"):
        edge_model.fit(X)

    # Побудова моделі кластеризації з використанням моделі Affinity Propagation
    _, labels = cluster.affinity_propagation(edge_model.covariance_)
    num_labels = labels.max()

    # Виведення результатів кластеризації
    print("\nКластеризація акцій на основі різниці між котируваннями при відкритті та закритті:\n")
    for i in range(num_labels + 1):
        cluster_indices = np.where(labels == i)[0]
        cluster_names = names[cluster_indices]
        if len(cluster_names) > 0:
            print("Кластер", i + 1, "==>", ", ".join(cluster_names))