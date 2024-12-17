import numpy as np
import tensorflow as tf

# Параметри
n_samples, batch_size, num_steps = 1000, 100, 20000

# Генеруємо дані
X_data = np.random.uniform(1, 10, (n_samples, 1)).astype(np.float32)
y_data = (2 * X_data + 1 + np.random.normal(0, 2, (n_samples, 1))).astype(np.float32)

# Ініціалізуємо параметри
k = tf.Variable(tf.random.normal((1, 1)), name='slope')
b = tf.Variable(tf.zeros((1,)), name='bias')

# Лінійна регресія
def model(X):
    return tf.matmul(X, k) + b

# Функція втрат (MSE)
def compute_loss(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

# Оптимізатор
optimizer = tf.optimizers.SGD(learning_rate=0.01)

# Навчальний цикл
display_step = 100
for i in range(num_steps):
    indices = np.random.choice(n_samples, batch_size)
    X_batch, y_batch = X_data[indices], y_data[indices]

    # Обчислення градієнтів та оптимізація
    with tf.GradientTape() as tape:
        y_pred = model(X_batch)
        loss = compute_loss(y_batch, y_pred)

    gradients = tape.gradient(loss, [k, b])
    optimizer.apply_gradients(zip(gradients, [k, b]))

    # Відображаємо прогрес
    if (i + 1) % display_step == 0:
        print(f'Похибка {i + 1}: {loss:.8f}, k={k.numpy()[0][0]:.4f}, b={b.numpy()[0]:.4f}')
