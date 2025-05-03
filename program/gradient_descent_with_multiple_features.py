import numpy as np

# Dữ liệu mẫu: diện tích, số phòng và giá nhà.
X = np.array([
    [50,2],
    [80,3],
    [120,4]
])

y = np.array([150,250,350])

# Khởi tạo tham số
n_samples, n_features = X.shape
w = np.random.randn(n_features)
b = np.random.randn()

learning_rate = 0.0001
epochs = 1000

# Huấn luyện
for epoch in range(epochs):
    y_pred = np.dot(X,w) + b
    loss = np.mean((y_pred - y) ** 2)

    dw = (2/n_samples) * np.dot(X.T, (y_pred - y))
    db = (2/n_samples) * np.sum(y_pred -y)

    w -= learning_rate * dw
    b -= learning_rate * db

    if epoch % 1000 == 0:
        print(f"Epoch {epoch}: Loss = {loss:.4f}, w = {w}, b = {b:.4f}")