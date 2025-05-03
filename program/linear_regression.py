import numpy as np

class LinearRegression:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0

        for epoch in range(self.epochs):
            y_pred = np.dot(X,self.w) + self.b
            dw = (2 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (2 / n_samples) * np.sum(y_pred - y)

            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db

            if epoch % 100 == 0:
                loss = np.mean((y_pred - y) ** 2)
                print(f"Epoch {epoch}: Loss={loss:.4f}")

    def predict(self, X):
        return np.dot(X, self.w) + self.b
    
# Tạo dữ liệu mẫu
X = np.array([
    [50, 2],
    [80, 3],
    [120, 4]
])
y = np.array([150, 250, 350])

# Tạo mô hình và huấn luyện
model = LinearRegression(learning_rate=0.0001, epochs=1000)
model.fit(X, y)

# Dự đoán trên dữ liệu mới
X_new = np.array([
    [100, 3],
    [60, 2]
])
predictions = model.predict(X_new)
print("Dự đoán giá nhà:", predictions)