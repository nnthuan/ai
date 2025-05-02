import numpy as np
import matplotlib.pyplot as plt

# Dữ liệu mẫu (diện tích nhà và giá nhà)
x = np.array([50, 80, 120])
y = np.array([150, 250, 350])

# Khởi tạo ngẫu nhiên w và b
w = np.random.randn()
b = np.random.randn()

# Các siêu tham số
learning_rate = 0.0001
epochs = 1000

# Huấn luyện mô hình
for epoch in range(epochs):
    # 1. Dự đoán
    y_pred = w * x + b

    # 2. Tính Loss
    loss = np.mean((y_pred - y) ** 2)
 
    # 3. Tính gradient
    dw = np.mean(2 * (y_pred - y) * x)
    db = np.mean(2 * (y_pred - y))

    # 4. Cập nhật tham số
    w -= learning_rate * dw
    b -= learning_rate * db

    # 5. In loss mỗi 100 epochs
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Loss = {loss:.4f}, b = {b:.4f}")

# Vẽ dữ liệu thật và đường thẳng dự đoán

plt.scatter(x,y,color='blue',label='Dữ liệu thật')
plt.plot(x, w * x + b, color='red', label='Đường dự đoán')
plt.xlabel('Diện tích nhà (m²)')
plt.ylabel('Giá nhà (nghìn USD)')
plt.legend()
plt.show()

# Giải thích nhanh:
# | Bước           | Ý nghĩa                                        |
# |:---------------|:-----------------------------------------------|
# | y_pred         | Tính giá trị dự đoán dựa trên w và b hiện tại  |
# | loss           | Tính sai số trung bình giữa dự đoán và thực tế |
# | dw, db         | Tính đạo hàm để biết hướng điều chỉnh w, b     |
# | cập nhật w, b  | Dịch w và b theo hướng giảm loss               |

# Kết quả mong đợi:
# - Ban đầu loss khá lớn.
# - Sau nhiều vòng (epoch), loss sẽ giảm dần.
# - Cuối cùng, đường thẳng (đường màu đỏ) sẽ fit sát dữ liệu thật (các điểm xanh).