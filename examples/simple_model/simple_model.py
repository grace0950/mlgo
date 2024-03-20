import torch
import torch.nn as nn
import torch.optim as optim


# 1. 定義模型
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(in_features=1, out_features=1)  # 單一輸入和輸出

    def forward(self, x):
        return self.linear(x)


# 2. 實例化模型、定義損失函數和優化器
model = LinearRegressionModel()
criterion = nn.MSELoss()  # 均方誤差損失
optimizer = optim.SGD(model.parameters(), lr=0.01)  # 隨機梯度下降優化器

# 3. 準備訓練數據
x_train = torch.tensor([[1.0], [2.0], [3.0], [4.0]])  # 輸入
y_train = torch.tensor([[2.0], [4.0], [6.0], [8.0]])  # 期望輸出

# 4. 訓練模型
epochs = 100  # 迭代次數
for epoch in range(epochs):
    # 前向傳播
    y_pred = model(x_train)
    loss = criterion(y_pred, y_train)

    # 反向傳播和優化
    optimizer.zero_grad()  # 清空過去的梯度
    loss.backward()  # 反向傳播計算當前梯度
    optimizer.step()  # 更新參數

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")


# 假設訓練過程已經完成

# 保存模型參數
torch.save(model.state_dict(), "models/simple_model.state_dict")

# 輸出信息表示已保存
print("Model state_dict saved to simple_model.state_dict")
