import torch
import numpy as np
from transformers import AutoModel, AutoConfig

# ====== 1️⃣ 生成一个模拟时间序列数据 ======
data = np.linspace(1, 10, 10)  # 示例：1,2,...,10
print("输入序列：", data)

# ====== 2️⃣ 模拟模型加载（可替换成自己的模型） ======
config = AutoConfig.from_pretrained("bert-base-uncased")
model = AutoModel.from_config(config)

# ====== 3️⃣ 构造输入张量（伪装成“时间序列特征”） ======
x = torch.tensor(data).unsqueeze(0).unsqueeze(-1)  # [batch, seq_len, feature_dim]

# ====== 4️⃣ 模拟预测过程 ======
with torch.no_grad():
    output = model(torch.ones(1, 10, dtype=torch.long))[0]
    prediction = float(torch.mean(output).item())

print(f"预测结果：{prediction:.4f}")
