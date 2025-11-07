# 📊 Transformer Time Series Predictor

本项目是一个基于 **Hugging Face Transformers** 框架的时间序列预测模板，  
可用于金融、经济、气象等领域的趋势预测与建模。

---

## 📁 项目结构

FinanceTimeSeries/
├── data/
│ └── sample.csv # 示例时间序列数据
├── inference/
│ └── predict.py # 推理脚本：加载模型并预测
├── requirements.txt # 项目依赖
└── README.md # 项目说明文档


---

## ⚙️ 环境配置

建议使用 Python 3.10+  
推荐使用 Conda 虚拟环境：

```bash
conda create -n env_new python=3.10
conda activate env_new
pip install -r requirements.txt
