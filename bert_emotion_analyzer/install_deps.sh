#!/bin/bash
# install_deps.sh - 在 Codespaces 中安装依赖

echo "在 Codespaces 中安装 Python 3.11 依赖..."

# 添加 deadsnakes PPA 以获取 Python 3.11（适用于 Ubuntu 24.04）
sudo apt update
sudo apt install software-properties-common -y
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update

# 安装 Python 3.11
sudo apt install python3.11 python3.11-dev python3.11-venv python3.11-distutils -y

# 确保 python3.11 可用
if ! command -v python3.11 &> /dev/null; then
    echo "Python 3.11 安装失败，尝试替代方案..."
    # 替代方案：使用 conda
    if command -v conda &> /dev/null; then
        conda create -n py311 python=3.11 -y
        conda activate py311
    else
        echo "请手动安装 Python 3.11"
        exit 1
    fi
fi

# 创建虚拟环境
echo "创建虚拟环境..."
python3.11 -m venv venv

# 激活虚拟环境
source venv/bin/activate

# 升级 pip
pip install --upgrade pip

# 首先安装一些基础包
echo "安装基础依赖..."
pip install numpy==1.24.3 scipy==1.10.1 pandas==2.0.3 scikit-learn==1.3.0

# 安装 TensorFlow
echo "安装 TensorFlow..."
pip install tensorflow==2.13.0 tf-keras==2.13.0

# 安装 bert4keras
echo "安装 BERT 相关..."
pip install bert4keras==0.11.5 transformers==4.30.2

# 安装其他依赖
echo "安装其他依赖..."
pip install gensim==4.3.2 jieba==0.42.1 regex==2022.10.31 PyYAML==6.0
pip install matplotlib==3.7.1 seaborn==0.12.2 tqdm==4.65.0 colorlog==6.7.0 requests==2.31.0

echo "=================================="
echo "安装完成！"
echo "虚拟环境已激活"
echo "Python 版本："
python --version
echo "TensorFlow 版本："
python -c "import tensorflow as tf; print(f'TensorFlow: {tf.__version__}')" 2>/dev/null || echo "TensorFlow 未安装成功"
echo "=================================="
echo "使用方法："
echo "source venv/bin/activate  # 激活环境"
echo "python your_script.py     # 运行脚本"