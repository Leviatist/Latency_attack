# 创建并激活环境
conda create -n latency_attack python=3.11 -y
conda activate latency_attack
# 安装必要库
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1  pytorch-cuda=11.8 -c pytorch -c nvidia
conda install numpy opencv
pip install ultralytics