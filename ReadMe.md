快速开始：
1. 首先创建一个虚拟环境
conda create -n python==3.12
2. 安装所需要的包
pip install -r requirements.txt
3. 安装verl
cd ver
pip intsall -e.

开始你的训练：
bash ./GRIL/train.sh
验证你的模型：
./GRIL/ragen/eval.py