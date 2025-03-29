# A Res-CSP Network for ISP Inverse and Image Super-Resolution



## 1. Environment

* We use python==3.10 and pytorch >= 2.5.1  with CUDA version 12.4
* Running in a NVIDIA GeForce RTX 3080 GPU
* Create environment:

```python
conda create --name isp python=3.10
conda activate isp
```

* Install the corresponding version pytorch:

```python
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=12.1 -c pytorch -c nvidia
```

* Install other dependency packages:

```python
pip install -r requirements.txt
```



## 2. How to Run

* Please prepare an environment such as pytorch in advance. You can use the following commands for inference: 

```cmd
python inference.py --folder test/ --output results/
```

