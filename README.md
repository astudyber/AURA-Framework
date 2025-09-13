<div align="center">

# AURA: YCbCr-Based Universal RAW-Reconstruction for Inverse ISP

 

<div align="center">

This is the official repository for **AURA**.

![made-for-VSCode](https://img.shields.io/badge/Made%20for-VSCode-1f425f.svg)

</div>

</div>







## :bulb:Learning features in YCbCr​  color space

* 🚩  We propose AURA, a parameter-agnostic inverse ISP framework leveraging **YCbCr** perceptual decoupling for cross-device generalization. 

* 🚩  We design a **CSP module** performing multi-dimensional residual fusion to recover lost ISP details. 



![Diagram of the AURA Framework](./img/1.jpg)

* 🚩  We introduce a **noise-aware composite loss** enforcing stronger constraints on difficult regions for higher-fidelity RAW reconstruction. 

$$
\begin{equation}
L_{total} = \lambda_{rec} L_{rec} + \lambda_{str} L_{str} + \lambda_{perc} L_{perc}\\

L_{rec} = \frac{1}{N}\sum_{i=1}^{N} |x_i - y_i| + \lambda_{hlog} L_{HLog}\\

L_{HLog} = \frac{1}{N}\sum_{i=1}^{N} -\log(1 - \min(|x_i - y_i|, 1) + \epsilon)\\

L_{str} = 1 - \text{SSIM}(X, Y)\\

L_{perc} = \text{LPIPS}(X, Y)

\end{equation}
$$

* where the trade-off weights $\lambda_{rec}$, $\lambda_{str}$, $\lambda_{perc}$, and $\lambda_{hlog}$ are empirically set to 1.0, 0.1, 0.1, and 0.05, respectively.




##  :hourglass: Environment

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



##  :white_check_mark: How to Run

* Please prepare an environment such as pytorch in advance. You can use the following commands for inference: 

```cmd
python inference.py --folder test/ --output results/
```









 :rocket:

 :clipboard: 

  :dart: 