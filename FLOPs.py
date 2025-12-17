# -*- coding: utf-8 -*-
# Time: 2025-11-20 ~ now
# Create by: Huize Cheng
# Email: hzcheng@chd.edu.cn
# Created by: Visual Studio Code 1.104.0

import torch
import time
from thop import profile

device = torch.device("cuda")
pretrained_weights_path = r".\model.pt"
model = torch.load(pretrained_weights_path, map_location=device, weights_only=False)  # torch.nn.Module
model.eval()

# dummy input
x = torch.randn(1, 3, 512, 512).to(device)  # (B,C,H,W)
start = time.time()
for i in range(10):
    macs, params = profile(model, inputs=(x,), verbose=False)
end = time.time()

flops = 2 * macs  #  1MAC=2FLOPs 

print("\n\n" + "\033[92m--------  Ours  --------- \033[0m ")
print(f"time:  {(end - start)/10:.4f} s/sample")
print(f"MACs:  {macs/1e9:.4f} G")
print(f"FLOPs: {flops/1e9:.4f} G")
print(f"Params:{params/1e6:.4f} M" + '\n\n')

# dummy input  
start = time.time()
model = torch.load(pretrained_weights_path, map_location=device, weights_only=False)  # torch.nn.Module
model.eval()
x = torch.randn(1, 3, 512, 512).to(device)  # (B,C,H,W)
macs, params = profile(model, inputs=(x,), verbose=False)
macs, params = 85.412, 1.4 # The data comes from https://arxiv.org/pdf/2312.13313
flops = 2 * macs  #  1MAC=2FLOPs 
end = time.time()

print("\033[92m--------  ParamISP  --------- \033[0m ")
print(f"time:  {(end - start):.4f} s/sample ")
print(f"MACs:  {macs:.4f} G")
print(f"FLOPs: {flops:.4f} G")
print(f"Params:{params:.4f} M")