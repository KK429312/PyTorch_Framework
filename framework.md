# PyTorch搭建
## train.py
### 进度条
```commandline
from tqdm import tqdm
```

### 参数设置
```commandline
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--learning_rate', type=float, default=0.01)
args = parser.parse_args()
```

### 加载数据
```commandline
import torch.utils.data as Data
ld = Load_Data(data_size) // 自定义读取数据函数，读取原始数据
data = torch.tensor(ld.data, dtype=torch.float)
labels = torch.tensor(ld.labels, dtype=torch.long)
dataset = Data.TensorDataset(data, labels)
data_iter = Data.DataLoader(train_dataset, args.batch_size, shuffle=True)
```

### 初始化模型实例
```commandline
model = EEG_graph_net(
                      batch_size=args.batch_size,
                      learning_rate=args.learning_rate,
                      delta1=args.delta1,
                      delta2=args.delta2,
                      data_iter= data_iter,
                      )
```

### 训练
```commandline
model.train()
```

### 打印模型参数
```commandline
for name, parms in model.named_parameters():
    print(cm.Fore.BLACK+'-->name:', name, '-->grad_requirs:', parms.requires_grad, '--weight', torch.mean(parms.data),
          ' -->grad_value:', torch.mean(parms.grad))
```

## model.py
### 结构
```commandline
import numpy as np
from tqdm import tqdm
import torch
import torch.utils.data as Data
import torch.optim as optim
from torch import nn


class EEG_graph_net(nn.Module):
    def __init__(self,*):
        super().__init__()
     	pass
    def forward(self, x):
        pass
    def Loss(self,y_pre,y):
   	pass
    def Train(self):
 	pass
    def Predict(self):
     	pass
```

### 初始化参数
```commandline
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
```
### 注册MODEL的自定义参数（有AUTO-GRAD）
```commandline
m = torch.rand((3,3))
nn.init.kaiming_uniform_(m, gain=1)
self.M = torch.nn.Parameter(m)
self.register_parameter('M', self.M)  # 手动注册参数
```

### XAVIER初始化参数
```commandline
# xavier_uniform_初始化参数
for m in self.modules():
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.constant_(m.bias, 0)
    if isinstance(m, nn.Embedding):
        torch.nn.init.xavier_uniform_(m.weight)
    if isinstance(m, nn.Conv1d):
        torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.constant_(m.bias, 0)
```

### 设置OPTIMIZER
```commandline
import torch.optim as optim
self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
```