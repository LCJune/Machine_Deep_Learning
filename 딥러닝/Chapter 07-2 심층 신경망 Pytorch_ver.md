```python
# 실행마다 동일한 결과를 얻기 위해 파이토치에 랜덤 시드를 지정하고 GPU 연산을 결정적으로 만듭니다.
import torch

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True

from torchvision.datasets import FashionMNIST

fm_train = FashionMNIST(root='.', train=True, download=True) # root = '.' >> 현재 폴더에 생성
fm_test = FashionMNIST(root='.', train=False, download=True)

type(fm_train.data)
#torch.Tensor(PyTorch의 기본 데이터 구조)

print(fm_train.data.shape, fm_test.data.shape)
# torch.Size([60000, 28, 28]) torch.Size([10000, 28, 28])

print(fm_train.targets.shape, fm_test.targets.shape)
# torch.Size([60000]) torch.Size([10000]) 타깃이 1차원 == 원 핫 인코딩이 아닌 일반 정수값

train_input = fm_train.data
train_target = fm_train.targets
# train_scaled = train_input / 255.0

from sklearn.model_selection import train_test_split

train_scaled, val_scaled, train_target, val_target = train_test_split(
    train_scaled, train_target, test_size=0.2, random_state=42)

print(train_scaled.shape, val_scaled.shape)
# torch.Size([48000, 28, 28]) torch.Size([12000, 28, 28])

import torch.nn as nn

model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 100),
    nn.ReLU(),
    nn.Linear(100, 10)
)

!pip install torchinfo
"""
Collecting torchinfo
  Downloading torchinfo-1.8.0-py3-none-any.whl.metadata (21 kB)
Downloading torchinfo-1.8.0-py3-none-any.whl (23 kB)
Installing collected packages: torchinfo
Successfully installed torchinfo-1.8.0
"""

from torchinfo import summary

summary(model, input_size=(32, 28, 28))
"""
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
Sequential                               [32, 10]                  --
├─Flatten: 1-1                           [32, 784] << 28*28        --
├─Linear: 1-2                            [32, 100]                 78,500 << 784 * 100 + 100(bias)
├─ReLU: 1-3                              [32, 100]                 --
├─Linear: 1-4                            [32, 10]                  1,010 << 출력층
==========================================================================================
Total params: 79,510
Trainable params: 79,510
Non-trainable params: 0
Total mult-adds (Units.MEGABYTES): 2.54
==========================================================================================
Input size (MB): 0.10
Forward/backward pass size (MB): 0.03
Params size (MB): 0.32
Estimated Total Size (MB): 0.45
==========================================================================================
"""

import torch

# Keras와 달리 PyTorch는 명시적으로 GPU로 모델을 이동해야 함
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
# to() 메서드: 텐서 또는 모듈을 특정 장치(device)나 자료형(dtype)으로 이동·변환하기 위해 사용

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())
'''
torch.Size([100, 784])
torch.Size([100])
torch.Size([10, 100])
torch.Size([10])
'''

epochs = 5
batches = int(len(train_scaled)/32)
for epoch in range(epochs):
    model.train() # 모델을 훈련 모드로 설정
    train_loss = 0
    for i in range(batches):
        inputs = train_scaled[i*32:(i+1)*32].to(device)
        targets = train_target[i*32:(i+1)*32].to(device)
        optimizer.zero_grad() # 옵티마이저 gradient 초기화
        outputs = model(inputs) # 모델에 입력 전달
        loss = criterion(outputs, targets) # 손실 계산: cross entropy
        loss.backward() # 손실 역전파: 손실 함수로부터 각 파라미터의 gradient를 계산
        optimizer.step() # 모델 파라미터 업데이트
        train_loss += loss.item() # 에포크 손실 기록
    print(f"에포크:{epoch + 1}, 손실:{train_loss/batches:.4f}")
"""
에포크:1, 손실:0.5428
에포크:2, 손실:0.4004
에포크:3, 손실:0.3594
에포크:4, 손실:0.3320
에포크:5, 손실:0.3119
"""

model.eval()
with torch.no_grad():
    val_scaled = val_scaled.to(device)
    val_target = val_target.to(device)
    outputs = model(val_scaled)
    predicts = torch.argmax(outputs, 1)
    corrects = (predicts == val_target).sum().item()

accuracy = corrects / len(val_target)
print(f"검증 정확도: {accuracy:.4f}")
# 검증 정확도: 0.8719
"""
with 구문 사용 시
1. 블록 진입 시 자원 획득 (__enter__())
2. 블록 종료 시 자원 해제 (__exit__())

자원 획득과 해제를 자동으로 관리해주는 구문이다.
__enter__(), __exit__()
메서드 를 통해 예외 여부와 관계없이 안전한 정리를 보장한다.
"""
```
