# Lab 1 Homework Template

### 1. Model Architecture (15%)
共五層 conv 的 blocks，每個 Block 包含 Convolution、BatchNorm 和 ReLU ，後跟 MaxPooling 來縮減特徵圖大小，最後使用Full Connect進行分類。
卷積層逐步增加通道數，從 32 到 256，通過 MaxPooling 最終展開成一維後進入全連接層。
在前三個 Block 中，Convolution 作為特徵截取，並在特徵截取後接一層 n->n Convolution，是為了模型在特徵擷取時能夠學習更深層的特徵，也利用BN提高訓練穩定、ReLU提高非線性特性，以上作為一個block，且總共加上四層 MaxPooling 對特徵圖降維(32->2)，減少最後全連接層的參數數量(256 * 2 * 2 -> 256)。
在經過五個 block 後，最後三層全連接層（1024 → 256 → 128 → 10）結合特徵進行最終分類。

    ----------------------------------------------------------------
            Layer (type)               Output Shape         Param #
    ================================================================
                Conv2d-1           [-1, 32, 32, 32]             896
           BatchNorm2d-2           [-1, 32, 32, 32]              64
                  ReLU-3           [-1, 32, 32, 32]               0
                Conv2d-4           [-1, 32, 32, 32]           9,248
           BatchNorm2d-5           [-1, 32, 32, 32]              64
                  ReLU-6           [-1, 32, 32, 32]               0
             MaxPool2d-7           [-1, 32, 16, 16]               0
                Conv2d-8           [-1, 64, 16, 16]          18,496
           BatchNorm2d-9           [-1, 64, 16, 16]             128
                 ReLU-10           [-1, 64, 16, 16]               0
               Conv2d-11           [-1, 64, 16, 16]          36,928
          BatchNorm2d-12           [-1, 64, 16, 16]             128
                 ReLU-13           [-1, 64, 16, 16]               0
            MaxPool2d-14             [-1, 64, 8, 8]               0
               Conv2d-15            [-1, 128, 8, 8]          73,856
          BatchNorm2d-16            [-1, 128, 8, 8]             256
                 ReLU-17            [-1, 128, 8, 8]               0
               Conv2d-18            [-1, 128, 8, 8]         147,584
          BatchNorm2d-19            [-1, 128, 8, 8]             256
                 ReLU-20            [-1, 128, 8, 8]               0
            MaxPool2d-21            [-1, 128, 4, 4]               0
               Conv2d-22            [-1, 256, 4, 4]         295,168
          BatchNorm2d-23            [-1, 256, 4, 4]             512
                 ReLU-24            [-1, 256, 4, 4]               0
               Conv2d-25            [-1, 256, 4, 4]         590,080
          BatchNorm2d-26            [-1, 256, 4, 4]             512
                 ReLU-27            [-1, 256, 4, 4]               0
            MaxPool2d-28            [-1, 256, 2, 2]               0
               Linear-29                  [-1, 256]         262,400
                 ReLU-30                  [-1, 256]               0
               Linear-31                  [-1, 128]          32,896
                 ReLU-32                  [-1, 128]               0
               Linear-33                   [-1, 10]           1,290
    ================================================================
    Total params: 1,470,762
    Trainable params: 1,470,762
    Non-trainable params: 0
    ----------------------------------------------------------------
    Input size (MB): 0.01
    Forward/backward pass size (MB): 2.94
    Params size (MB): 5.61
    Estimated Total Size (MB): 8.56
    ----------------------------------------------------------------



### 2. Loss/Epoch and Accuract/Epoch Plotting (15%)

Overfitting(during 0-20 Epoch)：
val_loss上升（從 1.8 到 2.0），而train_loss持續下降，
val_acc下降（從 0.65 到 0.5），而train_acc持續上升。
這表明模型在這段時間overfitting，導致泛化能力下降，
後續也有部分顯示存在overfitting的問題。

![image](../N26131960_lab1/image.png)



### 3. Accuracy Tuning (20%)
#### data preprocessing techniques and explain
- Data Augmentation
            transforms.RandomRotation(degrees=15),        
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.1)
- Loss Function
    -     criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
- Optimizer
    -     optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
- scheduler
    -     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
- Early Stopping
    -     early_stop_patience=10,  
```python
    best_val_acc = 0.0
    early_stop_count = 0
    
    
    # for epoch in epoch_loop:...
    
    # 若當前驗證準確率超越最佳值則儲存模型
    if _val_acc >= best_val_acc:
        best_val_acc = _val_acc
        early_stop_count = 0  # 重置 patience 計數器
        if _val_acc >= max(val_acc):
            save_model(model, save_path, existed="overwrite")
    else:
        early_stop_count += 1
        print(f"No improvement for {early_stop_count} epoch(s).")

    # 若超過 patience，則提前停止訓練
    if early_stop_count >= early_stop_patience:
        print(f"Early stopping triggered after {epoch+1} epochs.")
        break
```
#### hyperparameters and explain
| Hyperparameter | Loss function | Optimizer | Scheduler | Weight decay or Momentum | Epoch |
| -------------- | ------------- | --------- | --------- | ------------------------ | ----- |
| Value          |CrossEntropyLoss|SGD|CosineAnnealingLR|Weight decay : 5e-4, Momentum : 0.9|50|



Origin
嘗試用admin + ReduceLROnPlateau，訓練效果不好

    #Loss function
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    #Optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    #Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

Test: loss=1.1592, accuracy=0.7225
Model size: 5.91 MB
Plot saved at figure/cifar10/vgg-3.png
Time: 8050.48s

Newer
改用SGD + CosineAnnealingLR(參考[Reproducing ResNet + CIFAR 10 test error](https://discuss.pytorch.org/t/reproducing-resnet-cifar-10-test-error/56558) & 對 cifar10 訓練的參數)

    #Loss function
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    #Optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    #Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
Test: loss=0.8668, accuracy=0.8460
Model size: 5.91 MB
Plot saved at figure/cifar10/vgg-6.png
Time: 829.97s

### 4. Explain how the Power-of-Two Observer in your QConfig is implemented. (25%)
#### 1. Explain how to caluclate ***scale*** and ***zero-point***.
因為權重(weight)的值通常分佈在正負範圍內（−1,1)，所以weight的quant value使用 INT8（range [−128,127]）進行quantization。
反之Activation經過 ReLU：值是非負的（0,∞)，所以使用 UINT8（range [0,255])進行quantization。

設數據範圍 $[min_{value}, max_{value}]$、Output範圍$[r_{min}, r_{max}]$，
A. 考慮 weight ，令$r_{max} = max(abj(min_{value}), abj(max_{value}))$, 
mapping to $[-127, 127]$： 

- 計算 $scale$： 
    1. $2 * r_{max} / scale ≤ 127$ 
    2. $scale ≥ 2 * r_{max} / 127$
    3. $2 ^ {log_2(scale)} ≥ 2^{log_2(2 * r_{max} / 127)}$
    4. $scale = 2 ^ {log_2(scale)} ≈ 2^{round(log_2(2 * r_{max} / 127))}$
- 計算zero-point :
    採用 symetrical quantization。
    -> $ZeroPoint = 0$

B. Activation 直接取用$[min_{value}, max_{value}]$作計算，
mapping to UINT8 $[0,255]$： 
- 計算 $scale$： 
    1. $2 * (max_{value} - min_{value}) / scale ≤ 255$ 
    2. $scale ≥ 2 * (max_{value} - min_{value}) / 255$
    3. $2 ^ {log_2(scale)} ≥ 2^{log_2(2 * (max_{value} - min_{value}) / 255)}$
    4. $scale = 2 ^ {log_2(scale)} ≈ 2^{round(log_2(2 * (max_{value} - min_{value}) / 255))}$
- 計算zero-point :
    採用 symmetric quantization。
    -> $ZeroPoint = 128$

C. Quantization $q$(quantization value) :
    $q=round(x / scale) + ZeroPoint = round(x / 2^{ceil(log_2(2 * r_{max})}) + ZeroPoint = round(x × 2^{-ceil(log_2(2 * r_{max})}) + ZeroPoint$


#### 2. Explain how ```scale_approximate()``` function in ```class PowerOfTwoObserver()``` is implemented.

．Qconfig :
Using UINT8 for activations and INT8 for weights.
```python
class CustomQConfig(Enum):
    POWER2 = tq.QConfig(
        activation=PowerOfTwoObserver.with_args(
            dtype=torch.quint8, qscheme=torch.per_tensor_symmetric
        ),
        weight=PowerOfTwoObserver.with_args(
            dtype=torch.qint8, qscheme=torch.per_tensor_symmetric
        ),
    )
    DEFAULT = None
```


scale_approximate() :
```python
    def scale_approximate(self, scale: float, max_shift_amount=8) -> float:
        # ...
```

令 min_val, max_val 為 input data 之最小值和最大值。
```python
    min_val, max_val = self.min_val, self.max_val
```

令 r_min, r_max 為計算 input data 範圍之最小值和最大值。
$$
[r_{\min}, r_{\max}] =
\begin{cases}
[-r_{\max}, r_{\max}], & \text{if UINT} \\[2mm]
[\min_{\text{val}}, \max_{\text{val}}], & \text{otherwise}
\end{cases}
$$

```python        
    max_abs = max(abs(min_val), abs(max_val))
        # 避免除以0或inf
    if max_abs == 0 or max_abs == float("inf"):
        scale = 1.0
``` 

power-of-two uniform/scale, symmetric quantization：

・quantization range (q_min & q_max): 
$$
\begin{cases}
\text{INT8}[-128, 127], & \text{if INT} \\[2mm]
\text{UINT8}[0, 255], & \text{otherwise}
\end{cases}
$$


・zero point :
$$
ZeroPoint =
\begin{cases}
0, & \text{if INT} \\[2mm]
128, & \text{otherwise}
\end{cases}
$$

```python
    # when symmetric, [qmin, qmax] = [-127, 127]
    # when asymmetric, [qmin, qmax] = [0, 255]
    # r_max as max value of input data
    if self.dtype == torch.qint8:
        qmin, qmax = -127, 127
        zero_point = 0
    elif self.dtype == torch.quint8:
        qmin, qmax = 0, 255
        zero_point = 128
    else:
        raise ValueError("Unsupported dtype") 
```

・scale : 
$$
scale \geq
\begin{cases}
2 \cdot r_{\max} / 128, & \text{if UINT} \\[2mm]
2 \cdot (\max_{\text{value}} - \min_{\text{value}}) / 255, & \text{otherwise}
\end{cases}
$$


power-of-two uniform/scale(approximate),
$$
scale =
\begin{cases}
2^{\text{round}(\log_2(2 \cdot r_{\max} / 128))}, & \text{if UINT} \\[2mm]
2^{\text{round}(\log_2(2 \cdot (\max_{\text{value}} - \min_{\text{value}}) / 255))}, & \text{otherwise}
\end{cases}
$$


```python
    # 避免除以0或inf
    if max_abs == 0:
        scale = 0
    elif max_abs == float("inf"):
        scale = 1.0
    else:
        scale = 2 * max_abs / qmax
    scale = self.scale_approximate(scale)
    scale = torch.tensor(scale, dtype=torch.float32)
```
```python
def scale_approximate(self, scale: float, max_shift_amount=8) -> float:
    if scale == 0:
        return 0.0
    exponent = round(math.log(scale, 2))
    exponent = max(-max_shift_amount, min(exponent, 0))
    return 2 ** exponent
```



#### 3. When writing ```scale_approximate()```, is there a possibility of overflow? If so, how can it be handled?

- 如果 min_val == max_val，數據範圍為 0，無法有效量化，返回默認值 scale = 1.0 和 zero_point = 0 
- 如果max_abs==0 or == inf 會出現計算錯誤，令 scale = 0 跟 scale = 1 來避免前後兩個錯誤。
```python
    if max_abs == 0:
        scale = 0
    elif max_abs == float("inf"):
        scale = 1.0
```
- 如果使用INT8[-128, 127]在計算上會因為abj(-128)>127所以產生overflow，改用[-127, 127]來做 symmetric 的 scale 計算。
- 因為data type為int & uint，所以在計算zero point時用round將值四捨五入到最近的整數，並加上torch.clamp確保zero point在[q_min, q_max]範圍內。
```python
zero_point = torch.clamp(zero_point, qmin, qmax)
```

```python 
from enum import Enum
import math

import torch
import torch.ao.quantization as tq


class PowerOfTwoObserver(tq.MinMaxObserver):
    """
    Observer module for power-of-two quantization (dyadic quantization with b = 1).
    """

    def scale_approximate(self, scale: float, max_shift_amount=8) -> float:
        if scale == 0:
            return 0.0
        exponent = round(math.log(scale, 2))
        exponent = max(-max_shift_amount, min(exponent, 0))
        return 2 ** exponent
        
    def calculate_qparams(self):
        """Calculates the quantization parameters with scale as power of two."""
        min_val, max_val = self.min_val.item(), self.max_val.item()
        if self.dtype == torch.qint8:
            qmin, qmax = -127, 127
            zero_point = 0
        elif self.dtype == torch.quint8:
            qmin, qmax = 0, 255
            zero_point = 128
        else:
            raise ValueError("Unsupported dtype")
        
        max_abs = 2 * max(abs(min_val), abs(max_val))
        # 避免除以0或inf
        if max_abs == 0:
            scale = 0
        elif max_abs == float("inf"):
            scale = 1
        else:
            scale = max_abs / (qmax-qmin)
        scale = self.scale_approximate(scale)
        scale = torch.tensor(scale, dtype=torch.float32)
        zero_point = torch.tensor(zero_point, dtype=torch.int64)
        zero_point = torch.clamp(zero_point, qmin, qmax)
        # print(f"scale: {scale.item()}") 
        return scale, zero_point
    
class CustomQConfig(Enum):
    POWER2 = tq.QConfig(
        activation=PowerOfTwoObserver.with_args(
            dtype=torch.quint8, qscheme=torch.per_tensor_symmetric
        ),
        weight=PowerOfTwoObserver.with_args(
            dtype=torch.qint8, qscheme=torch.per_tensor_symmetric
        ),
    )
    DEFAULT = None
```

### 5. Comparison of Quantization Schemes (25%)

Given a **linear layer (128 → 10)** with an input shape of 1×128 and an output shape of 1×10, along with the energy costs for different data types, we will use the provided table to estimate the total energy consumption for executing such a fully connected layer during inference under the following two scenarios:

1. Full precision (FP32)
2. 8-bit integer, power-of-2, static, uniform symmetric quantization
    - activation: UINT8
    - weight: INT8


| Operation                        | Energy consumption (pJ)    |
| -------------------------------- | -------------------------- |
| FP32 Multiply                    | 3.7                        |
| FP32 Add                         | 0.9                        |
| <font color=red>INT32 Add</font> | <font color=red>0.1</font> |
| INT8 / UINT8 Multiply            | 0.2                        |
| INT8 / UINT8 Add                 | 0.03                       |
| Bit Shift                        | 0.01                       |

:::warning
#### 2025/02/19 Update

- The energy consumption of INT32 addition should also be considered. Each INT32 addition consumes 0.1 pJ of energy, as depicted in the figure of the lab hanout.
- Since we are using **static quantization** in this lab, the power-of-two scaling factors for input, weight, and output can fused into **ONE** integer before the inference.
- The summation is computed under **INT32** rather than INT16.
:::

You can ignore the energy consumption of type casting, memory movement, and other operations not listed in the table.

You can refer to the following formula previously-mentioned in the lab handout:



Write down your **calculation process** and **answer** in detail. Answers without the process will only get partial credit.

$$\tag{6}\bar y_i = \left( \text{ReLU}(\bar b_i + \sum_j (\bar x_j - 128) \cdot \bar w_{ji}) \gg \overbrace{(c_x + c_w - c_y)}^{\text{pre-computed offline}} \right) + 128$$

#### Your Answer

#### 1. FP32
與題目所給的quant後計算式不同，FP32 不需考慮預先使用zero point位移到指定範圍，也不用做scale shift，一個神經元的計算式如下。

$$
\begin{align}
y_i = \text{ReLU}(b_i + \sum_j x_j \cdot w_{ji})
\end{align}
$$


- i : 0~9
- j : 0~127

Data type of FP32 :


|           | Activation | weight | bias |
| --------- | ---------- | ------ | ---- |
| Data type |    FP32    |  FP32  | FP32 |




FP32 energe computation:
\begin{align}
\ & 10 \times \Biggl( 
\overbrace{(128 \times \text{FP32 Multiplication})}^{\text{Input tensor times weight}} \nonumber\\[1mm]
&\quad + \overbrace{(127 \times \text{FP32 Addition})}^{\text{Summation}} \nonumber\\[1mm]
&\quad + \overbrace{(1 \times \text{FP32 Addition})}^{\text{Add Bias}}\Biggr) \nonumber\\[1mm] 
&= 5888 \text{ pJ}
\end{align}


#### 2. INT8


$$\tag{6}\bar y_i = \left( \text{ReLU}(\bar b_i + \sum_j (\bar x_j - 128) \cdot \bar w_{ji}) \gg \overbrace{(c_x + c_w - c_y)}^{\text{pre-computed offline}} \right) + 128$$



Data type of INT8 :
|           | Activation | weight | bias | 
| --------- | ---------- | ------ | ---- |
| Data type |    UINT8   |  INT8  | INT32|

INT8 energe computation:
Given that the quantization scheme is power-of-2, static, uniform symmetric quantization , we use UINT8 for activations and INT8 for weights.

Consequently, when calculating the linear layer—which is performed in UINT8—the zero point is set to 128.

Additionally, because the quantization is static, we pre-calculate the scale and zero point before inference, therefore the sum of cx + cw - cy is pre-calculated.

$$
\begin{align}
\ & 10 \times \Biggl( 
\overbrace{(128 \times \text{INT8 Addition})}^{\text{subtract zero point}} \nonumber\\[1mm]
&\quad + \overbrace{(128 \times \text{INT8 Multiplication})}^{\text{Input tensor times weight}} \nonumber\\[1mm]
&\quad + \overbrace{(127 \times \text{INT32 Addition})}^{\text{Summation}} \nonumber\\[1mm]
&\quad + \overbrace{(1 \times \text{INT32 Addition})}^{\text{Add Bias}} \nonumber\\[1mm]
&\quad + \overbrace{(1 \times \text{Bit Shift})}^{\text{Power of Two Scale Shifting}} \nonumber\\[1mm]
&\quad + \overbrace{(1 \times \text{INT8 Addition})}^{\text{Add Zero Point}} \Biggr) \\ \nonumber\\[1mm]
&= 422.8 \text{ pJ}
\end{align}
$$


#### 3. Ans
|                         | Before quantization (FP32) | After quantization |
| ----------------------- | -------------------------- | --------- |
| Energy consumption (pJ) |    5888 pJ  |      422.8 pJ     |

