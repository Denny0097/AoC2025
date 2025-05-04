# Lab 4 Homework Template

## PASS Screenshot
### DLA (total 30%)
##### Testbench `dla0` 7.5%
![截圖 2025-04-26 上午11.43.48](https://hackmd.io/_uploads/rkJFLCtyxe.png)

##### Testbench `dla1` 7.5%
![截圖 2025-04-26 上午11.44.19](https://hackmd.io/_uploads/SkAq8CFJeg.png)

##### Testbench `dla2` 7.5%
![截圖 2025-04-26 上午11.44.36](https://hackmd.io/_uploads/rJCjLCFJex.png)

##### Testbench `dla3` 7.5%
![截圖 2025-04-26 上午11.44.46](https://hackmd.io/_uploads/rkKhLCt1ex.png)


### CPU (total 60%)
Each `original version` operation contributes **7.5** points to the overall score.
Each `Improved version` operation is based on **the degree of cycle reduction**, with a full score of 7.5 points.
Please refer to the section below for detailed evaluation criteria.
##### `qconv2d_relu_maxpool_cpu`
**Original**
![截圖 2025-04-26 上午11.54.58](https://hackmd.io/_uploads/Bk6fYCKyxx.png)

**Improve**
![截圖 2025-04-26 上午11.56.57](https://hackmd.io/_uploads/SJPcYRY1xg.png)
**cycle reduction :** 63.9%

##### `qconv2d_relu_cpu`
**Original**
![截圖 2025-04-26 上午11.54.44](https://hackmd.io/_uploads/rJmfKRFJle.png)

**Improve**
![截圖 2025-04-26 上午11.56.37](https://hackmd.io/_uploads/Sk5FtCt1ll.png)
**cycle reduction :** 64.5%

##### `qlinear_relu_cpu`
**Original**
![截圖 2025-04-26 上午11.56.13](https://hackmd.io/_uploads/ByFDYRY1gx.png)

**Improve**
![截圖 2025-04-26 上午11.57.43](https://hackmd.io/_uploads/HkUat0FJee.png)
**cycle reduction :** 64.2%

##### `qlinear_cpu`
**Original**
![截圖 2025-04-26 上午11.55.08](https://hackmd.io/_uploads/rJIQF0KJxe.png)

**Improve**
![截圖 2025-04-26 上午11.57.26](https://hackmd.io/_uploads/SyunF0K1gl.png)
**cycle reduction :** 57.1%

## Performance in CPU
Please fill in the number of execution cycles and the results from the Valgrind analysis.
| Function     |            D refs      |        D1 miss      |   Cycle (roughly), depend on server status  |
|------------------|--------------------|---------------------|-----------|
|`CONV_original`         | 2,787,220,439  (2,475,678,398 rd   + 311,542,041 wr)        | 22,263  (       21,156 rd   +       1,107 wr)           |   1054719531       |
|`CONV_improve`          | 806,313,749  (727,517,606 rd   + 78,796,143 wr)        | 22,272  (     21,162 rd   +      1,110 wr)           |  380533666        |
|`CONV_MAX_original`     | 2,787,831,450  (2,476,131,430 rd   + 311,700,020 wr)        | 23,314  (       21,994 rd   +       1,320 wr)           |  1072161804        |
|`CONV_MAX_improve`      | 806,924,760  (727,970,638 rd   + 78,954,122 wr)        | 23,320  (     21,998 rd   +      1,322 wr)           |  380407686        |
|`LINEAR_original`       | 12,634,803  (12,621,996 rd   + 12,807 wr)        | 18,478  (    17,883 rd   +    595 wr)           |   10279403       |
|`LINEAR_improve`        | 10,208,826  (8,873,047 rd   + 1,335,779 wr)        | 1,181,757  (1,181,134 rd   +       623 wr)           |   3674914       |
|`LINEAR_RELU_original`  | 12,635,325  (12,622,513 rd   + 12,812 wr)        | 18,473  (    17,882 rd   +    591 wr)           |   10599868       |
|`LINEAR_RELU_improve`   | 10,208,837  (8,873,053 rd   + 1,335,784 wr)        | 1,181,757  (1,181,136 rd   +       621 wr)           |   4537250       |

**Scoring Criteria for every improved operation:**

| Cycles reduced ratio  | score | Note                  |
| -------------------------------- | ----- | --------------------- |
| <20%                             | 2.5%   | Basic implement score |
| 20%                              | 3.5%   |                   |
| 30%                              | 5%   |                   |
| 40%                              | 6.5%   |                   |
| 50%                              | 7.5%   |                   |

Cycle Reduction Ratio, defined as:
$$
\frac{\text{Cycle}_{\text{original}} - \text{Cycle}_{\text{improved}}}{\text{Cycle}_{\text{original}}}
$$

:::info
**❗Additional information:**
Since the number of D refs and D1 misses does not vary with the server's state, these values will be used as a reference during grading. The cycle count, however, will be evaluated with a more lenient standard.
:::


## How to improve performance in CPU 10%
:::info
Explain your method.
:::
### Conv & Conv_Max
Because CPU cache line loads blocks of memory together. Sequential access maximizes cache hit rate, reducing cache miss stalls., use row-major access pattern instead of jumping across rows randomly.

Reduce the number of loop counters(unrolling), branching, and improve ILP.

```c
    for (; ic + 7 < input_C; ic += 8) {
        const uint8_t* __restrict act_ic0 = activation + (ic + 0) * input_HW;
        const uint8_t* __restrict act_ic1 = activation + (ic + 1) * input_HW;
        const uint8_t* __restrict act_ic2 = activation + (ic + 2) * input_HW;
        const uint8_t* __restrict act_ic3 = activation + (ic + 3) * input_HW;
        const uint8_t* __restrict act_ic4 = activation + (ic + 4) * input_HW;
        const uint8_t* __restrict act_ic5 = activation + (ic + 5) * input_HW;
        const uint8_t* __restrict act_ic6 = activation + (ic + 6) * input_HW;
        const uint8_t* __restrict act_ic7 = activation + (ic + 7) * input_HW;

        const int8_t* __restrict w_ic0 = w_oc + (ic + 0) * filter_HW;
        const int8_t* __restrict w_ic1 = w_oc + (ic + 1) * filter_HW;
        const int8_t* __restrict w_ic2 = w_oc + (ic + 2) * filter_HW;
        const int8_t* __restrict w_ic3 = w_oc + (ic + 3) * filter_HW;
        const int8_t* __restrict w_ic4 = w_oc + (ic + 4) * filter_HW;
        const int8_t* __restrict w_ic5 = w_oc + (ic + 5) * filter_HW;
        const int8_t* __restrict w_ic6 = w_oc + (ic + 6) * filter_HW;
        const int8_t* __restrict w_ic7 = w_oc + (ic + 7) * filter_HW;

        for (uint32_t kh = 0; kh < filter_H; ++kh) {
            int32_t ih = ih_base + kh;
            if ((uint32_t)ih >= input_H) continue;

            const uint8_t* row0 = act_ic0 + ih * input_W;
            const uint8_t* row1 = act_ic1 + ih * input_W;
            const uint8_t* row2 = act_ic2 + ih * input_W;
            const uint8_t* row3 = act_ic3 + ih * input_W;
            const uint8_t* row4 = act_ic4 + ih * input_W;
            const uint8_t* row5 = act_ic5 + ih * input_W;
            const uint8_t* row6 = act_ic6 + ih * input_W;
            const uint8_t* row7 = act_ic7 + ih * input_W;

            const int8_t* wrow0 = w_ic0 + kh * filter_W;
            const int8_t* wrow1 = w_ic1 + kh * filter_W;
            const int8_t* wrow2 = w_ic2 + kh * filter_W;
            const int8_t* wrow3 = w_ic3 + kh * filter_W;
            const int8_t* wrow4 = w_ic4 + kh * filter_W;
            const int8_t* wrow5 = w_ic5 + kh * filter_W;
            const int8_t* wrow6 = w_ic6 + kh * filter_W;
            const int8_t* wrow7 = w_ic7 + kh * filter_W;

            for (uint32_t kw = 0; kw < filter_W; ++kw) {
                int32_t iw = iw_base + kw;
                if ((uint32_t)iw >= input_W) continue;

                acc += ((int32_t)row0[iw] - 128) * (int32_t)wrow0[kw];
                acc += ((int32_t)row1[iw] - 128) * (int32_t)wrow1[kw];
                acc += ((int32_t)row2[iw] - 128) * (int32_t)wrow2[kw];
                acc += ((int32_t)row3[iw] - 128) * (int32_t)wrow3[kw];
                acc += ((int32_t)row4[iw] - 128) * (int32_t)wrow4[kw];
                acc += ((int32_t)row5[iw] - 128) * (int32_t)wrow5[kw];
                acc += ((int32_t)row6[iw] - 128) * (int32_t)wrow6[kw];
                acc += ((int32_t)row7[iw] - 128) * (int32_t)wrow7[kw];
            }
        }
    }
```

### Linear & Linear_relu
Computes multiple output neurons at once.
```c
    int O_BLK = 8; // output block size
```
It processes 8 output neurons simultaneously, reducing the number of outer loop iterations, Reduces loop control overhead.

Processes the input in small chunks.
```c
    int I_BLK = 32; // input block size
```
Each block of input data is loaded and reused efficiently before moving to the next block.

loop unrolling : 
```c
switch (o_bound) {
default: acc[7] += a * (int32_t)w_ptr[7 * input_size];
case 7:  acc[6] += a * (int32_t)w_ptr[6 * input_size];
case 6:  acc[5] += a * (int32_t)w_ptr[5 * input_size];
case 5:  acc[4] += a * (int32_t)w_ptr[4 * input_size];
case 4:  acc[3] += a * (int32_t)w_ptr[3 * input_size];
case 3:  acc[2] += a * (int32_t)w_ptr[2 * input_size];
case 2:  acc[1] += a * (int32_t)w_ptr[1 * input_size];
case 1:  acc[0] += a * (int32_t)w_ptr[0 * input_size];
case 0:  break;
}
```
Eliminates branches, allowing better CPU branch prediction and pipelining.

The optimization introduced output blocking and manual loop unrolling, which changed the memory access pattern. Instead of accessing memory sequentially, the code now accesses multiple output neurons at once, leading to more strided memory access. This slightly increased cache misses at the L1 data cache level.

Although the data cache (D1) miss rate increased after optimization, the overall execution performance significantly improved.
(Modern CPUs can tolerate increased L1 cache misses as long as the last-leve cache hit rate remains high, which was observed in the results.)
## Feedback bonus 10%
因為lab3難度很高，導致這次作業比較延後才開始，而且又撞到proposal的準備，這禮拜非常的匆忙，但也感謝助教幫忙debug，也感謝宣佑指點我們隨機的第15組，希望一切順利。