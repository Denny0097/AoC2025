#include "hardware_cpu.h"

void conv_maxpooling(uint32_t input_C, uint32_t input_H, uint32_t input_W,
                     uint8_t* activation, uint32_t filter_N, uint32_t filter_C,
                     uint32_t filter_H, uint32_t filter_W, int8_t* filter,
                     int32_t* bias, uint32_t padding, uint8_t* output,
                     uint32_t scale) {

    uint32_t output_H = input_H + 2 * padding - filter_H + 1;
    uint32_t output_W = input_W + 2 * padding - filter_W + 1;
    uint32_t pool_H = output_H / 2;
    uint32_t pool_W = output_W / 2;

    uint8_t* conv_result = (uint8_t*)malloc(filter_N * output_H * output_W);
    conv(input_C, input_H, input_W, activation, filter_N, filter_C,
         filter_H, filter_W, filter, bias, padding, conv_result, scale);

    for (uint32_t n = 0; n < filter_N; n++) {
        for (uint32_t h = 0; h < pool_H; h++) {
            for (uint32_t w = 0; w < pool_W; w++) {
                uint8_t max_val = 0;
                for (uint32_t kh = 0; kh < 2; kh++) {
                    for (uint32_t kw = 0; kw < 2; kw++) {
                        uint32_t ih = h * 2 + kh;
                        uint32_t iw = w * 2 + kw;
                        uint8_t val = conv_result[((n * output_H + ih) * output_W) + iw];
                        if (val > max_val)
                            max_val = val;
                    }
                }
                output[((n * pool_H + h) * pool_W) + w] = max_val;
            }
        }
    }
    free(conv_result);
    
};

void conv(uint32_t input_C, uint32_t input_H, uint32_t input_W,
          uint8_t* activation, uint32_t filter_N, uint32_t filter_C,
          uint32_t filter_H, uint32_t filter_W, int8_t* filter, int32_t* bias,
          uint32_t padding, uint8_t* output, uint32_t scale) {

    uint32_t output_H = input_H + 2 * padding - filter_H + 1;
    uint32_t output_W = input_W + 2 * padding - filter_W + 1;

    for (uint32_t n = 0; n < filter_N; n++) {
        for (uint32_t oh = 0; oh < output_H; oh++) {
            for (uint32_t ow = 0; ow < output_W; ow++) {
                int32_t acc = bias[n];
                for (uint32_t c = 0; c < input_C; c++) {
                    for (uint32_t kh = 0; kh < filter_H; kh++) {
                        for (uint32_t kw = 0; kw < filter_W; kw++) {
                            int32_t ih = oh + kh - padding;
                            int32_t iw = ow + kw - padding;
                            if (ih >= 0 && ih < (int32_t)input_H &&
                                iw >= 0 && iw < (int32_t)input_W) {
                                uint8_t input_val = activation[((c * input_H + ih) * input_W) + iw] - 128;
                                int8_t filter_val = filter[((((n * input_C + c) * filter_H + kh) * filter_W) + kw)];
                                acc += input_val * filter_val;
                            }
                        }
                    }
                }
                output[((n * output_H + oh) * output_W) + ow] = requant(relu(acc), scale);
            }
        }
    }
};

void linear_relu(uint32_t input_size, uint32_t output_size, uint8_t* activation,
                 uint8_t* output, int8_t* filter, int32_t* bias,
                 uint32_t scale) {
    for (uint32_t i = 0; i < output_size; i++) {
        int32_t acc = bias[i];
        for (uint32_t j = 0; j < input_size; j++) {
            acc += (activation[j]-128) * filter[i * input_size + j];
        }
        output[i] = requant(relu(acc), scale);
    }
};

void linear(uint32_t input_size, uint32_t output_size, uint8_t* activation,
            uint8_t* output, int8_t* filter, int32_t* bias, uint32_t scale) {
    for (uint32_t i = 0; i < output_size; i++) {
        int32_t acc = bias[i];
        for (uint32_t j = 0; j < input_size; j++) {
            acc += (activation[j]-128) * filter[i * input_size + j];
        }
        output[i] = requant(acc, scale);
    }
};

void quantize(float* input_in_DRAM, uint8_t* output_in_DRAM, uint32_t size,
              uint32_t scale) {
    float fp_scale = 1;
    for (uint32_t i = 0; i < scale; i++) {
        fp_scale *= 2;
    }
    for (uint32_t i = 0; i < size; i++) {
        float t = input_in_DRAM[i] * fp_scale;
        int32_t temp = (int32_t)t + 128;
        // clamp to 0 ~ 255
        if (temp < 0) {
            output_in_DRAM[i] = 0;
        } else if (temp > 255)
            output_in_DRAM[i] = 255;
        else
            output_in_DRAM[i] = (uint8_t)temp;
    }
};

void dequantize(uint8_t* input_in_DRAM, float* output_in_DRAM, uint32_t size,
                uint32_t scale) {
    float fp_scale = 1;
    for (uint32_t i = 0; i < scale; i++) {
        fp_scale *= 2;
    }
    for (uint32_t i = 0; i < size; i++) {
        float temp = *(input_in_DRAM + i) - 128;
        *(output_in_DRAM + i) = temp / fp_scale;
    }
};
