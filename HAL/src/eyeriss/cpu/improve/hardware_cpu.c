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
    uint32_t input_HW = input_H * input_W;
    uint32_t output_HW = output_H * output_W;
    uint32_t filter_HW = filter_H * filter_W;

    for (uint32_t oc = 0; oc < filter_N; ++oc) {
        const int8_t* __restrict w_oc = filter + oc * filter_C * filter_HW;
        int32_t bias_val = bias[oc];

        for (uint32_t oh = 0; oh < output_H; ++oh) {
            int32_t ih_base = (int32_t)oh - (int32_t)padding;

            for (uint32_t ow = 0; ow < output_W; ++ow) {
                int32_t iw_base = (int32_t)ow - (int32_t)padding;
                int32_t acc = bias_val;

                uint32_t ic = 0;
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

                // over 8 input channel
                for (; ic < input_C; ++ic) {
                    const uint8_t* act_ic = activation + ic * input_HW;
                    const int8_t* w_ic = w_oc + ic * filter_HW;

                    for (uint32_t kh = 0; kh < filter_H; ++kh) {
                        int32_t ih = ih_base + kh;
                        if ((uint32_t)ih >= input_H) continue;

                        const uint8_t* row_act = act_ic + ih * input_W;
                        const int8_t* row_w = w_ic + kh * filter_W;

                        for (uint32_t kw = 0; kw < filter_W; ++kw) {
                            int32_t iw = iw_base + kw;
                            if ((uint32_t)iw >= input_W) continue;

                            acc += ((int32_t)row_act[iw] - 128) * (int32_t)row_w[kw];
                        }
                    }
                }

                output[((oc * output_H + oh) * output_W) + ow] = requant(relu(acc), scale);
            }
        }
    }
}

void linear_relu(uint32_t input_size, uint32_t output_size, uint8_t* activation,
                 uint8_t* output, int8_t* filter, int32_t* bias,
                 uint32_t scale) {
    int O_BLK = 8; // output block size
    int I_BLK = 32; // input block size

    int32_t acc[O_BLK];
    for (uint32_t o_base = 0; o_base < output_size; o_base += O_BLK) {
        uint32_t o_bound = (o_base + O_BLK <= output_size) ? O_BLK : (output_size - o_base);

        for (uint32_t o = 0; o < o_bound; ++o)
            acc[o] = bias[o_base + o];

        for (uint32_t i_base = 0; i_base < input_size; i_base += I_BLK) {
            uint32_t i_bound = (i_base + I_BLK <= input_size) ? I_BLK : (input_size - i_base);
            uint8_t* act_ptr = activation + i_base;

            for (uint32_t i = 0; i < i_bound; ++i) {
                int32_t a = (int32_t)act_ptr[i] - 128;
                int8_t* w_ptr = filter + (o_base * input_size) + (i_base + i);

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
            }
        }

        for (uint32_t o = 0; o < o_bound; ++o)
            output[o_base + o] = requant(relu(acc[o]), scale);
    }
};

void linear(uint32_t input_size, uint32_t output_size, uint8_t* activation,
            uint8_t* output, int8_t* filter, int32_t* bias, uint32_t scale) {

    int O_BLK = 8; // output block size
    int I_BLK = 32; // input block size

    int32_t acc[O_BLK];
    for (uint32_t o_base = 0; o_base < output_size; o_base += O_BLK) {
        uint32_t o_bound = (o_base + O_BLK <= output_size) ? O_BLK : (output_size - o_base);

        for (uint32_t o = 0; o < o_bound; ++o)
            acc[o] = bias[o_base + o];

        for (uint32_t i_base = 0; i_base < input_size; i_base += I_BLK) {
            uint32_t i_bound = (i_base + I_BLK <= input_size) ? I_BLK : (input_size - i_base);
            uint8_t* act_ptr = activation + i_base;

            for (uint32_t i = 0; i < i_bound; ++i) {
                int32_t a = (int32_t)act_ptr[i] - 128;
                int8_t* w_ptr = filter + (o_base * input_size) + (i_base + i);

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
            }
        }

        for (uint32_t o = 0; o < o_bound; ++o)
            output[o_base + o] = requant(acc[o], scale);
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
