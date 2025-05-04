#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "hardware_dla.h"
#include "runtime.h"

/*  //////////      NOTICE      //////////
    all parameter used to set DLA are send in by function argument
*/

void dla_stop() {
    // set disable
    reg_write(DLA_ENABLE_OFFSET, 0);
}

void dla_init() {
#ifdef DLA_INFO
    fprintf(stdout, "DLA runtime info logging enabled.\n");
    dla_reset_runtime_info();
    create_dla_info_to_csv(DLA_INFO_CSV);
#endif
    hal_init();
}

void dla_final() {
#ifdef DLA_INFO
    fprintf(stdout, "Creating dla info file: %s\n", DLA_INFO_CSV);
#endif
    hal_final();
}

void dla_reset_runtime_info() { reset_runtime_info(); }

void create_dla_info_to_csv(const char *filename) {
    fprintf(stdout, "Creating dla info file: %s\n", filename);
    FILE *file = fopen(filename, "w");
    if (!file) {
        fprintf(stderr, "Create DLA info file failed.\n");
        return;
    }
    fprintf(file,
            "Operation,Cycles,Time(ns),Memory read,Memory "
            "write,m,e,p,q,r,t,PAD,U,R,S,C,M,W,H\n");
    fclose(file);
}

void dump_dla_info_to_csv(const char *filename, const char *operation_name,
                          // mapping parameter
                          uint32_t m, uint32_t e, uint32_t p, uint32_t q,
                          uint32_t r, uint32_t t,
                          // shape parameter
                          uint32_t PAD, uint32_t U, uint32_t R, uint32_t S,
                          uint32_t C, uint32_t M, uint32_t W, uint32_t H) {
    FILE *file = fopen(filename, "a");
    struct runtime_info info = get_runtime_info();
    fprintf(file, "%s,", operation_name);        // Operation
    fprintf(file, "%10d,", info.elapsed_cycle);  // Cycles
    fprintf(file, "%10d,", info.elapsed_time);   // Time (ns)
    fprintf(file, "%10d,", info.memory_read);    // Memory read
    fprintf(file, "%10d,", info.memory_write);   // Memory write
    fprintf(file, "%d,%d,%d,%d,%d,%d,", m, e, p, q, r, t);
    fprintf(file, "%d,%d,%d,%d,%d,%d,%d,%d\n", PAD, U, R, S, C, M, W, H);
    fclose(file);
}

int qconv2d_relu_maxpool(
    uint8_t *input_in_DRAM, int8_t *filter_in_DRAM, uint8_t *opsum_in_DRAM,
    int32_t *bias, uint32_t ofmap_len, uint32_t ifmap_len, uint32_t filter_len,
    // mapping parameter
    uint32_t m, uint32_t e, uint32_t p, uint32_t q, uint32_t r, uint32_t t,
    // shape parameter
    uint32_t PAD, uint32_t U, uint32_t R, uint32_t S, uint32_t C, uint32_t M,
    uint32_t W, uint32_t H,
    uint32_t scale) {  // int32_t scale_factor: merge ifmap and weight and ofmap
    // scale bit-shift

#ifdef DLA_INFO
    dla_reset_runtime_info();
#endif
    // Calculate m for GLB memory allocation
    /*! <<<========= Implement here =========>>>*/
    int GLB_size = 64 * 1024;  // GLB size in bytes
    // int W_padding = W + 2 * PAD;
    int F = (e + W - R) / U + 1;

    int ifmap_usage = (q * r) * (e + R - 1) * W;
    int filter_usage = (p * t) * (q * r) * R * S;
    //  calculate the largest power-of-two value for m
    int opsum_usage;
    int bias_usage;

    int best_m = 1;
    for (int i = 1; i <= M; i *= 2) {
        opsum_usage = i * e * F * 4;  // 4 bytes per int32
        bias_usage = i * 4;
        int total_usage = ifmap_usage + filter_usage + bias_usage + opsum_usage;
        if (total_usage <= GLB_size) {
            best_m = i;
            m = best_m;
        } else {
            break;
        }
    }
    bias_usage = m * 4;
    opsum_usage = m * e * F * 4;

    // call lower setting functions
    /*! <<<========= Implement here =========>>>*/
    // set GLB memory address
    set_glb_filter_addr(ifmap_usage);
    set_glb_bias_addr(ifmap_usage + filter_usage);
    set_glb_ofmap_addr(ifmap_usage + filter_usage + bias_usage);

    // set input data addr
    set_ifmap_addr(input_in_DRAM);
    set_filter_addr(filter_in_DRAM);
    set_bias_addr(bias);
    set_opsum_addr(opsum_in_DRAM);
    
    // set total data number
    set_input_activation_len(ifmap_len);
    set_output_activation_len(ofmap_len);
    
    // set mapping param
    set_mapping_param(m, e, p, q, r, t);
    set_shape_param1(PAD, U, R, S, C, M);
    set_shape_param2(W, H, PAD);
    
    // Enable DLA (with maxpool and relu)
    set_enable(scale, true, true, 0);  // maxpool, relu, operation

    wait_for_interrupt();
    dla_stop();
#ifdef DLA_INFO
    dump_dla_info_to_csv(DLA_INFO_CSV, "qconv2d_relu_maxpool", m, e, p, q, r, t,
                         PAD, U, R, S, C, M, W, H);
#endif
    return 0;
};

int qconv2d_relu(uint8_t *input_in_DRAM, int8_t *filter_in_DRAM,
                 uint8_t *opsum_in_DRAM, int32_t *bias, uint32_t ofmap_len,
                 uint32_t ifmap_len, uint32_t filter_len,
                 // mapping parameter
                 uint32_t m, uint32_t e, uint32_t p, uint32_t q, uint32_t r,
                 uint32_t t,
                 // shape parameter
                 uint32_t PAD, uint32_t U, uint32_t R, uint32_t S, uint32_t C,
                 uint32_t M, uint32_t W, uint32_t H,
                 uint32_t scale) {  // int32_t scale_factor: merge ifmap and
                                    // ofmap scale bit-shift
#ifdef DLA_INFO
    dla_reset_runtime_info();
#endif
    // Calculate m for GLB memory allocation
    /*! <<<========= Implement here =========>>>*/
    int GLB_size = 64 * 1024;  // GLB size in bytes
    // int W_padding = W + 2 * PAD;
    int F = (e + W - R) / U + 1;

    int ifmap_usage = (q * r) * (e + R - 1) * W;
    int filter_usage = (p * t) * (q * r) * R * S;

    //  calculate the largest power-of-two value for m
    int opsum_usage;
    int bias_usage;

    int best_m = 1;
    for (int i = 1; i<= M; i *= 2) {
        opsum_usage = i * e * F * 4;  // 4 bytes per int32
        bias_usage = i * 4;
        int total_usage = ifmap_usage + filter_usage + bias_usage + opsum_usage;
        if (total_usage <= GLB_size) {
            best_m = i;
            m = best_m;
        } else {
            break;
        }
    }
    
    bias_usage = m * 4;
    opsum_usage = m * e * F * 4;

    // call lower setting functions
    /*! <<<========= Implement here =========>>>*/
    // set GLB memory address
    set_glb_filter_addr(ifmap_usage);
    set_glb_bias_addr(ifmap_usage + filter_usage);
    set_glb_ofmap_addr(ifmap_usage + filter_usage + bias_usage);

    // set input data addr
    set_ifmap_addr(input_in_DRAM);
    set_filter_addr(filter_in_DRAM);
    set_bias_addr(bias);
    set_opsum_addr(opsum_in_DRAM);
    
    // set total data number
    set_input_activation_len(ifmap_len);
    set_output_activation_len(ofmap_len);
    
    // set mapping param
    set_mapping_param(m, e, p, q, r, t);
    set_shape_param1(PAD, U, R, S, C, M);
    set_shape_param2(W, H, PAD);

    // Enable DLA (with maxpool and relu)
    set_enable(scale, false, true, 0);

    wait_for_interrupt();
    dla_stop();
#ifdef DLA_INFO
    dump_dla_info_to_csv(DLA_INFO_CSV, "qconv2d_relu", m, e, p, q, r, t, PAD, U,
                         R, S, C, M, W, H);
#endif
    return 0;
};
