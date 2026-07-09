#ifndef __LN_H__
#define __LN_H__

#include <ap_int.h>
#include <ap_fixed.h>
#include <hls_math.h>
#include <hls_vector.h>
#include <hls_stream.h>
#include "ln_config.h"

using namespace hls;

// ---------------------------------------------------------
// 1. Type Definition
// ---------------------------------------------------------
typedef vector<float,      PACK_SIZE>     pack_t;
typedef vector<float,      PACK_SIZE_INT> pack_float_t;
typedef vector<ap_uint<8>, PACK_SIZE_INT> pack_uint8_t;
typedef vector<int16_t,    PACK_SIZE_INT> pack_int16_t;
typedef int32_t acc_t;

// fixed point definition
typedef ap_ufixed<20, 4>             mscale_t;
typedef ap_fixed<30, 14>             addacc_t;
typedef ap_fixed<14, 14, AP_RND_INF> qint_t;

typedef union {
    uint32_t i;
    float f;
} fp_conv_t;

// ---------------------------------------------------------
// 2. Prototypes of Functions
// ---------------------------------------------------------

void inp_a_loader(
    uint32_t b,
    ap_uint<32> seqlen,
    ap_uint<32> seqlen_pad,
    ap_uint<32> parts,
    pack_uint8_t* inp,
    stream<pack_uint8_t>& stream_inp);

void inp_b_loader(
    bool en_residual,
    uint32_t b,
    ap_uint<32> seqlen,
    ap_uint<32> seqlen_pad,
    ap_uint<32> parts,
    pack_uint8_t* inp,
    stream<pack_uint8_t>& stream_inp);

void add(
    bool        en_residual,
    ap_uint<32> seqlen,
    ap_uint<32> parts,
    float       scale_a,
    float       scale_b,
    float       scale_c,
    uint8_t     zp_a,
    uint8_t     zp_b,
    uint8_t     zp_c,
    stream<pack_uint8_t>& stream_inp_a,
    stream<pack_uint8_t>& stream_inp_b,
    stream<pack_uint8_t>& stream_out_1,
    stream<pack_uint8_t>& stream_out_2,
    stream<pack_uint8_t>& stream_out_a);

void accumulation(
    ap_uint<32> seqlen,
    ap_uint<32> parts,
    uint8_t     zp_in,
    stream<pack_uint8_t>& stream_inp,
    stream<acc_t>& stream_sum_x1,
    stream<acc_t>& stream_sum_x2);

void statistics(
    ap_uint<32>    seqlen,
    ap_uint<32>    dim,
    float          scale_in,
    float          eps,
    stream<acc_t>& stream_sum_x1,
    stream<acc_t>& stream_sum_x2,
    stream<float>& stream_mean,
    stream<float>& stream_rstd);

void normalization(
    ap_uint<32> seqlen,
    ap_uint<32> parts,
    ap_uint<32> dim,
    float       scale_in,
    uint8_t     zp_in,
    pack_t      mem_wght[NUM_PACKS],
    pack_t      mem_bias[NUM_PACKS],
    stream<pack_uint8_t>& stream_inp,
    stream<pack_float_t>& stream_norm,
    stream<float>& stream_mean,
    stream<float>& stream_rstd);

void out_quantization(
    ap_uint<32>           seqlen,
    ap_uint<32>           parts,
    float                 scale_o,
    uint8_t               zp_o,
    stream<pack_float_t>& stream_norm,
    stream<pack_uint8_t>& stream_out);

void out_a_writer(
    bool                  en_residual,
    uint32_t              b,
    ap_uint<32>           seqlen,
    ap_uint<32>           seqlen_pad,
    ap_uint<32>           parts,
    pack_uint8_t          zp_vec,
    stream<pack_uint8_t>& stream_out,
    pack_uint8_t*         out);

void out_b_writer(
    uint32_t              b,
    ap_uint<32>           seqlen,
    ap_uint<32>           seqlen_pad,
    ap_uint<32>           parts,
    pack_uint8_t          zp_vec,
    stream<pack_uint8_t>& stream_out,
    pack_uint8_t*         out);

void layernorm(
    ap_uint<32>   mode,
    // model parameters
    ap_uint<32>   batch,
    ap_uint<32>   seqlen,
    ap_uint<32>   dim,
    float         eps,
    // quantization parameters
    float         scale_a,
    float         scale_b,
    float         scale_c,
    float         scale_o,
    uint8_t       zp_a,
    uint8_t       zp_b,
    uint8_t       zp_c,
    uint8_t       zp_o,
    // AXI ports
    pack_uint8_t* inp_a,
    pack_uint8_t* inp_b,
    pack_t*       par_0,
    pack_uint8_t* out_a,
    pack_uint8_t* out_b);

// ---------------------------------------------------------
// 3. RTL Blackbox Prototypes
// ---------------------------------------------------------
void itofp_stage1(
    bool      is_signed,
    uint32_t  din,
    bool*     sign,
    uint32_t* abs,
    bool*     zero
);

void itofp_stage2(
    bool      in_sign,
    uint32_t  in_abs,
    bool      in_zero,
    uint32_t* out_abs,
    uint16_t* out_count_allz,
    bool* out_sign,
    bool* out_zero
);

void itofp_stage3(
    bool      in_sign,
    bool      in_zero,
    uint32_t  in_abs,
    uint16_t  in_count_allz,
    bool* out_sign,
    uint8_t* out_exponent,
    uint8_t* out_lzc,
    uint32_t* out_abs
);

void itofp_stage4(
    bool      in_sign,
    uint8_t   in_exponent,
    uint8_t   in_lzc,
    uint32_t  in_abs,
    uint32_t* out_data_tmp,
    bool* out_g,
    bool* out_r,
    bool* out_s
);

void itofp_stage5(
    uint32_t  in_data_tmp,
    bool      in_g,
    bool      in_r,
    bool      in_s,
    float*    dout
);

void itofp(bool is_signed, uint32_t din, float* dout);

void itofp_small(ap_int<9> in, float* dout);

#endif // __LN_H__