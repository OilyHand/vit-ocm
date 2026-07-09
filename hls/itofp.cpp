#include "ln.h"

void itofp_stage1(
    bool      is_signed,
    uint32_t  din,
    bool*     sign,
    uint32_t* abs,
    bool*     zero
){
#pragma HLS INLINE
    *sign = is_signed ? ((int32_t) din < 0) : false;
    *abs = (is_signed ) && ((int32_t)din < 0) ? -din : din;
    *zero = (din == 0);
}

void itofp_stage2(
    bool      in_sign,
    uint32_t  in_abs,
    bool      in_zero,
    uint32_t* out_abs,
    uint16_t* out_count_allz,
    bool* out_sign,
    bool* out_zero
){
#pragma HLS INLINE
    *out_sign = in_sign;
    *out_zero = in_zero;
    *out_abs  = in_abs;

    uint8_t d3 = (in_abs >> 24) & 0xFF;
    uint8_t d2 = (in_abs >> 16) & 0xFF;
    uint8_t d1 = (in_abs >> 8)  & 0xFF;
    uint8_t d0 = (in_abs)       & 0xFF;

    uint8_t b3_cnt, b2_cnt, b1_cnt, b0_cnt;
    
    if      (d3 & 0x80) b3_cnt = 0;
    else if (d3 & 0x40) b3_cnt = 1;
    else if (d3 & 0x20) b3_cnt = 2;
    else if (d3 & 0x10) b3_cnt = 3;
    else if (d3 & 0x08) b3_cnt = 4;
    else if (d3 & 0x04) b3_cnt = 5;
    else if (d3 & 0x02) b3_cnt = 6;
    else if (d3 & 0x01) b3_cnt = 7;
    else                b3_cnt = 0;

    if      (d2 & 0x80) b2_cnt = 0;
    else if (d2 & 0x40) b2_cnt = 1;
    else if (d2 & 0x20) b2_cnt = 2;
    else if (d2 & 0x10) b2_cnt = 3;
    else if (d2 & 0x08) b2_cnt = 4;
    else if (d2 & 0x04) b2_cnt = 5;
    else if (d2 & 0x02) b2_cnt = 6;
    else if (d2 & 0x01) b2_cnt = 7;
    else                b2_cnt = 0;

    if      (d1 & 0x80) b1_cnt = 0;
    else if (d1 & 0x40) b1_cnt = 1;
    else if (d1 & 0x20) b1_cnt = 2;
    else if (d1 & 0x10) b1_cnt = 3;
    else if (d1 & 0x08) b1_cnt = 4;
    else if (d1 & 0x04) b1_cnt = 5;
    else if (d1 & 0x02) b1_cnt = 6;
    else if (d1 & 0x01) b1_cnt = 7;
    else                b1_cnt = 0;

    if      (d0 & 0x80) b0_cnt = 0;
    else if (d0 & 0x40) b0_cnt = 1;
    else if (d0 & 0x20) b0_cnt = 2;
    else if (d0 & 0x10) b0_cnt = 3;
    else if (d0 & 0x08) b0_cnt = 4;
    else if (d0 & 0x04) b0_cnt = 5;
    else if (d0 & 0x02) b0_cnt = 6;
    else if (d0 & 0x01) b0_cnt = 7;
    else                b0_cnt = 0;
    
    bool b3_z = (d3 == 0);
    bool b2_z = (d2 == 0);
    bool b1_z = (d1 == 0);
    bool b0_z = (d0 == 0);

    *out_count_allz = ((b3_cnt & 0x7) << 13) | ((b3_z ? 1 : 0) << 12) |
                      ((b2_cnt & 0x7) << 9)  | ((b2_z ? 1 : 0) << 8)  |
                      ((b1_cnt & 0x7) << 5)  | ((b1_z ? 1 : 0) << 4)  |
                      ((b0_cnt & 0x7) << 1)  | ((b0_z ? 1 : 0) << 0);
}

void itofp_stage3(
    bool      in_sign,
    bool      in_zero,
    uint32_t  in_abs,
    uint16_t  in_count_allz,
    bool* out_sign,
    uint8_t* out_exponent,
    uint8_t* out_lzc,
    uint32_t* out_abs
){
#pragma HLS INLINE
    uint8_t b3_cnt = (in_count_allz >> 13) & 0x7; bool b3_z = (in_count_allz >> 12) & 1;
    uint8_t b2_cnt = (in_count_allz >> 9)  & 0x7; bool b2_z = (in_count_allz >> 8)  & 1;
    uint8_t b1_cnt = (in_count_allz >> 5)  & 0x7; bool b1_z = (in_count_allz >> 4)  & 1;
    uint8_t b0_cnt = (in_count_allz >> 1)  & 0x7; bool b0_z = (in_count_allz >> 0)  & 1;

    uint8_t block_sel = 0;
    uint8_t local_count = 0;

    // Priority Multiplexer
    if      (!b3_z) { block_sel = 0; local_count = b3_cnt; }
    else if (!b2_z) { block_sel = 1; local_count = b2_cnt; }
    else if (!b1_z) { block_sel = 2; local_count = b1_cnt; }
    else            { block_sel = 3; local_count = b0_cnt; }

    uint8_t final_lzc = (block_sel << 3) | local_count; // 5-bit LZC

    uint8_t exp_calc = 158 - final_lzc;
    uint8_t final_exponent = in_zero ? 0 : exp_calc;

    *out_sign     = in_sign;
    *out_exponent = final_exponent;
    *out_lzc      = final_lzc;
    *out_abs      = in_abs;
}

void itofp_stage4(
    bool      in_sign,
    uint8_t   in_exponent,
    uint8_t   in_lzc,
    uint32_t  in_abs,
    uint32_t* out_data_tmp,
    bool* out_g,
    bool* out_r,
    bool* out_s
){
#pragma HLS INLINE
    uint32_t shift_16 = (in_lzc & 0x10) ? (in_abs << 16) : in_abs;
    uint32_t shift_8  = (in_lzc & 0x08) ? (shift_16 << 8) : shift_16;
    uint32_t shift_4  = (in_lzc & 0x04) ? (shift_8 << 4) : shift_8;
    uint32_t shift_2  = (in_lzc & 0x02) ? (shift_4 << 2) : shift_4;
    uint32_t shift_1  = (in_lzc & 0x01) ? (shift_2 << 1) : shift_2;

    uint32_t mantissa = (shift_1 >> 8) & 0x7FFFFF;
    
    *out_g = (shift_1 >> 7) & 1;
    *out_r = (shift_1 >> 6) & 1;
    *out_s = (shift_1 & 0x3F) != 0;

    *out_data_tmp = ((in_sign ? 1 : 0) << 31) | 
                    ((in_exponent & 0xFF) << 23) | 
                    mantissa;
}

void itofp_stage5(
    uint32_t  in_data_tmp,
    bool      in_g,
    bool      in_r,
    bool      in_s,
    float*    dout
){
#pragma HLS INLINE
    bool     in_sign     = (in_data_tmp >> 31) & 1;
    uint8_t  in_exponent = (in_data_tmp >> 23) & 0xFF;
    uint32_t in_mantissa = (in_data_tmp & 0x7FFFFF);

    bool l_bit = in_mantissa & 1;
    bool round_up = in_g && (in_r || in_s || l_bit);

    uint32_t mantissa_add = in_mantissa + (round_up ? 1 : 0);
    bool mantissa_overflow = (mantissa_add >> 23) & 1;

    uint32_t final_mantissa = mantissa_overflow ? 0 : (mantissa_add & 0x7FFFFF);
    uint8_t  final_exponent = mantissa_overflow ? (in_exponent + 1) : in_exponent;

    // floating point single precision IEEE 754 format
    uint32_t packed_fp = ((in_sign ? 1 : 0) << 31) | 
                         ((final_exponent & 0xFF) << 23) | 
                         final_mantissa;

    fp_conv_t conv;
    conv.i = packed_fp;
    *dout = conv.f;
}

void itofp(bool is_signed, uint32_t din, float* dout) {
    #pragma HLS INLINE
    bool s1_sign, s1_zero; uint32_t s1_abs;
    itofp_stage1(is_signed, din, &s1_sign, &s1_abs, &s1_zero);

    uint32_t s2_abs; uint16_t s2_count_allz; bool s2_sign, s2_zero;
    itofp_stage2(s1_sign, s1_abs, s1_zero, &s2_abs, &s2_count_allz, &s2_sign, &s2_zero);

    bool s3_sign; uint8_t s3_exponent, s3_lzc; uint32_t s3_abs;
    itofp_stage3(s2_sign, s2_zero, s2_abs, s2_count_allz, &s3_sign, &s3_exponent, &s3_lzc, &s3_abs);

    uint32_t s4_data_tmp; bool s4_g, s4_r, s4_s;
    itofp_stage4(s3_sign, s3_exponent, s3_lzc, s3_abs, &s4_data_tmp, &s4_g, &s4_r, &s4_s);

    itofp_stage5(s4_data_tmp, s4_g, s4_r, s4_s, dout);
}

// Narrow int->float for operands with |value| <= 255 (the int8/int9 datapath).
// Same IEEE-754 bits as itofp() over this range, but with an 8-bit LZC, a fixed
// small shift, and NO guard/round/sticky (the value is always exact).
void itofp_small(ap_int<9> in, float* dout) {
#pragma HLS INLINE
    bool       sign = (in < 0);
    ap_uint<8> a    = sign ? (ap_uint<8>)(-in) : (ap_uint<8>)in;   // |in| in [0,255]

    fp_conv_t conv;
    if (a == 0) { conv.i = 0; *dout = conv.f; return; }            // +0.0f

    ap_uint<3> n;                                                   // MSB position, 0..7
    if      (a[7]) n = 7;  else if (a[6]) n = 6;
    else if (a[5]) n = 5;  else if (a[4]) n = 4;
    else if (a[3]) n = 3;  else if (a[2]) n = 2;
    else if (a[1]) n = 1;  else           n = 0;

    ap_uint<8> exponent = (ap_uint<8>)(127 + n);                    // 127..134
    uint32_t   mantissa = ((uint32_t)a << (23 - n)) & 0x7FFFFF;     // drop leading 1, exact

    conv.i = ((uint32_t)(sign ? 1u : 0u) << 31) |
             ((uint32_t)exponent << 23) | mantissa;
    *dout  = conv.f;
}