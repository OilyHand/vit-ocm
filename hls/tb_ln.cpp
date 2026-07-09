#include <iostream>
#include <random>
#include <cmath>
#include <vector>
#include <iomanip>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include "ln.h"

static_assert(PART_WIDTH == PACK_SIZE_INT, "packing logic assumes 1 pack == 1 part");

// ============================================================================
// Quantization helpers (asymmetric, uint8, range must include 0.0f)
// ============================================================================
static void calculate_quant_params(const std::vector<float>& src, float& scale, uint8_t& zp) {
    float min_val = src[0];
    float max_val = src[0];
    for (float val : src) {
        if (val < min_val) min_val = val;
        if (val > max_val) max_val = val;
    }
    if (min_val > 0.0f) min_val = 0.0f;
    if (max_val < 0.0f) max_val = 0.0f;

    scale = (max_val - min_val) / 255.0f;
    if (scale == 0.0f) scale = 1.0f;

    int zp_i = static_cast<int>(std::round(-min_val / scale));
    if (zp_i < 0)   zp_i = 0;
    if (zp_i > 255) zp_i = 255;
    zp = static_cast<uint8_t>(zp_i);
}

static uint8_t quantize_value(float val, float scale, uint8_t zp) {
    int q_i = static_cast<int>(std::round(val / scale)) + zp;
    if (q_i < 0)   q_i = 0;
    if (q_i > 255) q_i = 255;
    return static_cast<uint8_t>(q_i);
}

static float dequantize_value(uint8_t q, float scale, uint8_t zp) {
    return static_cast<float>(static_cast<int>(q) - static_cast<int>(zp)) * scale;
}

// ============================================================================
// Blocked memory layout matching inp_a_loader / inp_b_loader / out_*_writer.
//
// The modified kernel uses a TOKEN-MAJOR layout with a padded row stride: each
// loader walks tokens (l) on the outer loop and dim-parts (d) on the inner
// loop over inp[b*(parts*seqlen_pad) + l*parts + d] (see ln.cpp inp_a_loader
// and out_b_writer). So:
//   pack_index(b,t,s) = b*(parts*seqlen_pad) + s*parts + t
//   - t : dim-part index      [0, parts)
//   - s : sequence position   [0, seqlen)   (rows [seqlen, seqlen_pad) are pad)
//   one pack (PACK_SIZE_INT == PART_WIDTH elements) covers dim block t, row s
//
// The earlier kernel this TB was written for used a transposed, unpadded layout
// (b*(parts*seqlen) + t*seqlen + s); both the transpose and the seqlen_pad
// stride changed, so the packing below was updated to match.
// ============================================================================

// Padded sequence length, computed exactly like the kernel:
//   en_padpass = mode[1]  -> seqlen_pad = seqlen        (caller pre-padded)
//   otherwise             -> seqlen_pad = round-up-16(seqlen)
static inline uint32_t compute_seqlen_pad(uint32_t mode, uint32_t seqlen) {
    bool en_padpass = ((mode >> 1) & 0x1) != 0;
    return en_padpass ? seqlen : (((seqlen + 15) / 16) * 16);
}

static inline size_t pack_index(size_t b, size_t t, size_t s, size_t parts, size_t seqlen_pad) {
    return b * (parts * seqlen_pad) + s * parts + t;
}

// Packs a flat [batch][seqlen][dim] uint8 tensor into the HLS AXI memory layout.
// Only the valid rows [0, seqlen) are written; padding rows [seqlen, seqlen_pad)
// are left at their default (0) since the kernel never reads them.
static void pack_input_tensor(
    const std::vector<uint8_t>& flat,
    uint32_t batch, uint32_t seqlen, uint32_t seqlen_pad, uint32_t dim, uint32_t parts,
    std::vector<pack_uint8_t>& packed)
{
    packed.assign((size_t)batch * parts * seqlen_pad, pack_uint8_t());
    for (uint32_t b = 0; b < batch; b++) {
        for (uint32_t t = 0; t < parts; t++) {
            for (uint32_t s = 0; s < seqlen; s++) {
                pack_uint8_t vec;
                for (uint32_t e = 0; e < PACK_SIZE_INT; e++) {
                    uint32_t d = t * PART_WIDTH + e;
                    size_t flat_idx = ((size_t)b * seqlen + s) * dim + d;
                    vec[e] = flat[flat_idx];
                }
                packed[pack_index(b, t, s, parts, seqlen_pad)] = vec;
            }
        }
    }
}

// Unpacks the HLS AXI output memory layout back into a flat [batch][seqlen][dim]
// uint8 tensor. Only the valid rows [0, seqlen) are read back for comparison.
static void unpack_output_tensor(
    const std::vector<pack_uint8_t>& packed,
    uint32_t batch, uint32_t seqlen, uint32_t seqlen_pad, uint32_t dim, uint32_t parts,
    std::vector<uint8_t>& flat)
{
    flat.assign((size_t)batch * seqlen * dim, 0);
    for (uint32_t b = 0; b < batch; b++) {
        for (uint32_t t = 0; t < parts; t++) {
            for (uint32_t s = 0; s < seqlen; s++) {
                pack_uint8_t vec = packed[pack_index(b, t, s, parts, seqlen_pad)];
                for (uint32_t e = 0; e < PACK_SIZE_INT; e++) {
                    uint32_t d = t * PART_WIDTH + e;
                    size_t flat_idx = ((size_t)b * seqlen + s) * dim + d;
                    flat[flat_idx] = vec[e];
                }
            }
        }
    }
}

// Packs weight/bias (length dim each) into the par_0 AXI layout: weights first, then biases.
static void pack_params(
    const std::vector<float>& weight, const std::vector<float>& bias, uint32_t dim,
    std::vector<pack_t>& packed)
{
    uint32_t num_packs = dim / PACK_SIZE;
    packed.assign((size_t)num_packs * 2, pack_t());
    for (uint32_t pk = 0; pk < num_packs; pk++) {
        pack_t w_vec, b_vec;
        for (uint32_t i = 0; i < PACK_SIZE; i++) {
            uint32_t d = pk * PACK_SIZE + i;
            w_vec[i] = weight[d];
            b_vec[i] = bias[d];
        }
        packed[pk]             = w_vec;
        packed[num_packs + pk] = b_vec;
    }
}

// ============================================================================
// Golden Reference: replicates the HLS datapath in the float domain.
//
// The modified kernel exposes two outputs:
//   out_a = residual add result (requantized into scale_c/zp_c domain)
//           -- ONLY produced in residual mode (mode[0]==1); in bypass mode
//              out_a_writer early-returns and nothing is written.
//   out_b = layernorm output    (quantized into scale_o/zp_o domain)
// ============================================================================

// Residual path: requantizes (a - zp_a)*scale_a + (b - zp_b)*scale_b into the
// layernorm's uint8 input domain (scale_c/zp_c), exactly mirroring add().
static void golden_residual_add_row(
    const uint8_t* a_row, const uint8_t* b_row, uint32_t dim,
    float scale_a, float scale_b, float scale_c,
    uint8_t zp_a, uint8_t zp_b, uint8_t zp_c,
    uint8_t* c_row)
{
    float inv_scale_c = 1.0f / scale_c;
    for (uint32_t d = 0; d < dim; d++) {
        float a_f = static_cast<float>(static_cast<int>(a_row[d]) - static_cast<int>(zp_a));
        float b_f = static_cast<float>(static_cast<int>(b_row[d]) - static_cast<int>(zp_b));
        float add_f = a_f * scale_a + b_f * scale_b;
        float q_f = std::round(add_f * inv_scale_c);
        int q_i = static_cast<int>(q_f) + zp_c;
        if (q_i < 0)   q_i = 0;
        if (q_i > 255) q_i = 255;
        c_row[d] = static_cast<uint8_t>(q_i);
    }
}

// One full row (single sequence position) of accumulate -> statistics -> normalize -> out_quantization
static void golden_layernorm_row(
    const uint8_t* in_row, uint32_t dim,
    float scale_in, uint8_t zp_in,
    const std::vector<float>& weight, const std::vector<float>& bias,
    float eps, float scale_o, uint8_t zp_o,
    uint8_t* out_row)
{
    // accumulation (exact int32 arithmetic, matches acc_t)
    int64_t sum_x1 = 0;
    int64_t sum_x2 = 0;
    for (uint32_t d = 0; d < dim; d++) {
        int v = static_cast<int>(in_row[d]) - static_cast<int>(zp_in);
        sum_x1 += v;
        sum_x2 += (int64_t)v * (int64_t)v;
    }

    // statistics
    float dim_f = static_cast<float>(dim);
    float inv_dim = 1.0f / dim_f;
    float scale_mean = scale_in * inv_dim;
    float scale_var  = scale_in * scale_in * inv_dim;

    float sum_x1_f = static_cast<float>(static_cast<int32_t>(sum_x1));
    float sum_x2_f = static_cast<float>(static_cast<int32_t>(sum_x2));

    float mean  = sum_x1_f * scale_mean;
    float mean2 = mean * mean;
    float var   = sum_x2_f * scale_var - mean2;
    float rstd  = 1.0f / std::sqrt(var + eps);

    // normalization + out_quantization
    float inv_scale_o = 1.0f / scale_o;
    for (uint32_t d = 0; d < dim; d++) {
        int v = static_cast<int>(in_row[d]) - static_cast<int>(zp_in);
        float x = static_cast<float>(v) * scale_in;
        float norm = (x - mean) * rstd * weight[d] + bias[d];

        float q_f = std::round(norm * inv_scale_o);
        int q_i = static_cast<int>(q_f) + zp_o;
        if (q_i < 0)   q_i = 0;
        if (q_i > 255) q_i = 255;
        out_row[d] = static_cast<uint8_t>(q_i);
    }
}

// Golden model.
//   ln_flat  : layernorm output (compare against out_b, always valid)
//   add_flat : residual add result (compare against out_a, valid only when
//              en_residual == true)
static void golden_layernorm(
    bool en_residual,
    const std::vector<uint8_t>& a_q, const std::vector<uint8_t>& b_q,
    uint32_t batch, uint32_t seqlen, uint32_t dim,
    float scale_a, float scale_b, float scale_c,
    uint8_t zp_a, uint8_t zp_b, uint8_t zp_c,
    const std::vector<float>& weight, const std::vector<float>& bias,
    float eps, float scale_o, uint8_t zp_o,
    std::vector<uint8_t>& ln_flat,
    std::vector<uint8_t>& add_flat)
{
    ln_flat.assign((size_t)batch * seqlen * dim, 0);
    add_flat.assign((size_t)batch * seqlen * dim, 0);
    std::vector<uint8_t> in_row(dim);

    for (uint32_t b = 0; b < batch; b++) {
        for (uint32_t s = 0; s < seqlen; s++) {
            size_t row_off = ((size_t)b * seqlen + s) * dim;

            if (en_residual) {
                golden_residual_add_row(
                    &a_q[row_off], &b_q[row_off], dim,
                    scale_a, scale_b, scale_c, zp_a, zp_b, zp_c,
                    in_row.data());
                std::memcpy(&add_flat[row_off], in_row.data(), dim);
                golden_layernorm_row(
                    in_row.data(), dim, scale_c, zp_c,
                    weight, bias, eps, scale_o, zp_o,
                    &ln_flat[row_off]);
            } else {
                golden_layernorm_row(
                    &a_q[row_off], dim, scale_a, zp_a,
                    weight, bias, eps, scale_o, zp_o,
                    &ln_flat[row_off]);
            }
        }
    }
}

// ============================================================================
// Validation
// ============================================================================
struct ValidationStats {
    int    max_int_diff;
    double rmse;
    double sqnr;
    bool   pass;
};

static ValidationStats validate(
    const std::vector<uint8_t>& hls_res, const std::vector<uint8_t>& ref_res,
    float scale, uint8_t zp, int tolerance_lsb)
{
    ValidationStats st{0, 0.0, 0.0, true};
    double err_energy = 0.0, sig_energy = 0.0;

    for (size_t i = 0; i < hls_res.size(); i++) {
        int diff = std::abs(static_cast<int>(hls_res[i]) - static_cast<int>(ref_res[i]));
        if (diff > st.max_int_diff) st.max_int_diff = diff;

        float hls_f = dequantize_value(hls_res[i], scale, zp);
        float ref_f = dequantize_value(ref_res[i], scale, zp);
        double d = hls_f - ref_f;
        err_energy += d * d;
        sig_energy += static_cast<double>(ref_f) * ref_f;
    }

    st.rmse = std::sqrt(err_energy / hls_res.size());
    st.sqnr = 10.0 * std::log10(sig_energy / (err_energy + 1e-15));
    st.pass = (st.max_int_diff <= tolerance_lsb);
    return st;
}

static void print_stats(const char* label, const ValidationStats& st) {
    std::cout << "  -> " << label << " validation:\n";
    std::cout << "     - max |int| diff  : " << st.max_int_diff << " LSB\n";
    std::cout << "     - RMSE            : " << std::fixed << std::setprecision(6) << st.rmse << "\n";
    std::cout << "     - SQNR            : " << std::fixed << std::setprecision(4) << st.sqnr << " dB\n";
    std::cout << (st.pass ? "     - [PASS]\n" : "     - [FAIL]\n") << std::endl;
}

// ============================================================================
// Main
// ============================================================================
int main() {
    std::cout << "==================================================\n";
    std::cout << " LayerNorm + Residual-Add HLS C-Simulation\n";
    std::cout << " (out_a = residual sum [residual mode only], out_b = LN)\n";
    std::cout << "==================================================\n" << std::endl;

    const uint32_t batch  = BATCH;
    const uint32_t seqlen = LN_SEQLEN;
    const uint32_t dim    = LN_DIM;
    const uint32_t parts  = LN_DIM / PART_WIDTH;
    const float    eps    = LN_EPS;

    const size_t total_valid = (size_t)batch * seqlen * dim;

    // mode bit layout (see ln.cpp): bit0 = en_residual, bit1 = en_padpass.
    const ap_uint<32> MODE_BYPASS   = 0;  // en_residual=0, en_padpass=0
    const ap_uint<32> MODE_RESIDUAL = 1;  // en_residual=1, en_padpass=0

    // Both test modes leave en_padpass=0, so the kernel pads internally to a
    // multiple of 16 rows; this stride drives the packed-buffer allocation.
    const uint32_t seqlen_pad = compute_seqlen_pad((uint32_t)MODE_BYPASS, seqlen);

    // ------------------------------------------------------------------
    // (1) Tensor Generation: float tensors uniform in [-10, 10]
    // ------------------------------------------------------------------
    std::mt19937 gen(12345);
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
    std::uniform_real_distribution<float> wdist(0.8f, 1.2f);
    std::uniform_real_distribution<float> bdist(-0.2f, 0.2f);

    std::vector<float> tensor_a(total_valid), tensor_b(total_valid);
    for (size_t i = 0; i < total_valid; i++) {
        tensor_a[i] = dist(gen);
        tensor_b[i] = dist(gen);
    }

    std::vector<float> weight(dim), bias(dim);
    for (uint32_t d = 0; d < dim; d++) {
        weight[d] = wdist(gen);
        bias[d]   = bdist(gen);
    }

    // ------------------------------------------------------------------
    // (2) Tensor Quantization
    // ------------------------------------------------------------------
    float scale_a, scale_b;
    uint8_t zp_a, zp_b;
    calculate_quant_params(tensor_a, scale_a, zp_a);
    calculate_quant_params(tensor_b, scale_b, zp_b);

    // Residual Connection input quant params (layernorm's post-add input):
    // fixed calibration constant. a,b in [-10,10] => a+b in [-20,20], so this
    // range is chosen to just cover the true sum range without clipping.
    const float   scale_c = 40.0f / 255.0f;
    const uint8_t zp_c    = 128;

    // Output activation quant params: fixed calibration constant (normalized +
    // affine output stays well within +/-8 for this weight/bias range).
    const float   scale_o = 16.0f / 255.0f;
    const uint8_t zp_o    = 128;

    std::vector<uint8_t> a_q(total_valid), b_q(total_valid);
    for (size_t i = 0; i < total_valid; i++) {
        a_q[i] = quantize_value(tensor_a[i], scale_a, zp_a);
        b_q[i] = quantize_value(tensor_b[i], scale_b, zp_b);
    }

    std::cout << "Shape: batch=" << batch << " seqlen=" << seqlen
              << " (pad=" << seqlen_pad << ") dim=" << dim
              << " parts=" << parts << " | layout=token-major\n";
    std::cout << "Quant params: scale_a=" << scale_a << " zp_a=" << (int)zp_a
              << " | scale_b=" << scale_b << " zp_b=" << (int)zp_b
              << " | scale_c=" << scale_c << " zp_c=" << (int)zp_c
              << " | scale_o=" << scale_o << " zp_o=" << (int)zp_o << "\n" << std::endl;

    // ------------------------------------------------------------------
    // (3) Run HLS Design
    // ------------------------------------------------------------------
    std::vector<pack_uint8_t> inp_a, inp_b;
    pack_input_tensor(a_q, batch, seqlen, seqlen_pad, dim, parts, inp_a);
    pack_input_tensor(b_q, batch, seqlen, seqlen_pad, dim, parts, inp_b);

    std::vector<pack_t> par_0;
    pack_params(weight, bias, dim, par_0);

    const size_t packed_elems = (size_t)batch * parts * seqlen_pad;

    // out_a = residual add result, out_b = layernorm output.
    // out_a is only written in residual mode; the bypass-call out_a buffer is a
    // required argument but the kernel leaves it untouched, so it is not checked.
    std::vector<pack_uint8_t> out_a_bypass_pack(packed_elems);   // unused (kernel skips it)
    std::vector<pack_uint8_t> out_b_bypass_pack(packed_elems);
    std::vector<pack_uint8_t> out_a_residual_pack(packed_elems);
    std::vector<pack_uint8_t> out_b_residual_pack(packed_elems);

    // The kernel reloads weight/bias from par_0 on every call (LOAD_PARAM runs
    // unconditionally), so no separate parameter-load mode is required.

    // 1. Bypass Layernorm Mode (mode=0): tensor_a only
    layernorm(
        MODE_BYPASS, batch, seqlen, dim, eps,
        scale_a, scale_b, scale_c, scale_o,
        zp_a, zp_b, zp_c, zp_o,
        inp_a.data(), inp_b.data(), par_0.data(),
        out_a_bypass_pack.data(), out_b_bypass_pack.data());

    // 2. Residual Connection Mode (mode=1): tensor_a + tensor_b
    layernorm(
        MODE_RESIDUAL, batch, seqlen, dim, eps,
        scale_a, scale_b, scale_c, scale_o,
        zp_a, zp_b, zp_c, zp_o,
        inp_a.data(), inp_b.data(), par_0.data(),
        out_a_residual_pack.data(), out_b_residual_pack.data());

    std::vector<uint8_t> out_b_bypass_hls;
    std::vector<uint8_t> out_a_residual_hls, out_b_residual_hls;
    unpack_output_tensor(out_b_bypass_pack,   batch, seqlen, seqlen_pad, dim, parts, out_b_bypass_hls);
    unpack_output_tensor(out_a_residual_pack, batch, seqlen, seqlen_pad, dim, parts, out_a_residual_hls);
    unpack_output_tensor(out_b_residual_pack, batch, seqlen, seqlen_pad, dim, parts, out_b_residual_hls);

    // ------------------------------------------------------------------
    // (4) Golden Reference (built from the same quantized tensors used by HLS)
    // ------------------------------------------------------------------
    std::vector<uint8_t> out_b_bypass_ref,   dummy_add_bypass;
    std::vector<uint8_t> out_b_residual_ref, out_a_residual_ref;

    golden_layernorm(
        /*en_residual*/ false, a_q, b_q, batch, seqlen, dim,
        scale_a, scale_b, scale_c, zp_a, zp_b, zp_c,
        weight, bias, eps, scale_o, zp_o,
        out_b_bypass_ref, dummy_add_bypass);

    golden_layernorm(
        /*en_residual*/ true, a_q, b_q, batch, seqlen, dim,
        scale_a, scale_b, scale_c, zp_a, zp_b, zp_c,
        weight, bias, eps, scale_o, zp_o,
        out_b_residual_ref, out_a_residual_ref);

    // ------------------------------------------------------------------
    // (5) Validation
    // ------------------------------------------------------------------
    // out_b (layernorm): tolerance 1 LSB accounts for hls::rsqrt / hls::recip
    // hardware approximation versus the plain software sqrt/divide reference.
    // out_a (residual sum): tolerance 1 LSB accounts for the fixed-point
    // requantization (mscale_t/qint_t) versus the float round() reference.
    const int tol_ln  = 1;
    const int tol_add = 1;

    std::cout << ">>> Mode 0: Bypass Layernorm\n";
    ValidationStats st_bypass_b = validate(out_b_bypass_hls, out_b_bypass_ref, scale_o, zp_o, tol_ln);
    print_stats("Bypass out_b (layernorm)", st_bypass_b);

    std::cout << ">>> Mode 1: Residual Connection + Layernorm\n";
    ValidationStats st_res_a = validate(out_a_residual_hls, out_a_residual_ref, scale_c, zp_c, tol_add);
    print_stats("Residual out_a (add result)", st_res_a);
    ValidationStats st_res_b = validate(out_b_residual_hls, out_b_residual_ref, scale_o, zp_o, tol_ln);
    print_stats("Residual out_b (layernorm)", st_res_b);

    bool all_pass = st_bypass_b.pass && st_res_a.pass && st_res_b.pass;
    std::cout << "==================================================\n";
    std::cout << (all_pass ? " ALL TESTS PASSED\n" : " TESTS FAILED\n");
    std::cout << "==================================================\n";

    return all_pass ? 0 : -1;
}
