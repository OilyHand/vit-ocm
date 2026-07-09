#include "ln.h"
#include <cstdint>

void inp_a_loader(
    uint32_t b,
    ap_uint<32> seqlen,
    ap_uint<32> seqlen_pad,
    ap_uint<32> parts,
    pack_uint8_t* inp,
    stream<pack_uint8_t>& stream_inp)
{
    uint32_t batch_offset = b * (parts * seqlen_pad);

    LOOP_TOKEN: for(int l = 0; l < seqlen_pad; l++){
    #pragma HLS loop_flatten
    #pragma HLS loop_tripcount min=LN_SEQLEN_PAD \
                               max=LN_SEQLEN_PAD \
                               avg=LN_SEQLEN_PAD
        uint32_t token_offset = batch_offset + l * parts;

        LOOP_DIM: for(int d = 0; d < parts; d++){
        #pragma HLS loop_flatten
        #pragma HLS loop_tripcount min=NUM_PARTS \
                                   max=NUM_PARTS \
                                   avg=NUM_PARTS
        #pragma HLS PIPELINE II=1
            pack_uint8_t inp_vec = inp[token_offset + d];

            if (l < seqlen){
                stream_inp.write(inp_vec);
            }

        } /* LOOP_DIM */
    } /* LOOP_TOKEN */
}

void inp_b_loader(
    bool en_residual,
    uint32_t b,
    ap_uint<32> seqlen,
    ap_uint<32> seqlen_pad,
    ap_uint<32> parts,
    pack_uint8_t* inp,
    stream<pack_uint8_t>& stream_inp)
{
    if (!en_residual) return;

    uint32_t batch_offset = b * (parts * seqlen_pad);

    LOOP_TOKEN: for(int l = 0; l < seqlen_pad; l++){
    #pragma HLS loop_flatten
    #pragma HLS loop_tripcount min=LN_SEQLEN_PAD \
                               max=LN_SEQLEN_PAD \
                               avg=LN_SEQLEN_PAD
        uint32_t token_offset = batch_offset + l * parts;

        LOOP_DIM: for(int d = 0; d < parts; d++){
        #pragma HLS loop_flatten
        #pragma HLS loop_tripcount min=NUM_PARTS \
                                   max=NUM_PARTS \
                                   avg=NUM_PARTS
        #pragma HLS PIPELINE II=1
            pack_uint8_t inp_vec = inp[token_offset + d];

            if (l < seqlen){
                stream_inp.write(inp_vec);
            }

        } /* LOOP_DIM */
    } /* LOOP_TOKEN */
}

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
    stream<pack_uint8_t>& stream_out_a)
{
    mscale_t MA = (mscale_t) (scale_a / scale_c);
    mscale_t MB = (mscale_t) (scale_b / scale_c);
    addacc_t K = -((addacc_t)(int)zp_a*MA + (addacc_t)(int)zp_b*MB);


    LOOP_TENSOR: for(int i = 0; i < seqlen; i++){
    #pragma HLS loop_flatten
    #pragma HLS loop_tripcount min=LN_SEQLEN \
                               max=LN_SEQLEN \
                               avg=LN_SEQLEN
        LOOP_ADD: for(int d = 0; d < parts; d++){
        #pragma HLS loop_flatten
        #pragma HLS loop_tripcount min=NUM_PARTS \
                                   max=NUM_PARTS \
                                   avg=NUM_PARTS
        #pragma HLS PIPELINE II=1
            pack_uint8_t inp_a_vec = stream_inp_a.read();
            pack_uint8_t out_vec;

            if(en_residual){
                pack_uint8_t inp_b_vec = stream_inp_b.read();
                DEQUANT: for(int e = 0; e < PACK_SIZE_INT; e++){
                #pragma HLS UNROLL
                    float a_f, b_f;

                    ap_uint<8> QA = inp_a_vec[e];
                    ap_uint<8> QB = inp_b_vec[e];
                    addacc_t Q = QA*MA + QB*MB + K;

                    qint_t R = Q;
                    int q_i = R.to_int() + (int)zp_c;

                    uint8_t q_c = (uint8_t) ((q_i < 0) ? 0 : (q_i > 255) ? 255 : q_i);
                    out_vec[e] = q_c;
                }
            } else {
                out_vec = inp_a_vec;
            } /* endif(en_residual) */

            stream_out_1.write(out_vec);
            stream_out_2.write(out_vec);
            
            if(en_residual) stream_out_a.write(out_vec);
            
        } /* LOOP_ADD */
    } /* LOOP_TENSOR */
}

void accumulation(
    ap_uint<32> seqlen,
    ap_uint<32> parts,
    uint8_t     zp_in,
    stream<pack_uint8_t>& stream_inp,
    stream<acc_t>& stream_sum_x1,
    stream<acc_t>& stream_sum_x2)
{
    LOOP_TOKEN: for(int l = 0; l < seqlen; l++){   
    #pragma HLS loop_tripcount min=LN_SEQLEN \
                               max=LN_SEQLEN \
                               avg=LN_SEQLEN
        acc_t sum_x1 = 0;
        acc_t sum_x2 = 0;

        LOOP_DIM: for(int d = 0; d < parts; d++){
        #pragma HLS loop_tripcount min=NUM_PARTS \
                                   max=NUM_PARTS \
                                   avg=NUM_PARTS
        #pragma HLS PIPELINE II=1

            pack_uint8_t inp_vec = stream_inp.read();

            acc_t local_sum    = 0;
            acc_t local_sum_sq = 0;

            LOOP_VECTOR: for(int k = 0; k < PACK_SIZE_INT; k++){
            #pragma HLS UNROLL
                ap_int<9> inp_int = (ap_int<9>) inp_vec[k] - (ap_int<9>) zp_in;
                local_sum    += (acc_t) inp_int;
                local_sum_sq += (acc_t)(inp_int*inp_int);
            }
            sum_x1 += local_sum;
            sum_x2 += local_sum_sq;

        } /* LOOP_LAYER */
        stream_sum_x1.write(sum_x1);
        stream_sum_x2.write(sum_x2);
        
    } /* LOOP_TENSOR */
}

void statistics(
    ap_uint<32>    seqlen,
    ap_uint<32>    dim,
    float          scale_in,
    float          eps,
    stream<acc_t>& stream_sum_x1,
    stream<acc_t>& stream_sum_x2,
    stream<float>& stream_mean,
    stream<float>& stream_rstd)
{
    float dim_f;
    itofp(false, (uint32_t)dim, &dim_f);

    float inv_dim = hls::recip(dim_f);
    float scale_mean = scale_in * inv_dim;
    float scale_var  = scale_in * scale_in * inv_dim;

    LOOP_STATS: for(int l = 0; l < seqlen; l++){
    #pragma HLS loop_tripcount min=LN_SEQLEN \
                               max=LN_SEQLEN \
                               avg=LN_SEQLEN
    #pragma HLS PIPELINE II=1
        // read accumulated data from stream
        acc_t sum_x1_int = stream_sum_x1.read();
        acc_t sum_x2_int = stream_sum_x2.read();

        float sum_x1_f, sum_x2_f;
        itofp(true, (uint32_t)sum_x1_int, &sum_x1_f);
        itofp(true, (uint32_t)sum_x2_int, &sum_x2_f);

        float mean = sum_x1_f * scale_mean;
        float mean2 = mean * mean;
        float var  = sum_x2_f * scale_var - mean2;
        float rstd = hls::rsqrt(var + eps);

        stream_mean.write(mean);
        stream_rstd.write(rstd);
    }
}

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
    stream<float>& stream_rstd)
{
    LOOP_TOKEN: for(int l = 0; l < seqlen; l++){
    #pragma HLS loop_tripcount min=LN_SEQLEN \
                               max=LN_SEQLEN \
                               avg=LN_SEQLEN
        float mean = stream_mean.read();
        float rstd = stream_rstd.read();

        LOOP_DIM: for(int d = 0; d < parts; d++){
        #pragma HLS loop_tripcount min=NUM_PARTS \
                                   max=NUM_PARTS \
                                   avg=NUM_PARTS
        #pragma HLS PIPELINE II=1

            pack_uint8_t inp_vec = stream_inp.read();
            pack_float_t out_vec;

            LOOP_PACK: for(int p = 0; p < (PACK_SIZE_INT / PACK_SIZE); p++){ 
            #pragma HLS UNROLL

                int weight_idx = d * (PART_WIDTH / PACK_SIZE) + p; 
                pack_t wght_vec = mem_wght[weight_idx];
                pack_t bias_vec = mem_bias[weight_idx];

                LOOP_NORM: for(int i = 0; i < PACK_SIZE; i++){
                    #pragma HLS UNROLL
                    int flat_idx = p * PACK_SIZE + i;

                    uint8_t   inp_uint = inp_vec[flat_idx];
                    ap_int<9> inp_int  = inp_uint - zp_in;
                    
                    float inp_f;
                    itofp_small(inp_int, &inp_f);

                    float x = inp_f * scale_in;

                    out_vec[flat_idx] = (x - mean) * rstd * wght_vec[i] + bias_vec[i];

                } /* LOOP_NORM */
            } /* LOOP_PACK */
            stream_norm.write(out_vec);

        } /* LOOP_DIM */
    } /* LOOP_TOKEN */
}

void out_quantization(
    ap_uint<32>           seqlen,
    ap_uint<32>           parts,
    float                 scale_o,
    uint8_t               zp_o,
    stream<pack_float_t>& stream_norm,
    stream<pack_uint8_t>& stream_out)
{
    float inv_scale = hls::recip(scale_o);

    LOOP_LAYER: for(int l = 0; l < seqlen; l++){
    #pragma HLS loop_tripcount min=LN_SEQLEN_PAD \
                               max=LN_SEQLEN_PAD \
                               avg=LN_SEQLEN_PAD
        LOOP_TENSOR: for(int d = 0; d < parts; d++){
        #pragma HLS loop_tripcount min=NUM_PARTS \
                                   max=NUM_PARTS \
                                   avg=NUM_PARTS
        #pragma HLS PIPELINE II=1

            pack_float_t float_vec = stream_norm.read();
            pack_uint8_t uint8_vec;

            LOOP_QUANT: for(int e = 0; e < PACK_SIZE_INT; e++){
            #pragma HLS UNROLL
                float   q_f = hls::round(float_vec[e] * inv_scale);
                int16_t q_i = (int16_t) q_f + (int16_t) zp_o;
                uint8_t q_c = (uint8_t) ((q_i < 0) ? 0 : (q_i > 255) ? 255 : q_i);
                uint8_vec[e] = q_c;
            } /* LOOP_QUANT */
            stream_out.write(uint8_vec);
        } /* LOOP_LAYER */
    } /* LOOP_TENSOR */
}

void out_a_writer(
    bool                  en_residual,
    uint32_t              b,
    ap_uint<32>           seqlen,
    ap_uint<32>           seqlen_pad,
    ap_uint<32>           parts,
    pack_uint8_t          zp_vec,
    stream<pack_uint8_t>& stream_out,
    pack_uint8_t*         out)
{
    if (!en_residual) return;
    
    uint32_t batch_offset = b * (parts * seqlen_pad);

    LOOP_LAYER: for(int l = 0; l < seqlen_pad; l++){
    #pragma HLS loop_flatten 
    #pragma HLS loop_tripcount min=LN_SEQLEN_PAD \
                               max=LN_SEQLEN_PAD \
                               avg=LN_SEQLEN_PAD
        uint32_t token_offset = batch_offset + l * parts;

        LOOP_TENSOR: for(int d = 0; d < parts; d++){
        #pragma HLS loop_tripcount min=NUM_PARTS \
                                   max=NUM_PARTS \
                                   avg=NUM_PARTS
        #pragma HLS PIPELINE II=1
            pack_uint8_t out_vec;
            if (l < seqlen) {
                out_vec = stream_out.read();
            } else {
                // TODO: make zero point vector
                out_vec = zp_vec;
            }
            out[token_offset + d] = out_vec;

        } /* LOOP_LAYER_0 */
    } /* LOOP_TENSOR_0 */
}


void out_b_writer(
    uint32_t              b,
    ap_uint<32>           seqlen,
    ap_uint<32>           seqlen_pad,
    ap_uint<32>           parts,
    pack_uint8_t          zp_vec,
    stream<pack_uint8_t>& stream_out,
    pack_uint8_t*         out)
{
    uint32_t batch_offset = b * (parts * seqlen_pad);

    LOOP_LAYER: for(int l = 0; l < seqlen_pad; l++){
    #pragma HLS loop_flatten 
    #pragma HLS loop_tripcount min=LN_SEQLEN \
                               max=LN_SEQLEN \
                               avg=LN_SEQLEN
        uint32_t token_offset = batch_offset + l * parts;

        LOOP_TENSOR: for(int d = 0; d < parts; d++){
        #pragma HLS loop_tripcount min=NUM_PARTS \
                                   max=NUM_PARTS \
                                   avg=NUM_PARTS
        #pragma HLS PIPELINE II=1
            pack_uint8_t out_vec;
            if (l < seqlen) {
                out_vec = stream_out.read();
            } else {
                // TODO: make zero point vector
                out_vec = zp_vec;
            }
            out[token_offset + d] = out_vec;
            
        } /* LOOP_LAYER_0 */
    } /* LOOP_TENSOR_0 */
}



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
    pack_uint8_t* out_b)
{
    // ============================================================ //
    //   Block-level Control Interfaces                             //
    // ============================================================ //
    #pragma HLS INTERFACE ap_ctrl_hs port=return

    // ============================================================ //
    //   AXI4-lite Slave Interfaces                                 //
    // ============================================================ //
    #pragma HLS INTERFACE s_axilite port=return  bundle=ctrl_bus
    #pragma HLS INTERFACE s_axilite port=mode    bundle=ctrl_bus
    #pragma HLS INTERFACE s_axilite port=batch   bundle=ctrl_bus
    #pragma HLS INTERFACE s_axilite port=seqlen  bundle=ctrl_bus
    #pragma HLS INTERFACE s_axilite port=dim     bundle=ctrl_bus
    #pragma HLS INTERFACE s_axilite port=eps     bundle=ctrl_bus
    #pragma HLS INTERFACE s_axilite port=scale_a bundle=ctrl_bus
    #pragma HLS INTERFACE s_axilite port=zp_a    bundle=ctrl_bus
    #pragma HLS INTERFACE s_axilite port=scale_b bundle=ctrl_bus
    #pragma HLS INTERFACE s_axilite port=zp_b    bundle=ctrl_bus
    #pragma HLS INTERFACE s_axilite port=scale_c bundle=ctrl_bus
    #pragma HLS INTERFACE s_axilite port=zp_c    bundle=ctrl_bus
    #pragma HLS INTERFACE s_axilite port=scale_o bundle=ctrl_bus
    #pragma HLS INTERFACE s_axilite port=zp_o    bundle=ctrl_bus
    #pragma HLS INTERFACE s_axilite port=inp_a   bundle=ctrl_bus
    #pragma HLS INTERFACE s_axilite port=inp_b   bundle=ctrl_bus
    #pragma HLS INTERFACE s_axilite port=par_0   bundle=ctrl_bus
    #pragma HLS INTERFACE s_axilite port=out_a   bundle=ctrl_bus
    #pragma HLS INTERFACE s_axilite port=out_b   bundle=ctrl_bus

    // ============================================================ //
    //   AXI4 Master Interfaces                                     //
    // ============================================================ //
    #pragma HLS INTERFACE m_axi port=inp_a offset=slave bundle=bus_0 \
        depth=BATCH*NUM_PARTS*LN_SEQLEN_PAD
    #pragma HLS INTERFACE m_axi port=inp_b offset=slave bundle=bus_1 \
        depth=BATCH*NUM_PARTS*LN_SEQLEN_PAD
    #pragma HLS INTERFACE m_axi port=par_0 offset=slave bundle=bus_0 \
        depth=NUM_PACKS*2
    #pragma HLS INTERFACE m_axi port=out_a offset=slave bundle=bus_1 \
        depth=BATCH*NUM_PARTS*LN_SEQLEN_PAD
    #pragma HLS INTERFACE m_axi port=out_b offset=slave bundle=bus_0 \
        depth=BATCH*NUM_PARTS*LN_SEQLEN_PAD

    #pragma HLS stable variable=mode
    #pragma HLS stable variable=batch
    #pragma HLS stable variable=seqlen
    #pragma HLS stable variable=dim
    #pragma HLS stable variable=eps
    #pragma HLS stable variable=scale_a
    #pragma HLS stable variable=scale_b
    #pragma HLS stable variable=scale_c
    #pragma HLS stable variable=scale_o
    #pragma HLS stable variable=zp_a
    #pragma HLS stable variable=zp_b
    #pragma HLS stable variable=zp_c
    #pragma HLS stable variable=zp_o
    
    bool en_residual = (mode[0]==1);
    bool en_padpass  = (mode[1]==1);

    ap_uint<32> seqlen_pad = en_padpass ? (ap_uint<32>) seqlen
                                        : (ap_uint<32>) (((seqlen+15)/16)*16);

    ap_uint<32> parts = dim / 16;
    ap_uint<32> packs = dim / 4;

    float   scale_in = en_residual ? scale_c : scale_a;
    uint8_t zp_in    = en_residual ? zp_c    : zp_a;

    pack_uint8_t zp_c_vec, zp_o_vec;
    LOOP_ZP_INIT: for(int e = 0; e < PACK_SIZE_INT; e++){
    #pragma HLS UNROLL
        zp_c_vec[e] = zp_c;
        zp_o_vec[e] = zp_o;
    }

    // local memory
    static pack_t mem_wght[NUM_PACKS];
    static pack_t mem_bias[NUM_PACKS];
    #pragma HLS BIND_STORAGE variable=mem_wght type=ram_1p impl=bram
    #pragma HLS BIND_STORAGE variable=mem_bias type=ram_1p impl=bram
    #pragma HLS stable variable=mem_wght
    #pragma HLS stable variable=mem_bias

    LOAD_PARAM: for(int i = 0; i < packs * 2; i++){
    #pragma HLS loop_tripcount min=384 max=384 avg=384
    #pragma HLS PIPELINE II=1
        if(i < packs){
            mem_wght[i] = par_0[i];
        } else {
            mem_bias[i - packs] = par_0[i];
        }
    } /* LOAD_PARAM */

    LOOP_BATCH: for(uint32_t b = 0; b < batch; b++){
    #pragma HLS loop_tripcount min=BATCH \
                               max=BATCH \
                               avg=BATCH
    #pragma HLS DATAFLOW

        hls::stream<pack_uint8_t> stream_a("stream_a");
        hls::stream<pack_uint8_t> stream_b("stream_b");
        #pragma HLS STREAM variable=stream_a depth=1001
        #pragma HLS STREAM variable=stream_b depth=65

        hls::stream<pack_uint8_t> stream_in_1("stream_in_1");
        hls::stream<pack_uint8_t> stream_in_2("stream_in_2");
        #pragma HLS STREAM variable=stream_in_1 depth=1001
        #pragma HLS STREAM variable=stream_in_2 depth=8513
        #pragma HLS BIND_STORAGE variable=stream_in_2 type=fifo impl=uram

        hls::stream<acc_t> stream_sum_x1("stream_sum_x1");
        hls::stream<acc_t> stream_sum_x2("stream_sum_x2");
        #pragma HLS STREAM variable=stream_sum_x1 depth=2
        #pragma HLS STREAM variable=stream_sum_x2 depth=2

        hls::stream<float> stream_mean("stream_mean");
        hls::stream<float> stream_rstd("stream_rstd");
        #pragma HLS STREAM variable=stream_mean depth=123
        #pragma HLS STREAM variable=stream_rstd depth=123
        
        hls::stream<pack_float_t> stream_norm ("stream_norm");
        hls::stream<pack_uint8_t> stream_out_a("stream_out_a");
        hls::stream<pack_uint8_t> stream_out_b("stream_out_b");
        #pragma HLS STREAM variable=stream_norm depth=3
        #pragma HLS STREAM variable=stream_out_a depth=1001
        #pragma HLS STREAM variable=stream_out_b depth=21

        inp_a_loader(
            b,
            seqlen,
            seqlen_pad,
            parts,
            inp_a,
            stream_a
        );

        inp_b_loader(
            en_residual,
            b,
            seqlen,
            seqlen_pad,
            parts,
            inp_b,
            stream_b
        );

        add(
            en_residual,
            seqlen,
            parts,
            scale_a,
            scale_b,
            scale_c,
            zp_a,
            zp_b,
            zp_c,
            stream_a,
            stream_b,
            stream_in_1,
            stream_in_2,
            stream_out_a
        );

        accumulation(
            seqlen,
            parts,
            zp_in,
            stream_in_1,
            stream_sum_x1,
            stream_sum_x2
        );

        statistics(
            seqlen,
            dim,
            scale_in,
            eps,
            stream_sum_x1,
            stream_sum_x2,
            stream_mean,
            stream_rstd
        );

        normalization(
            seqlen,
            parts,
            dim,
            scale_in,
            zp_in,
            mem_wght,
            mem_bias,
            stream_in_2,
            stream_norm,
            stream_mean,
            stream_rstd
        );

        out_quantization(
            seqlen,
            parts,
            scale_o,
            zp_o,
            stream_norm,
            stream_out_b
        );

        out_a_writer(
            en_residual,
            b,
            seqlen,
            seqlen_pad,
            parts,
            zp_c_vec,
            stream_out_a,
            out_a
        );

        out_b_writer(
            b,
            seqlen,
            seqlen_pad,
            parts,
            zp_o_vec,
            stream_out_b,
            out_b
        );
        
    } /* LOOP_BATCH */
}
