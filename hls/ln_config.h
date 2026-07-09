#ifndef __LN_CONFIG_H__
#define __LN_CONFIG_H__

#define BATCH 4

// ---------------------------------------------------------
// 1. Model Specification - ViT-Base
// ---------------------------------------------------------
constexpr int   LN_DIM        = 1024;
constexpr int   LN_SEQLEN     = 256;
constexpr int   LN_SEQLEN_PAD = 256;
constexpr float LN_EPS        = 1e-6f;
constexpr float LN_REP_DIM    = 1.0f / static_cast<float>(LN_DIM);

// ---------------------------------------------------------
// 2. Hardware Interface - ZCU-104 AXI
// ---------------------------------------------------------
constexpr int   BUS_WIDTH      = 128;
constexpr int   ELEM_WIDTH     = 32;
constexpr int   ELEM_WIDTH_INT = 8;
constexpr int   PART_WIDTH     = 16;

// ---------------------------------------------------------
// 3. Parallelism & Loop Control
// ---------------------------------------------------------
constexpr int   PACK_SIZE   = BUS_WIDTH  / ELEM_WIDTH; // 4
constexpr int   NUM_VECTORS = PART_WIDTH / PACK_SIZE;  // 4
constexpr int   NUM_PARTS   = LN_DIM     / PART_WIDTH; // 48
constexpr int   NUM_PACKS   = LN_DIM     / PACK_SIZE;  // 192
constexpr int   UNIT_ITER   = LN_SEQLEN * NUM_PARTS * NUM_VECTORS;

constexpr int   PACK_SIZE_INT   = BUS_WIDTH / ELEM_WIDTH_INT;
constexpr int   NUM_VECTORS_INT = PART_WIDTH / PACK_SIZE_INT;

constexpr int Q_MIN_INT8  = -128;
constexpr int Q_MAX_INT8  =  127;
constexpr int Q_MIN_UINT8 =    0;
constexpr int Q_MAX_UINT8 =  255;

#endif // __LN_CONFIG_H__