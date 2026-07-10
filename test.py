#!/usr/bin/env python3
"""test.py — 단일 TPU_PROCESSOR가 N=768 출력을 한 번에 지원하는지 검증.

배경
-----
TPU_PROJLinear의 BRAM 직결(row-partition) 방식은 하나의 TPU_PROCESSOR가
출력 폭 N=768을 한 번에 계산할 수 있어야 성립한다. 그러나 현재 production
경로(QKV / 기존 PROJ)는 항상 출력 채널을 192씩 4분할해서만 돌려봤기 때문에
N=768 단일 실행이 실제로 동작하는지는 미검증 상태다.

검증 아이디어 (하드웨어를 스스로의 oracle로 사용)
-------------------------------------------------
동일한 활성값 X, 동일한 weight/파라미터에 대해
  (A) 검증된 192폭 경로를 4회 실행해 이어붙인 결과            → golden
  (B) 768폭을 한 번에 실행한 결과                              → test
를 비교한다. N=768이 정상 지원되면 두 결과는 **완전히 동일**해야 한다.
fixed-point 반올림을 재현할 필요 없이, 이미 검증된 192폭 경로를 기준으로
등가성만 확인하므로 신뢰도가 높다.

실패 양상 해석
  - 인터럽트 timeout           → FSM 정지, N=768 미지원 가능성
  - 특정 컬럼부터 값 불일치     → 배열이 일정 폭까지만 처리(잘림/aliasing)
  - 산발적 불일치               → 내부 주소 wrap 등

실행 (ZCU 보드에서, root 권한 필요)
  sudo -E python3 test.py --hw_path ../hardware/FINAL.xsa
"""
import argparse
import time

import numpy as np
import torch
from pynq import Overlay, allocate

from model.tpu import run_sa, preprocess_weight_for_tpu
from hardware.interrupt import Interrupt_write

# ----------------------------------------------------------------------------
# 상수
# ----------------------------------------------------------------------------
K       = 768     # in_features
N_FULL  = 768     # 테스트 대상 출력 폭
N_SPLIT = 192     # production 분할 폭 (검증된 값)
X_ZP    = 128     # 입력 zero-point
OUT_ZP  = 128     # 출력 zero-point (uint8 중앙)


# ----------------------------------------------------------------------------
# 헬퍼
# ----------------------------------------------------------------------------
def alloc_weight(w_out_in):
    """weight[out, in](int8) → (shape_src[K,N] numpy, concat pynq버퍼).

    preprocess_weight_for_tpu가 내부에서 transpose([out,in]→[in,out]=[K,N])하고
    16폭 타일 concat 포맷을 만든다. production _preprocess_weight와 동일 함수라
    하드웨어가 기대하는 포맷이 그대로 보장된다.
    """
    W, W_c = preprocess_weight_for_tpu(torch.from_numpy(w_out_in).to(torch.int8))
    s2_c = allocate(shape=W_c.shape, dtype=np.int8)
    s2_c[:] = W_c
    s2_c.flush()
    return W, s2_c  # W(numpy [K,N])는 run_sa에서 shape(K,N) 추출용으로만 사용


def alloc_param(m_scale, bias):
    """per-channel m_scale/bias를 interleave한 파라미터 버퍼 생성."""
    n = len(m_scale)
    inter = np.empty(n * 2, dtype=np.float32)
    inter[0::2] = m_scale.astype(np.float32)
    inter[1::2] = bias.astype(np.float32)
    p = allocate(shape=(n * 2,), dtype=np.float32)
    p[:] = inter
    p.flush()
    return p


def run_one(tpu, intc, act_addr, W_shape_src, s2_c, dst, param, M, N,
            x_zp=X_ZP, out_zp=OUT_ZP, timeout=5.0):
    """단일 TPU_PROCESSOR 실행 후 [M, N] uint8 결과를 반환."""
    Interrupt_write(intc)          # IER/MER enable
    intc.write(0x0C, 0xF)          # 이전 잔류 인터럽트 클리어 (false early-exit 방지)

    src_act = np.empty((M, W_shape_src.shape[0]), dtype=np.int8)  # [M, K] (shape 용도)
    run_sa(tpu, src_act, act_addr, W_shape_src, s2_c, dst, param, x_zp, out_zp)

    t0 = time.perf_counter()
    while not (intc.read(0x00) & 0x1):        # ISR bit0 = TPU_PROCESSOR_0 완료
        if time.perf_counter() - t0 > timeout:
            intc.write(0x0C, intc.read(0x00))
            raise TimeoutError(
                f"TPU 인터럽트 timeout (N={N}). FSM 정지 → N={N} 미지원 가능성.")
        time.sleep(5e-5)
    intc.write(0x0C, 0x1)                     # bit0 클리어
    try:
        dst.invalidate()                      # 캐시 무효화 (cacheable=False면 no-op)
    except Exception:
        pass
    return np.asarray(dst).reshape(-1)[:M * N].reshape(M, N).copy()


# ----------------------------------------------------------------------------
# 메인
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="단일 TPU_PROCESSOR N=768 지원 검증")
    ap.add_argument("--hw_path", default="../hardware/FINAL.xsa",
                    help="비트스트림 xsa 경로")
    ap.add_argument("--M", type=int, default=208, help="행(토큰) 수 (16으로 정렬됨)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    M = ((args.M + 15) // 16) * 16
    print(f"[setup] M={M}, K={K}, N_full={N_FULL}, seed={args.seed}")
    print(f"[setup] Overlay 로드: {args.hw_path}")
    ol   = Overlay(args.hw_path)
    tpu  = ol.TPU_PROCESSOR_0
    intc = ol.axi_intc_0

    # ── 입력/가중치 생성 ─────────────────────────────────────────────
    X = rng.integers(0, 256, size=(M, K), dtype=np.uint8)          # 활성값
    W = rng.integers(-127, 128, size=(N_FULL, K), dtype=np.int8)   # weight[out, in]

    # 출력이 uint8 전 구간에 고르게 퍼지도록 per-channel scale 설정
    acc     = (X.astype(np.int32) - X_ZP) @ W.T.astype(np.int32)   # [M, N] 누적합
    col_std = acc.std(axis=0) + 1e-6
    m_scale = (100.0 / col_std).astype(np.float32)                 # [N]
    bias    = np.zeros(N_FULL, dtype=np.float32)

    # numpy 참고 모델 (informational: 하드웨어 반올림과 ±1 차이 가능)
    ref = np.clip(np.rint(acc * m_scale + bias) + OUT_ZP, 0, 255).astype(np.uint8)

    # ── 공용 버퍼 ───────────────────────────────────────────────────
    #   act: cacheable=False (production ip_buf_act과 동일, CPU 쓰기가 DRAM 직행)
    #   dst: cacheable=True  (production ip_buf_dst과 동일) + 읽기 전 invalidate로
    #        캐시 무효화 → ACP 배치 없이도 DMA 결과를 coherent하게 읽음
    act = allocate(shape=(M * K,), dtype=np.uint8, cacheable=False)
    act[:] = X.reshape(-1)
    act.flush()
    dst = allocate(shape=(M * N_FULL,), dtype=np.uint8, cacheable=True)

    tmp_bufs = []  # 정리용
    try:
        # ── (A) golden: 검증된 192폭 × 4 ────────────────────────────
        print("[A] 검증된 192폭 경로 4회 실행 …")
        golden = np.empty((M, N_FULL), dtype=np.uint8)
        for j in range(4):
            cols = slice(j * N_SPLIT, (j + 1) * N_SPLIT)
            Wsrc, s2c = alloc_weight(W[cols, :])              # [192, 768]
            param     = alloc_param(m_scale[cols], bias[cols])
            golden[:, cols] = run_one(tpu, intc, act.device_address,
                                      Wsrc, s2c, dst, param, M, N_SPLIT)
            s2c.freebuffer()
            param.freebuffer()

        # ── (B) test: 768폭 × 1 ─────────────────────────────────────
        print("[B] 768폭 단일 실행 …")
        Wsrc, s2c = alloc_weight(W)                          # [768, 768]
        param     = alloc_param(m_scale, bias)
        tmp_bufs += [s2c, param]
        try:
            test = run_one(tpu, intc, act.device_address,
                           Wsrc, s2c, dst, param, M, N_FULL)
        except TimeoutError as e:
            print("❌ FAIL:", e)
            return 1

        # ── 비교 ────────────────────────────────────────────────────
        eq          = (golden == test)
        n_mismatch  = int((~eq).sum())
        col_ok      = eq.all(axis=0)

        print("\n================ 결과 ================")
        print(f"출력 shape       : {test.shape}  (M={M}, N={N_FULL})")
        print(f"불일치 원소 수   : {n_mismatch} / {test.size}")
        if n_mismatch == 0:
            print("✅ PASS: 768폭 단일 == 192폭×4 (완전 일치)")
            print("        → 단일 TPU_PROCESSOR가 N=768을 지원함")
            ret = 0
        else:
            first_bad = int(np.argmax(~col_ok))
            print("❌ FAIL: 768폭 결과가 검증된 192폭 경로와 다름")
            print(f"   최초 불일치 컬럼 : {first_bad}")
            print("   128-컬럼 블록별 일치율:")
            for b in range(0, N_FULL, 128):
                blk = eq[:, b:b + 128]
                print(f"     col[{b:3d}:{b + 128:3d}] {100 * blk.mean():6.2f}%")
            ret = 1

        # informational: numpy 참고 모델과 비교 (192폭 경로도 함께 깨진 경우 감지)
        within1_test   = (np.abs(test.astype(int)   - ref.astype(int)) <= 1).mean()
        within1_golden = (np.abs(golden.astype(int) - ref.astype(int)) <= 1).mean()
        print(f"[참고] numpy 모델 ±1 이내 : 768폭 {100*within1_test:5.2f}% / "
              f"192폭 {100*within1_golden:5.2f}%")
        print("=====================================")
        return ret

    finally:
        for b in tmp_bufs:
            try:
                b.freebuffer()
            except Exception:
                pass
        act.freebuffer()
        dst.freebuffer()


if __name__ == "__main__":
    raise SystemExit(main())
