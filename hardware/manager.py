from pynq import Overlay, MMIO, allocate
import numpy as np
import torch
import struct
from collections import namedtuple

HIDDEN_DIM  = 768
SEQ_LEN     = 197
SEQ_LEN_PAD = 208
PACK        = 16
NPARTS      = HIDDEN_DIM // PACK

class BRAMBuffer:
    def __init__(self, phys_addr, mmio):
        self.device_address = phys_addr
        self._mmio = mmio
        self._arr = np.frombuffer(mmio.array, dtype=np.uint8)

    def __array__(self):
        return self._arr

    def __getitem__(self, idx):
        return self._arr[idx]

    def __setitem__(self, idx, val):
        self._arr[idx] = val

    def reshape(self, *args, **kwargs):
        return self._arr.reshape(*args, **kwargs)

    def flatten(self, *args, **kwargs):
        return self._arr.flatten(*args, **kwargs)

class BufferManager:
    ACP_START    = 0x58000000
    ACP_END      = 0x5A000000
    WEIGHT_START = 0x5A000000

    def __init__(self):
        self._pre_acp_dummies  = []
        self._post_acp_dummies = []

    def reserve_pre_acp(self):
        """CMA를 0x38000000 직전까지 채움"""
        print("Pre-ACP 더미 채우는 중...")
        while True:
            buf = allocate(shape=(1024*4), dtype=np.uint8)
            if buf.device_address >= self.ACP_START:
                break
            self._pre_acp_dummies.append(buf)
        print(f"✅ Pre-ACP 완료: {len(self._pre_acp_dummies)}MB 점유")

    def reserve_post_acp(self):
        """ACP 이후를 0x40000000까지 채움"""
        print("Post-ACP 더미 채우는 중...")
        while True:
            buf = allocate(shape=(1024*4), dtype=np.uint8)
            if buf.device_address >= self.WEIGHT_START:
                break
            self._post_acp_dummies.append(buf)
        print(f"✅ Post-ACP 완료: {len(self._post_acp_dummies)}MB 점유")

    def free_dummies(self):
        """더미 해제 (모든 중요 버퍼 할당 후)"""
        for b in self._pre_acp_dummies + self._post_acp_dummies:
            b.freebuffer()
        self._pre_acp_dummies.clear()
        self._post_acp_dummies.clear()

class FPGAManager:
    def __init__(self, ip_path=None, batch=8):
        print("[FPGA Manager] Initializing Hardware...")
        row_nums = 208
        self.batch_size = batch
        self.num_heads =12
        self.d_k = 64
        self.ip_ol = Overlay(ip_path)
        self.buf_mgr = BufferManager()
        self.buf_mgr.reserve_pre_acp()


        # --------------------------------------------------------------------------------
        #  TPU
        # --------------------------------------------------------------------------------
        ''' ACP Region Memories '''
        import math
        PhysAddr = namedtuple('PhysAddr', ['device_address'])

        self.ip_buf_dst  = [allocate(shape=(row_nums*batch, 1024), dtype=np.uint8 ,cacheable = True) for _ in range(4)]
        self.ip_qbuf_dst = [allocate(shape=(row_nums*batch, 1024), dtype=np.uint8 ,cacheable = True) for _ in range(4)]
        self.ip_kbuf_dst = [allocate(shape=(row_nums*batch, 1024), dtype=np.uint8 ,cacheable = True) for _ in range(4)]
        self.ip_vbuf_dst = [allocate(shape=(row_nums*batch, 1024), dtype=np.uint8 ,cacheable = True) for _ in range(4)]

        BRAM_BASE_ADDR = 0xB000_0000
        BRAM_SIZE      = row_nums * row_nums


        # 각 버퍼를 BRAM 고정 오프셋에 매핑
        self.ip_buf_mm_OCM_list = []
        for i in range(16):
            offset = i * row_nums * row_nums
            phys_addr = BRAM_BASE_ADDR + offset
            mmio = MMIO(phys_addr, BRAM_SIZE)
            self.ip_buf_mm_OCM_list.append(BRAMBuffer(phys_addr, mmio))

        self.ocm_np = [buf.reshape(row_nums, row_nums)
                for buf in self.ip_buf_mm_OCM_list[:16]]

        # torch도 각각 from_numpy로 view 유지
        self.ocm_u8_torch = [torch.from_numpy(v) for v in self.ocm_np]


        self.slot   = (row_nums // 16) * 64  * 16  # 13 * 64 * 16 = 13312
        self.slots  = batch * 12             # 24
        total  = self.slots * self.slot                   # 24 * 13312 = 319488
        # 4KB 정렬 slot_size
        self.slot_aligned = math.ceil(self.slot / 4096) * 4096  # 16384


        # ─ 3) ACP 이후 더미로 채우기 ──────────────────────────
        self.buf_mgr.reserve_post_acp()

        #########DDR REGION
        self.ip_buf_act = allocate(shape=(row_nums*batch*3072), dtype=np.uint8,cacheable = False)

        self.ip_buf_mm_KT_all = allocate(
            shape=(self.slots * self.slot_aligned,), dtype=np.int8
        )
        # KT scratch: [2, 12, 13, 64, 16] contiguous
        self._KT_scratch = np.empty(
            (batch, 12, row_nums//16, 64, 16),
            dtype=np.int8
        )

        self.ip_buf_mm_KT_list = [
            PhysAddr(device_address=
                self.ip_buf_mm_KT_all.device_address + idx * self.slot_aligned)
            for idx in range(self.slots)
        ]

        self.kt_all_np = np.frombuffer(
            self.ip_buf_mm_KT_all, dtype=np.int8
        )
        self.kt_strided = np.lib.stride_tricks.as_strided(
            self.kt_all_np,
            shape=(self.slots, self.slot),           # [24, 13312]
            strides=(self.slot_aligned, 1)      # head간 16384 bytes 점프
        )
        ########################################    QQQ            #######################################
        self.slot_q         = row_nums * 64          # 208 * 64 = 13312
        self.slot_q_aligned = math.ceil(self.slot_q / 4096) * 4096  # 16384

        self.ip_buf_mm_Q_all = allocate(
            shape=(batch * 12 * self.slot_q_aligned,),
            dtype=np.uint8
        )
        self.ip_buf_mm_Q_list = [
            PhysAddr(device_address=
                self.ip_buf_mm_Q_all.device_address + idx * self.slot_q_aligned)
            for idx in range(batch * 12)
        ]

        self._Q_slot         = self.slot_q
        self._Q_slot_aligned = self.slot_q_aligned
        self._Q_slots        = batch * self.num_heads

        q_all_np = np.asarray(self.ip_buf_mm_Q_all)
        self.q_strided = np.lib.stride_tricks.as_strided(
            q_all_np,
            shape=(self._Q_slots, row_nums, self.d_k),  # (24, 208, 64)
            strides=(self.slot_q_aligned, self.d_k, 1)
        )

        ######################################     VVVVVV  ####################################
        self.slot_v         = row_nums * 64          # 208 * 64 = 13312
        self.slot_v_aligned = math.ceil(self.slot_v / 4096) * 4096  # 16384

        self.ip_buf_mm_V_all = allocate(
            shape=(batch * self.num_heads * self.slot_v_aligned,),
            dtype=np.int8
        )
        self._V_scratch = np.empty(
            (batch, self.num_heads, row_nums, 64//16, 16),
            dtype=np.int8
        )
        self.ip_buf_mm_V_list = [
            PhysAddr(device_address=
                self.ip_buf_mm_V_all.device_address + idx * self.slot_v_aligned)
            for idx in range(batch * self.num_heads)
        ]
        self._V_slot         = self.slot_v
        self._V_slot_aligned = self.slot_v_aligned
        self._V_slots        = batch * self.num_heads

        self.v_all_np  = np.asarray(self.ip_buf_mm_V_all).view(np.int8)
        self.v_strided = np.lib.stride_tricks.as_strided(
            self.v_all_np,
            shape=(self.batch_size, self.num_heads, 64//16, row_nums, 16),
            strides=(
                self.num_heads * self._V_slot_aligned,
                self._V_slot_aligned,
                row_nums*16,
                16,
                1
            )
        )

        self.slot_P  = row_nums * row_nums
        self.slots_P = batch * 12

        self.ip_buf_mm_P_list = [
            PhysAddr(...) for i in range(self.slots_P)  # 24개
        ]
        self.ip_buf_mm_P_all = allocate(
            shape=(batch, 12, row_nums, row_nums),
            dtype=np.int8
        )
        self.ip_buf_mm_P_list = [
            PhysAddr(device_address=
                self.ip_buf_mm_P_all.device_address + i * self.slot_P
            )
            for i in range(self.slots_P)  # ← 여기서 slots_P가 제대로 설정됐는지 확인
        ]
        P_all_torch = torch.from_numpy(np.asarray(self.ip_buf_mm_P_list))
        self.P_strided = torch.from_numpy(  np.asarray(self.ip_buf_mm_P_all)).view(torch.qint8)


        self.pv_result_memory = np.empty(
            (batch * 12, 208, 64), dtype=np.uint8)
        self._pv_result_torch = torch.from_numpy(self.pv_result_memory)

        self._pv_result_view = self._pv_result_torch.reshape(batch, 12, 208, 64)

        # Per-thread scratch buffers (사전할당, 평생 재사용)
        self._softmax_scratch_f32 = np.empty((4, 208, 208), dtype=np.float32)
        self._softmax_scratch_u8  = np.empty((4, 208, 208), dtype=np.uint8)
        self._softmax_scratch_f32_torch = torch.from_numpy(self._softmax_scratch_f32)
        self._softmax_scratch_u8_torch  = torch.from_numpy(self._softmax_scratch_u8)

        self._ort_np_buf = np.zeros(
            (4,) + self._softmax_scratch_f32_torch.shape[1:],
            dtype=np.float32
        )

        # --------------------------------------------------------------------------------
        #  LayerNorm
        # --------------------------------------------------------------------------------

        batch = batch

        self.data_a_buf = allocate(shape=(batch*SEQ_LEN_PAD*HIDDEN_DIM,),
                                    dtype=np.uint8, cacheable=False)
        self.data_b_buf = allocate(shape=(batch*SEQ_LEN_PAD*HIDDEN_DIM,),
                                    dtype=np.uint8, cacheable=False)
        self.data_c_buf = allocate(shape=(batch*SEQ_LEN_PAD*HIDDEN_DIM,),
                                    dtype=np.uint8, cacheable=False)
        self.param_buf  = allocate(shape=(HIDDEN_DIM*2,),
                                    dtype=np.float32, cacheable=False)
        self.result_buf = allocate(shape=(batch*SEQ_LEN_PAD*HIDDEN_DIM,),
                                    dtype=np.uint8, cacheable=True)

        # token-major view: [batch, seqlen_pad, hidden] (hidden contiguous, no transpose)
        self.data_a_np = np.asarray(self.data_a_buf).reshape(
            batch, SEQ_LEN_PAD, HIDDEN_DIM)
        self.data_b_np = np.asarray(self.data_b_buf).reshape(
            batch, SEQ_LEN_PAD, HIDDEN_DIM)
        self.data_c_np = np.asarray(self.data_c_buf).reshape(
            batch, SEQ_LEN_PAD, HIDDEN_DIM)
        self.result_np = np.asarray(self.result_buf).reshape(
            batch, SEQ_LEN_PAD, HIDDEN_DIM)

        self.param_buf_np = np.asarray(self.param_buf)

        self.ln_result_np    = np.empty((batch, SEQ_LEN, HIDDEN_DIM), dtype=np.uint8)
        self.ln_result_torch = torch.from_numpy(self.ln_result_np)


        # LayerNorm OCM Configuration
        LN_BRAM_BASE = 0xB000_0000
        LN_BRAM_SIZE = batch * SEQ_LEN_PAD * HIDDEN_DIM

        self.ln_in_bram_mmio = MMIO(LN_BRAM_BASE, LN_BRAM_SIZE)
        self.ln_in_bram_addr = LN_BRAM_BASE
        _ln_bram_u8 = np.frombuffer(self.ln_in_bram_mmio.array, dtype=np.uint8)

        self.ln_in_bram_np    = _ln_bram_u8.reshape(batch, SEQ_LEN_PAD, HIDDEN_DIM)
        self.ln_in_bram_flat  = _ln_bram_u8.reshape(batch * SEQ_LEN_PAD, HIDDEN_DIM)
        self.ln_in_bram_torch = torch.from_numpy(self.ln_in_bram_flat)
        
        self._ln_input_in_bram = False

        ln_ip = self.ip_ol.layernorm_0

        self.ln_addr_a = self.data_a_buf.device_address
        self.ln_addr_b = self.data_b_buf.device_address
        self.ln_addr_c = self.data_c_buf.device_address

        ln_ip.register_map.inp_a  = self.ln_addr_a
        ln_ip.register_map.out_b  = self.result_buf.device_address
        ln_ip.register_map.par_0  = self.param_buf.device_address
        ln_ip.register_map.batch  = batch
        ln_ip.register_map.seqlen = SEQ_LEN
        ln_ip.register_map.dim    = HIDDEN_DIM
        ln_ip.register_map.eps    = struct.unpack('<I', struct.pack('<f', float(1e-6)))[0]

        print("[FPGA Manager] Initialization Complete")

    def free(self):
        """free allocated buffers"""
        pass
