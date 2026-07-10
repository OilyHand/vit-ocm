from pynq import allocate
import numpy as np
import torch
import torch.nn as nn
import time, struct, threading, asyncio, ctypes, math
from concurrent.futures import ThreadPoolExecutor
from collections import namedtuple

from hardware.interrupt import Interrupt_write, interrupt_monitor

# run_sa()는 목적지로 `.device_address` 속성만 참조하므로,
# BRAM 물리주소를 그대로 목적지로 넘기기 위한 경량 래퍼.
PhysAddr = namedtuple('PhysAddr', ['device_address'])

def preprocess_weight_for_tpu(weight_tensor, TRANS = True):
    if hasattr(weight_tensor, 'detach'):
        w_np = weight_tensor.T.detach().cpu().numpy() if TRANS else weight_tensor.detach().cpu().numpy()
    else:
        w_np = weight_tensor.T if TRANS else weight_tensor

    W = w_np
    remainder = w_np.shape[1] % 16
    if remainder != 0:
        padding_size = 16 - remainder
        W = np.pad(w_np, ((0, 0), (0, padding_size)), mode='constant', constant_values=0)

    rows, cols = w_np.shape
    W_concat = w_np.reshape(rows, cols // 16, 16) .transpose(1, 0, 2).reshape(-1, 16)

    return w_np, W_concat

def pack_cont1(relu: int, weight_width: int, weight_height: int) -> int:
    relu = 1 if relu else 0
    return ((relu & 1) << 31) | ((weight_width & 0x7FFF) << 16) | (weight_height & 0xFFFF)

def pack_cont2(relu: int, act_height: int) -> int:
    relu = 1 if relu else 0
    return ((relu & 1) << 31) | (act_height & 0xFFFF)

def run_sa(
    tpu_node,
    src_act,
    src1_1,
    src1_2,
    src1_2_CONCAT,
    dst1,
    src3_param,
    x_zp: int = 128,
    out_zp: int = 0,
    relu: int = 0,
    sa_start_val: int = 0x80000000,
    timeout_s: float = 2.0,
    do_flush: bool = True,
    do_invalidate: bool = True,
    poll: bool = True,
    LITE = False
):
    CSRA_CONTROL   = 0x00
    SA_SOURCE1     = 0x04
    SA_SOURCE2     = 0x08
    SA_CONT1       = 0x0C
    SA_CONT2       = 0x10
    SA_DESTINATION = 0x14
    SA_Parameter1  = 0x18
    SA_Parameter2  = 0x20

    M, K1 = src_act.shape
    if isinstance(src1_2, tuple):
        K2, N = src1_2
    else:
        K2, N = src1_2.shape

    if K1 != K2:
        raise ValueError(
            f"Shape mismatch: src1_1 is (M,K)=({M},{K1}) "
            f"but src1_2 is (K,N)=({K2},{N})"
        )
    K = K1
    src1_1_phys = int(src1_1)
    src1_2_phys = int(src1_2_CONCAT.device_address)
    dst1_phys = int(dst1.device_address)
    src3_phys     = int(src3_param.device_address)


    tpu_node.write(CSRA_CONTROL,sa_start_val)
    tpu_node.write(SA_DESTINATION, dst1_phys)
    tpu_node.write(SA_SOURCE1, src1_1_phys)
    tpu_node.write(SA_SOURCE2, src1_2_phys)
    if LITE == False:
        tpu_node.write(SA_CONT1, pack_cont1(relu, N, K))
        tpu_node.write(SA_CONT2, pack_cont2(relu, M))
        tpu_node.write(SA_Parameter2,  out_zp <<8 | x_zp & 0xFF)
    tpu_node.write(SA_Parameter1,  src3_phys)
    tpu_node.write(CSRA_CONTROL, sa_start_val | 0x1)


    return {
        "M":           M,
        "K":           K,
        "N":           N,
        "src1_1_phys": src1_1_phys,
        "src1_2_phys": src1_2_phys,
        "dst1_phys":   dst1_phys,
        "src3_phys":   src3_phys,
        "x_zp":        x_zp,
    }



def run_softmax(
    softmax_node,
    dst,
    src,
    height: int,
    width:  int,
    scale:  int  = 0x3F800000,
    softmax_scale: int = 0x3E000000,
    start_val: int = 0x80000003,
    poll:  bool  = True,
):
    CSRA_CONTROL = 0x40
    CSRA_SCALE1  = 0x54
    CSRA_SCALE2  = 0x58
    CSRA_RDADDR  = 0x5c
    CSRA_WRADDR  = 0x60
    CSRA_MATRIX  = 0x50

    rd_phys = int(src.device_address)
    wr_phys = int(dst.device_address)

    # 1. CONTROL
    softmax_node.write(CSRA_CONTROL, start_val)
    # 2. SCALE
    softmax_node.write(CSRA_SCALE1,   scale & 0xFFFF_FFFF)
    softmax_node.write(CSRA_SCALE2,   softmax_scale & 0xFFFF_FFFF)
    # 3. RDADDR
    softmax_node.write(CSRA_RDADDR,  rd_phys)
    # 4. WRADDR
    softmax_node.write(CSRA_WRADDR,  wr_phys)
    # 5. MATRIX (GO → FSM 시작)
    matrix_val = (
        (1                   << 31) |
        ((height & 0xFFF)    << 16) |
        ( width  & 0xFFFF)
    )
    softmax_node.write(CSRA_MATRIX, matrix_val)

    return {"height": height, "width": width,
            "scale": scale, "rd_phys": rd_phys, "wr_phys": wr_phys}


def start_irq_loop(loop):
    asyncio.set_event_loop(loop)
    loop.run_forever()



class TPUPatchEmbedding(nn.Module):
    def __init__(self, name, x_scale, weight_tensor, bias_tensor, out_scale, out_zp, hw):
        super().__init__()
        self.hw        = hw
        self.name      = name
        self.out_scale = float(out_scale)
        self.out_zp    = out_zp

        self.P     = 16
        self.H     = 224
        self.W_img = 224
        self.C     = 3
        self.N     = (self.H // self.P) * (self.W_img // self.P)
        self.N_pad = (self.N + 7) // 8 * 8
        self.K     = self.C * self.P * self.P
        self.M     = self.N * self.hw.batch_size

        self.INTERRUPT1 = hw.ip_ol.axi_intc_0
        self.weight_ori = weight_tensor
        self.weight = weight_tensor.int_repr().detach().cpu().numpy().reshape(768, -1)
        if not hasattr(hw, 'irq_loop'):
            new_loop = asyncio.new_event_loop()
            hw.irq_loop = new_loop
            t = threading.Thread(target=start_irq_loop, args=(new_loop,), daemon=True)
            t.start()
            print("🌐 [INFO] Shared IRQ loop started.")

        # ── Weight 추출 (TPULinear와 동일) ─────────────────────
        if hasattr(self.weight_ori, 'int_repr'):
            w_np = self.weight_ori.int_repr().detach().cpu().numpy()
            self.w_scale      = self.weight_ori.q_per_channel_scales()
            self.w_zero_point = self.weight_ori.q_per_channel_zero_points()
        elif hasattr(self.weight_ori, 'detach'):
            w_np = self.weight_ori.detach().cpu().numpy()
            self.w_scale      = 1.0
            self.w_zero_point = 0
        else:
            w_np = self.weight_ori
            self.w_scale      = 1.0
            self.w_zero_point = 0

        # Conv → GEMM reshape: [768, 3, 16, 16] → [768, 768]
        w_np = w_np.reshape(w_np.shape[0], -1)

        self.out_C       = w_np.shape[0]   # 768
        self.in_features = w_np.shape[1]   # 768

        # ── m_scale_per_channel 계산 (TPULinear와 동일) ────────
        if hasattr(self.w_scale, 'detach'):
            w_scale_np = self.w_scale.detach().cpu().numpy().astype(np.float32)
        else:
            w_scale_np = np.asarray(self.w_scale, dtype=np.float32)

        if w_scale_np.ndim == 0:
            w_scale_np = np.full(self.out_C, float(w_scale_np), dtype=np.float32)

        x_scale_f   = float(x_scale)
        out_scale_f = float(self.out_scale)
        m_scale_per_channel = (x_scale_f * w_scale_np / out_scale_f).astype(np.float32)
        self.bias_tensor = bias_tensor
        # ── bias 처리 (TPULinear와 동일) ───────────────────────
        if bias_tensor is not None:
            if hasattr(bias_tensor, 'detach'):
                self.bias = (bias_tensor.detach().cpu().numpy().astype(np.float32)
                             / out_scale_f)
            else:
                self.bias = (np.asarray(bias_tensor, dtype=np.float32)
                             / out_scale_f)
        else:
            self.bias = np.zeros(self.out_C, dtype=np.float32)

        # ── 4분할 + param_buf (TPULinear와 동일) ───────────────
        w_slices = np.vsplit(w_np, 4)
        m_slices = np.split(m_scale_per_channel, 4)
        b_slices = np.split(self.bias, 4)

        self.src2_list      = []
        self.src2_c_list    = []
        self.param_buf_list = []

        for w_s, m_s, b_s in zip(w_slices, m_slices, b_slices):
            current_rows = w_s.shape[0]
            remainder = current_rows % 16
            if remainder != 0:
                pad = 16 - remainder
                w_s = np.pad(w_s, ((0, pad), (0, 0)), mode='constant', constant_values=0)
                m_s = np.pad(m_s, (0, pad), mode='constant')
                b_s = np.pad(b_s, (0, pad), mode='constant')
                print(f"Padding added: {current_rows} -> {w_s.shape[0]}")

            w_s_tensor = torch.from_numpy(w_s).to(torch.int8)
            W_proc, W_c = preprocess_weight_for_tpu(w_s_tensor)

            s2_c = allocate(shape=W_c.shape, dtype=np.int8)
            s2_c[:] = W_c
            self.src2_list.append(W_proc.shape)
            self.src2_c_list.append(s2_c)

            num_ch = w_s.shape[0]
            interleaved = np.empty(num_ch * 2, dtype=np.float32)
            interleaved[0::2] = m_s
            interleaved[1::2] = b_s
            param_buf = allocate(shape=(num_ch * 2,), dtype=np.float32)
            param_buf[:] = interleaved
            param_buf.flush()
            self.param_buf_list.append(param_buf)

        self.m_scale_all = np.concatenate([
            np.asarray(pb)[0::2] for pb in self.param_buf_list
        ]).astype(np.float32)                                  # [768]

        self.bias_all = np.concatenate([
            np.asarray(pb)[1::2] for pb in self.param_buf_list
        ]).astype(np.float32)                                  # [768]

        self.weight_T = self.weight.astype(np.int32).T.copy()
        # ── 결과/입력 버퍼 ─────────────────────────────────────
        self.result_buf   = np.empty((self.M, self.out_C), dtype=np.int8)
        self.result_torch = torch.from_numpy(self.result_buf)
        self.patch_buf    = np.zeros((self.M, self.K), dtype=np.uint8)
        self.patches_buf = np.empty(
            (self.hw.batch_size * self.N, self.K), dtype=np.uint8
        )

    @staticmethod
    def _im2col(x_np):
        B, C, H, W = x_np.shape
        P     = 16
        H_out = H // P
        W_out = W // P
        x = x_np.reshape(B, C, H_out, P, W_out, P)   # [B, C, 14, 16, 14, 16]
        x = x.transpose(0, 2, 4, 1, 3, 5)             # [B, 14, 14, C, 16, 16]
        x = x.reshape(B, H_out * W_out, C * P * P)    # [B, 196, 768]
        return np.ascontiguousarray(x)

    def forward(self, x):
        """
        x: quint8 tensor [B, 3, 224, 224]
        return: quint8 tensor [B, 196, 768]  (TPULinear와 동일하게 quint8 반환)
        """
        B     = x.shape[0]
        in_zp = x.q_zero_point()

        # ① int_repr
        start = time.perf_counter()
        x_np = x.int_repr().cpu().numpy()
        print(f'[0] INT_REPR: {(time.perf_counter()-start)*1000:.4f}')

        # ② im2col
        start = time.perf_counter()
        patches = self._im2col(x_np)
        print(f'[1] IM2COL: {(time.perf_counter()-start)*1000:.4f}')

        start = time.perf_counter()
        patches = patches.reshape(B * self.N, self.K)
        print(f'[2] IM2COL RESHAPE: {(time.perf_counter()-start)*1000:.4f}')

        # ③ 버퍼 복사
        start = time.perf_counter()
        self.patch_buf[:self.M, :] = patches
        self.patch_buf[self.M:, :] = 0
        print(f'[3] BUF COPY: {(time.perf_counter()-start)*1000:.4f}')

        # ④ memmove
        start = time.perf_counter()
        ctypes.memmove(
            self.hw.ip_buf_act.ctypes.data,
            self.patch_buf.ctypes.data,
            self.patch_buf.nbytes
        )
        print(f'[4] MEMMOVE: {(time.perf_counter()-start)*1000:.4f}')

        # ⑤ flush
        start = time.perf_counter()
        self.hw.ip_buf_act.flush()
        print(f'[5] FLUSH: {(time.perf_counter()-start)*1000:.4f}')

        # ── ④ 인터럽트 준비 ────────────────────────────────────
        start = time.perf_counter()
        Interrupt_write(self.INTERRUPT1)
        irq_future = asyncio.run_coroutine_threadsafe(
            anext(interrupt_monitor(self.INTERRUPT1, num_events=4)),
            self.hw.irq_loop
        )
        print(f'[6] INTR SET: {(time.perf_counter()-start)*1000:.4f}')

        # ── ⑤ TPU GEMM 실행 ────────────────────────────────────
        start = time.perf_counter()
        for i in range(4):
            tpu_node = getattr(self.hw.ip_ol, f'TPU_PROCESSOR_{i}')
            run_sa(
                tpu_node,
                patches,
                self.hw.ip_buf_act.device_address,
                self.src2_list[i],
                self.src2_c_list[i],
                self.hw.ip_buf_dst[i],
                self.param_buf_list[i],
                int(in_zp),
                int(self.out_zp)
            )

        # ── ⑥ 인터럽트 대기 ────────────────────────────────────
        status = irq_future.result(timeout=5000)

        if status is None:
            read_value = self.INTERRUPT1.read(0x00)
            raise RuntimeError(f"TPU Timeout! status: {hex(read_value)}")
        print(f'[7] RUN TPU: {(time.perf_counter()-start)*1000:.4f}')

        # ── ⑦ 결과 수집 (TPULinear와 동일) ─────────────────────
        start = time.perf_counter()
        col_size = self.out_C // 4
        for i, d in enumerate(self.hw.ip_buf_dst):
            arr = np.asarray(d).ravel()
            arr = arr[:self.M * col_size].reshape(self.M, col_size)
            self.result_buf[:self.M, i*col_size:(i+1)*col_size] = arr[:self.M, :col_size]
        print(f'[8] COLLECT: {(time.perf_counter()-start)*1000:.4f}')

        # ── ⑧ quint8 반환 (TPULinear와 동일) ───────────────────
        start = time.perf_counter()
        res_torch = self.result_torch[:self.M, :self.out_C].reshape(B, self.N, self.out_C)
        out_np = res_torch.numpy().transpose(0, 2, 1).reshape(B, self.out_C, 14, 14)
        print(f'[9] QUANT CONVERT: {(time.perf_counter()-start)*1000:.4f}')

        start = time.perf_counter()
        out_int   = torch.from_numpy(out_np.copy()).to(torch.uint8)
        print(f'[A] OUT_INT: {(time.perf_counter()-start)*1000:.4f}')

        start = time.perf_counter()
        out_quant = torch._make_per_tensor_quantized_tensor(
            out_int,
            scale      = float(self.out_scale),
            zero_point = int(self.out_zp)
            )
        print(f'[B] MAKE QTENSOR: {(time.perf_counter()-start)*1000:.4f}')

        return out_quant   # [B, 196, 768] quint8


class TPUMultiHeadAttention(nn.Module):

    MAX_OUT_FEATURES = 208

    def __init__(self, qkv_module, proj_module,
                 qkv_act_scale,qkv_input_act_zero,proj_act_scale,
                 energy_scale, energy_zero,attention_input_scale, attention_input_zero, attention_output_scale, attention_output_zero, num_heads,  hw):
        super().__init__()
        from pynq import Interrupt
        self.INTERRUPT1 = hw.ip_ol.axi_intc_0
        self.tpu_irq = Interrupt('TPU_PROCESSOR_3/interrupt')
        self.num_heads  = num_heads
        self.qkv        = qkv_module
        self.proj       = proj_module
        self.hw         = hw
        self.original_row_nums = 197
        self.energy_scale = energy_scale
        self.energy_zero = energy_zero
        # __init__에서
        self.d_k = qkv_module._packed_params._weight_bias()[0].shape[0] // 3 // num_heads
        self.q_concat_memory = np.full( (self.hw.batch_size, 208, 768),  self.qkv.zero_point, dtype = np.uint8 )
        self._q_concat_torch = torch.from_numpy(self.q_concat_memory)
        self.k_concat_memory = np.full( (self.hw.batch_size, 208, 768),  self.qkv.zero_point, dtype = np.uint8 )
        self._k_concat_torch = torch.from_numpy(self.k_concat_memory)
        self.v_concat_memory = np.full( (self.hw.batch_size, 208, 768),  self.qkv.zero_point, dtype = np.uint8 )
        self._v_concat_torch = torch.from_numpy(self.v_concat_memory)

        self._head_pool = ThreadPoolExecutor(
            max_workers=3
        )

        self._attn_scale = float(1.0 / math.sqrt(self.d_k))

        if not hasattr(hw, 'irq_loop'):
            new_loop = asyncio.new_event_loop()
            hw.irq_loop = new_loop
            t = threading.Thread(target=start_irq_loop,
                                 args=(new_loop,), daemon=True)
            t.start()

        # QKV preprocess
        k_module, q_module, v_module = self._reorder_qkv_to_kqv(qkv_module)

        self.k_src2_list, self.k_src2_c_list, self.k_param_buf_list = self._preprocess_weight(k_module, qkv_act_scale)
        self.q_src2_list, self.q_src2_c_list, self.q_param_buf_list = self._preprocess_weight(q_module, qkv_act_scale)
        self.v_src2_list, self.v_src2_c_list, self.v_param_buf_list = self._preprocess_weight(v_module, qkv_act_scale)
        self.k_out_features= k_module._packed_params._weight_bias()[0].shape[0]
        self.q_out_features= q_module._packed_params._weight_bias()[0].shape[0]
        self.v_out_features= v_module._packed_params._weight_bias()[0].shape[0]
        self.k_shape = np.empty((self.hw.batch_size,12,64,208),dtype = np.int8)
        # PROJ preprocess
        self.proj_src2_list, self.proj_src2_c_list, self.proj_param_buf_list = self._preprocess_weight(proj_module, proj_act_scale)
        self.proj_out_features = proj_module._packed_params._weight_bias()[0].shape[0]

        # PROJ preprocess (row-partition / BRAM 직결용 풀 웨이트)
        #   기존 열분할(vsplit) 대신, 4개의 TPU가 모두 동일한 전체 weight[768,768]를
        #   사용하고 활성값을 행(토큰)으로 나눠 처리 → 각 TPU가 완전한 768폭 블록을
        #   BRAM에 연속 기록하므로 CPU gather(열 인터리빙)가 불필요해진다.
        self.proj_full_src2, self.proj_full_src2_c, self.proj_full_param = \
            self._preprocess_weight_full(proj_module, proj_act_scale)

        proj_num_rows         = self.hw.batch_size * 208
        self.proj_padded_rows = (proj_num_rows)
        self.proj_padded_input = np.zeros((self.proj_padded_rows, 768), dtype=np.int8)
        self.proj_result_buf  = np.empty((proj_num_rows, self.proj_out_features), dtype=np.uint8)
        self.proj_result_torch = torch.from_numpy(self.proj_result_buf)
        self.proj_col_size    = self.proj_out_features // 4
        self.proj_actual_elements = self.proj_padded_rows * self.proj_src2_list[0].shape[1]
        self.proj_zp_int      = int(self.proj.zero_point)

        self.mha_result_buf = np.empty(
                (208*self.hw.batch_size, 768), dtype=np.uint8
            )
        self.mha_result_torch = torch.from_numpy(self.mha_result_buf)
        self.mha_col_size = 768 // 4



        #Matmul preprocess
        self.combined_scale = float(self.energy_scale) * self._attn_scale
        self.attention_input_scale = attention_input_scale
        self.attention_input_zero = attention_input_zero
        self.attention_output_scale = attention_output_scale
        self.attention_output_zero = attention_output_zero

        self._preprocess_matmul_param(
            qkv_module = qkv_module,
            p_scale    = energy_scale,    # matmul_24 출력 scale
            p_zp       = energy_zero,
            v_scale    = float(qkv_module.scale),  # V scale = QKV scale
            attn_scale = attention_input_scale, # matmul_25 출력 scale
            attn_zp    = attention_input_zero,
            row_nums   = 208
        )

        #softmax_process
        self._softmax_first_run = True
        self.inv_out_scale    = 1.0 /(float(self.attention_input_scale))
        self._valid_mask_scaled = torch.zeros(4, 208, 208)
        self._valid_mask_scaled[:, :, :] = self.inv_out_scale
        self.combined_scale = float(self.energy_scale) * self._attn_scale
        self.p_zp_f         = float(self.energy_zero)

        self.neg_zp = np.uint8((256 - self.qkv.zero_point) & 0xFF)
        self.v3_buf_u8 = np.empty((self.hw.batch_size, 208, 768), dtype=np.uint8)
        self.v3_buf_i8 = self.v3_buf_u8.view(np.int8)
        self.scale_128 = float( 128 * self.attention_input_scale  * self.qkv.scale / self.attention_output_scale)

    def _reorder_qkv_to_kqv(self, qkv_module):
        weight, bias = qkv_module._packed_params._weight_bias()
        w_np     = weight.int_repr().detach().cpu().numpy()
        w_scales = weight.q_per_channel_scales().detach().numpy()
        w_zp     = weight.q_per_channel_zero_points().detach().numpy()
        b_np     = bias.detach().cpu().numpy().astype(np.float32)

        out_features = w_np.shape[0]
        chunk        = out_features // 3

        Q_w = w_np[:chunk, :];   K_w = w_np[chunk:2*chunk, :];   V_w = w_np[2*chunk:, :]
        Q_b = b_np[:chunk];      K_b = b_np[chunk:2*chunk];      V_b = b_np[2*chunk:]
        Q_s = w_scales[:chunk];  K_s = w_scales[chunk:2*chunk];  V_s = w_scales[2*chunk:]
        Q_z = w_zp[:chunk];      K_z = w_zp[chunk:2*chunk];      V_z = w_zp[2*chunk:]

        def make_module(w, b, s, z):
            import copy
            w_tensor = torch._make_per_channel_quantized_tensor(
                torch.from_numpy(w).to(torch.int8),
                torch.from_numpy(s).double(),
                torch.from_numpy(z).int(),
                axis=0
            )
            b_tensor = torch.nn.Parameter(
                torch.from_numpy(b), requires_grad=False
            )
            module = torch.ao.nn.quantized.Linear(
                in_features  = w.shape[1],  # 768
                out_features = w.shape[0],  # 768 (K만)
            )
            module.scale       = qkv_module.scale
            module.zero_point  = qkv_module.zero_point
            module._packed_params._weight_bias = lambda: (w_tensor, b_tensor)

            return module
        k_module = make_module(K_w, K_b, K_s, K_z)
        q_module = make_module(Q_w, Q_b, Q_s, Q_z)
        v_module = make_module(V_w, V_b, V_s, V_z)

        return k_module, q_module, v_module

    def _preprocess_matmul_param(self, qkv_module, p_scale, p_zp, v_scale, attn_scale, attn_zp, row_nums):
        """
        Q@K^T (energy) 와 P@V (attention) 연산을 위한 param_buf 생성

        energy (Q@K^T):
          M_scale = Q_scale * K_scale / P_scale
          bias    = 0

        attention (P@V):
          M_scale = attn_scale * V_scale / out_scale
          bias    = 0
        """
        qkv_scale = float(qkv_module.scale)

        # ── Energy (Q@K^T) param ───────────────────
        energy_M_scale = (qkv_scale * qkv_scale) / float(self.energy_scale)
        interleaved_energy       = np.empty(row_nums * 2, dtype=np.float32)
        interleaved_energy[0::2] = energy_M_scale
        interleaved_energy[1::2] = 0.0

        self.MM_energy_param_buf_list = [
            allocate(shape=(row_nums * 2,), dtype=np.float32)
            for _ in range(4)
        ]

        for i in range(4):
            self.MM_energy_param_buf_list[i][:] = interleaved_energy
        self.MM_energy_param_buf_list[i].flush()

        # ── Attention (P@V) param ──────────────────
        attn_M_scale = (float(self.attention_input_scale) * float(v_scale)) / float(self.attention_output_scale)

        interleaved_attn       = np.empty(row_nums * 2, dtype=np.float32)
        interleaved_attn[0::2] = attn_M_scale
        interleaved_attn[1::2] = 0.0

        self.MM_attn_param_buf_all = allocate(
                shape=(self.hw.batch_size * self.num_heads * row_nums * 2,),
                dtype=np.float32
            )

        self.mm_attn_param_np = np.asarray(self.MM_attn_param_buf_all).reshape(
                self.hw.batch_size * self.num_heads, row_nums * 2
            )
        from collections import namedtuple

        PhysAddr = namedtuple('PhysAddr', ['device_address'])

        self.MM_attn_param_buf_list = [
                PhysAddr(device_address=
                    self.MM_attn_param_buf_all.device_address + idx * row_nums * 2 * 4)  # float32=4bytes
                for idx in range(self.hw.batch_size * self.num_heads)
            ]

        for i in range(self.hw.batch_size*12):
            self.mm_attn_param_np[i] = interleaved_attn #broadcasting
        self.MM_attn_param_buf_all.flush()



        # scale, zp 저장
        self.p_scale   = float(p_scale)
        self.p_zp      = int(p_zp)

    def _preprocess_weight (self, module, act_scale):
        weight, bias = module._packed_params._weight_bias()
        w_np     = weight.int_repr().detach().cpu().numpy()
        w_slices = np.vsplit(w_np, 4)

        m_scale = (act_scale * weight.q_per_channel_scales()
                   / module.scale)
        m_slices = np.split(m_scale.detach().numpy(), 4)

        bias_fused = (bias / module.scale).detach().cpu().numpy()
        b_slices   = np.split(bias_fused, 4)

        src2_list, src2_c_list, param_buf_list = [], [], []
        for w_s, m_s, b_s in zip(w_slices, m_slices, b_slices):
            current_rows = w_s.shape[0]
            remainder    = current_rows % 16
            if remainder != 0:
                padding_size = 16 - remainder
                w_s = np.pad(w_s, ((0, padding_size), (0, 0)),
                             mode='constant', constant_values=0)
                m_s = np.pad(m_s, (0, padding_size), mode='constant')
                b_s = np.pad(b_s, (0, padding_size), mode='constant')

            W, W_c = preprocess_weight_for_tpu(
                torch.from_numpy(w_s).to(torch.int8))

            s2   = allocate(shape=W.shape,   dtype=np.int8)
            s2_c = allocate(shape=W_c.shape, dtype=np.int8)
            s2[:]   = W
            s2_c[:] = W_c
            src2_list.append(s2)
            src2_c_list.append(s2_c)

            num_ch      = w_s.shape[0]
            interleaved = np.empty(num_ch * 2, dtype=np.float32)
            interleaved[0::2] = m_s
            interleaved[1::2] = b_s
            param_buf    = allocate(shape=(num_ch * 2,), dtype=np.float32)
            param_buf[:] = interleaved
            param_buf.flush()
            param_buf_list.append(param_buf)

        return src2_list, src2_c_list, param_buf_list


    def _preprocess_weight_full(self, module, act_scale):
        """_preprocess_weight의 무분할(no vsplit) 버전.

        전체 weight [out=768, in=768]을 그대로 하나의 concat 버퍼로 만든다.
        row-partition 방식에서는 4개의 TPU가 이 동일한 버퍼를 공유하고,
        각자 활성값의 행(토큰) 1/4만 처리하여 N=768 전폭 출력을 낸다.
        """
        weight, bias = module._packed_params._weight_bias()
        w_np = weight.int_repr().detach().cpu().numpy()                 # [768(out), 768(in)]

        m_scale = (act_scale * weight.q_per_channel_scales()
                   / module.scale).detach().numpy()                     # [768]
        bias_fused = (bias / module.scale).detach().cpu().numpy()       # [768]

        # 출력 채널 수(768)는 이미 16의 배수라 padding 불필요
        W, W_c = preprocess_weight_for_tpu(torch.from_numpy(w_np).to(torch.int8))

        s2   = allocate(shape=W.shape,   dtype=np.int8);  s2[:]   = W    # [768, 768]
        s2_c = allocate(shape=W_c.shape, dtype=np.int8);  s2_c[:] = W_c

        num_ch      = w_np.shape[0]                                      # 768
        interleaved = np.empty(num_ch * 2, dtype=np.float32)
        interleaved[0::2] = m_scale
        interleaved[1::2] = bias_fused
        param_buf    = allocate(shape=(num_ch * 2,), dtype=np.float32)
        param_buf[:] = interleaved
        param_buf.flush()

        return s2, s2_c, param_buf


    def TPU_QKVLinear(self, x, mode,q_zero_point, DATA_COPY = False):
        src2_list     = getattr(self, f'{mode}_src2_list')
        src2_c_list   = getattr(self, f'{mode}_src2_c_list')
        dst_list = getattr(self.hw,f'ip_{mode}buf_dst')
        param_buf_list = getattr(self, f'{mode}_param_buf_list')
        out_features  = getattr(self, f'{mode}_out_features')
        concat_memory = getattr(self, f'{mode}_concat_memory')
        concat_memory_torch = getattr(self, f'_{mode}_concat_torch')

        original_shape = x.shape
        x_2d = x.reshape(-1, x.shape[-1])
        num_rows = x_2d.shape[0] #197
        in_features = x_2d.shape[1] #768 or 3072
        padded_rows = (num_rows + 15) // 16 * 16

        if DATA_COPY == True:
            flat_data = x_2d.int_repr().cpu().numpy().flatten()
            num_elements = flat_data.size
            self.hw.ip_buf_act.flat[:num_elements] = flat_data
            current_input = flat_data.reshape(num_rows,in_features)
            pad_amt = padded_rows - current_input.shape[0] # 200 - 197 = 3
            current_input = np.pad(current_input, ((0, pad_amt), (0, 0)), mode='constant', constant_values=0)
        # 인터럽트 감시 시작
        Interrupt_write(self.INTERRUPT1)
        for i in range(4):
            tpu_node = getattr(self.hw.ip_ol, f'TPU_PROCESSOR_{i}')
            run_sa(tpu_node,x_2d, self.hw.ip_buf_act.device_address, src2_list[i], src2_c_list[i], dst_list[i], param_buf_list[i],q_zero_point,self.qkv.zero_point)

        actual_results_elements = padded_rows * src2_list[0].shape[1]
        tile_row_nums = src2_list[0].shape[1]
        tile_col_nums = x.shape[1]
        B = x.shape[0]
        results = [None] * 4
        done_mask = 0
        target_mask = 0b1111

        start_time = time.perf_counter()

        while done_mask != target_mask:
            if (time.perf_counter() - start_time) > 5.0:
                read_value = self.INTERRUPT1.read(0x00)
                print(f"TPU Timeout! done={bin(done_mask)} reg={hex(read_value)}")
                breakpoint()
                raise RuntimeError(f"TPU Timeout! done={bin(done_mask)}")

            reg_val = self.INTERRUPT1.read(0x00)
            for i in range(4):
                bit = (1 << i)
                if (reg_val & bit) and not (done_mask & bit):
                    buf = dst_list[i]
                    arr = np.asarray(buf).reshape(-1)[:actual_results_elements].reshape(padded_rows, tile_row_nums)
                    needed = arr[:num_rows, :tile_row_nums]            # non-contig view
                    col_start = i * tile_row_nums
                    col_end   = col_start + tile_row_nums
                    for b in range(B):
                        np.copyto(
                            concat_memory[b, :self.original_row_nums, col_start:col_end],
                            needed[(b*padded_rows//B):(b*padded_rows//B)+self.original_row_nums, :]
                        )
                    done_mask |= bit
                else:
                    time.sleep(0.00005)
                    i=3
        self.INTERRUPT1.write(0x0C, 0xF)

        out_quant = torch._make_per_tensor_quantized_tensor(
            concat_memory_torch,
            scale      = float(self.qkv.scale),
            zero_point = int(self.qkv.zero_point)
        )
        return out_quant

    ########    ##########  ##########
    ########    ##########  ##########
    ##      ##  ##          ##
    ##      ##  ##          ##
    ########    ########    ########
    ########    ########    ########
    ##    ##    ##          ##
    ##    ##    ##          ##
    ##      ##  ##########  ##
    ##      ##  ##########  ##

    def TPU_PROJLinear(self, x):
        if x.dim() == 4:
            x = x.transpose(1, 2)
            x = x.reshape(x.shape[0], x.shape[1], -1)

        x_2d        = x.reshape(-1, x.shape[-1])
        num_rows    = x_2d.shape[0]
        padded_rows = (num_rows + 15) // 16 * 16
        col         = self.proj_src2_list[0].shape[1]
        BRAM_BASE   = 0xB000_0000

        # 1. ravel (no copy)
        _t_load0 = time.perf_counter()
        flat_data    = x_2d.int_repr().cpu().numpy().ravel()
        num_elements = flat_data.size

        # 2. ctypes memmove
        ctypes.memmove(
            self.hw.ip_buf_act.ctypes.data,
            flat_data.ctypes.data,
            flat_data.nbytes
        )
        _t_load1 = time.perf_counter()

        # 3. pre-allocated padded input (no np.pad)
        current_input = self.proj_padded_input

        # TPU 4개 실행: 연산은 기존 그대로, 목적지만 BRAM으로 변경
        #   TPU i → BRAM_BASE + i*(padded_rows*192) 에 [padded_rows,192] compact 기록.
        _t_issue0 = time.perf_counter()
        Interrupt_write(self.INTERRUPT1)
        for i in range(4):
            tpu_node = getattr(self.hw.ip_ol, f'TPU_PROCESSOR_{i}')
            dst_obj  = PhysAddr(device_address=BRAM_BASE + i * padded_rows * col)
            run_sa(tpu_node,
                   current_input,
                   self.hw.ip_buf_act.device_address,
                   self.proj_src2_list[i],
                   self.proj_src2_c_list[i],
                   dst_obj,
                   self.proj_param_buf_list[i],
                   x.q_zero_point(),
                   self.proj.zero_point)
        _t_issue1 = time.perf_counter()

        # 인터럽트 대기만 (CPU gather/reshape 전면 제거)
        _t_wait0 = time.perf_counter()
        done_mask   = 0
        target_mask = 0b1111
        start_time  = time.perf_counter()

        while done_mask != target_mask:
            if (time.perf_counter() - start_time) > 5.0:
                read_value = self.INTERRUPT1.read(0x00)
                print(f"TPU Timeout! done={bin(done_mask)} reg={hex(read_value)}")
                breakpoint()
                raise RuntimeError(f"TPU Timeout! done={bin(done_mask)}")

            reg_val   = self.INTERRUPT1.read(0x00)
            done_mask |= reg_val
            if done_mask != target_mask:
                time.sleep(0.00005)
        _t_wait1 = time.perf_counter()

        self.INTERRUPT1.write(0x0C, 0b1111)
        self.hw._ln_input_in_bram = True

        # 반환 (파이프라인 유지용 shape/scale/zp only; 데이터 정확도는 무시)
        res_torch = self.mha_result_torch[:num_rows, :self.proj_out_features].reshape(
            x.shape[:-1] + (self.proj_out_features,)
        )

        # 구간별 지연 계측
        print(
            f"[PROJ] load={(_t_load1-_t_load0)*1000:7.3f}ms  "
            f"issue={(_t_issue1-_t_issue0)*1000:7.3f}ms  "
            f"TPU_wait={(_t_wait1-_t_wait0)*1000:7.3f}ms")

        out_quant = torch._make_per_tensor_quantized_tensor(
            res_torch,
            scale      = float(self.proj.scale),
            zero_point = int(self.proj.zero_point)
        )

        return out_quant

    def TPU_Matmul(self, a_shape, b_shape, a_zero_point, mode='QK'):
        """
        mode: 'QK' = Q@K^T → dequant + scale + softmax + 재양자화까지 처리
              'PV' = P@V → 결과 수집만

        QK mode:
            - TPU matmul 후, 4 head를 thread pool로 병렬 처리
            - head별 fused pipeline: read → dequant → ×(1/√d_k) → softmax → quantize
            - 결과는 self.hw.ip_buf_mm_P_list (PV 입력)와 qk_result_memory에 write
            - return: softmax + quantize된 quantized tensor (attention_input scale/zp)

        PV mode:
            - TPU matmul 후 결과만 copy
            - return: attention_output scale/zp로 wrap된 quantized tensor
        """
        # ─────────────────────────────────────────────
        # 0) Shape 및 공통 변수
        # ─────────────────────────────────────────────
        B, heads, M, K = a_shape.shape
        N = b_shape.shape[-1]
        padded_rows = M
        cols   = b_shape.shape[3]
        n_elem = padded_rows * cols
        # buf_list: mode에 따라 다르지만 group과는 무관 (루프 밖에서 한 번만 결정)
        if mode == 'QK':
            buf_list = self.hw.ip_buf_mm_OCM_list
        elif mode == 'PV':
            buf_list = self.hw.ip_buf_mm_Q_list
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # ─────────────────────────────────────────────
        # 1) QK mode 양자화 파라미터 미리 계산
        # ─────────────────────────────────────────────
        if mode == 'QK':
            def process_qk_head(group_):
                selected_group = group_%8
                BRAM_BASE  = 0xB000_0000
                H, W = 208, 208
                group_size = H * W

                b       = group_ // 12
                h_start = group_ % 12
                src_addr = self.hw.ip_buf_mm_OCM_list[selected_group]
                dst_addr = self.hw.ip_buf_mm_P_list[group_]
                if self._softmax_first_run:
                    run_softmax(
                        self.hw.ip_ol.softmax_module_0,
                        dst           = dst_addr,
                        src           = src_addr,
                        height        = H*4,
                        width         = W,
                        scale = struct.unpack('<I', struct.pack('<f', self.inv_out_scale))[0],
                        softmax_scale = struct.unpack('<I', struct.pack('<f', self.combined_scale))[0],
                        poll          = False,
                    )
                    self._softmax_first_run = False
                else:
                    run_softmax(
                        self.hw.ip_ol.softmax_module_0,
                        dst           = dst_addr,
                        src           = src_addr,
                        height        = H * 4,
                        width         = W,
                        scale = struct.unpack('<I', struct.pack('<f', self.inv_out_scale))[0],
                        softmax_scale = struct.unpack('<I', struct.pack('<f', self.combined_scale))[0],
                        poll          = False,
                    )

        else:
            def process_pv_head(group_):
                np.copyto(
                    self.hw.pv_result_memory[group_:group_+4, :208, :self.hw.d_k],
                    self.hw.q_strided[group_:group_+4, :208, :self.hw.d_k]
                )

        # ─────────────────────────────────────────────
        # 2) Group별 루프 (TPU 4개 동시 실행 + head 4개 병렬 후처리)
        # ─────────────────────────────────────────────

        futures = []
        prev_group  = None
        self.tpu_nodes = [
            getattr(self.hw.ip_ol, f'TPU_PROCESSOR_{i}')
            for i in range(4)
        ]
        q_addrs = [self.hw.ip_buf_mm_Q_list[h].device_address
           for h in range(B * heads)]

        # TPU_Matmul 시작 전에
        a_zp_int      = int(a_zero_point)
        energy_zp_int = int(self.energy_zero)
        attn_zp_int   = int(self.attention_output_zero)
        for group in range(0, B * heads, 4):
            if mode == 'PV':
                b       = group // 12
                h_start = group % 12
            Interrupt_write(self.INTERRUPT1)
            irq_future = asyncio.run_coroutine_threadsafe(
                anext(interrupt_monitor(self.INTERRUPT1, num_events=4)),
                self.hw.irq_loop
            )

            for i in range(4):
                head_idx = group + i
                dst_idx = head_idx % 8
                if mode == 'QK':
                    debug=run_sa(self.tpu_nodes[i],
                           a_shape[0][0],
                           q_addrs[head_idx],
                           b_shape[0][0],
                           self.hw.ip_buf_mm_KT_list[head_idx],
                           self.hw.ip_buf_mm_OCM_list[dst_idx],
                           self.MM_energy_param_buf_list[i],
                           a_zp_int, int(self.energy_zero) )
                else:
                    run_sa(self.tpu_nodes[i],
                           a_shape[0][0],
                           self.hw.ip_buf_mm_P_list[head_idx].device_address,
                           b_shape[0][0],
                           self.hw.ip_buf_mm_V_list[head_idx],
                           self.hw.ip_buf_mm_Q_list[head_idx],
                           self.MM_attn_param_buf_list[head_idx],
                           a_zp_int, attn_zp_int)
            # ── (c) 4-head 병렬 처리 (thread pool) ──
            if prev_group is not None:
                if mode == 'QK':
                    process_qk_head(prev_group)
                else:
                    process_pv_head(prev_group)
            else:
                self.hw._softmax_scratch_f32_torch.zero_()

            status = irq_future.result(timeout=5000)
            while (self.hw.ip_ol.softmax_module_0.read(0x50) >> 31) & 0x1:
                time.sleep(0.0001)

            if status is None:
                breakpoint()
                raise RuntimeError("TPU Timeout!")
            prev_group     = group

        if prev_group is not None:
            if mode == 'QK':
                process_qk_head(prev_group)
            else:
                process_pv_head(prev_group)

        # ─────────────────────────────────────────────
        # 3) 결과 반환 (zero copy view wrap)
        # ─────────────────────────────────────────────
        if mode == 'QK':
            # softmax + quantize된 결과 → PV 입력 형식 (attention_input scale/zp)
            out_quant = self.hw.P_strided
        else:  # PV
            out_quant = torch._make_per_tensor_quantized_tensor(
                self.hw._pv_result_view,
                scale      = float(self.attention_output_scale),
                zero_point = int(self.attention_output_zero)
            )
        return out_quant




    def preprocess_k(self, k_raw, x_shape):
        import ctypes
        B, N, C = x_shape
        N_pad = (N + 15) // 16 * 16

        # ① dtype 변환
        neg_zp = np.uint8((256 - self.qkv.zero_point) & 0xFF)
        k3_np = (self.k_concat_memory[:, :N, :] + neg_zp).view(np.int8)
        k3 = torch.from_numpy(k3_np)

        # ② padding
        if N != N_pad:
            k3 = torch.nn.functional.pad(k3, (0, 0, 0, N_pad - N))

        # ③ 한번에 reshape/permute (loop 제거)
        k3 = k3.reshape(B, N_pad, self.num_heads, self.d_k)   # (B, N_pad, heads, d_k)
        k3 = k3.permute(0, 2, 1, 3)                            # (B, heads, N_pad, d_k)
        k3 = k3.reshape(B, self.num_heads, N_pad//16, 16, self.d_k)
        k_np = k3.numpy()
        np.copyto(   self.hw._KT_scratch,   k_np.transpose(0, 1, 2, 4, 3) ) # 2,12,13,64,16

        # ⑤ memmove loop
        # 한번에 복사 (padding 위치는 건드리지 않음)
        np.copyto(
            self.hw.kt_strided,
            self.hw._KT_scratch.reshape(self.hw.slots, self.hw.slot)
        )

        self.hw.ip_buf_mm_KT_all.flush()

        return k3
    def preprocess_q(self, q_raw, x_shape):
        B, N, C = x_shape

        # ❌ 기존 문제들:
        # 1. q3.astype(np.uint8) → 이미 uint8인데 복사 발생
        # 2. torch.permute → non-contiguous tensor
        # 3. 이중 for loop → 느림
        # 4. q3[b,h].reshape(-1) → 매번 reshape

        # ✅ 최적화:
        # 1) zero-copy numpy view (이미 uint8)
        q_np = self.q_concat_memory[:, :N, :]          # [B, N, 768] view

        # 2) reshape → [B, N, 12, 64] view (복사 없음)
        q_np = q_np.reshape(B, N, self.hw.num_heads, self.d_k)

        q3 = q_np.transpose(0, 2, 1, 3)
        # ③ transpose + copy 한번에
        np.copyto(self.hw.q_strided, q3.reshape(self.hw.batch_size*12,208,64))
        self.hw.ip_buf_mm_Q_all.flush()
        return q3  # [B, 12, N, 64]

    def preprocess_v(self, v_raw, x_shape):
        B, N, C = x_shape
        N_pad = (N + 15) // 16 * 16
        # ① dtype 변환
        np.add(self.v_concat_memory[:, :N, :], self.neg_zp,out=self.v3_buf_u8[:B, :N, :])   # ← 새 배열 할당 없음
        v3_np = self.v3_buf_i8[:B, :N, :]

        # [B, N, 768] → [B, N, heads, d_k] → [B, heads, N, d_k//16, 16]
        v_for_sum = v3_np.reshape(self.hw.batch_size, N, self.hw.num_heads, 64) #(2,208,12,64)

        v_sum = v_for_sum.sum(axis=1, dtype=np.int32)
        correction = (v_sum * self.scale_128).reshape(B*self.hw.num_heads,64)
        self.mm_attn_param_np[:,1::2][:,:64] =  correction

        self.MM_attn_param_buf_all.flush()

        if N != N_pad:
            v3_np = np.pad(v3_np, ((0,0),(0,N_pad-N),(0,0)))

        v_np = v3_np.reshape(B, N_pad, self.hw.num_heads, 64)

        # transpose [B,N,heads,d_k] → [B,heads,N,d_k//16,16]
        v_src = np.ascontiguousarray(
            v_np.transpose(0, 2, 1, 3)              # [2,12,208,64]
            .reshape(self.hw.batch_size, self.hw.num_heads, N_pad, 4, 16)  # [2,12,208,4,16]
            .transpose(0, 1, 3, 2, 4)              # [2,12,4,208,16]
        )

        np.copyto(self.hw.v_strided, v_src)
        self.hw.ip_buf_mm_V_all.flush()

        return torch.from_numpy(v_np.transpose(0, 2, 1, 3))  # [B, heads, N, d_k]

    import ctypes
    def preprocess_k_wrapper(self,k_raw, x_shape):
        return self.preprocess_k(k_raw, x_shape)

    def preprocess_q_wrapper(self,q_raw, x_shape):
        return self.preprocess_q(q_raw, x_shape)

    def preprocess_v_wrapper(self,v_raw, x_shape):
        return self.preprocess_v(v_raw, x_shape)

    def forward(self, x):
        B, N, C = x.shape
        d_k = C // self.num_heads
        scale= d_k ** -0.5
        q_zero_point = x.q_zero_point()

        if N%16 !=0:
            N=(N+15)//16 * 16
            pad_amt = N - x.shape[1] # 200 - 197 = 3
            x = x.int_repr()
            x = np.pad(x, ((0,0), (0, pad_amt), (0, 0)), mode='constant', constant_values=q_zero_point)

        # ── 1. QKV Linear (TPU) ───────────────────
        import ctypes
        ctypes.memmove(
            self.hw.ip_buf_act.ctypes.data,  # dst
            x.ctypes.data,                     # src
            x.nbytes                           # size
        )

        k2 = self.TPU_QKVLinear(x, 'k',q_zero_point)
        kt_thread = self._head_pool.submit(self.preprocess_k_wrapper, k2, k2.shape)

        q2 = self.TPU_QKVLinear(x, 'q',q_zero_point)  # K preprocess와 overlap!
        qt_thread = self._head_pool.submit(self.preprocess_q_wrapper, q2, q2.shape)

        v2 = self.TPU_QKVLinear(x, 'v',q_zero_point)  # K,Q preprocess와 overlap!
        vt_thread = self._head_pool.submit(self.preprocess_v_wrapper, v2, v2.shape)

        k_shape=kt_thread.result()
        q_shape= qt_thread.result()
        # --TPU Q@ K^T + SOFTMAX----------------------------------------
        attn = self.TPU_Matmul(q_shape,self.k_shape,q2.q_zero_point())

        v_shape = vt_thread.result()
        # ── 5. P @ V (CPU torch.matmul) ───────────
        x2= self.TPU_Matmul(attn, v_shape, self.attention_input_zero, mode = 'PV')

        x2 = x2.transpose(1, 2).contiguous()
        x2 = x2.reshape(B, N, C)

        # ── 7. proj Linear (TPU) ──────────────────
        x = self.TPU_PROJLinear(x2)

        x  = x[:, :197, :]

        return x


##          ######  ##      ##  ##########    ######    ########
##          ######  ##      ##  ##########    ######    ########
##            ##    ####    ##  ##          ##      ##  ##      ##
##            ##    ####    ##  ##          ##      ##  ##      ##
##            ##    ##  ##  ##  ########    ##########  ########
##            ##    ##  ##  ##  ########    ##########  ########
##            ##    ##    ####  ##          ##      ##  ##    ##
##            ##    ##    ####  ##          ##      ##  ##    ##
##########  ######  ##      ##  ##########  ##      ##  ##      ##
##########  ######  ##      ##  ##########  ##      ##  ##      ##

class TPULinear1(nn.Module):
    def __init__(
        self,
        name,
        x_scale,
        weight_tensor,
        bias_tensor,
        out_scale,
        out_zp,
        hw
    ):
        super().__init__()

        # 1. default config
        self.name = name
        self.hw = hw
        self.out_scale  = float(out_scale)
        self.out_zp = out_zp
        self.INTERRUPT1 = hw.ip_ol.axi_intc_0

        # 2. activate shared irq loop
        if not hasattr(hw, 'irq_loop'):
            new_loop = asyncio.new_event_loop()
            hw.irq_loop = new_loop
            t = threading.Thread(target=start_irq_loop, args=(new_loop,), daemon=True)
            t.start()
            print("[TPULinear] Shared IRQ loop started.")

        # 3. parse weights and scale,zp
        if hasattr(weight_tensor, 'int_repr'):
            w_np = weight_tensor.int_repr().detach().cpu().numpy()
            self.w_scale = weight_tensor.q_per_channel_scales()
            self.w_zero_point = weight_tensor.q_per_channel_zero_points()
        else:
            w_np = weight_tensor.detach().cpu().numpy() \
                if hasattr(weight_tensor, 'detach') else weight_tensor
            self.w_scale = 1.0
            self.w_zero_point = 0

        self.out_features, self.in_features= w_np.shape

        w_scale_np = self.w_scale.detach().cpu().numpy().astype(np.float32) \
            if hasattr(self.w_scale, 'detach') else np.asarray(self.w_scale, dtype=np.float32)
        if w_scale_np.ndim == 0:
            w_scale_np = np.full(self.out_features, float(w_scale_np), dtype=np.float32)

        m_scale_per_channel = (float(x_scale) * w_scale_np / self.out_scale).astype(np.float32)

        # process bias
        if bias_tensor is not None:
            b_np = bias_tensor.detach().cpu().numpy().astype(np.float32) \
                if hasattr(bias_tensor, 'detach') else np.asarray(bias_tensor, dtype=np.float32)
            self.bias = b_np / self.out_scale
        else:
            self.bias = np.zeros(self.out_features, dtype=np.float32)

        # split allocated TPU buffer
        w_slices = np.vsplit(w_np, 4)
        m_slices = np.split(m_scale_per_channel, 4)
        b_slices = np.split(self.bias, 4)

        self.src2_list = []
        self.src2_c_list = []
        self.param_buf_list = []

        for w_s, m_s, b_s in zip(w_slices, m_slices, b_slices):
            current_rows = w_s.shape[0]
            remainder = current_rows % 16
            if remainder != 0:
                padding_size = 16 - remainder
                w_s = np.pad(w_s, ((0, padding_size), (0, 0)), mode='constant', constant_values=0)
                m_s = np.pad(m_s, (0, padding_size), mode='constant')
                b_s = np.pad(b_s, (0, padding_size), mode='constant')
                print(f"Padding added: {current_rows} -> {w_s.shape[0]}")

            w_s_tensor = torch.from_numpy(w_s).to(torch.int8)
            W, W_c = preprocess_weight_for_tpu(w_s_tensor)

            s2_c = allocate(shape=W_c.shape, dtype=np.int8)
            s2_c[:] = W_c

            self.src2_list.append(W.shape)
            self.src2_c_list.append(s2_c)

            num_ch = w_s.shape[0]
            interleaved = np.empty(num_ch * 2, dtype=np.float32)
            interleaved[0::2] = m_s
            interleaved[1::2] = b_s

            param_buf = allocate(shape=(num_ch * 2,), dtype=np.float32)
            param_buf[:] = interleaved
            param_buf.flush()
            self.param_buf_list.append(param_buf)

        self.result_buf = np.empty(
            (197*self.hw.batch_size, self.out_features), dtype=np.int8)
        self.result_torch = torch.from_numpy(self.result_buf)

        padded_rows = (197 * self.hw.batch_size + 7) // 8 * 8
        self.padded_input_map = {
            768:  np.zeros((padded_rows, 768),  dtype=np.int8),
            3072: np.zeros((padded_rows, 3072), dtype=np.int8),
        }

    def forward(self, x):
        # input data fetch
        x_2d = x.reshape(-1, x.shape[-1])
        num_rows, in_features = x_2d.shape
        padded_rows = (num_rows + 7) // 8 * 8

        # 1. copy data
        flat_data = x_2d.int_repr().numpy().ravel()
        ctypes.memmove(
            self.hw.ip_buf_act.ctypes.data,
            flat_data.ctypes.data,
            flat_data.nbytes
        )

        # 2. activate interrupt monitor
        Interrupt_write(self.INTERRUPT1)
        irq_future = asyncio.run_coroutine_threadsafe(
            anext(interrupt_monitor(self.INTERRUPT1, num_events=4)),
            self.hw.irq_loop
        )

        # 3. execute tpu cores
        padded_input = self.padded_input_map[in_features]
        for i in range(4):
            tpu_node = getattr(self.hw.ip_ol, f'TPU_PROCESSOR_{i}')
            run_sa(
                tpu_node,
                padded_input,
                self.hw.ip_buf_act.device_address,
                self.src2_list[i],
                self.src2_c_list[i],
                self.hw.ip_buf_dst[i],
                self.param_buf_list[i],
                x.q_zero_point(),
                self.out_zp
            )

        # 4. wait interrupt
        status = irq_future.result(timeout=5000)
        if status is None:
            read_value = self.INTERRUPT1.read(0x00)
            print(f"read_value is {read_value}")
            breakpoint()
            raise RuntimeError(f"TPU HW Timeout! Interrupt status: {hex(read_value)}")

        # 5. collect result data
        col_size = self.out_features // 4
        actual_results_elements = padded_rows * self.src2_list[0][1]

        for i, d in enumerate(self.hw.ip_buf_dst):
            arr = np.asarray(d).ravel()[:actual_results_elements].reshape(padded_rows, self.src2_list[0][1])
            self.result_buf[:num_rows, i*col_size:(i+1)*col_size] = arr[:num_rows, :col_size]

        # 6. quantization wrapping
        res_torch = (self.result_torch[:num_rows, :self.out_features]
                    .reshape(x.shape[:-1] + (self.out_features,)).to(x.device))
        out_int = res_torch.to(torch.uint8)

        return torch._make_per_tensor_quantized_tensor(
            out_int,
            scale      = float(self.out_scale),
            zero_point = int(self.out_zp)
        )

# TPULinear2 (= mlp_3, MLP 출력 Linear 3072→768):
#   결과를 DRAM(ip_buf_dst)에 모아 CPU gather로 반환하던 방식 대신,
#   TPU_PROJLinear과 동일하게 BRAM(0xB000_0000)에 직접 기록하고
#   hw._ln_input_in_bram 플래그를 세워 뒤따르는 Residual-LayerNorm이
#   BRAM에서 바로 읽게 한다(CPU gather/reshape 제거).
#   분할 방식은 TPU_PROJLinear과 동일한 열분할(N=192×4)이며,
#   각 TPU i는 BRAM_BASE + i*(padded_rows*192) 블록에 기록한다.
class TPULinear2(nn.Module):
    def __init__(
        self,
        name,
        x_scale,
        weight_tensor,
        bias_tensor,
        out_scale,
        out_zp,
        hw
    ):
        super().__init__()

        # 1. default config
        self.name = name
        self.hw = hw
        self.out_scale  = float(out_scale)
        self.out_zp = out_zp
        self.INTERRUPT1 = hw.ip_ol.axi_intc_0

        # 2. activate shared irq loop
        if not hasattr(hw, 'irq_loop'):
            new_loop = asyncio.new_event_loop()
            hw.irq_loop = new_loop
            t = threading.Thread(target=start_irq_loop, args=(new_loop,), daemon=True)
            t.start()
            print("[TPULinear] Shared IRQ loop started.")

        # 3. parse weights and scale,zp
        if hasattr(weight_tensor, 'int_repr'):
            w_np = weight_tensor.int_repr().detach().cpu().numpy()
            self.w_scale = weight_tensor.q_per_channel_scales()
            self.w_zero_point = weight_tensor.q_per_channel_zero_points()
        else:
            w_np = weight_tensor.detach().cpu().numpy() \
                if hasattr(weight_tensor, 'detach') else weight_tensor
            self.w_scale = 1.0
            self.w_zero_point = 0

        self.out_features, self.in_features= w_np.shape

        w_scale_np = self.w_scale.detach().cpu().numpy().astype(np.float32) \
            if hasattr(self.w_scale, 'detach') else np.asarray(self.w_scale, dtype=np.float32)
        if w_scale_np.ndim == 0:
            w_scale_np = np.full(self.out_features, float(w_scale_np), dtype=np.float32)

        m_scale_per_channel = (float(x_scale) * w_scale_np / self.out_scale).astype(np.float32)

        # process bias
        if bias_tensor is not None:
            b_np = bias_tensor.detach().cpu().numpy().astype(np.float32) \
                if hasattr(bias_tensor, 'detach') else np.asarray(bias_tensor, dtype=np.float32)
            self.bias = b_np / self.out_scale
        else:
            self.bias = np.zeros(self.out_features, dtype=np.float32)

        # split allocated TPU buffer
        w_slices = np.vsplit(w_np, 4)
        m_slices = np.split(m_scale_per_channel, 4)
        b_slices = np.split(self.bias, 4)

        self.src2_list = []
        self.src2_c_list = []
        self.param_buf_list = []

        for w_s, m_s, b_s in zip(w_slices, m_slices, b_slices):
            current_rows = w_s.shape[0]
            remainder = current_rows % 16
            if remainder != 0:
                padding_size = 16 - remainder
                w_s = np.pad(w_s, ((0, padding_size), (0, 0)), mode='constant', constant_values=0)
                m_s = np.pad(m_s, (0, padding_size), mode='constant')
                b_s = np.pad(b_s, (0, padding_size), mode='constant')
                print(f"Padding added: {current_rows} -> {w_s.shape[0]}")

            w_s_tensor = torch.from_numpy(w_s).to(torch.int8)
            W, W_c = preprocess_weight_for_tpu(w_s_tensor)

            s2_c = allocate(shape=W_c.shape, dtype=np.int8)
            s2_c[:] = W_c

            self.src2_list.append(W.shape)
            self.src2_c_list.append(s2_c)

            num_ch = w_s.shape[0]
            interleaved = np.empty(num_ch * 2, dtype=np.float32)
            interleaved[0::2] = m_s
            interleaved[1::2] = b_s

            param_buf = allocate(shape=(num_ch * 2,), dtype=np.float32)
            param_buf[:] = interleaved
            param_buf.flush()
            self.param_buf_list.append(param_buf)

        # 반환용 placeholder(데이터는 BRAM에 있으므로 shape/scale/zp만 의미).
        # LayerNorm 입력은 quint8이므로 uint8 버퍼로 준비한다.
        self.result_buf = np.empty(
            (197*self.hw.batch_size, self.out_features), dtype=np.uint8)
        self.result_torch = torch.from_numpy(self.result_buf)

        # TPU_PROJLinear과 동일한 16-정렬(shape(M,K) 전달용 zero 버퍼).
        padded_rows = (197 * self.hw.batch_size + 15) // 16 * 16
        self.padded_input_map = {
            768:  np.zeros((padded_rows, 768),  dtype=np.int8),
            3072: np.zeros((padded_rows, 3072), dtype=np.int8),
        }

        # LayerNorm이 읽는 BRAM 베이스 (manager의 ln_in_bram_addr과 동일)
        self.BRAM_BASE = 0xB000_0000

    def forward(self, x):
        # ── 입력 2D 정리 ───────────────────────────────────────
        x_2d        = x.reshape(-1, x.shape[-1])
        num_rows    = x_2d.shape[0]
        in_features = x_2d.shape[1]
        padded_rows = (num_rows + 15) // 16 * 16       # proj와 동일한 16-정렬
        col         = self.src2_list[0][1]             # TPU 1개가 내는 출력 폭(=192)
        BRAM_BASE   = self.BRAM_BASE

        # 1. 활성값 → ip_buf_act(DRAM) 복사
        _t_load0 = time.perf_counter()
        flat_data = x_2d.int_repr().cpu().numpy().ravel()
        ctypes.memmove(
            self.hw.ip_buf_act.ctypes.data,
            flat_data.ctypes.data,
            flat_data.nbytes
        )
        _t_load1 = time.perf_counter()

        # shape(M,K) 전달용 (내용 무의미, 실제 활성값은 ip_buf_act에 있음)
        padded_input = self.padded_input_map[in_features]

        # 2. TPU 4개 실행: 목적지를 ip_buf_dst(DRAM) 대신 BRAM으로 지정.
        #    TPU i → BRAM_BASE + i*(padded_rows*col) 에 [padded_rows, col] 블록 기록.
        #    → CPU gather 없이 결과가 곧바로 LayerNorm 입력 BRAM에 놓인다.
        _t_issue0 = time.perf_counter()
        Interrupt_write(self.INTERRUPT1)
        for i in range(4):
            tpu_node = getattr(self.hw.ip_ol, f'TPU_PROCESSOR_{i}')
            dst_obj  = PhysAddr(device_address=BRAM_BASE + i * padded_rows * col)
            run_sa(
                tpu_node,
                padded_input,
                self.hw.ip_buf_act.device_address,
                self.src2_list[i],
                self.src2_c_list[i],
                dst_obj,
                self.param_buf_list[i],
                x.q_zero_point(),
                self.out_zp
            )
        _t_issue1 = time.perf_counter()

        # 3. 인터럽트 대기만 (CPU gather/reshape 전면 제거)
        _t_wait0 = time.perf_counter()
        done_mask   = 0
        target_mask = 0b1111
        start_time  = time.perf_counter()

        while done_mask != target_mask:
            if (time.perf_counter() - start_time) > 5.0:
                read_value = self.INTERRUPT1.read(0x00)
                print(f"TPU Timeout! done={bin(done_mask)} reg={hex(read_value)}")
                breakpoint()
                raise RuntimeError(f"TPU Timeout! done={bin(done_mask)}")

            reg_val   = self.INTERRUPT1.read(0x00)
            done_mask |= reg_val
            if done_mask != target_mask:
                time.sleep(0.00005)
        _t_wait1 = time.perf_counter()

        self.INTERRUPT1.write(0x0C, 0b1111)

        # 4. LayerNorm이 BRAM에서 직접 읽도록 플래그 세팅
        self.hw._ln_input_in_bram = True

        # 구간별 지연 계측 (proj와 동일 포맷)
        print(
            f"[MLP3] load={(_t_load1-_t_load0)*1000:7.3f}ms  "
            f"issue={(_t_issue1-_t_issue0)*1000:7.3f}ms  "
            f"TPU_wait={(_t_wait1-_t_wait0)*1000:7.3f}ms")

        # 5. 반환 (파이프라인 유지용 shape/scale/zp only; 데이터는 BRAM에 있음)
        res_torch = self.result_torch[:num_rows, :self.out_features].reshape(
            x.shape[:-1] + (self.out_features,)
        )

        return torch._make_per_tensor_quantized_tensor(
            res_torch,
            scale      = float(self.out_scale),
            zero_point = int(self.out_zp)
        )