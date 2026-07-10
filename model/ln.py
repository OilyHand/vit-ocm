import torch
import torch.nn as nn
import numpy as np
import struct


def float_packint(value):
    return struct.unpack('<I', struct.pack('<f', float(value)))[0]

def run_layernorm_hw(
    x,
    hw,
    src_addr,
    dst_addr,
    src_np,
    dst_np,
    mode,
    scale_a, zp_a,
    scale_b, zp_b,
    scale_c, zp_c,
    scale_o, zp_o,
):
    ln_ip = hw.ip_ol.layernorm_0

    B = hw.batch_size
    N = 197

    # input fetch (token-major: hidden is contiguous, no part/token transpose)
    use_bram = getattr(hw, "_ln_input_in_bram", False)
    print("use_bram",use_bram)

    if use_bram:
        # 입력은 proj gather가 이미 BRAM(0xB000_0000)에 token-major로 적재함.
        # → CPU 복사·BRAM 재읽기 전면 생략, inp_a만 BRAM으로 지정.
        ln_ip.register_map.inp_a = hw.ln_in_bram_addr
    else:
        data_a = x.int_repr().reshape(B, N, -1).contiguous()

        # copy input into the first N(=197) valid rows of the padded token-major buffer
        if mode == 0x00:
            np.copyto(dst_np[:, :N, :], data_a.numpy())
            ln_ip.register_map.inp_a = dst_addr
        else:
            np.copyto(hw.data_a_np[:, :N, :], data_a.numpy())

    # register write
    ln_ip.write(0x10, mode)
    ln_ip.write(0x38, float_packint(scale_a))
    ln_ip.write(0x40, float_packint(scale_b))
    ln_ip.write(0x48, float_packint(scale_c))
    ln_ip.write(0x50, float_packint(scale_o))
    ln_ip.write(0x58, zp_a)
    ln_ip.write(0x60, zp_b)
    ln_ip.write(0x68, zp_c)
    ln_ip.write(0x70, zp_o)
    # source and destination
    ln_ip.write(0x80, src_addr)
    ln_ip.write(0x90, dst_addr)

    # hardware run
    ln_ip.write(0x00, 0x01)
    while (ln_ip.read(0x00) & 0x02) == 0:
        pass
    hw.result_buf.invalidate()

    # token-major output: take the first N(=197) valid rows directly (no transpose)
    np.copyto(hw.ln_result_np, hw.result_np[:, :N, :])

    if use_bram:
        ln_ip.register_map.inp_a = hw.ln_addr_a
        hw._ln_input_in_bram = False
    elif mode == 0x00:
        ln_ip.register_map.inp_a = hw.ln_addr_a

    return torch._make_per_tensor_quantized_tensor(
        hw.ln_result_torch, scale_o, zp_o)


class fusedResidualLayerNorm(nn.Module):
    def __init__(
        self,
        normalized_shape,
        hw,
        src_addr,
        dst_addr,
        src_np,
        dst_np,
        mode,
        scale_a, zp_a,
        scale_b, zp_b,
        scale_c, zp_c,
        scale_o, zp_o,
    ):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.hw               = hw

        # add input
        self.scale_a, self.zp_a = scale_a, zp_a
        self.scale_b, self.zp_b = scale_b, zp_b

        # add output === layernorm input
        self.scale_c, self.zp_c = scale_c, zp_c

        # layernorm output
        self.scale_o, self.zp_o = scale_o, zp_o

        self.src_addr = src_addr
        self.dst_addr = dst_addr
        self.src_np = src_np
        self.dst_np = dst_np

        self.mode = mode

        # layernorm parameters
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias   = nn.Parameter(torch.zeros(normalized_shape))

        # layernorm parameters numpy
        self._weight_np = None
        self._bias_np   = None

    def sync_params(self):
        self._weight_np = self.weight.detach().cpu().numpy().astype(np.float32)
        self._bias_np   = self.bias.detach().cpu().numpy().astype(np.float32)

    def forward(self, x):
        if self._weight_np is None:
            self.sync_params()

        C = self.normalized_shape[0]
        np.copyto(self.hw.param_buf_np[:C], self._weight_np)
        np.copyto(self.hw.param_buf_np[C:], self._bias_np)

        out_tensor = run_layernorm_hw(
            x,
            self.hw,
            self.src_addr,
            self.dst_addr,
            self.src_np,
            self.dst_np,
            self.mode,
            self.scale_a, self.zp_a,
            self.scale_b, self.zp_b,
            self.scale_c, self.zp_c,
            self.scale_o, self.zp_o
        )

        return out_tensor
