import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.ao.quantization as tq
from torch.ao.quantization import quantize_fx
from torch.ao.quantization.qconfig_mapping import QConfigMapping
from torch.ao.nn.quantized import FloatFunctional

# ==============================================================================
#  Custom Quantized Multihead Attention Class
# ------------------------------------------------------------------------------
class QuantMultiheadAttention(nn.Module):
    def __init__(self, dim, num_heads, attn_drop=0.0, proj_drop=0.0, use_hw=False):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.dim       = dim
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.scale     = self.head_dim ** -0.5
        self.use_hw    = use_hw

        # Layers
        self.qkv       = nn.Linear(dim, 3 * dim, bias=True)
        self.proj      = nn.Linear(dim, dim, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        self.func = FloatFunctional()

        # Buffers for Sparsity Masks (persistent=False: state_dict에 저장 안 함)
        self.qkv.register_buffer("_mixed_sparsity_mask", None, persistent=False)
        self.proj.register_buffer("_mixed_sparsity_mask", None, persistent=False)

    def _enforce_weight_mask(self):
        # QKV Mask
        qkv_mask = getattr(self.qkv, "qkv_mask", None)
        if qkv_mask is not None:
            self.qkv.weight.data.mul_(qkv_mask)

        # Proj Mask
        proj_mask = getattr(self.proj, "proj_mask", None)
        if proj_mask is not None:
            self.proj.weight.data.mul_(proj_mask)

    def forward(self, query, key, value, need_weights=False, attn_mask=None, **kwargs):
        # Apply mask
        self._enforce_weight_mask()
        x = query
        B, N, C = x.shape
        H, D = self.num_heads, self.head_dim

        # QKV Projection and Reshape
        qkv = self.qkv(x)  # [B, N, 3*C]
        if hasattr(x, 'node'):
            print("Currently in Tracing Mode (Proxy)")
        else:
            print(f"Actual Data Flow - Type: {x.dtype}")
            print(f"Actual Data Flow - Type: {qkv.dtype}")
            if x.is_quantized:
                print(f"Quantized: {x.qscheme()}, Scale: {x.q_scale()}")
                print(f"Quantized: {qkv.qscheme()}, Scale: {qkv.q_scale()}")

        qkv = qkv.reshape(B, N, 3, H, D).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, H, N, D]

        # Attention Score
        attn = self.func.matmul(q, k.transpose(-2, -1))
        attn = self.func.mul(attn, self.scale)

        if attn_mask is not None:
            attn = self.func.add(attn, attn_mask)
        # Softmax
        if self.use_hw:
            attn = F.softmax(attn, dim=-1)
        else:
            attn = F.softmax(attn, dim=-1)

        attn = self.attn_drop(attn)

        # Output Projection
        out = self.func.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().reshape(B, N, C)  # [B,N,C]
        out = self.proj(out)
        out = self.proj_drop(out)

        return out, None


def apply_mixed_sparsity_(parent_module: nn.Module,
                          layer_name: str,
                          embed_dim: int = None,
                          dense_ratio=0.1,
                          dense_sparsity=0.1,
                          sparse_sparsity=0.9):
    linear = getattr(parent_module, layer_name)

    with torch.no_grad():
        W = linear.weight.data
        
        def _create_mask(W_sub, d_ratio, d_sparse, s_sparse):
            out_ch, in_ch = W_sub.shape
            split = int(out_ch * d_ratio)
            mask = torch.ones_like(W_sub)
            
            if split > 0:
                mask[:, :split] = (torch.rand(out_ch, split, device=W_sub.device) > d_sparse).float()
            if split < in_ch:
                mask[:, split:] = (torch.rand(out_ch, in_ch - split, device=W_sub.device) > s_sparse).float()
            return mask

        if embed_dim is not None and W.shape[0] == 3 * embed_dim:
            D = embed_dim
            mask_q = _create_mask(W[0:D], dense_ratio, dense_sparsity, sparse_sparsity)
            mask_k = _create_mask(W[D:2*D], dense_ratio, dense_sparsity, sparse_sparsity)
            mask_v = _create_mask(W[2*D:3*D], dense_ratio, dense_sparsity, sparse_sparsity)
            final_mask = torch.cat([mask_q, mask_k, mask_v], dim=0)
        else:
            final_mask = _create_mask(W, dense_ratio, dense_sparsity, sparse_sparsity)

        linear.weight.copy_(W * final_mask)

        buffer_name = f"{layer_name}_mask"
        if hasattr(parent_module, buffer_name):
            delattr(parent_module, buffer_name)
        parent_module.register_buffer(buffer_name, final_mask)


# =========================================================================
# [Model Conversion Utils]
#   * Float-based Model -> Quantizable Model
#   * Targets : nn.MultiheadAttention -> QuantMultiheadAttention
#               nn.LayerNorm -> QuantLayerNorm
# =========================================================================

def replace_mha(model: nn.Module, use_hw=False):
    replaced = 0

    for name, child in model.named_children():
        replaced += replace_mha(child, use_hw)

        if isinstance(child, nn.MultiheadAttention):
            embed_dim = child.embed_dim
            num_heads = child.num_heads
            dropout   = float(child.dropout)

            batch_first = getattr(child, "batch_first", False)
            if not batch_first:
                print(f"[Warning] Layer {name} has batch_first=False. HW implementation assumes True.")
            
            new_attn = QuantMultiheadAttention(
                dim=embed_dim,
                num_heads=num_heads,
                attn_drop=dropout,
                proj_drop=0.0,
                use_hw=False
            )

            # copy weights
            with torch.no_grad():
                new_attn.qkv.weight.copy_(child.in_proj_weight)
                if child.in_proj_bias is not None:
                    new_attn.qkv.bias.copy_(child.in_proj_bias)
                else:
                    new_attn.qkv.bias.zero_()

                new_attn.proj.weight.copy_(child.out_proj.weight)
                if child.out_proj.bias is not None:
                    new_attn.proj.bias.copy_(child.out_proj.bias)
                else:
                    new_attn.proj.bias.zero_()

            # apply sparsity
            apply_mixed_sparsity_(new_attn, "qkv", embed_dim=embed_dim, 
                                  dense_ratio=0.5, dense_sparsity=0.0, sparse_sparsity=0.6)
            apply_mixed_sparsity_(new_attn, "proj", 
                                  dense_ratio=0.5, dense_sparsity=0.0, sparse_sparsity=0.6)

            setattr(model, name, new_attn)
            replaced += 1

    return replaced


# =========================================================================
#  QAT Build Functions
# =========================================================================

def buildQuant(model_fp32: nn.Module, batch_size=1, use_hw=False, qbackend="qnnpack"):
    model_quant = copy.deepcopy(model_fp32).cpu()

    n_mha = replace_mha(model_quant, use_hw=use_hw)
    print(f"[Info] Replaced {n_mha} MultiheadAttention layers.")

    qconfig = tq.get_default_qat_qconfig(qbackend)
    qconfig_mapping = QConfigMapping().set_global(qconfig)

    example_input = torch.randn(batch_size, 3, 224, 224).cpu()
    
    quant_model = quantize_fx.prepare_qat_fx(
        model_quant,
        qconfig_mapping,
        example_input
    )

    return quant_model
