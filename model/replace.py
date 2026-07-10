import torch
import torch.nn as nn
import torch.ao.nn.quantized.modules.linear as qlinear

from model.tpu import TPULinear, TPUMultiHeadAttention, TPUPatchEmbedding
from model.ln  import fusedResidualLayerNorm

# ====================================================================
#  Quantization Parameter Finders
# ====================================================================
def find_scale_in_args(args, model):
    """args에서 scale get_attr 노드 찾기"""
    for arg in args:
        if hasattr(arg, 'op') and arg.op == 'get_attr':
            try:
                val = model.get_buffer(arg.target)
                # scalar tensor이면 scale!
                if val.numel() == 1:
                    return float(val)
            except:
                continue
    return None

def find_output_scale(node, model, debug=True):
    # matmul 노드는 args[2]=scale, args[3]=zero_point를 직접 들고 있음

    scale_node = node.args[2]
    zp_node    = node.args[3]

    if isinstance(scale_node, torch.fx.Node) and scale_node.op == 'get_attr':
        out_scale = float(getattr(model, scale_node.target).item())
    elif isinstance(scale_node, (float, int)):
        out_scale = float(scale_node)
    else:
        raise ValueError(f'scale 타입 미지원: {type(scale_node)}')

    if isinstance(zp_node, torch.fx.Node) and zp_node.op == 'get_attr':
        out_zp = int(getattr(model, zp_node.target).item())
    elif isinstance(zp_node, (float, int)):
        out_zp = int(zp_node)
    else:
        raise ValueError(f'zp 타입 미지원: {type(zp_node)}')

    return out_scale, out_zp

def find_input_scale(node, model):
    curr = node.args[0]

    skip_ops = ['reshape', 'view', 'transpose',
                'permute', 'contiguous', 'flatten']
    while curr is not None:
        if any(op in str(curr.target) for op in skip_ops):
            curr = curr.args[0]
            continue

        if curr.op == 'call_function':
            fname = str(curr.target)

            # matmul 발견!
            if 'matmul' in fname or 'quantize_per_tensor' in fname:
                scale = find_scale_in_args(curr.args, model)
                if scale is not None:
                    return scale

        if curr.op == 'call_module':
            submod = model.get_submodule(curr.target)
            if hasattr(submod, 'scale'):
                return float(submod.scale)

        if curr.op == 'get_attr':
            return float(model.get_buffer(curr.target))

        curr = curr.args[0] if curr.args else None

    return None

# ====================================================================
#  FPGA Layer Replacement Functions
# ====================================================================
def transform_quantized_model_to_tpu(model, hw):
    graph = model.graph
    # 1. 대상 노드(matmul) 수집
    nodes_to_replace1 = []
    for k in graph.nodes:
        if k.op == 'call_module':
            try:
                submod = model.get_submodule(k.target)
                if isinstance(submod, qlinear.Linear):
                    nodes_to_replace1.append(k)
            except AttributeError:
                continue

    for node in nodes_to_replace1:
        # 2. 가중치 모듈 이름 추적
        submod = model.get_submodule(node.target)
        weight, bias = submod._packed_params._weight_bias()

        # [2] Output Scale/ZP (모듈 속성에 직접 있음)
        out_scale = submod.scale
        out_zp = submod.zero_point

        # [3] Input Scale (위에서 만든 안전한 함수 사용)
        # node.args[0]은 qkv의 입력인 ln_1이나 dropout일 것임
        input_scale = find_input_scale(node, model)
        x_scale = input_scale

        # 4. TPULinear 모듈 생성 및 가중치 로드
        if "mlp_3" in node.name:
            tpu_linear = TPULinear2(
                node.target,
                x_scale,
                weight,
                bias,
                out_scale,
                out_zp,
                hw)
        else:
            tpu_linear = TPULinear1(
                node.target,
                x_scale,
                weight,
                bias,
                out_scale,
                out_zp,
                hw)

        # 5. 모델에 등록 (이름은 중복 안 되게 유니크하게)
        tpu_module_name = f"tpu_{node.name}"
        setattr(model, tpu_module_name, tpu_linear)

        # 6. 그래프 노드 교체 (설계도 수정)
        with graph.inserting_before(node):
            # 입력 노드를 명확히 지정 (튜플 형태 유지)
            input_node = node.args[0]
            new_node = graph.call_module(tpu_module_name, args=(input_node,))

            # 7. 기존 노드의 사용처를 새 노드로 교체
            node.replace_all_uses_with(new_node)
        # 기존 matmul 노드 삭제
        graph.erase_node(node)

    # 7. 변경된 그래프로 모델 재컴파일
    model.graph.lint()
    model.recompile()

    print("[Replace] Linear-Nodes Complete")
    return model


def transform_conv_to_tpu(model, hw):
    graph = model.graph
    nodes_to_replace_conv = []

    for k in graph.nodes:
        if k.op == 'call_module' and k.name == 'conv_proj':
            try:
                submod = model.get_submodule(k.target)
                if isinstance(submod, torch.nn.quantized.Conv2d):
                    nodes_to_replace_conv.append(k)
            except AttributeError:
                pass
    for k in nodes_to_replace_conv:
        submod = model.get_submodule(k.target)
        # quantize_per_tensor 노드에서 x_scale 추출
        qpt_node = k.args[0]
        x_scale  = qpt_node.args[1]   # quantize_per_tensor의 scale

        tpu_module = TPUPatchEmbedding(
            name          = k.target,
            x_scale       = getattr(model,x_scale.target),
            weight_tensor = submod.weight(),    # per-channel quant 정보 포함
            bias_tensor   = submod.bias(),
            out_scale     = submod.scale,
            out_zp        = submod.zero_point,
            hw            = hw
        )

        tpu_target = k.target + '_tpu'
        model.add_submodule(tpu_target, tpu_module)

        with graph.inserting_after(k):
            input_node = k.args[0]
            new_node   = graph.call_module(tpu_target, args=(input_node,))
        k.replace_all_uses_with(new_node)
        graph.erase_node(k)

    model.recompile()
    model.graph.lint()

    print("[Replace] Conv-Nodes Complete")
    return model


def transform_mha_to_tpu(model, hw):
    graph = model.graph
    for layer_idx in range(12):
        qkv_target  = f'encoder.layers.encoder_layer_{layer_idx}.self_attention.qkv'
        proj_target = f'encoder.layers.encoder_layer_{layer_idx}.self_attention.proj'
        attn_drop_target = f'encoder.layers.encoder_layer_{layer_idx}.self_attention.attn_drop'
        # ── 노드 찾기 ──────────────────────────────
        qkv_node  = None
        proj_node = None
        for node in graph.nodes:
            if node.op == 'call_module':
                if node.target == qkv_target:
                    qkv_node  = node
                if node.target == proj_target:
                    proj_node = node

        if qkv_node is None or proj_node is None:
            continue
        # ── MHA 입력 = qkv의 입력 ─────────────────
        mha_input = qkv_node.args[0]
        proj_input = proj_node.args[0]
        energy_layer = proj_node.args[0].args[0].args[0].args[0].args[0].args[0].args[0].args[0].args[0].args[0]
        attention_output_layer = proj_node.args[0].args[0].args[0].args[0]
        attention_input_layer = proj_node.args[0].args[0].args[0].args[0].args[0].args[0]
        # ── TPUMultiHeadAttention 생성 ────────────
        qkv_mod       = model.get_submodule(qkv_target)
        proj_mod      = model.get_submodule(proj_target)
        tpu_mha = TPUMultiHeadAttention(
            qkv_module  = qkv_mod,
            proj_module = proj_mod,
            qkv_act_scale = model.get_submodule(mha_input.target).scale,
            qkv_input_act_zero = model.get_submodule(mha_input.target).zero_point,

            proj_act_scale = find_output_scale(proj_node.args[0].args[0].args[0].args[0],model)[0],
            energy_scale=model.get_buffer(energy_layer.args[2].target),
            energy_zero=model.get_buffer(energy_layer.args[3].target),
            attention_input_scale = model.get_buffer(attention_input_layer.args[1].target) ,
            attention_input_zero = model.get_buffer(attention_input_layer.args[2].target),
            attention_output_scale=model.get_buffer(attention_output_layer.args[2].target),
            attention_output_zero=model.get_buffer(attention_output_layer.args[3].target),
            num_heads   = 12,
            hw          = hw
        )

        tpu_name = f'tpu_mha_{layer_idx}'
        setattr(model, tpu_name, tpu_mha)

        # ── qkv ~ proj 사이 노드 수집 ────────────
        nodes_between = []
        collecting = False
        for node in graph.nodes:
            if node == qkv_node:
                collecting = True
            if collecting:
                nodes_between.append(node)
            if node == proj_node:
                break

        # ── 새 노드 삽입 ──────────────────────────
        with graph.inserting_after(proj_node):
            new_node = graph.call_module(
                tpu_name,
                args=(mha_input,)
            )
            proj_node.replace_all_uses_with(new_node)

        # ── 중간 노드 역순으로 제거 ───────────────
        for node in reversed(nodes_between):
            if len(node.users) == 0:
                graph.erase_node(node)

    graph.lint()
    model.recompile()
    return model

# ------------------------------------------------------------------------------
#  Residual Connection + LayerNorm
# ------------------------------------------------------------------------------

def collect_ln_params(model, hw):
    params = {}
    scale_prev = 1.0
    zp_prev    = 0
    for node in model.graph.nodes:
        is_start = "encoder_layer_0" in node.name
        is_ln1   = "ln_1" in node.name and not "fpga" in node.name
        is_ln2   = "ln_2" in node.name and not "fpga" in node.name
        is_end   = node.name == 'encoder_ln'

        if is_start and is_ln1:
            # LayerNorm Only
            mode = 0x00

            # Buffer Address Map
            #  - inp_a: data_b_buf (output of prev_module)
            #  - inp_b: data_c_buf (src_addr, prev add input)
            #  - out_a: data_b_buf (dst_addr, current add output)
            #  - out_b: result_buf
            src_addr = hw.ln_addr_c
            dst_addr = hw.ln_addr_b
            src_np = hw.data_c_np
            dst_np = hw.data_b_np

            # Quant Parameter Map
            #  - scale_a, zp_a: 이전 layer (add)의 출력 파라미터
            #  - scale_b, zp_b: 없음 (1.0, 0)
            #  - scale_c, zp_c: 없음 (1.0, 0)
            #  - scale_o, zp_o: LayerNorm 출력 파라미터

            add_node = node.args[0].args[0]
            ln_mod   = model.get_submodule(node.target)

            scale_a = float(getattr(model, add_node.args[2].target))
            zp_a    = int(getattr(model, add_node.args[3].target))

            scale_prev, zp_prev = scale_a, zp_a

            scale_b = 1.0
            zp_b    = 0

            scale_c = 1.0
            zp_c    = 0

            scale_o = float(ln_mod.scale)
            zp_o    = int(ln_mod.zero_point)

            params[node.name] = {
                'mode' : mode,
                'add_node' : add_node,
                'src_addr' : src_addr, 'dst_addr' : dst_addr,
                'src_np' : src_np,   'dst_np' : dst_np,
                'scale_a' : scale_a,  'zp_a' : zp_a,
                'scale_b' : scale_b,  'zp_b' : zp_b,
                'scale_c' : scale_c,  'zp_c' : zp_c,
                'scale_o' : scale_o,  'zp_o' : zp_o
            }

            continue

        if is_start and is_ln2:
            # Residual + LayerNorm Mode
            mode = 0x01

            # Buffer Address Map
            #  - inp_a: data_a_buf (output of prev_module)
            #  - inp_b: data_b_buf (src_addr, prev add input)
            #  - out_a: data_c_buf (dst_addr, current add output)
            #  - out_b: result_buf
            src_addr = hw.ln_addr_b
            dst_addr = hw.ln_addr_c
            src_np = hw.data_b_np
            dst_np = hw.data_c_np

            # Quant Parameter Map
            #  - scale_a, zp_a: MHA 출력 파라미터
            #  - scale_b, zp_b: 이전 ADD의 출력 파라미터
            #  - scale_c, zp_c: add 출력
            #  - scale_o, zp_o: LayerNorm 출력 파라미터
            add_node = node.args[0]
            ln_mod   = model.get_submodule(node.target)
            mha_mod  = model.get_submodule(add_node.args[0].args[0].args[0].target)

            scale_a = float(mha_mod.scale)
            zp_a    = int(mha_mod.zero_point)

            scale_b = scale_prev
            zp_b    = zp_prev

            scale_c = float(getattr(model, add_node.args[2].target))
            zp_c    = int(getattr(model, add_node.args[3].target))

            scale_prev, zp_prev = scale_c, zp_c

            scale_o = float(ln_mod.scale)
            zp_o    = int(ln_mod.zero_point)

            params[node.name] = {
                'mode' : mode,
                'add_node' : add_node,
                'src_addr' : src_addr, 'dst_addr' : dst_addr,
                'src_np' : src_np,   'dst_np' : dst_np,
                'scale_a' : scale_a,  'zp_a' : zp_a,
                'scale_b' : scale_b,  'zp_b' : zp_b,
                'scale_c' : scale_c,  'zp_c' : zp_c,
                'scale_o' : scale_o,  'zp_o' : zp_o
            }

            continue

        if is_ln1:
            # Residual + LayerNorm Mode
            mode = 0x01

            # Buffer Address Map
            #  - inp_a: data_a_buf (output of prev_module)
            #  - inp_b: data_c_buf (src_addr, prev add input)
            #  - out_a: data_b_buf (dst_addr, current add output)
            #  - out_b: result_buf
            src_addr = hw.ln_addr_c
            dst_addr = hw.ln_addr_b
            src_np = hw.data_c_np
            dst_np = hw.data_b_np

            # Quant Parameter Map
            #  - scale_a, zp_a: MHA 출력 파라미터
            #  - scale_b, zp_b: 이전 ADD의 출력 파라미터
            #  - scale_c, zp_c: add 출력
            #  - scale_o, zp_o: LayerNorm 출력 파라미터
            add_node = node.args[0]
            ln_mod   = model.get_submodule(node.target)
            mlp_out  = model.get_submodule(add_node.args[1].args[0].target)

            scale_a = float(mlp_out.scale)
            zp_a    = int(mlp_out.zero_point)

            scale_b = scale_prev
            zp_b    = zp_prev

            scale_c = float(getattr(model, add_node.args[2].target))
            zp_c    = int(getattr(model, add_node.args[3].target))

            scale_prev, zp_prev = scale_c, zp_c

            scale_o = float(ln_mod.scale)
            zp_o    = int(ln_mod.zero_point)

            params[node.name] = {
                'mode' : mode,
                'add_node' : add_node,
                'src_addr' : src_addr, 'dst_addr' : dst_addr,
                'src_np' : src_np,   'dst_np' : dst_np,
                'scale_a' : scale_a,  'zp_a' : zp_a,
                'scale_b' : scale_b,  'zp_b' : zp_b,
                'scale_c' : scale_c,  'zp_c' : zp_c,
                'scale_o' : scale_o,  'zp_o' : zp_o
            }

            continue

        if is_ln2:
            # Residual + LayerNorm Mode
            mode = 0x01

            # Buffer Address Map
            #  - inp_a: data_a_buf (output of prev_module)
            #  - inp_b: data_b_buf (src_addr, prev add input)
            #  - out_a: data_c_buf (dst_addr, current add output)
            #  - out_b: result_buf
            src_addr = hw.ln_addr_b
            dst_addr = hw.ln_addr_c
            src_np = hw.data_b_np
            dst_np = hw.data_c_np

            # Quant Parameter Map
            #  - scale_a, zp_a: MHA 출력 파라미터
            #  - scale_b, zp_b: 이전 ADD의 출력 파라미터
            #  - scale_c, zp_c: add 출력
            #  - scale_o, zp_o: LayerNorm 출력 파라미터
            add_node = node.args[0]
            ln_mod   = model.get_submodule(node.target)
            mha_out  = model.get_submodule(add_node.args[0].args[0].args[0].target)

            scale_a = float(mha_out.scale)
            zp_a    = int(mha_out.zero_point)

            scale_b = scale_prev
            zp_b    = zp_prev

            scale_c = float(getattr(model, add_node.args[2].target))
            zp_c    = int(getattr(model, add_node.args[3].target))

            scale_prev, zp_prev = scale_c, zp_c

            scale_o = float(ln_mod.scale)
            zp_o    = int(ln_mod.zero_point)

            params[node.name] = {
                'mode' : mode,
                'add_node' : add_node,
                'src_addr' : src_addr, 'dst_addr' : dst_addr,
                'src_np' : src_np,   'dst_np' : dst_np,
                'scale_a' : scale_a,  'zp_a' : zp_a,
                'scale_b' : scale_b,  'zp_b' : zp_b,
                'scale_c' : scale_c,  'zp_c' : zp_c,
                'scale_o' : scale_o,  'zp_o' : zp_o
            }

            continue

        if is_end:
            # Residual + LayerNorm Mode
            mode = 0x01

            # Buffer Address Map
            #  - inp_a: data_a_buf (output of prev_module)
            #  - inp_b: data_c_buf (src_addr, prev add input)
            #  - out_a: data_b_buf (dst_addr, current add output)
            #  - out_b: result_buf
            src_addr = hw.ln_addr_c
            dst_addr = hw.ln_addr_b
            src_np = hw.data_c_np
            dst_np = hw.data_b_np

            # Quant Parameter Map
            #  - scale_a, zp_a: MHA 출력 파라미터
            #  - scale_b, zp_b: 이전 ADD의 출력 파라미터
            #  - scale_c, zp_c: add 출력
            #  - scale_o, zp_o: LayerNorm 출력 파라미터
            add_node = node.args[0]
            ln_mod   = model.get_submodule(node.target)
            mlp_out  = model.get_submodule(add_node.args[1].args[0].target)

            scale_a = float(mlp_out.scale)
            zp_a    = int(mlp_out.zero_point)

            scale_b = scale_prev
            zp_b    = zp_prev

            scale_c = float(getattr(model, add_node.args[2].target))
            zp_c    = int(getattr(model, add_node.args[3].target))

            scale_prev, zp_prev = scale_c, zp_c

            scale_o = float(ln_mod.scale)
            zp_o    = int(ln_mod.zero_point)

            params[node.name] = {
                'mode' : mode,
                'add_node' : add_node,
                'src_addr' : src_addr, 'dst_addr' : dst_addr,
                'src_np'   : src_np,   'dst_np'   : dst_np,
                'scale_a' : scale_a, 'zp_a' : zp_a,
                'scale_b' : scale_b, 'zp_b' : zp_b,
                'scale_c' : scale_c, 'zp_c' : zp_c,
                'scale_o' : scale_o, 'zp_o' : zp_o
            }

            continue

    return params


def replace_ln_to_fpga(model, hw, params):
    for node in model.graph.nodes:
        is_start = "encoder_layer_0" in node.name
        is_ln1   = "ln_1" in node.name and not "fpga" in node.name
        is_ln2   = "ln_2" in node.name and not "fpga" in node.name
        is_end   = "encoder_ln" == node.name

        if is_ln1 or is_ln2 or is_end:
            param_layer = params[node.name]
            mode     = param_layer['mode']
            add_node = param_layer['add_node']
            src_addr = param_layer['src_addr']
            dst_addr = param_layer['dst_addr']
            src_np   = param_layer['src_np']
            dst_np   = param_layer['dst_np']
            scale_a  = param_layer['scale_a']
            scale_b  = param_layer['scale_b']
            scale_c  = param_layer['scale_c']
            scale_o  = param_layer['scale_o']
            zp_a     = param_layer['zp_a']
            zp_b     = param_layer['zp_b']
            zp_c     = param_layer['zp_c']
            zp_o     = param_layer['zp_o']

            ln_module = model.get_submodule(node.target)
            normalized_shape = ln_module.normalized_shape
            weight = ln_module.weight.data
            bias   = ln_module.bias.data

            fpga_ln = fusedResidualLayerNorm(
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
            )

            fpga_ln.weight.data.copy_(weight)
            fpga_ln.bias.data.copy_(bias)

            new_module_name = f"fpga_layernorm_{node.name}"
            model.add_module(new_module_name, fpga_ln)

            if is_ln1:
                if is_start:
                    arg = node.args[0]
                else:
                    arg = add_node.args[1]
            elif is_ln2:
                arg = add_node.args[0]
            elif is_end:
                arg = add_node.args[1]

            with model.graph.inserting_after(node):
                new_node = model.graph.call_module(
                    new_module_name,
                    args=(arg,))
            node.replace_all_uses_with(new_node)

            if is_start:
                model.graph.erase_node(node)
            else:
                nodes = []
                node_target = add_node
                while node_target.prev.name != node.name:
                    nodes.append(node_target)
                    node_target = node_target.next

                for n in reversed(nodes):
                    if len(n.users) == 0:
                        model.graph.erase_node(n)

    model.graph.eliminate_dead_code()
    model.graph.lint()
    model.recompile()

    print("[Replace] LayerNorm-Nodes Complete")
    return model