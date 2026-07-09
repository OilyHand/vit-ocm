import torch, torchvision
import torchvision.models as tvm
import torch.ao.quantization as tq
from   torch.utils.data import DataLoader, Subset
import argparse
import time
import os
import re
import sys
import traceback

from hardware.manager import FPGAManager
from model.quantization import buildQuant
from model.replace import transform_mha_to_tpu, transform_conv_to_tpu, \
                          transform_quantized_model_to_tpu, \
                          collect_ln_params, replace_ln_to_fpga
from collections import defaultdict



torch.backends.quantized.engine = "qnnpack"

# ==============================================================================
#  Setup Hooks
# ==============================================================================

block_times = defaultdict(list)
_block_start = {}

def make_pre_hook(name):
    def hook(module, input):
        _block_start[name] = time.perf_counter()
    return hook

def make_post_hook(name):
    def hook(module, input, output):
        elapsed = time.perf_counter() - _block_start[name]
        block_times[name].append(elapsed)
    return hook


# === Per-module latency profiler ===
class ModuleLatencyProfiler:
    """GraphModule 이 실제로 호출하는 call_module 노드마다 forward hook 을 걸어
    모듈별 latency 를 누적/집계한다 (모듈 코드 수정 불필요, 동기 실행이라 wall-clock 정확)."""

    # encoder block 내부 모듈 이름 규칙 -> (encoder layer 번호, 역할 라벨) 매핑용 정규식
    _MHA_RE           = re.compile(r'^tpu_mha_(\d+)$')
    _ENCODER_LAYER_RE = re.compile(r'encoder_layer_(\d+)')
    # encoder block 바깥(패치 임베딩/최종 LN/분류 head)의 개별 모듈 라벨
    _NON_BLOCK_ROLES = {
        'conv_proj_tpu':             'Conv2d (patch_embedding)',
        'encoder.dropout':           'Dropout (embedding)',
        'fpga_layernorm_encoder_ln': 'LayerNorm (encoder_ln, final)',
        'tpu_heads_head':            'Linear (classifier_head)',
    }
    # encoder block 하나의 내부 모듈 표준 실행 순서 (role 라벨 기준)
    _ENCODER_ROLE_ORDER = [
        "LayerNorm (ln_1)",
        "MHA (self_attention)",
        "Dropout (attn.proj_drop)",
        "Dropout (residual)",
        "LayerNorm (ln_2)",
        "Linear (mlp.fc1)",
        "GELU (mlp.act)",
        "Dropout (mlp.drop1)",
        "Linear (mlp.fc2)",
        "Dropout (mlp.drop2)",
    ]

    def __init__(self):
        self.times    = defaultdict(list)   # target -> [elapsed_sec]
        self.klass    = {}                  # target -> class name
        self.order    = []                  # 실제 forward 호출(그래프) 순서대로의 target 목록
        self._start   = {}
        self._handles = []

    @classmethod
    def _classify(cls, target):
        """모듈 경로(target) 문자열을 (encoder_layer_idx, 역할 라벨) 로 분류한다.
        encoder_layer_idx 가 None 이면 encoder block 바깥의 모듈이라는 뜻이다."""
        if target in cls._NON_BLOCK_ROLES:
            return None, cls._NON_BLOCK_ROLES[target]

        m = cls._MHA_RE.match(target)
        if m:
            return int(m.group(1)), "MHA (self_attention)"

        m = cls._ENCODER_LAYER_RE.search(target)
        if not m:
            return None, None
        layer_idx = int(m.group(1))

        if 'ln_1' in target:
            role = "LayerNorm (ln_1)"
        elif 'ln_2' in target:
            role = "LayerNorm (ln_2)"
        elif 'proj_drop' in target:
            role = "Dropout (attn.proj_drop)"
        elif target.endswith('.dropout') or target.endswith('_dropout'):
            role = "Dropout (residual)"
        elif 'mlp_0' in target or 'mlp.0' in target:
            role = "Linear (mlp.fc1)"
        elif 'mlp_1' in target or 'mlp.1' in target:
            role = "GELU (mlp.act)"
        elif 'mlp_2' in target or 'mlp.2' in target:
            role = "Dropout (mlp.drop1)"
        elif 'mlp_3' in target or 'mlp.3' in target:
            role = "Linear (mlp.fc2)"
        elif 'mlp_4' in target or 'mlp.4' in target:
            role = "Dropout (mlp.drop2)"
        else:
            role = None
        return layer_idx, role

    def _pre(self, name):
        def hook(module, inp):
            self._start[name] = time.perf_counter()
        return hook

    def _post(self, name):
        def hook(module, inp, out):
            self.times[name].append(time.perf_counter() - self._start[name])
        return hook

    def attach_called_modules(self, model):
        """그래프의 call_module 타깃(=실제 실행 단위)에만 훅을 건다."""
        seen = set()
        for node in model.graph.nodes:
            if node.op != "call_module" or node.target in seen:
                continue
            seen.add(node.target)
            try:
                sub = model.get_submodule(node.target)
            except AttributeError:
                continue
            self.klass[node.target] = type(sub).__name__
            self.order.append(node.target)
            self._handles.append(sub.register_forward_pre_hook(self._pre(node.target)))
            self._handles.append(sub.register_forward_hook(self._post(node.target)))
        return len(self._handles) // 2

    def report(self, skip=1, top=None):
        rows = []
        for name, ts in self.times.items():
            ts = ts[skip:] if len(ts) > skip else ts   # warm-up 배치 제외
            if not ts:
                continue
            n = len(ts)
            rows.append([name, self.klass.get(name, "?"), n,
                         sum(ts) / n * 1e3, min(ts) * 1e3, max(ts) * 1e3, sum(ts) * 1e3])
        if not rows:
            print("\n[Profiler] no samples collected")
            return
        rows.sort(key=lambda r: r[6], reverse=True)     # total 내림차순
        grand = sum(r[6] for r in rows) or 1e-9

        print("\n=== Per-module latency (total 기준 정렬, warm-up 제외) ===")
        print(f"{'module':44s} {'type':20s} {'cnt':>5} {'mean':>8} {'min':>8} {'max':>8} {'total':>10} {'%':>6}")
        for name, kls, n, mean, mn, mx, tot in (rows if top is None else rows[:top]):
            disp = name if len(name) <= 44 else "..." + name[-41:]
            print(f"{disp:44s} {kls:20s} {n:>5} {mean:>8.3f} {mn:>8.3f} {mx:>8.3f} {tot:>10.2f} {tot / grand * 100:>5.1f}%")

        type_tot = defaultdict(float)
        type_cnt = defaultdict(int)
        for name, kls, n, mean, mn, mx, tot in rows:
            type_tot[kls] += tot
            type_cnt[kls] += n
        print("\n=== Per-type latency (모듈 종류별 합계) ===")
        print(f"{'type':24s} {'calls':>7} {'total(ms)':>12} {'%':>7}")
        for kls in sorted(type_tot, key=type_tot.get, reverse=True):
            print(f"{kls:24s} {type_cnt[kls]:>7} {type_tot[kls]:>12.2f} {type_tot[kls] / grand * 100:>6.1f}%")

    def report_by_encoder(self, skip=1, log_path=None, block_times=None):
        """Hook 으로 추출한 latency 를, encoder 레이어별로 실제 forward 실행 순서
        (MHA -> LayerNorm -> GELU -> Dropout ... 순) 그대로 정리해서 보여준다.
        log_path 를 주면 동일한 내용을 텍스트 파일로도 저장한다."""

        def stat(target):
            ts = self.times.get(target, [])
            ts = ts[skip:] if len(ts) > skip else ts   # warm-up 배치 제외
            if not ts:
                return None
            n = len(ts)
            return n, sum(ts) / n * 1e3, min(ts) * 1e3, max(ts) * 1e3, sum(ts) * 1e3

        # 1) hook 이 걸린 모듈들을 그래프 실행 순서 그대로 encoder layer 별로 묶는다
        per_layer = defaultdict(list)   # layer_idx -> [(target, role)]
        others    = []                  # encoder block 바깥 모듈: [(target, role)]
        for target in self.order:
            layer_idx, role = self._classify(target)
            role = role or self.klass.get(target, "?")
            if layer_idx is None:
                others.append((target, role))
            else:
                per_layer[layer_idx].append((target, role))

        header = (f"  {'#':>3} {'role(역할)':30s} {'class':23s} {'module':42s} "
                  f"{'cnt':>5} {'mean(ms)':>9} {'min(ms)':>8} {'max(ms)':>8} {'total(ms)':>10}")
        sep = "-" * len(header)

        lines = []
        lines.append("=" * len(header))
        lines.append("Encoder 내부 모듈별 Latency 추출 과정 (hook 실행 순서 기준, warm-up 배치 제외)")
        lines.append("=" * len(header))

        def emit_block(title, entries):
            lines.append(f"\n[{title}]")
            lines.append(header)
            lines.append(sep)
            block_total = 0.0   # 이 구간 모든 모듈의 total(ms) 합
            mean_sum    = 0.0   # 이 구간 모든 모듈의 mean(ms) 합 (=1회 forward 근사 시간)
            for i, (target, role) in enumerate(entries, start=1):
                st = stat(target)
                if st is None:
                    continue
                n, mean, mn, mx, tot = st
                block_total += tot
                mean_sum    += mean
                kls  = self.klass.get(target, "?")
                disp = target if len(target) <= 42 else "..." + target[-39:]
                lines.append(f"  {i:>3} {role:30s} {kls:23s} {disp:42s} "
                             f"{n:>5} {mean:>9.3f} {mn:>8.3f} {mx:>8.3f} {tot:>10.2f}")
            lines.append(f"  -> {title}: 모듈별 mean 합계 = {mean_sum:.3f} ms/iter "
                         f"(warm-up 제외 전체 배치 total = {block_total:.2f} ms)")
            return block_total

        if not per_layer and not others:
            lines.append("\n[Profiler] 수집된 샘플이 없습니다.")
            report_text = "\n".join(lines)
            print(report_text)
            return report_text

        grand_total = 0.0
        for layer_idx in sorted(per_layer):
            title = f"Encoder Layer {layer_idx}"
            grand_total += emit_block(title, per_layer[layer_idx])

            # 참고용: encoder block 전체를 감싼 pre/post hook 실측값과 교차 검증
            blk_name = f"encoder.layers.encoder_layer_{layer_idx}"
            if block_times is not None and block_times.get(blk_name):
                ts = block_times[blk_name]
                ts = ts[skip:] if len(ts) > skip else ts
                if ts:
                    blk_mean = sum(ts) / len(ts) * 1e3
                    lines.append(f"     (참고) 이 encoder block 을 통째로 감싼 forward hook 실측 평균 "
                                 f"= {blk_mean:.3f} ms/iter")

        if others:
            grand_total += emit_block("Encoder 외부 모듈 (Patch Embedding / Head 등)", others)

        lines.append(f"\n전체 모듈 total 시간 합계: {grand_total:.2f} ms")

        report_text = "\n".join(lines)
        print("\n" + report_text)

        if log_path:
            out_dir = os.path.dirname(log_path)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            with open(log_path, "w", encoding="utf-8") as f:
                f.write(report_text + "\n")
            print(f"\n[Profiler] Encoder별 latency 분석 결과를 {log_path} 에 저장했습니다.")

        return report_text

    def report_encoder_min(self, acc=None, cur_latency_ms=None, min_latency_ms=None):
        """encoder 전체(12개 layer)를 통틀어, 각 내부 모듈(role)별로 지금까지 관측된
        '최소 latency' 를 찾아 실행 순서대로 한 표에 출력한다.
        같은 role(예: ln_1) 을 가진 12개 layer 모듈 중 가장 빠른 샘플을 대표로 뽑는다.
        매 iteration 마다 호출해서 갱신되는 최소값을 실시간으로 보여주는 용도다.
        반환값: (표 문자열, role별 min 합계 TOTAL(ms))"""

        # role -> 해당 role 에 속한 모든 layer 모듈 target 목록
        role_targets = defaultdict(list)
        for target in self.order:
            layer_idx, role = self._classify(target)
            if layer_idx is None:                 # patch embedding / head 등 encoder 바깥 모듈 제외
                continue
            if role in self._ENCODER_ROLE_ORDER:
                role_targets[role].append(target)

        header = ("  {:>3} {:30s} {:23s} {:42s} {:>5} {:>10}"
                  .format("#", "role", "class", "module", "cnt", "min(ms)"))
        sep = "-" * len(header)

        lines = []
        if acc is not None:
            lines.append(f"Accuracy: {acc:.4f} | Current Latency: {cur_latency_ms:.3f} ms | "
                         f"Minimum Latency: {min_latency_ms:.3f} ms")
        lines.append(sep)
        lines.append(header)
        lines.append(sep)

        total = 0.0
        idx = 0
        for role in self._ENCODER_ROLE_ORDER:
            # 이 role 을 가진 12개 layer 모듈 중, 단일 최소 latency 샘플이 가장 작은 모듈을 대표로 선택
            best_min = None
            best_target = None
            best_cnt = 0
            for target in role_targets.get(role, []):
                ts = self.times.get(target, [])
                if not ts:
                    continue
                m = min(ts)
                if best_min is None or m < best_min:
                    best_min, best_target, best_cnt = m, target, len(ts)
            if best_min is None:
                continue
            idx += 1
            min_ms = best_min * 1e3
            total += min_ms
            kls  = self.klass.get(best_target, "?")
            disp = best_target if len(best_target) <= 42 else "..." + best_target[-39:]
            lines.append("  {:>3} {:30s} {:23s} {:42s} {:>5} {:>10.3f}"
                         .format(idx, role, kls, disp, best_cnt, min_ms))

        lines.append(f"TOTAL: {total:.3f}".rjust(len(header)))
        text = "\n".join(lines)
        print(text)
        return text, total

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()


def parse_args():
    parser = argparse.ArgumentParser(description="ViT INT8 Inference Script")
    parser.add_argument("--model_path", type=str, default="models/vit_qat_int8_custom.pt", help="Path to the converted INT8 checkpoint")
    parser.add_argument("--batch_size", type=int, default=8,                                help="Batch size for inference")
    parser.add_argument("--device",     type=str, default="cpu",                           help="Device to run inference on (cpu/cuda)")
    parser.add_argument("--log_path",   type=str, default="./log/infer_int8.csv",          help="Path to save inference logs")
    parser.add_argument("--latency_log_path", type=str, default="./log/latency.log",       help="Path to save per-module/per-encoder latency analysis")
    parser.add_argument("--use_hw",     action="store_true",                               help="Enable FPGA Hardware Acceleration")
    parser.add_argument("--hw_path", type=str, default="../hardware/FINAL.xsa",             help="Path to hardware xsa file")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # -------------------------------------------------------------------------
    # 1. Setup Environment
    # -------------------------------------------------------------------------
    device = torch.device(args.device)
    print(f"[Init] Device: {device}")
    print(f"[Init] Loading Model from: {args.model_path}")

    # -------------------------------------------------------------------------
    # 2. Load Data
    # -------------------------------------------------------------------------
    preprocess = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5071, 0.4867, 0.4408),
                                         (0.2675, 0.2565, 0.2761))
    ])

    test_set = torchvision.datasets.CIFAR100(
        root="../",
        train=False,
        download=False,
        transform=preprocess)

    indices = list(range(4096))
    test_set = Subset(test_set, indices)

    test_loader = DataLoader(
        dataset=test_set,
        batch_size=args.batch_size,
        shuffle=False)

    # -------------------------------------------------------------------------
    # 3. Load Model
    # -------------------------------------------------------------------------
    hw = FPGAManager(args.hw_path, args.batch_size)

    model = tvm.vit_b_16(weights=None, num_classes=100)
    model = buildQuant(model, batch_size=args.batch_size, qbackend="qnnpack")
    model = tq.quantize_fx.convert_fx(model)

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Checkpoint not found at {args.model_path}")

    state_dict = torch.load(args.model_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)
    model.to(device).eval()

    try:
        ln_params = collect_ln_params(model, hw)
        model = transform_mha_to_tpu(model, hw)
        model = transform_conv_to_tpu(model,hw)
        model = transform_quantized_model_to_tpu(model,hw)
        model = replace_ln_to_fpga(model, hw, ln_params)

        with open('log/model.txt', 'w', encoding='utf-8') as f:
            print(model.code, file=f)

        import gc
        gc.collect()

    except Exception as e:
        traceback.print_exc()
        exit()

    except KeyboardInterrupt:
        del hw
        print("[Keyboard Interrupt] Exit")
        exit()

    # -------------------------------------------------------------------------
    # 4. Inference Loop
    # -------------------------------------------------------------------------
    blocks_to_hook = []
    for name, module in model.named_modules():
        if re.match(r'^encoder\.layers\.encoder_layer_\d+$', name):
            blocks_to_hook.append((name, module))

    print(f"Hooked {len(blocks_to_hook)} transformer blocks")

    hooks = []
    for name, block in blocks_to_hook:
        h1 = block.register_forward_pre_hook(make_pre_hook(name))
        h2 = block.register_forward_hook(make_post_hook(name))
        hooks.extend([h1, h2])

    print(f"Registered {len(hooks)} hooks")

    # 그래프가 실제 호출하는 모든 call_module 에 per-module 프로파일러 부착
    mod_prof = ModuleLatencyProfiler()
    n_mods = mod_prof.attach_called_modules(model)
    print(f"[Profiler] Hooked {n_mods} call_module targets")

    correct = 0
    total = 0
    total_inference_time = 0.0
    min_latency_ms = float('inf')   # 전체 iteration 을 통틀어 관측된 최소(샘플당) latency


    print("===============================================================")
    print(" ***             Starting Inference on TestSet             *** ")
    print("===============================================================")

    try:
        with torch.inference_mode():
            warmup_done = False
            total_batches = len(test_loader)
            batch_count = 0
            for imgs, labels in test_loader:
                batch_count += 1
                imgs = imgs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                # 시간 측정
                start_time = time.perf_counter()
                preds = model(imgs)
                end_time = time.perf_counter()

                # 배치 처리 시간 누적
                batch_time = end_time - start_time
                total_inference_time += batch_time

                # 정확도 계산
                pred_cls = preds.argmax(dim=1)
                correct += (pred_cls == labels).sum().item()
                total += labels.size(0)

                # 현재 정확도 / 현재(샘플당) latency / 지금까지의 최소 latency 계산
                current_acc    = correct / total
                cur_latency_ms = (batch_time * 1000) / labels.size(0)   # 이번 배치의 샘플당 latency
                min_latency_ms = min(min_latency_ms, cur_latency_ms)    # 전체 iteration 최소 latency

                # === encoder 전체를 통틀어 각 내부 모듈(role)별 최소 latency 표 출력 (매 iteration) ===
                mod_prof.report_encoder_min(
                    acc=current_acc,
                    cur_latency_ms=cur_latency_ms,
                    min_latency_ms=min_latency_ms,
                )

            # === 최종 block별 평균 출력 ===
            print("\n=== Per-block timing (avg) ===")
            for name, _ in blocks_to_hook:
                ts = block_times[name]
                if ts:
                    avg_ms = sum(ts) / len(ts) * 1000
                    print(f"  {name}: {avg_ms:.3f} ms")

            # 전체 transformer 평균
            all_block_times = [t for ts in block_times.values() for t in ts]
            if all_block_times:
                print(f"\nAll blocks avg: {sum(all_block_times)/len(all_block_times)*1000:.3f} ms")

    except KeyboardInterrupt:
        del hw
        print("[Keyboard Interrupt] Exit")
        exit()

    except Exception as e:
        traceback.print_exc()

    finally:
        # 모듈별 latency 요약 (성공/중단/예외 관계없이 항상 출력)
        mod_prof.report(skip=1, top=40)
        # encoder 레이어별로 내부 모듈(MHA, LayerNorm, GELU, Dropout ...)의 latency 추출 과정을
        # 실행 순서대로 출력하고, log/latency.log 로 저장
        mod_prof.report_by_encoder(skip=1, log_path=args.latency_log_path, block_times=block_times)
        for h in hooks:
            h.remove()
        mod_prof.remove()
        print(f"\n[Cleanup] Removed {len(hooks)} block hooks + module profiler hooks")

    # -------------------------------------------------------------------------
    # 5. Final Report
    # -------------------------------------------------------------------------
    final_acc = correct / total
    final_avg_latency_ms = (total_inference_time / total) * 1000

    print("\n===============================================================")
    print(f" [Result] Final Accuracy: {final_acc:.4f} ({correct}/{total})")
    print(f" [Result] Avg Latency   : {final_avg_latency_ms:.4f} ms/sample")
    print(f" [Result] Total Time    : {total_inference_time:.4f} sec")
    print("===============================================================")
