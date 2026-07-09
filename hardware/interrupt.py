import time
import asyncio


def Interrupt_write(INTERRUPT):

    IER_OFFSET = 0x08
    MER_OFFSET = 0x1c
    INTERRUPT.write(IER_OFFSET, 0x1)
    INTERRUPT.write(MER_OFFSET, 0x3)
    read_val1 = INTERRUPT.read(IER_OFFSET)
    read_val2 = INTERRUPT.read(MER_OFFSET)

    if read_val1 == 0x1 and read_val2 == 0x3:
        return 1
    else:
        return 0


async def interrupt_monitor(INTERRUPT, num_events, timeout_ms=5000, ocm_regions=None):
    """
    ocm_regions: [(ptr, size), ...] 각 TPU의 OCM 버퍼 주소/크기
    신호 포착 시 해당 OCM cache invalidate
    """
    ISR_OFFSET = 0x00
    IAR_OFFSET = 0x0C
    check_interval = 0.0001
    start_wait = time.perf_counter()
    acc_reg    = 0
    prev_acc   = 0
    target_flag = (1 << num_events) - 1

    while True:
        reg_val  = INTERRUPT.read(ISR_OFFSET)
        acc_reg |= reg_val

        # 새로 포착된 비트만 확인
        new_bits = acc_reg & ~prev_acc
        if new_bits and ocm_regions:
            for i in range(num_events):
                if new_bits & (1 << i):
                    ptr, size = ocm_regions[i]
                    _invalidate_cache(ptr, size)   # ← 해당 OCM만 invalidate

        prev_acc = acc_reg

        if (acc_reg & target_flag) == target_flag:
            INTERRUPT.write(IAR_OFFSET, acc_reg)
            yield acc_reg
            return

        if (time.perf_counter() - start_wait) > (timeout_ms / 1000):
            print(f"[IRQ] Polling Timeout {bin(acc_reg)}")
            yield None
            return

        await asyncio.sleep(check_interval)


# ARM64 전용 cache invalidate
def _invalidate_cache(ptr, size):
    # PYNQ의 libxlnk_cma.so 사용
    import ctypes

    libc = ctypes.CDLL("libc.so.6")
    # msync로 동기화
    libc.msync(ctypes.c_void_p(ptr),
               ctypes.c_size_t(size),
               ctypes.c_int(4))  # MS_INVALIDATE = 4
