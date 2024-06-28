import torch
from tqdm import tqdm
import colorama


def num2suf(number):
    suffixes = ['', 'K', 'M', 'B', 'T']
    i = 0
    while number >= 1000 and i < len(suffixes)-1:
        i += 1
        number /= 1000
    return f'{number:.1f}{suffixes[i]}'


def bitcount_i16(n: torch.Tensor):
    """Primitive way of bitcount"""
    assert n.dtype == torch.int16
    mask = (1 << torch.arange(16))
    return ((n & mask) != 0).int().sum()


def bf16_mask(x0: torch.Tensor):
    """Count non-uniqe bits of 1D BF16 tensor"""
    # `assert x0.dtype == torch.bfloat16
    assert x0.dim() == 1
    x0i16 = x0.view(dtype=torch.int16)
    mask = torch.tensor(0, dtype=torch.int32)
    stats = [0 for _ in range(16)]

    for i in range(16):
        # calculate number of bits set at i-th position
        summed = ((x0i16 & (1 << i)) != 0).int().sum()
        stats[i] = float(summed) / len(x0i16)
        if summed == 0:  # not a single bit was set
            mask |= 1 << i
        if summed == len(x0i16):  # all bits were set
            mask |= 1 << i
    return mask, stats


def format_stats(stats):
    for i in range(len(stats)):
        fg = ""
        if stats[i] > 0.98 or stats[i] < 0.02:
            fg = colorama.Fore.LIGHTCYAN_EX
        if stats[i] in (0.0, 1.0):
            fg = colorama.Fore.LIGHTGREEN_EX
        stats[i] = f"{fg}{stats[i]:.3f}{colorama.Fore.RESET}"
    return stats

def calc_strides(layer, n):
    assert layer.numel() % n == 0
    z = layer.reshape(-1 ,n)
    neg = z <= 0
    pos = z >= 0
    neg = neg.all(-1)
    pos = pos.all(-1)
    res = neg | pos
    return (res.sum().item(), len(res))


def stats_of_file(fname):
    """Load model dict and count everything per layer"""
    m = torch.load(fname)
    if isinstance(m, torch.Tensor):
        m = {fname: m}
    del_me = []
    print("Prep")
    for layer_name in tqdm(m):
        layer: torch.Tensor = m[layer_name]
        if not layer.is_floating_point():
            del_me.append(layer_name)
        m[layer_name] = layer.to(dtype=torch.bfloat16).ravel()
    for layer_name in del_me:
        del m[layer_name]
    print("Calc stat")
    for layer_name, layer in m.items():
        bitmask, stats = bf16_mask(layer)
        strides = calc_strides(layer, 16)
        bitmask = f"{bitmask:016b}"[::-1]
        bitmask += colorama.Fore.RESET
        bitmask = bitmask.replace("1",
                                  f"{colorama.Fore.LIGHTGREEN_EX}1{colorama.Fore.RESET}")
        print(f"{layer_name:42}: {num2suf(layer.numel()):6} {bitmask}", end="")
        for fmt in format_stats(stats):
            print(f" {fmt}", end=f"")
        print(strides)



if __name__ == "__main__":
    import sys
    assert len(sys.argv) == 2, "bloat16 <filename>"
    stats_of_file(sys.argv[1])
