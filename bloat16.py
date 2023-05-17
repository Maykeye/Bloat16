from typing import Any
from typing_extensions import SupportsIndex
import torch


def bitcount_i16(n: torch.Tensor):
    """Primitive way of bitcount"""
    assert n.dtype == torch.int16
    mask = (1 << torch.arange(16))
    return ((n & mask) != 0).int().sum()


def bf16_to_i16_with_common_mask(x0: torch.Tensor):
    """Bitcast 1D tensor of BF16 to 1D tensor of 
    I16 and a common mask. 
    A common mask is such that 
        (x0 & mask) == common 
    for every element of x0, in other words - 
    common mask contains bits(set or not set) that
    are constant across whole i16
    """
    assert x0.dtype == torch.bfloat16
    assert x0.dim() == 1
    x0i16 = x0.view(dtype=torch.int16)
    mask = torch.tensor(0, dtype=torch.int16)
    common = torch.tensor(0, dtype=torch.int16)

    for i in range(16):
        # calculate number of bits set at i-th position
        summed = ((x0i16 & (1 << i)) != 0).int().sum()
        if summed == 0:  # not a single bit was set
            mask |= 1 << i
        if summed == len(x0i16):  # all bits were set
            mask |= 1 << i
            common |= 1 << i
    return x0i16, mask, common


def pack_by_mask(xi: torch.IntTensor, pack_mask: torch.int16):
    """Pack tensor according to pack_mask.
    xi need to be divisible by 8

    Assume pack_mask ask to pack bits 0,1,3, then
    buffer[0...N] contains bit 0 of xi[0], xi[1]...
    buffer[N...2N] contains bit 1,
    buffer[2N...3N] contains bit 3
    Bit 2, 4-16 are skipped and not stored in the buffer 
    according to pack_mask
    """
    assert xi.dtype == torch.int16
    assert xi.dim() == 1
    assert len(xi) % 8 == 0

    coeffs = 1 << torch.arange(8)
    bit_row_length = len(xi) // 8
    used_bytes = bitcount_i16(pack_mask) * bit_row_length
    buffer = torch.zeros(used_bytes, dtype=torch.uint8)
    n = 0
    for i in range(16):
        bit_mask = 1 << i
        if (pack_mask & bit_mask) == 0:
            continue
        sbit = ((xi & bit_mask) != 0).int()
        sbit = (sbit.reshape(-1, 8) * coeffs).sum(-1)
        buffer[n:n+bit_row_length] = sbit
        n += bit_row_length
    assert n == len(buffer), "buffer underflow"
    return buffer


def unpack_by_mask(x: torch.ByteTensor, unpack_mask: torch.int16):
    """Reverse pack_by_mask"""
    n = 0
    assert (8 * len(x) % bitcount_i16(unpack_mask)) == 0
    used_i16 = 8 * len(x) // bitcount_i16(unpack_mask)
    xi = torch.zeros(used_i16, dtype=torch.int16)
    assert used_i16 % 8 == 0
    bit_row_length = used_i16 // 8
    n = 0
    coeffs = (1 << torch.arange(8, dtype=torch.uint8))
    for i in range(16):
        bit_mask = 1 << i
        if (unpack_mask & bit_mask) == 0:
            continue
        ith_bit = x[n:n+bit_row_length]
        ith_bit = ith_bit.reshape(-1, 1).repeat(1, 8)
        ith_bit = ith_bit & coeffs
        ith_bit = ith_bit.ravel()
        xi[ith_bit != 0] |= bit_mask
        n += bit_row_length
    assert n == len(x), "buffer underflow"
    return xi


class Bloat16:
    def __init__(self, xi, common_mask, common_bits, orig_shape, dtype) -> None:
        assert xi.dtype == torch.uint8
        self.xi = xi
        self.common_mask = common_mask
        self.common_bits = common_bits
        self.orig_shape = orig_shape
        self.dtype = dtype

    def __reduce_ex__(self, __protocol: SupportsIndex) -> str | tuple[Any, ...]:
        return (Bloat16.unpack,
                (self.xi,
                 self.common_mask,
                 self.common_bits,
                 self.orig_shape,
                 self.dtype))

    @staticmethod
    def unpack(xi, common_mask, common_bits, shape, dtype):
        xi = unpack_by_mask(xi, ~common_mask) | common_bits
        x = xi.reshape(*shape).view(dtype=torch.bfloat16)
        x = x.to(dtype=dtype)
        return x

    @staticmethod
    def from_tensor(t: torch.Tensor, dtype=None):
        assert t.dtype.is_floating_point
        if dtype:
            assert dtype.is_floating_point
        x0 = t.bfloat16().ravel()
        xi16, mask, common = bf16_to_i16_with_common_mask(x0)
        xi8 = pack_by_mask(xi16, ~mask)
        return Bloat16(xi8, mask, common, t.shape, dtype or t.dtype)


def run():
    import sys
    assert len(sys.argv), "usage: python bloat16 <pytorchmodel.bin filename>"
    fname = sys.argv[1]
    m = torch.load(fname)
    for name, layer in m.items():
        print(name)
        assert isinstance(layer, torch.Tensor)
        if layer.dtype.is_floating_point:
            if layer.numel() % 8 != 0:
                print(f"*** {name} SKIPPED: {layer.numel()} not divisible by 8")
                continue
            m[name] = Bloat16.from_tensor(layer, torch.float16)
    torch.save(m, fname+".out")


if __name__ == "__main__":
    run()

# transformer.h.12.mlp.fc_in.weight: nan + inf
