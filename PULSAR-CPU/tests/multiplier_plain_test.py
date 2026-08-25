#!/usr/bin/env python3
import random


SLOTS = 32768
WIDTHS = (16, 32, 64, 128, 256)


def lazy_passes(digits: int) -> int:
    maximum = digits * 15 * 15
    passes = 0
    while maximum >= 31:
        maximum = maximum // 16 + 15
        passes += 1
    return passes


def normalize(values, passes):
    for _ in range(passes):
        values = [
            (value & 15) + (values[index - 1] >> 4 if index else 0)
            for index, value in enumerate(values)
        ]
    return values


def gate_width(bits: int) -> None:
    native_batch = SLOTS // bits
    aggregate_batch = 2 * native_batch
    digits = bits // 4
    radix_slice = 2 * digits
    radix_batch = SLOTS // radix_slice
    passes = lazy_passes(digits)
    prefix_layers = digits.bit_length() - 1

    assert radix_batch == aggregate_batch
    assert radix_slice * radix_batch == SLOTS
    assert passes == (2 if bits <= 64 else 3)

    seen = [0] * SLOTS
    for group in range(2):
        for digit in range(digits):
            for residue in range(4):
                for word in range(native_batch):
                    source = (4 * digit + residue) * native_batch + word
                    target = digit * radix_batch + group * native_batch + word
                    assert 0 <= source < SLOTS
                    assert 0 <= target < SLOTS // 2
                    seen[target] += 1
    assert all(count == 4 for count in seen[: SLOTS // 2])
    assert all(count == 0 for count in seen[SLOTS // 2 :])

    rng = random.Random(0xD35E0000 + bits)
    mask = (1 << bits) - 1
    cases = [
        (0, 0),
        (mask, 0),
        (0, mask),
        (mask, 1),
        (mask, mask),
        ((1 << (bits - 1)) - 1, 2),
    ]
    cases.extend((rng.getrandbits(bits), rng.getrandbits(bits))
                 for _ in range(1000))
    for left, right in cases:
        a = [(left >> (4 * i)) & 15 for i in range(digits)]
        b = [(right >> (4 * i)) & 15 for i in range(digits)]
        convolution = [
            sum(a[j] * b[i - j] for j in range(i + 1))
            for i in range(digits)
        ]
        lazy = normalize(convolution, passes)
        assert max(lazy) <= 30
        carry = 0
        recovered = 0
        for digit, value in enumerate(lazy):
            residue = value & 15
            generate = int(value >= 16)
            propagate = int(value == 15)
            corrected = (residue + carry) & 15
            recovered |= corrected << (4 * digit)
            carry = generate | (propagate & carry)
        assert recovered == (left * right) & mask

    raw_level = 18 + prefix_layers + 3 + 1
    assert raw_level <= 30
    print(f"{bits:3d}-bit: native={native_batch:4d} aggregate={aggregate_batch:4d} "
          f"digits={digits:2d} LC={passes} prefix={prefix_layers} "
          f"raw_level={raw_level} PASS")


def main() -> None:
    print("PULSAR Q33/43 dense radix multiplier gate")
    print("===========================================")
    for bits in WIDTHS:
        gate_width(bits)
    print("overall verdict: PASS")


if __name__ == "__main__":
    main()
