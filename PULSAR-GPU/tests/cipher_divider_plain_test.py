#!/usr/bin/env python3

from __future__ import annotations

import argparse
import random


def midpoint_gp_subtract(left: int, right: int, width: int) -> tuple[int, int]:
    """Model the split generate/propagate path used by each trial."""
    a = [(left >> bit) & 1 for bit in range(width)]
    b = [(right >> bit) & 1 for bit in range(width)]
    generate = [(1 - a[bit]) * b[bit] for bit in range(width)]
    propagate = [1 - (a[bit] ^ b[bit]) for bit in range(width)]

    layers = width.bit_length() - 1
    for layer in range(layers):
        distance = 1 << layer
        old_g = generate
        old_p = propagate
        generate = old_g.copy()
        propagate = old_p.copy()
        for bit in range(distance, width):
            generate[bit] = old_g[bit] | (
                old_p[bit] & old_g[bit - distance]
            )
            propagate[bit] = old_p[bit] & old_p[bit - distance]
        if layer == 3:
            # Bootstrap followed by Boolean projection is the identity over
            # exact plaintext bits; this marks the encrypted noise boundary.
            assert all(value in (0, 1) for value in generate)
            assert all(value in (0, 1) for value in propagate)

    borrow_in = [0] + generate[:-1]
    difference = [
        a[bit] ^ b[bit] ^ borrow_in[bit]
        for bit in range(width)
    ]
    value = sum(bit << index for index, bit in enumerate(difference))
    return value, generate[-1]


def radix16_divide(dividend: int, divisor: int, bits: int) -> tuple[int, int]:
    quotient = 0
    remainder = 0
    rounds = bits // 4
    for round_index in range(rounds):
        source = bits - 4 * (round_index + 1)
        nibble = (dividend >> source) & 0xF
        state = (remainder << 4) | nibble
        digit = 0
        for bit in range(3, -1, -1):
            weight = 1 << bit
            candidate = state - weight * divisor
            borrow = int(candidate < 0)
            if not borrow:
                state = candidate
                digit |= weight
        quotient |= digit << source
        remainder = state

    if divisor == 0:
        assert quotient == (1 << bits) - 1
        assert remainder == dividend
        quotient = 0
    return quotient, remainder


def verify_cursor_schedule(dividend: int, bits: int) -> None:
    mask = (1 << bits) - 1
    cursor = dividend
    for round_index in range(bits // 4):
        source = bits - 4 * (round_index + 1)
        expected = (dividend >> source) & 0xF
        observed = (cursor >> (bits - 4)) & 0xF
        assert observed == expected
        cursor = (cursor << 4) & mask


def run_width(bits: int, trials: int, seed: int) -> None:
    rng = random.Random(seed ^ bits)
    limit = 1 << bits
    cases = {
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
        (limit - 1, 0),
        (limit - 1, 1),
        (limit - 1, limit - 1),
        (limit - 2, limit - 1),
    }
    for _ in range(trials):
        cases.add((rng.randrange(limit), rng.randrange(limit)))

    for dividend, divisor in cases:
        verify_cursor_schedule(dividend, bits)
        quotient, remainder = radix16_divide(
            dividend, divisor, bits
        )
        if divisor == 0:
            expected_q, expected_r = 0, dividend
        else:
            expected_q, expected_r = divmod(dividend, divisor)
        assert quotient == expected_q
        assert remainder == expected_r
        assert 0 <= quotient < limit
        assert 0 <= remainder < limit

    wide_limit = 1 << (2 * bits)
    for _ in range(min(trials, 128)):
        left = rng.randrange(wide_limit)
        right = rng.randrange(wide_limit)
        difference, borrow = midpoint_gp_subtract(
            left, right, 2 * bits
        )
        assert difference == (left - right) % wide_limit
        assert borrow == int(left < right)

    print(f"{bits:>3}-bit: PASS  cases={len(cases)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test the exact radix-16 encrypted-divider recurrence."
    )
    parser.add_argument("--trials", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0x4349504845524449)
    args = parser.parse_args()

    print("Exact radix-16 ciphertext-divider plaintext test")
    print("================================================")
    for bits in (8, 16, 32, 64, 128, 256):
        run_width(bits, args.trials, args.seed)
    print("\nrecurrence: T=16R+nibble; trials=8D,4D,2D,D")
    print("cursor schedule: PASS")


if __name__ == "__main__":
    main()
