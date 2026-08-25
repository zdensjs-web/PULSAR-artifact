#!/usr/bin/env python3

from __future__ import annotations

import argparse
import random


def reciprocal_divide(n: int, d: int, bits: int) -> tuple[int, int, int]:
    mu = (1 << bits) // d
    q0 = (n * mu) >> bits
    r0 = n - q0 * d
    correction = int(r0 >= d)
    q = q0 + correction
    r = r0 - correction * d
    return q, r, correction


def blocked_public_product(
    value: int,
    scalar: int,
    scalar_bits: int,
    output_bits: int,
    blocks: int,
) -> int:
    result = 0
    blocks = min(blocks, scalar_bits)
    for block in range(blocks):
        begin = scalar_bits * block // blocks
        end = scalar_bits * (block + 1) // blocks
        shifted = value << begin
        for position in range(begin, end):
            if (scalar >> position) & 1:
                result += shifted
            shifted <<= 1
    return result & ((1 << output_bits) - 1)


def run_width(bits: int, trials: int, seed: int) -> None:
    rng = random.Random(seed ^ bits)
    limit = 1 << bits
    divisors = {
        1,
        2,
        3,
        5,
        7,
        limit // 2,
        limit - 2,
        limit - 1,
    }
    divisors.update(rng.randrange(1, limit) for _ in range(trials))

    checked = 0
    corrections = 0
    for d in divisors:
        samples = {
            0,
            1,
            d - 1,
            d,
            min(limit - 1, d + 1),
            limit - 2,
            limit - 1,
        }
        samples.update(rng.randrange(limit) for _ in range(trials))
        for n in samples:
            if not 0 <= n < limit:
                continue
            mu = limit // d
            for blocks in (1, 2, 4, 8):
                wide = blocked_public_product(
                    n, mu, bits + 1, 2 * bits, blocks
                )
                assert wide == n * mu
                q0_from_blocks = wide >> bits
                low_product = blocked_public_product(
                    q0_from_blocks, d, bits, bits, blocks
                )
                assert low_product == q0_from_blocks * d
            q, r, corrected = reciprocal_divide(n, d, bits)
            assert q == n // d
            assert r == n % d
            assert corrected in (0, 1)
            assert 0 <= r < d
            checked += 1
            corrections += corrected

    print(
        f"{bits:>3}-bit: PASS  cases={checked:<8} "
        f"one-step corrections={corrections}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test the exact one-correction public-scalar divider."
    )
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0x5343414C41524449)
    args = parser.parse_args()

    print("Exact reciprocal scalar-divider plaintext test")
    print("================================================")
    for bits in (8, 16, 32, 64, 128, 256):
        run_width(bits, args.trials, args.seed)
    print("\nidentity: q0=floor(N*floor(2^w/d)/2^w), q=q0 or q0+1")


if __name__ == "__main__":
    main()
