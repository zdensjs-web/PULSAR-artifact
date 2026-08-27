#!/usr/bin/env python3
from __future__ import annotations

import random


SLOTS = 1 << 15
WIDTHS = (16, 32, 64, 128, 256)


def encode(values: list[int], width: int) -> list[int]:
    words = SLOTS // width
    result = [0] * SLOTS
    for word, value in enumerate(values):
        for bit in range(width):
            result[bit * words + word] = (value >> bit) & 1
    return result


def decode(bits: list[int], width: int) -> list[int]:
    words = SLOTS // width
    result = [0] * words
    for word in range(words):
        for bit in range(width):
            result[word] |= bits[bit * words + word] << bit
    return result


def broadcast(values: list[int], width: int) -> list[int]:
    result = [0] * SLOTS
    words = len(values)
    for bit in range(width):
        start = bit * words
        result[start:start + words] = values
    return result


def prefix_equal(left: list[int], right: list[int], width: int) -> list[int]:
    words = SLOTS // width
    generate = [a ^ b for a, b in zip(left, right)]
    propagate = [1 - value for value in generate]
    propagate[:words] = [0] * words

    distance = 1
    while distance < width:
        next_generate = [0] * SLOTS
        next_propagate = [0] * SLOTS
        for bit in range(width):
            source_bit = (bit - distance) % width
            for word in range(words):
                target = bit * words + word
                source = source_bit * words + word
                next_generate[target] = (
                    generate[target]
                    | (propagate[target] & generate[source]))
                next_propagate[target] = (
                    propagate[target] & propagate[source])
        generate = next_generate
        propagate = next_propagate
        distance <<= 1

    tail = (width - 1) * words
    return broadcast([1 - generate[tail + word] for word in range(words)], width)


def main() -> None:
    randomizer = random.Random(0x50554C534152)
    for width in WIDTHS:
        words = SLOTS // width
        mask = (1 << width) - 1
        left = [randomizer.getrandbits(width) for _ in range(words)]
        right = [randomizer.getrandbits(width) for _ in range(words)]
        third = [randomizer.getrandbits(width) for _ in range(words)]
        for word in (4, words // 2, words - 1):
            right[word] = left[word]
        left_bits = encode(left, width)
        right_bits = encode(right, width)
        third_bits = encode(third, width)

        add_bits = encode([(a + b) & mask for a, b in zip(left, right)], width)
        xor_bits = [a ^ b for a, b in zip(left_bits, right_bits)]
        gt_words = [int(a > b) for a, b in zip(left, right)]
        gt_bits = broadcast(gt_words, width)
        eq_bits = prefix_equal(left_bits, right_bits, width)
        eq_words = [int(a == b) for a, b in zip(left, right)]
        mul_words = [(a * b) & mask for a, b in zip(left, right)]
        mixed_value_bits = [a ^ b for a, b in zip(add_bits, third_bits)]
        mixed_values = decode(mixed_value_bits, width)
        mixed_words = [int(value > a) for value, a in zip(mixed_values, left)]
        mixed_bits = broadcast(mixed_words, width)

        assert decode(add_bits, width) == [
            (a + b) & mask for a, b in zip(left, right)]
        assert decode(xor_bits, width) == [a ^ b for a, b in zip(left, right)]
        assert decode(encode(mul_words, width), width) == mul_words
        for bit in range(width):
            start = bit * words
            assert gt_bits[start:start + words] == gt_words
            assert eq_bits[start:start + words] == eq_words
            assert mixed_bits[start:start + words] == mixed_words
        print(
            f"width={width:3d} native_batch={words:4d} "
            "ADD/GT/EQ/XOR/MIXED/MUL PASS")
    print("overall verdict: PASS")


if __name__ == "__main__":
    main()
