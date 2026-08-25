#!/usr/bin/env python3

import argparse
import math
import random


WIDTHS = (8, 16, 32, 64, 128, 256)


def rotate(values, index):
    size = len(values)
    return [values[(slot + index) % size] for slot in range(size)]


def vector_add(lhs, rhs):
    return [a + b for a, b in zip(lhs, rhs)]


def vector_sub(lhs, rhs):
    return [a - b for a, b in zip(lhs, rhs)]


def vector_mul(lhs, rhs):
    return [a * b for a, b in zip(lhs, rhs)]


def bits_of(value, width):
    return [(value >> bit) & 1 for bit in range(width)]


def value_of(bits):
    return sum(bit << index for index, bit in enumerate(bits))


def place_words(words, width):
    stride = 2 * width
    values = [0] * (len(words) * stride)
    for word, value in enumerate(words):
        values[word * stride : word * stride + width] = bits_of(value, width)
    return values


def active_values(values, width):
    stride = 2 * width
    return [
        value_of(values[word * stride : word * stride + width])
        for word in range(len(values) // stride)
    ]


def scalar_shift(values, width, amount, left):
    if amount >= width:
        return [0] * len(values)
    if amount == 0:
        return list(values)

    rotated = rotate(values, -amount if left else amount)
    mask = [0] * len(values)
    stride = 2 * width
    for base in range(0, len(values), stride):
        if left:
            for bit in range(amount, width):
                mask[base + bit] = 1
        else:
            for bit in range(width - amount):
                mask[base + bit] = 1
    return vector_mul(rotated, mask)


def encrypted_shift(values, shift_words, width, left):
    words = len(shift_words)
    stride = 2 * width
    control_bits = int(math.log2(width))
    block = 1 << (control_bits - 1).bit_length()

    encrypted_control = [0] * len(values)
    low_mask = [0] * len(values)
    for word, shift in enumerate(shift_words):
        base = word * stride
        for bit in range(control_bits):
            encrypted_control[base + bit] = (shift >> bit) & 1
            low_mask[base + bit] = 1
        if word & 1:
            encrypted_control[base + control_bits] = 1

    pattern = vector_mul(encrypted_control, low_mask)
    distance = block
    while distance < width:
        pattern = vector_add(pattern, rotate(pattern, -distance))
        distance *= 2

    current = list(values)
    for layer in range(control_bits):
        lane_mask = [0] * len(values)
        for word in range(words):
            base = word * stride
            for bit in range(layer, width, block):
                lane_mask[base + bit] = 1
        control = vector_mul(pattern, lane_mask)
        if layer:
            control = rotate(control, layer)

        distance = 1
        while distance < block:
            control = vector_add(control, rotate(control, -distance))
            distance *= 2

        shift_distance = 1 << layer
        candidate = rotate(
            current, -shift_distance if left else shift_distance
        )
        boundary_mask = [0] * len(values)
        for word in range(words):
            base = word * stride
            if left:
                for bit in range(shift_distance, width):
                    boundary_mask[base + bit] = 1
            else:
                for bit in range(width - shift_distance):
                    boundary_mask[base + bit] = 1
        candidate = vector_mul(candidate, boundary_mask)
        current = vector_add(
            current,
            vector_mul(control, vector_sub(candidate, current)),
        )
    return current


def check_equal(label, actual, expected):
    if actual != expected:
        raise AssertionError(
            f"{label}: actual={actual!r}, expected={expected!r}"
        )


def run_width(width, trials, generator):
    mask = (1 << width) - 1
    fixed_shifts = (0, 1, 2, width // 4, width // 2, width - 1)

    for trial in range(trials):
        words = 8
        lhs = [generator.getrandbits(width) for _ in range(words)]
        rhs = [generator.getrandbits(width) for _ in range(words)]
        slots = place_words(lhs, width)

        amount = (
            fixed_shifts[trial % len(fixed_shifts)]
            if trial < len(fixed_shifts)
            else generator.randrange(0, width + 2)
        )
        expected_left = [
            ((value << amount) & mask) if amount < width else 0
            for value in lhs
        ]
        expected_right = [
            (value >> amount) if amount < width else 0
            for value in lhs
        ]
        check_equal(
            f"scalar left width={width} trial={trial}",
            active_values(scalar_shift(slots, width, amount, True), width),
            expected_left,
        )
        check_equal(
            f"scalar right width={width} trial={trial}",
            active_values(scalar_shift(slots, width, amount, False), width),
            expected_right,
        )

        shifts = [
            fixed_shifts[word % len(fixed_shifts)]
            if trial == 0
            else generator.randrange(width)
            for word in range(words)
        ]
        check_equal(
            f"encrypted left width={width} trial={trial}",
            active_values(
                encrypted_shift(slots, shifts, width, True), width
            ),
            [(value << shift) & mask for value, shift in zip(lhs, shifts)],
        )
        check_equal(
            f"encrypted right width={width} trial={trial}",
            active_values(
                encrypted_shift(slots, shifts, width, False), width
            ),
            [value >> shift for value, shift in zip(lhs, shifts)],
        )

        check_equal(
            f"xor width={width} trial={trial}",
            active_values(
                [
                    a + b - 2 * a * b
                    for a, b in zip(
                        place_words(lhs, width),
                        place_words(rhs, width),
                    )
                ],
                width,
            ),
            [a ^ b for a, b in zip(lhs, rhs)],
        )
        check_equal(
            f"and width={width} trial={trial}",
            active_values(
                vector_mul(
                    place_words(lhs, width),
                    place_words(rhs, width),
                ),
                width,
            ),
            [a & b for a, b in zip(lhs, rhs)],
        )
        check_equal(
            f"or width={width} trial={trial}",
            active_values(
                [
                    a + b - a * b
                    for a, b in zip(
                        place_words(lhs, width),
                        place_words(rhs, width),
                    )
                ],
                width,
            ),
            [a | b for a, b in zip(lhs, rhs)],
        )
        check_equal(
            f"not width={width} trial={trial}",
            active_values(
                [
                    active - value
                    for active, value in zip(
                        place_words([mask] * words, width),
                        place_words(lhs, width),
                    )
                ],
                width,
            ),
            [(~value) & mask for value in lhs],
        )


def main():
    parser = argparse.ArgumentParser(
        description="Plaintext model for runtime CKKS word operators"
    )
    parser.add_argument("--trials", type=int, default=200)
    args = parser.parse_args()
    if args.trials < 1:
        parser.error("--trials must be positive")

    generator = random.Random(0x574F52444F505338)
    print("Runtime word-operator plaintext test")
    print("====================================")
    for width in WIDTHS:
        run_width(width, args.trials, generator)
        print(
            f"width={width:3d}: scalar/encrypted shifts and "
            "XOR/AND/OR/NOT PASS"
        )
    print(f"random trials per width: {args.trials}")
    print("overall verdict        : PASS")


if __name__ == "__main__":
    main()
