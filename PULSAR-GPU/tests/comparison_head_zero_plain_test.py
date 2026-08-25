#!/usr/bin/env python3

import argparse
import random


WIDTHS = (8, 16, 32, 64, 128, 256)


def bits_lsb_first(value, width):
    return [(value >> bit) & 1 for bit in range(width)]


def verify_complex_identity():
    for q in (-1, 0):
        for g in (0, 1):
            for rotated_q in (-1, 0):
                for rotated_g in (0, 1):
                    current = q / 2.0 + 1j * g
                    previous = rotated_q / 2.0 + 1j * rotated_g
                    conjugate = current.conjugate()
                    actual = (
                        (current - conjugate) / 2.0
                        - (current + conjugate) * previous
                    )
                    expected = (
                        (-q * rotated_q) / 2.0
                        + 1j * (g - q * rotated_g)
                    )
                    if actual != expected:
                        raise RuntimeError(
                            "complex/integer channel identity failed")


def evaluate_underflow(width, minuend_words, subtrahend_words):
    words = len(minuend_words)
    minuend = []
    subtrahend = []
    for minuend_word, subtrahend_word in zip(
            minuend_words, subtrahend_words):
        minuend.extend(bits_lsb_first(minuend_word, width))
        subtrahend.extend(bits_lsb_first(subtrahend_word, width))

    slots = width * words
    g = [
        (1 - minuend[index]) * subtrahend[index]
        for index in range(slots)
    ]
    q = [
        (minuend[index] ^ subtrahend[index]) - 1
        for index in range(slots)
    ]

    # Q=0 is an absorbing delimiter for the affine prefix product.
    for word in range(words):
        q[word * width] = 0.0
    distance = 1
    while distance < width:
        old_q = q
        old_g = g
        q = [
            -old_q[index] * old_q[(index - distance) % slots]
            for index in range(slots)
        ]
        g = [
            old_g[index]
            - old_q[index] * old_g[(index - distance) % slots]
            for index in range(slots)
        ]
        distance *= 2

    underflow = []
    for word in range(words):
        msb = word * width + width - 1
        underflow.append(g[msb])
    return underflow


def evaluate_relations(width, a_words, b_words):
    # underflow(B-A) is A>B; underflow(A-B) is A<B.
    gt = evaluate_underflow(width, b_words, a_words)
    lt = evaluate_underflow(width, a_words, b_words)
    # The GPU implementation extracts -G in the existing multiplication and
    # then adds the public one-hot mask.
    le = [1 + (-value) for value in gt]
    ge = [1 + (-value) for value in lt]
    return gt, le, lt, ge


def evaluate_equality(width, a_words, b_words):
    words = len(a_words)
    state = []
    for a_word, b_word in zip(a_words, b_words):
        state.extend([
            a_bit ^ b_bit
            for a_bit, b_bit in zip(
                bits_lsb_first(a_word, width),
                bits_lsb_first(b_word, width))
        ])

    slots = width * words
    distance = 1
    while distance < width:
        previous = state
        state = [
            previous[index]
            | previous[(index - distance) % slots]
            for index in range(slots)
        ]
        distance *= 2

    not_equal = [
        state[word * width + width - 1]
        for word in range(words)
    ]
    # The encrypted EQ path extracts -NE and adds a public one-hot 1.
    equal = [1 + (-value) for value in not_equal]
    return equal, not_equal


def edge_cases(width):
    maximum = (1 << width) - 1
    high_bit = 1 << (width - 1)
    return [
        (0, 0),
        (0, 1),
        (1, 0),
        (maximum, maximum),
        (maximum, 0),
        (0, maximum),
        (high_bit, high_bit - 1),
        (high_bit - 1, high_bit),
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=1000)
    args = parser.parse_args()
    if args.trials < 1:
        raise ValueError("--trials must be positive")

    rng = random.Random(0x47544C4548454144)
    verify_complex_identity()
    for width in WIDTHS:
        words = 8
        fixed = edge_cases(width)
        for trial in range(args.trials):
            a_words = [rng.getrandbits(width) for _ in range(words)]
            b_words = [rng.getrandbits(width) for _ in range(words)]
            if trial == 0:
                for index, (a_value, b_value) in enumerate(fixed):
                    a_words[index] = a_value
                    b_words[index] = b_value

            actual_gt, actual_le, actual_lt, actual_ge = evaluate_relations(
                width, a_words, b_words)
            actual_eq, actual_ne = evaluate_equality(
                width, a_words, b_words)
            expected_gt = [
                int(a_value > b_value)
                for a_value, b_value in zip(a_words, b_words)
            ]
            expected_le = [
                int(a_value <= b_value)
                for a_value, b_value in zip(a_words, b_words)
            ]
            expected_lt = [
                int(a_value < b_value)
                for a_value, b_value in zip(a_words, b_words)
            ]
            expected_ge = [
                int(a_value >= b_value)
                for a_value, b_value in zip(a_words, b_words)
            ]
            expected_eq = [
                int(a_value == b_value)
                for a_value, b_value in zip(a_words, b_words)
            ]
            expected_ne = [
                int(a_value != b_value)
                for a_value, b_value in zip(a_words, b_words)
            ]
            if (actual_gt != expected_gt
                    or actual_le != expected_le
                    or actual_lt != expected_lt
                    or actual_ge != expected_ge
                    or actual_eq != expected_eq
                    or actual_ne != expected_ne):
                mismatch = next(
                    index for index in range(words)
                    if actual_gt[index] != expected_gt[index]
                    or actual_le[index] != expected_le[index]
                    or actual_lt[index] != expected_lt[index]
                    or actual_ge[index] != expected_ge[index]
                    or actual_eq[index] != expected_eq[index]
                    or actual_ne[index] != expected_ne[index]
                )
                raise RuntimeError(
                    f"width={width} trial={trial} word={mismatch} "
                    f"A={a_words[mismatch]:x} B={b_words[mismatch]:x} "
                    f"GT={actual_gt[mismatch]}/{expected_gt[mismatch]} "
                    f"LE={actual_le[mismatch]}/{expected_le[mismatch]} "
                    f"LT={actual_lt[mismatch]}/{expected_lt[mismatch]} "
                    f"GE={actual_ge[mismatch]}/{expected_ge[mismatch]} "
                    f"EQ={actual_eq[mismatch]}/{expected_eq[mismatch]} "
                    f"NE={actual_ne[mismatch]}/{expected_ne[mismatch]}"
                )
        print(
            f"{width:3d}-bit, {words:2d} packed words: "
            "GT/LE/LT/GE/EQ/NE PASS"
        )

    print("six-relation guard-free comparison equivalence: PASS")


if __name__ == "__main__":
    main()
