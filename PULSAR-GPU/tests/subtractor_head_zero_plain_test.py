#!/usr/bin/env python3

import argparse
import random


def rotate_from_lower(values, distance):
    size = len(values)
    return [values[(index - distance) % size] for index in range(size)]


def evaluate(bits, words, a, b):
    slots = bits * words
    g = [float((1 - a[i]) * b[i]) for i in range(slots)]
    q = [float((a[i] ^ b[i]) - 1) for i in range(slots)]

    for word in range(words):
        q[word * bits] = 0.0
    state = [q[i] / 2.0 + 1j * g[i] for i in range(slots)]

    distance = 1
    while distance < bits:
        rotated = rotate_from_lower(state, distance)
        updated = []
        for current, previous in zip(state, rotated):
            conjugate = current.conjugate()
            product = (current + conjugate) * previous
            keep_generate = (current - conjugate) / 2.0
            updated.append(keep_generate - product)
        state = updated
        distance *= 2

    prefix_g = [round(value.imag) for value in state]
    result = [0] * slots
    for word in range(words):
        base = word * bits
        for bit in range(bits):
            index = base + bit
            borrow_in = 0 if bit == 0 else prefix_g[index - 1]
            result[index] = a[index] - b[index] - borrow_in + 2 * prefix_g[index]
    return result


def reference(bits, words, a, b):
    result = [0] * (bits * words)
    for word in range(words):
        borrow = 0
        base = word * bits
        for bit in range(bits):
            index = base + bit
            value = a[index] - b[index] - borrow
            if value < 0:
                value += 2
                borrow = 1
            else:
                borrow = 0
            result[index] = value
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=1000)
    args = parser.parse_args()

    rng = random.Random(0x535542484541445A)
    for bits in (8, 16, 32, 64, 128, 256):
        words = max(2, min(32, 65536 // bits))
        for trial in range(args.trials):
            slots = bits * words
            a = [rng.randrange(2) for _ in range(slots)]
            b = [rng.randrange(2) for _ in range(slots)]
            if trial == 0:
                a[:bits] = [0] * bits
                b[:bits] = [0] * bits
                b[0] = 1
            actual = evaluate(bits, words, a, b)
            expected = reference(bits, words, a, b)
            if actual != expected:
                mismatch = next(
                    i for i, pair in enumerate(zip(actual, expected))
                    if pair[0] != pair[1]
                )
                raise RuntimeError(
                    f"bits={bits} trial={trial} slot={mismatch} "
                    f"actual={actual[mismatch]} expected={expected[mismatch]}"
                )
        print(f"{bits:3d}-bit, {words:2d} packed words: PASS")

    print("head-zero complex borrow-prefix equivalence: PASS")


if __name__ == "__main__":
    main()
