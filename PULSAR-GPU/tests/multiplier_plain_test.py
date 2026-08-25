#!/usr/bin/env python3

import argparse
import random


WIDTHS = (8, 16, 32, 64, 128, 256)
EXPECTED_256_TARGETS = (211, 141, 94, 63, 42, 28, 19, 13, 9, 6, 4, 3, 2)
BOOTSTRAP_OUTPUT_LEVEL = 21
BOOLEAN_PROJECTION_LEVELS = 3
CANONICAL_BOOLEAN_LEVEL = (
    BOOTSTRAP_OUTPUT_LEVEL + BOOLEAN_PROJECTION_LEVELS
)
EXPECTED_DEPTHS = {
    8: 33,
    16: 34,
    32: 35,
    64: 36,
    128: 37,
    256: 38,
}


def bits_of(value, width):
    return [(value >> bit) & 1 for bit in range(width)]


def value_of(bits):
    return sum(bit << index for index, bit in enumerate(bits))


def dadda_targets(rows):
    ascending = [2]
    while ascending[-1] < rows:
        ascending.append((3 * ascending[-1]) // 2)
    if ascending[-1] >= rows:
        ascending.pop()
    return list(reversed(ascending))


def purification_stages(width):
    stages = len(dadda_targets(width))
    first = (stages + 2) // 3 - 1
    second = (2 * stages + 2) // 3 - 1
    if not (0 <= first < second < stages - 1):
        raise AssertionError("invalid purification boundaries")
    return first, second


def level_schedule(width):
    stages = len(dadda_targets(width))
    projections = len(purification_stages(width))
    dadda_pre_refresh = (
        1
        + 2 * stages
        + BOOLEAN_PROJECTION_LEVELS * projections
    )
    final_adder_cost = width.bit_length() - 1 + 4
    minimum_depth = max(
        dadda_pre_refresh + 2,
        CANONICAL_BOOLEAN_LEVEL + final_adder_cost + 2,
    )
    raw_product = minimum_depth - 2
    final_adder_input = raw_product - final_adder_cost
    return {
        "boundaries": purification_stages(width),
        "dadda_pre_refresh": dadda_pre_refresh,
        "final_adder_input": final_adder_input,
        "raw_product": raw_product,
        "depth": minimum_depth,
    }


def shift_left(bits, distance):
    return [0] * distance + bits[: len(bits) - distance]


def compressor_3_to_2(x, y, z):
    pair = [a * b for a, b in zip(x, y)]
    parity = [a + b - 2 * ab for a, b, ab in zip(x, y, pair)]
    parity_z = [p * c for p, c in zip(parity, z)]
    total_parity = [
        p + c - 2 * pc for p, c, pc in zip(parity, z, parity_z)
    ]
    carry = [ab + pc for ab, pc in zip(pair, parity_z)]
    return total_parity, shift_left(carry, 1)


def project_boolean(bits):
    return [3 * bit * bit - 2 * bit * bit * bit for bit in bits]


def partial_product_rows(a, b, width):
    a_bits = bits_of(a, width)
    b_bits = bits_of(b, width)
    rows = []
    for source_bit in range(width):
        row = [0] * width
        for output_bit in range(source_bit, width):
            row[output_bit] = (
                a_bits[output_bit - source_bit] * b_bits[source_bit]
            )
        rows.append(row)
    return rows


def reduce_dadda(rows):
    width = len(rows)
    targets = dadda_targets(width)
    current = rows
    boundaries = set(purification_stages(width))
    for stage, target in enumerate(targets):
        input_rows = len(current)
        full_adders = input_rows - target
        passthrough = 3 * target - 2 * input_rows
        if 3 * full_adders + passthrough != input_rows:
            raise AssertionError("Dadda input-row identity failed")
        if 2 * full_adders + passthrough != target:
            raise AssertionError("Dadda output-row identity failed")

        output = []
        compressed_rows = 3 * full_adders
        for offset in range(0, compressed_rows, 3):
            summed, carry = compressor_3_to_2(
                current[offset],
                current[offset + 1],
                current[offset + 2],
            )
            output.extend((summed, carry))
        output.extend(current[compressed_rows:])
        if len(output) != target:
            raise AssertionError("Dadda target-height mismatch")
        if stage in boundaries:
            output = [project_boolean(row) for row in output]
        current = output
    if len(current) != 2:
        raise AssertionError("Dadda reduction did not produce two rows")
    return current


def prefix_add(lhs, rhs):
    width = len(lhs)
    generate = [a * b for a, b in zip(lhs, rhs)]
    propagate = [
        a + b - 2 * g for a, b, g in zip(lhs, rhs, generate)
    ]
    group = generate

    distance = 1
    while distance < width:
        old_group = group
        old_propagate = propagate
        rotated_group = shift_left(old_group, distance)
        rotated_propagate = shift_left(old_propagate, distance)
        term = [
            rotated * local
            for rotated, local in zip(rotated_group, old_propagate)
        ]
        group = [
            old + extension
            for old, extension in zip(old_group, term)
        ]
        if distance * 2 < width:
            propagate = [
                rotated * local
                for rotated, local in zip(
                    rotated_propagate, old_propagate
                )
            ]
        distance *= 2

    carry = shift_left(group, 1)
    return [
        a + b + c - 2 * g
        for a, b, c, g in zip(lhs, rhs, carry, group)
    ]


def complex_twisted_add(lhs, rhs):
    width = len(lhs)
    generate = [a * b for a, b in zip(lhs, rhs)]
    propagate = [
        a + b - 2 * g for a, b, g in zip(lhs, rhs, generate)
    ]
    state = [
        p / 2.0 + 1j * g for p, g in zip(propagate, generate)
    ]

    distance = 1
    while distance < width:
        conjugate = [value.conjugate() for value in state]
        rotated = shift_left(state, distance)
        state = [
            (value + conjugated) * shifted
            + (value - conjugated) / 2.0
            for value, conjugated, shifted in zip(
                state, conjugate, rotated
            )
        ]
        distance *= 2

    final_generate = [round(value.imag) for value in state]
    carry = shift_left(final_generate, 1)
    return [
        a + b + c - 2 * g
        for a, b, c, g in zip(
            lhs, rhs, carry, final_generate
        )
    ]


def multiply_model(a, b, width):
    rows = partial_product_rows(a, b, width)
    reduced = reduce_dadda(rows)
    baseline = prefix_add(reduced[0], reduced[1])
    twisted = complex_twisted_add(reduced[0], reduced[1])
    if twisted != baseline:
        raise AssertionError(
            "complex-twisted final adder differs from baseline prefix adder"
        )
    return value_of(twisted)


def run_width(width, trials, generator):
    mask = (1 << width) - 1
    cases = [
        (0, 0),
        (mask, 1),
        (1 << (width // 2), 1 << (width // 2 - 1)),
        (mask, mask),
        (1, mask),
    ]
    cases.extend(
        (generator.getrandbits(width), generator.getrandbits(width))
        for _ in range(trials)
    )

    for case, (lhs, rhs) in enumerate(cases):
        actual = multiply_model(lhs, rhs, width)
        expected = (lhs * rhs) & mask
        if actual != expected:
            raise AssertionError(
                f"width={width} case={case}: "
                f"0x{lhs:x} * 0x{rhs:x} -> "
                f"0x{actual:x}, expected 0x{expected:x}"
            )

    targets = dadda_targets(width)
    compressor_count = sum(
        source - target
        for source, target in zip([width] + targets[:-1], targets)
    )
    if compressor_count != width - 2:
        raise AssertionError(
            f"width={width}: compressor count {compressor_count}"
        )
    return len(targets), compressor_count


def main():
    parser = argparse.ArgumentParser(
        description="Plaintext Dadda/Kogge-Stone multiplier model"
    )
    parser.add_argument("--trials", type=int, default=100)
    args = parser.parse_args()
    if args.trials < 1:
        parser.error("--trials must be positive")

    if tuple(dadda_targets(256)) != EXPECTED_256_TARGETS:
        raise AssertionError("dynamic 256-bit Dadda targets changed")

    generator = random.Random(0x44414444414D554C)
    print("Runtime Dadda multiplier plaintext test")
    print("=======================================")
    for width in WIDTHS:
        stages, compressors = run_width(width, args.trials, generator)
        schedule = level_schedule(width)
        if schedule["depth"] != EXPECTED_DEPTHS[width]:
            raise AssertionError(
                f"width={width}: unexpected depth schedule {schedule}"
            )
        print(
            f"width={width:3d}: stages={stages:2d}, "
            f"compressors={compressors:3d}, "
            f"purify={schedule['boundaries']}, "
            f"preboot={schedule['dadda_pre_refresh']:2d}, "
            f"depth={schedule['depth']:2d}, verdict=PASS"
        )
    print(f"random trials per width: {args.trials}")
    print("overall verdict        : PASS")


if __name__ == "__main__":
    main()
