#!/usr/bin/env python3

from __future__ import annotations

import math


BOOTSTRAP_OUTPUT_LEVEL = 21
BOOLEAN_PROJECTION_LEVELS = 3
CANONICAL_STATE_LEVEL = (
    BOOTSTRAP_OUTPUT_LEVEL + BOOLEAN_PROJECTION_LEVELS
)
DEFAULT_DEPTH = 36


def pad_to_level(
    level: int, scale_degree: int, target_level: int
) -> tuple[int, int, int]:
    """Model FIDES identity-plaintext multiplication transitions."""
    operations = 0
    while level < target_level or scale_degree != 2:
        if scale_degree == 1:
            scale_degree = 2
        elif scale_degree == 2 and level < target_level:
            level += 1
        else:
            raise AssertionError(
                f"invalid pad state {level}/{scale_degree} -> {target_level}/2"
            )
        operations += 1
    return level, scale_degree, operations


def fides_plaintext_minus_ciphertext(
    level: int, scale_degree: int
) -> tuple[int, int]:
    """Model the GPU API's internal multScalar(-1) before plaintext add."""
    if scale_degree == 2:
        level += 1
    return level, 2


def audit(bits: int) -> tuple[int, ...]:
    wide_layers = int(math.log2(2 * bits))
    layers_before_refresh = min(4, wide_layers)
    layers_after_refresh = wide_layers - layers_before_refresh
    input_level = CANONICAL_STATE_LEVEL
    pre_refresh_level = (
        CANONICAL_STATE_LEVEL + 1 + layers_before_refresh
    )
    final_gp_level = (
        CANONICAL_STATE_LEVEL + layers_after_refresh
    )
    difference_level = final_gp_level + 1
    raw_borrow_level = final_gp_level + 1
    projected_borrow_level = (
        raw_borrow_level + BOOLEAN_PROJECTION_LEVELS
    )
    quotient_bit_level = projected_borrow_level
    selected_state_level = projected_borrow_level + 1
    minimum_depth = max(
        pre_refresh_level, selected_state_level
    ) + 2
    depth = minimum_depth
    raw_level = depth - 2
    trial_cost = raw_level - input_level
    dividend_cursor_level = BOOTSTRAP_OUTPUT_LEVEL
    nibble_level = dividend_cursor_level + 1
    zero_gate_level = quotient_bit_level + 1

    # Model every persistent path, not only one candidate subtraction.
    divisor_level = CANONICAL_STATE_LEVEL
    zero_flag_level = CANONICAL_STATE_LEVEL
    dividend_cursor_level = BOOTSTRAP_OUTPUT_LEVEL
    remainder_level = CANONICAL_STATE_LEVEL
    quotient_level = dividend_cursor_level
    maximum_direct_rotation_ksks = 0
    for _round in range(bits // 4):
        destination = bits - 4 * (_round + 1)
        maximum_direct_rotation_ksks = max(
            maximum_direct_rotation_ksks,
            destination.bit_count(),
        )
        nibble_level = dividend_cursor_level + 1
        state_level = max(remainder_level, nibble_level)
        for _trial in range(4):
            state_level = max(state_level, input_level)
            assert state_level == input_level
            # Every selected trial state reaches the bottom, is Bootstrapped,
            # and is then Boolean-projected and lane-masked.
            state_level = CANONICAL_STATE_LEVEL
        remainder_level = state_level
        digit_level = quotient_bit_level
        quotient_level = max(quotient_level, digit_level)

    zero_flag_level = max(zero_flag_level, quotient_level)
    quotient_level = zero_flag_level + 1
    # The encrypted nonzero mask is already zero outside the result lane.
    # The last trial's post-Bootstrap projection uses the word-width mask.

    assert input_level == CANONICAL_STATE_LEVEL
    assert nibble_level == BOOTSTRAP_OUTPUT_LEVEL + 1
    assert input_level >= nibble_level
    assert pre_refresh_level <= raw_level
    assert projected_borrow_level == raw_level - 1
    assert selected_state_level == raw_level
    assert depth - (selected_state_level + 2) == 0
    assert quotient_level == raw_level
    assert remainder_level == CANONICAL_STATE_LEVEL
    assert maximum_direct_rotation_ksks <= int(math.log2(bits))
    # The removed Plaintext-Ciphertext form would advance quotient bits by one
    # level. The encrypted-mask construction keeps all linear complements at
    # their input state.
    assert fides_plaintext_minus_ciphertext(
        quotient_bit_level, 2
    ) == (quotient_bit_level + 1, 2)
    assert pad_to_level(0, 1, input_level) == (
        input_level,
        2,
        input_level + 1,
    )
    assert pad_to_level(pre_refresh_level, 2, raw_level) == (
        raw_level,
        2,
        raw_level - pre_refresh_level,
    )

    return (
        bits,
        depth,
        trial_cost,
        input_level,
        pre_refresh_level,
        difference_level,
        projected_borrow_level,
        selected_state_level,
    )


def main() -> None:
    print("Encrypted divider bottom-level schedule audit")
    print("=" * 72)
    print(
        "bits depth cost input prefix difference borrow mux remaining"
    )
    for bits in (8, 16, 32, 64, 128, 256):
        row = audit(bits)
        print(
            f"{row[0]:4d} {row[1]:5d} {row[2]:4d} {row[3]:5d} "
            f"{row[4]:6d} {row[5]:10d} {row[6]:3d} "
            f"{row[7]:3d} {0:9d}"
        )
    print("\nverdict: PASS")


if __name__ == "__main__":
    main()
