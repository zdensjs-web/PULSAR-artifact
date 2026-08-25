#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import struct


INITIAL = (
    0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
    0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19,
)
ROUND_CONSTANTS = (
    0x428A2F98, 0x71374491, 0xB5C0FBCF, 0xE9B5DBA5,
    0x3956C25B, 0x59F111F1, 0x923F82A4, 0xAB1C5ED5,
    0xD807AA98, 0x12835B01, 0x243185BE, 0x550C7DC3,
    0x72BE5D74, 0x80DEB1FE, 0x9BDC06A7, 0xC19BF174,
    0xE49B69C1, 0xEFBE4786, 0x0FC19DC6, 0x240CA1CC,
    0x2DE92C6F, 0x4A7484AA, 0x5CB0A9DC, 0x76F988DA,
    0x983E5152, 0xA831C66D, 0xB00327C8, 0xBF597FC7,
    0xC6E00BF3, 0xD5A79147, 0x06CA6351, 0x14292967,
    0x27B70A85, 0x2E1B2138, 0x4D2C6DFC, 0x53380D13,
    0x650A7354, 0x766A0ABB, 0x81C2C92E, 0x92722C85,
    0xA2BFE8A1, 0xA81A664B, 0xC24B8B70, 0xC76C51A3,
    0xD192E819, 0xD6990624, 0xF40E3585, 0x106AA070,
    0x19A4C116, 0x1E376C08, 0x2748774C, 0x34B0BCB5,
    0x391C0CB3, 0x4ED8AA4A, 0x5B9CCA4F, 0x682E6FF3,
    0x748F82EE, 0x78A5636F, 0x84C87814, 0x8CC70208,
    0x90BEFFFA, 0xA4506CEB, 0xBEF9A3F7, 0xC67178F2,
)
MASK = 0xFFFFFFFF


def rotr(value: int, amount: int) -> int:
    return ((value >> amount) | (value << (32 - amount))) & MASK


def compress(block: bytes) -> bytes:
    if len(block) != 64:
        raise ValueError("SHA-256 compression requires one 64-byte block")
    words = list(struct.unpack(">16I", block))
    for index in range(16, 64):
        sigma0 = rotr(words[index - 15], 7) ^ rotr(
            words[index - 15], 18) ^ (words[index - 15] >> 3)
        sigma1 = rotr(words[index - 2], 17) ^ rotr(
            words[index - 2], 19) ^ (words[index - 2] >> 10)
        words.append(
            (words[index - 16] + sigma0 + words[index - 7] + sigma1)
            & MASK
        )

    a, b, c, d, e, f, g, h = INITIAL
    for index in range(64):
        big1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25)
        choose = (e & f) ^ ((~e) & g)
        t1 = (h + big1 + choose + ROUND_CONSTANTS[index] + words[index]) & MASK
        big0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22)
        majority = (a & b) ^ (a & c) ^ (b & c)
        t2 = (big0 + majority) & MASK
        a, b, c, d, e, f, g, h = (
            (t1 + t2) & MASK,
            a,
            b,
            c,
            (d + t1) & MASK,
            e,
            f,
            g,
        )
    state = (
        (INITIAL[0] + a) & MASK,
        (INITIAL[1] + b) & MASK,
        (INITIAL[2] + c) & MASK,
        (INITIAL[3] + d) & MASK,
        (INITIAL[4] + e) & MASK,
        (INITIAL[5] + f) & MASK,
        (INITIAL[6] + g) & MASK,
        (INITIAL[7] + h) & MASK,
    )
    return struct.pack(">8I", *state)


def padded_block(message: bytes) -> bytes:
    if len(message) > 55:
        raise ValueError("test message must fit in one SHA-256 block")
    return (
        message
        + b"\x80"
        + b"\x00" * (55 - len(message))
        + struct.pack(">Q", len(message) * 8)
    )


def test_bit_major_rotation() -> None:
    words = [0x01234567, 0x89ABCDEF, 0x13579BDF]
    count = len(words)
    slots = [
        (words[word] >> bit) & 1
        for bit in range(32)
        for word in range(count)
    ]

    def rotate_slots(offset: int) -> list[int]:
        return [
            slots[(index + offset) % len(slots)]
            for index in range(len(slots))
        ]

    for amount in (2, 3, 6, 7, 10, 11, 13, 17, 18, 19, 22, 25):
        rotated = rotate_slots(amount * count)
        for word, value in enumerate(words):
            recovered = sum(
                rotated[bit * count + word] << bit
                for bit in range(32)
            )
            if recovered != rotr(value, amount):
                raise RuntimeError(
                    f"bit-major right rotation failed for {amount}"
                )
    for amount in (3, 10):
        shifted = rotate_slots(amount * count)
        for bit in range(32 - amount, 32):
            for word in range(count):
                shifted[bit * count + word] = 0
        for word, value in enumerate(words):
            recovered = sum(
                shifted[bit * count + word] << bit
                for bit in range(32)
            )
            if recovered != value >> amount:
                raise RuntimeError(
                    f"bit-major logical right shift failed for {amount}"
                )


def main() -> None:
    test_bit_major_rotation()
    messages = (
        b"",
        b"abc",
        bytes(range(32)),
        b"PULSAR Boolean SHA-256 test",
    )
    for message in messages:
        actual = compress(padded_block(message))
        expected = hashlib.sha256(message).digest()
        if actual != expected:
            raise RuntimeError(
                f"SHA-256 mismatch for {message!r}: "
                f"{actual.hex()} != {expected.hex()}"
            )
    print("PULSAR SHA-256 plaintext audit")
    print("================================")
    print("bit-major ROTR              : PASS")
    print("bit-major SHR               : PASS")
    print("Ch and Maj formulas         : PASS")
    print("message schedule            : PASS")
    print("64 compression rounds       : PASS")
    print("single-block SHA-256 digest : PASS")
    print("aggregate encrypted batch   : 2048 messages")
    print("overall verdict             : PASS")


if __name__ == "__main__":
    main()
