#pragma once

#include <fideslib.hpp>

#include <array>
#include <chrono>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace ks256gpu {

using fideslib::Ciphertext;
using fideslib::CryptoContext;
using fideslib::DCRTPoly;
using fideslib::Plaintext;

constexpr uint32_t kWordBits   = 256;
constexpr uint32_t kGuardBits  = 256;
constexpr uint32_t kStride     = kWordBits + kGuardBits;  // 512 slots/word
constexpr uint32_t kNumWords   = 128;
constexpr uint32_t kTotalSlots = kNumWords * kStride;     // 65536 slots
constexpr uint32_t kNumLayers  = 8;                       // log2(256)

static_assert(
    kTotalSlots <= 65536,
    "This test is configured for ring dimension 131072, so the slot count must not exceed 65536");
static_assert(
    (kTotalSlots & (kTotalSlots - 1)) == 0,
    "The CKKS slot count must be a power of two");

// These timing structures intentionally keep the original public interface.
// The existing CPP files and their INFO output therefore remain unchanged.
// prefixLayers now measure the Kogge-Stone borrow-generate/propagate stages;
// the obsolete highest-different-bit/reduction fields remain zero.
struct PrefixLayerTiming {
    double rotatePMs    = 0.0;
    double prefixMultMs = 0.0;
    double wallMs       = 0.0;
};

struct ReductionLayerTiming {
    double rotateMs = 0.0;
    double addMs    = 0.0;
    double wallMs   = 0.0;
};

struct TimingReport {
    double preprocessMs = 0.0;

    std::array<PrefixLayerTiming, kNumLayers> prefixLayers{};

    double higherEqualRotateMs = 0.0;
    double highestDifferentMs  = 0.0;
    double selectAMs       = 0.0;

    std::array<ReductionLayerTiming, kNumLayers> reductionLayers{};

    double resultMaskMs = 0.0;
    double totalMs      = 0.0;
};

struct Masks {
    // 1 on the 256 active Boolean slots and 0 on all guard slots.
    // It supplies the Boolean constant one for XNOR while forcing the
    // propagate state in every guard slot to zero.
    Plaintext activeMask;

    // 1 only at bit 0 of every packed 512-slot word.
    Plaintext resultMask;
};

namespace detail {

using Clock = std::chrono::steady_clock;

inline double Milliseconds(const Clock::time_point& begin,
                           const Clock::time_point& end) {
    return std::chrono::duration<double, std::milli>(end - begin).count();
}

inline std::vector<double> MakeActiveMaskValues() {
    std::vector<double> mask(kTotalSlots, 0.0);

    for (uint32_t word = 0; word < kNumWords; ++word) {
        const uint32_t base = word * kStride;
        for (uint32_t bit = 0; bit < kWordBits; ++bit) {
            mask[base + bit] = 1.0;
        }
    }

    return mask;
}

inline std::vector<double> MakeResultMaskValues() {
    std::vector<double> mask(kTotalSlots, 0.0);

    for (uint32_t word = 0; word < kNumWords; ++word) {
        mask[word * kStride] = 1.0;
    }

    return mask;
}

inline void Synchronize(const CryptoContext<DCRTPoly>& cc) {
    cc->Synchronize();
}

template <class Fn>
inline auto TimedGpuCall(const CryptoContext<DCRTPoly>& cc,
                         double* elapsedMs,
                         Fn&& fn) {
    Synchronize(cc);
    const auto begin = Clock::now();
    auto result = fn();
    Synchronize(cc);
    *elapsedMs = Milliseconds(begin, Clock::now());
    return result;
}

}  // namespace detail

/**
 * Creates and explicitly loads the comparison masks on the selected GPU.
 * Call this only after cc->LoadContext(...).
 */
inline Masks BuildAndLoadMasks(const CryptoContext<DCRTPoly>& cc) {
    if (!cc) {
        throw std::invalid_argument(
            "BuildAndLoadMasks received a null context");
    }

    Masks masks;

    masks.activeMask = cc->MakeCKKSPackedPlaintext(
        detail::MakeActiveMaskValues(), 1, 0, nullptr, kTotalSlots);

    masks.resultMask = cc->MakeCKKSPackedPlaintext(
        detail::MakeResultMaskValues(), 1, 0, nullptr, kTotalSlots);

    masks.activeMask->SetLength(kTotalSlots);
    masks.resultMask->SetLength(kTotalSlots);

    cc->LoadPlaintext(masks.activeMask);
    cc->LoadPlaintext(masks.resultMask);
    detail::Synchronize(cc);

    return masks;
}

/**
 * Evaluates 128 independent unsigned 256-bit greater-than comparisons.
 *
 * Layout:
 *   [256 Boolean data slots][256 zero guard slots] x 128.
 *
 * Inputs use LSB-first order inside every word:
 *   base+0   = bit 0,
 *   base+255 = bit 255.
 *
 * The circuit uses the same Boolean generate/propagate states and associative
 * Kogge-Stone prefix operator as the borrow network in the subtractor:
 *
 *   g_i = a_i & (~b_i) = a_i - a_i b_i
 *   p_i = XNOR(a_i,b_i)
 *       = 1 - a_i - b_i + 2a_i b_i.
 *
 * For one interval, G records whether its most significant differing bit
 * proves A > B, and P records whether the whole interval is equal.
 * When a lower interval L and a more-significant interval H are combined:
 *
 *   G(H:L) = G(H) OR (P(H) AND G(L)),
 *   P(H:L) = P(H) AND P(L).
 *
 * G(H) and P(H)G(L) are mutually exclusive Boolean states, so arithmetic
 * addition implements OR without another ciphertext multiplication.
 *
 * Positive rotations are used because the unchanged CPP files generate only
 * +1,+2,...,+128 rotation keys.  With the FIDESlib convention used here,
 * Rotate(X,+step) places the more-significant state X[i+step] at slot i.
 * After eight stages, G[base+0] is exactly the unsigned A > B
 * result.  This Boolean function is equivalent to the final subtraction
 * borrow test, but it places the answer directly in the existing result slot.
 *
 * The caller still performs exactly one bootstrap after the complete circuit.
 * The public function name is retained so the CPP code and INFO output remain
 * source-compatible.
 */
inline Ciphertext<DCRTPoly> EvalGreaterThanPOnly(
    const CryptoContext<DCRTPoly>& cc,
    const Ciphertext<DCRTPoly>& cA,
    const Ciphertext<DCRTPoly>& cB,
    Masks& masks,
    TimingReport* timing = nullptr) {

    if (!cc || !cA || !cB) {
        throw std::invalid_argument(
            "EvalGreaterThanPOnly received a null context or ciphertext");
    }

    if (!masks.activeMask || !masks.resultMask) {
        throw std::invalid_argument(
            "EvalGreaterThanPOnly received invalid masks");
    }

    TimingReport local;
    TimingReport& report = timing ? *timing : local;
    report = TimingReport{};

    detail::Synchronize(cc);
    const auto totalBegin = detail::Clock::now();

    // 1. Build the per-bit borrow/comparison generate and equality-propagate
    // states with one ciphertext-ciphertext multiplication.
    //
    //   ab  = A*B
    //   xor = A + B - 2AB
    //   p   = activeMask - xor = XNOR(A,B) on active slots and 0 on guards
    //   g   = A - AB = A & (~B)
    detail::Synchronize(cc);
    auto begin = detail::Clock::now();

    auto ab = cc->EvalMult(cA, cB);
    auto xorAB = cc->EvalSub(
        cc->EvalAdd(cA, cB),
        cc->EvalAdd(ab, ab));

    auto G = cc->EvalSub(cA, ab);
    auto Pk = cc->EvalSub(masks.activeMask, xorAB);

    detail::Synchronize(cc);
    report.preprocessMs =
        detail::Milliseconds(begin, detail::Clock::now());

    // 2. Eight MSB-priority Kogge-Stone generate/propagate stages.
    // At destination i, Rotate(...,+step) supplies the more-significant
    // interval H, while the unrotated state is the lower interval L:
    //
    //   Gnew = GH + PH*GL
    //   Pnew = PH*PL
    //
    // P is not needed after the final layer.
    for (uint32_t layer = 0; layer < kNumLayers; ++layer) {
        detail::Synchronize(cc);
        const auto layerBegin = detail::Clock::now();

        const int32_t step =
            static_cast<int32_t>(1U << layer);

        const auto Gold = G;
        const auto Pold = Pk;

        detail::Synchronize(cc);
        begin = detail::Clock::now();

        auto Ghigh = cc->EvalRotate(Gold, step);
        auto Phigh = cc->EvalRotate(Pold, step);

        detail::Synchronize(cc);
        report.prefixLayers[layer].rotatePMs =
            detail::Milliseconds(begin, detail::Clock::now());

        detail::Synchronize(cc);
        begin = detail::Clock::now();

        auto lowerTerm = cc->EvalMult(Phigh, Gold);

        if (layer + 1 < kNumLayers) {
            Pk = cc->EvalMult(Phigh, Pold);
        }

        G = cc->EvalAdd(Ghigh, lowerTerm);

        detail::Synchronize(cc);
        report.prefixLayers[layer].prefixMultMs =
            detail::Milliseconds(begin, detail::Clock::now());

        report.prefixLayers[layer].wallMs =
            detail::Milliseconds(layerBegin, detail::Clock::now());
    }

    // 3. G already contains the final comparison bit at bit 0 of each word.
    // Clear all other active and guard slots.  The legacy highest-different-bit
    // and rotate-add timing fields intentionally remain zero.
    auto result = detail::TimedGpuCall(
        cc,
        &report.resultMaskMs,
        [&]() {
            return cc->EvalMult(
                G,
                masks.resultMask);
        });

    report.totalMs =
        detail::Milliseconds(
            totalBegin,
            detail::Clock::now());

    return result;
}

}  // namespace ks256gpu
