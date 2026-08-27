#pragma once

#include <fideslib.hpp>

#include <array>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace erc7984gpu {

using fideslib::Ciphertext;
using fideslib::CryptoContext;
using fideslib::DCRTPoly;
using fideslib::Plaintext;

// One CKKS ciphertext packs 128 independent uint256 values.
// Each value uses 256 Boolean slots followed by 256 zero guard slots.
constexpr uint32_t kWordBits   = 256;
constexpr uint32_t kGuardBits  = 256;
constexpr uint32_t kStride     = kWordBits + kGuardBits;
constexpr uint32_t kNumWords   = 128;
constexpr uint32_t kTotalSlots = kNumWords * kStride;
constexpr uint32_t kNumLayers  = 8;
constexpr int kDirectionSign   = -1;  // base+0 is the LSB

// Conservative multiplicative-depth budgets derived from the original
// Kogge-Stone operator implementations supplied with this project:
//   preprocess multiplication: 1
//   eight prefix layers:       8
//   final plaintext masking:   1
constexpr uint32_t kCompareDepthBudget = 10;
constexpr uint32_t kKsDepthBudget      = 10;
constexpr uint32_t kSelectDepthBudget  = 1;
constexpr uint32_t kBorrowExtractDepthBudget = 1;

// Fresh-input output-level estimates. They are used only to decide whether
// bootstrap setup/key generation can be skipped. Runtime decisions always use
// Ciphertext::GetLevel().
constexpr uint32_t kWhitepaperFreshOutputLevelEstimate = 21;
constexpr uint32_t kNoCmuxFreshOutputLevelEstimate     = 21;
constexpr uint32_t kOverflowFreshOutputLevelEstimate   = 22;

static_assert(kTotalSlots == 65536,
              "The ERC7984 benchmark expects 65536 CKKS slots");

struct Masks {
    Plaintext activeMask;  // 1 on active bits, 0 on guards
    Plaintext carryMask;   // 0 on each word's bit 0 and guards
    Plaintext resultMask;  // 1 only on each word's bit 0
};

enum class Scenario {
    Whitepaper,
    NoCmux,
    Overflow,
};

enum class CipherLayout {
    ActiveVector,
    BitZeroOnly,
};

struct BootstrapPolicy {
    uint32_t multiplicativeDepth = 25;
    uint32_t safetyMargin = 1;
    uint32_t bootstrapInputMaxLevel = 23;
    uint32_t bootstrapOutputLevelHint = 18;
    uint32_t requiredOutputDepth = 0;
    bool bootstrapPrepared = true;
    bool allowBootstrap = true;
    bool verboseLevels = true;
};

struct TransferTiming {
    double compareOrOverflowSubMs = 0.0;
    double conditionPrepareMs     = 0.0;
    double transferSelectMs       = 0.0;
    double senderUpdateMs         = 0.0;
    double receiverUpdateMs       = 0.0;
    double conditionalBootstrapMs = 0.0;
    double totalMs                = 0.0;
    uint32_t bootstrapCount       = 0;
};

struct TransferResult {
    Ciphertext<DCRTPoly> newSender;
    Ciphertext<DCRTPoly> newReceiver;
};

namespace detail {

using Clock = std::chrono::steady_clock;

inline double Milliseconds(const Clock::time_point& begin,
                           const Clock::time_point& end) {
    return std::chrono::duration<double, std::milli>(end - begin).count();
}

inline void Synchronize(const CryptoContext<DCRTPoly>& cc) {
    cc->Synchronize();
}

inline uint32_t LevelOf(const Ciphertext<DCRTPoly>& ciphertext,
                        const char* label) {
    if (!ciphertext) {
        throw std::invalid_argument(std::string(label) + " is null");
    }
    return ciphertext->GetLevel();
}

inline void TraceLevel(const BootstrapPolicy& policy,
                       const std::string& label,
                       const Ciphertext<DCRTPoly>& ciphertext) {
    if (!policy.verboseLevels) {
        return;
    }
    std::cerr << "[ERC7984 level] " << label
              << " = " << LevelOf(ciphertext, label.c_str())
              << std::endl;
}

template <class Fn>
inline auto TimedCall(const CryptoContext<DCRTPoly>& cc,
                      double* elapsedMs,
                      const char* stage,
                      Fn&& fn) {
    std::cerr << "[ERC7984 stage] BEGIN " << stage << std::endl;
    Synchronize(cc);
    const auto begin = Clock::now();
    auto result = fn();
    Synchronize(cc);
    *elapsedMs += Milliseconds(begin, Clock::now());
    std::cerr << "[ERC7984 stage] END   " << stage << std::endl;
    return result;
}

template <class Fn>
inline auto SynchronizedCall(const CryptoContext<DCRTPoly>& cc, Fn&& fn) {
    Synchronize(cc);
    auto result = fn();
    Synchronize(cc);
    return result;
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

inline std::vector<double> MakeCarryMaskValues() {
    std::vector<double> mask(kTotalSlots, 0.0);
    for (uint32_t word = 0; word < kNumWords; ++word) {
        const uint32_t base = word * kStride;
        for (uint32_t bit = 1; bit < kWordBits; ++bit) {
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

inline uint32_t UsableMaxLevel(const BootstrapPolicy& policy) {
    if (policy.safetyMargin >= policy.multiplicativeDepth) {
        throw std::invalid_argument(
            "bootstrap safety margin must be smaller than multiplicative depth");
    }
    return policy.multiplicativeDepth - policy.safetyMargin;
}

inline bool HasDepthBudget(const Ciphertext<DCRTPoly>& value,
                           uint32_t requiredDepth,
                           const BootstrapPolicy& policy) {
    const uint64_t projected =
        static_cast<uint64_t>(LevelOf(value, "ciphertext")) + requiredDepth;
    return projected <= UsableMaxLevel(policy);
}

class LevelBootstrapManager {
public:
    LevelBootstrapManager(const CryptoContext<DCRTPoly>& cc,
                          Masks& masks,
                          const BootstrapPolicy& policy,
                          TransferTiming* timing)
        : cc_(cc), masks_(masks), policy_(policy), timing_(timing) {}

    Ciphertext<DCRTPoly> EnsureVector(Ciphertext<DCRTPoly> value,
                                      uint32_t requiredDepth,
                                      const std::string& label) {
        return Ensure(std::move(value), requiredDepth,
                      CipherLayout::ActiveVector, label);
    }

    Ciphertext<DCRTPoly> EnsureBitZero(Ciphertext<DCRTPoly> value,
                                       uint32_t requiredDepth,
                                       const std::string& label) {
        return Ensure(std::move(value), requiredDepth,
                      CipherLayout::BitZeroOnly, label);
    }

private:
    Ciphertext<DCRTPoly> Ensure(Ciphertext<DCRTPoly> value,
                                uint32_t requiredDepth,
                                CipherLayout layout,
                                const std::string& label) {
        if (!value) {
            throw std::invalid_argument(label + " is null");
        }

        const uint32_t before = value->GetLevel();
        const uint32_t usable = UsableMaxLevel(policy_);
        if (static_cast<uint64_t>(before) + requiredDepth <= usable) {
            if (policy_.verboseLevels) {
                std::cerr << "[ERC7984 boot] " << label
                          << ": level=" << before
                          << ", need=" << requiredDepth
                          << ", usable=" << usable
                          << " -> no bootstrap" << std::endl;
            }
            return value;
        }

        if (!policy_.allowBootstrap) {
            throw std::runtime_error(
                label + ": insufficient depth and bootstrap is disabled; level=" +
                std::to_string(before) + ", required=" +
                std::to_string(requiredDepth) + ", usable=" +
                std::to_string(usable));
        }
        if (!policy_.bootstrapPrepared) {
            throw std::runtime_error(
                label + ": insufficient depth, but bootstrap setup/key generation "
                "was skipped by the program configuration");
        }
        if (before > policy_.bootstrapInputMaxLevel) {
            throw std::runtime_error(
                label + ": bootstrap requested too late; level=" +
                std::to_string(before) + ", latest=" +
                std::to_string(policy_.bootstrapInputMaxLevel));
        }

        // One plaintext multiplication is used after bootstrap to clean the
        // inactive slots. Reject configurations in which a known post-bootstrap
        // level still cannot support the next circuit.
        constexpr uint32_t remaskDepth = 1;
        const uint64_t hintedProjected =
            static_cast<uint64_t>(policy_.bootstrapOutputLevelHint) +
            remaskDepth + requiredDepth;
        if (hintedProjected > usable) {
            throw std::runtime_error(
                label + ": current context is too shallow for bootstrap followed "
                "by the requested circuit. bootstrap-output hint=" +
                std::to_string(policy_.bootstrapOutputLevelHint) +
                ", remask depth=1, required=" +
                std::to_string(requiredDepth) + ", usable=" +
                std::to_string(usable) +
                ". Increase the compile-time multiplicative depth");
        }

        std::cerr << "[ERC7984 boot] " << label
                  << ": level=" << before
                  << ", need=" << requiredDepth
                  << " -> bootstrap" << std::endl;

        Synchronize(cc_);
        const auto begin = Clock::now();

        // Keep the source alive until asynchronous GPU bootstrap completes.
        auto source = value;
        auto refreshed = cc_->EvalBootstrap(value);
        Synchronize(cc_);

        if (layout == CipherLayout::BitZeroOnly) {
            refreshed = cc_->EvalMult(refreshed, masks_.resultMask);
        }
        else {
            refreshed = cc_->EvalMult(refreshed, masks_.activeMask);
        }
        Synchronize(cc_);

        if (timing_) {
            timing_->conditionalBootstrapMs +=
                Milliseconds(begin, Clock::now());
            ++timing_->bootstrapCount;
        }

        const uint32_t after = refreshed->GetLevel();
        if (policy_.verboseLevels) {
            std::cerr << "[ERC7984 boot] " << label
                      << ": level " << before << " -> " << after
                      << std::endl;
        }
        if (static_cast<uint64_t>(after) + requiredDepth > usable) {
            throw std::runtime_error(
                label + ": bootstrap completed but still left insufficient depth; "
                "level=" + std::to_string(after) + ", required=" +
                std::to_string(requiredDepth) + ", usable=" +
                std::to_string(usable));
        }
        return refreshed;
    }

    const CryptoContext<DCRTPoly>& cc_;
    Masks& masks_;
    const BootstrapPolicy& policy_;
    TransferTiming* timing_;
};

}  // namespace detail

inline const char* ScenarioName(Scenario scenario) {
    switch (scenario) {
        case Scenario::Whitepaper: return "whitepaper";
        case Scenario::NoCmux:     return "no_cmux";
        case Scenario::Overflow:   return "overflow";
    }
    return "unknown";
}

inline Scenario ParseScenario(const std::string& name) {
    if (name == "whitepaper") {
        return Scenario::Whitepaper;
    }
    if (name == "no_cmux" || name == "no-cmux") {
        return Scenario::NoCmux;
    }
    if (name == "overflow") {
        return Scenario::Overflow;
    }
    throw std::invalid_argument(
        "scenario must be whitepaper, no_cmux, overflow, or all");
}

inline uint32_t FreshOutputLevelEstimate(Scenario scenario) {
    switch (scenario) {
        case Scenario::Whitepaper:
            return kWhitepaperFreshOutputLevelEstimate;
        case Scenario::NoCmux:
            return kNoCmuxFreshOutputLevelEstimate;
        case Scenario::Overflow:
            return kOverflowFreshOutputLevelEstimate;
    }
    return std::numeric_limits<uint32_t>::max();
}

inline Masks BuildAndLoadMasks(const CryptoContext<DCRTPoly>& cc) {
    if (!cc) {
        throw std::invalid_argument("BuildAndLoadMasks received a null context");
    }

    Masks masks;
    masks.activeMask = cc->MakeCKKSPackedPlaintext(
        detail::MakeActiveMaskValues(), 1, 0, nullptr, kTotalSlots);
    masks.carryMask = cc->MakeCKKSPackedPlaintext(
        detail::MakeCarryMaskValues(), 1, 0, nullptr, kTotalSlots);
    masks.resultMask = cc->MakeCKKSPackedPlaintext(
        detail::MakeResultMaskValues(), 1, 0, nullptr, kTotalSlots);

    masks.activeMask->SetLength(kTotalSlots);
    masks.carryMask->SetLength(kTotalSlots);
    masks.resultMask->SetLength(kTotalSlots);

    cc->LoadPlaintext(masks.activeMask);
    cc->LoadPlaintext(masks.carryMask);
    cc->LoadPlaintext(masks.resultMask);
    detail::Synchronize(cc);
    return masks;
}

// The following three circuits intentionally mirror the original
// adder_ks_gpu.h, subtractor_ks_gpu.h, and greaterequal_ks_gpu.h algorithms.
// Each GPU operation is synchronized before temporary ciphertexts leave scope.

inline Ciphertext<DCRTPoly> EvalAddKS(
    const CryptoContext<DCRTPoly>& cc,
    const Ciphertext<DCRTPoly>& cA,
    const Ciphertext<DCRTPoly>& cB,
    Masks& masks) {

    if (!cc || !cA || !cB || !masks.activeMask || !masks.carryMask) {
        throw std::invalid_argument("EvalAddKS received an invalid argument");
    }

    auto g = detail::SynchronizedCall(cc, [&]() {
        return cc->EvalMult(cA, cB);
    });
    auto p = detail::SynchronizedCall(cc, [&]() {
        return cc->EvalSub(cc->EvalAdd(cA, cB), cc->EvalAdd(g, g));
    });

    auto G = g;
    auto P = p;
    for (uint32_t layer = 0; layer < kNumLayers; ++layer) {
        const int32_t rotation =
            kDirectionSign * static_cast<int32_t>(1U << layer);
        const auto Gold = G;
        const auto Pold = P;

        auto Grot = detail::SynchronizedCall(cc, [&]() {
            return cc->EvalRotate(Gold, rotation);
        });
        auto Prot = detail::SynchronizedCall(cc, [&]() {
            return cc->EvalRotate(Pold, rotation);
        });
        auto term = detail::SynchronizedCall(cc, [&]() {
            return cc->EvalMult(Grot, Pold);
        });
        if (layer + 1 < kNumLayers) {
            P = detail::SynchronizedCall(cc, [&]() {
                return cc->EvalMult(Prot, Pold);
            });
        }
        G = detail::SynchronizedCall(cc, [&]() {
            return cc->EvalAdd(Gold, term);
        });
    }

    auto carry = detail::SynchronizedCall(cc, [&]() {
        return cc->EvalRotate(G, kDirectionSign);
    });
    carry = detail::SynchronizedCall(cc, [&]() {
        return cc->EvalMult(carry, masks.carryMask);
    });
    G = detail::SynchronizedCall(cc, [&]() {
        return cc->EvalMult(G, masks.activeMask);
    });

    return detail::SynchronizedCall(cc, [&]() {
        auto result = cc->EvalAdd(cA, cB);
        result = cc->EvalAdd(result, carry);
        result = cc->EvalSub(result, cc->EvalAdd(G, G));
        return result;
    });
}

inline Ciphertext<DCRTPoly> EvalSubKS(
    const CryptoContext<DCRTPoly>& cc,
    const Ciphertext<DCRTPoly>& cA,
    const Ciphertext<DCRTPoly>& cB,
    Masks& masks,
    Ciphertext<DCRTPoly>* finalBorrowPrefix = nullptr) {

    if (!cc || !cA || !cB || !masks.activeMask || !masks.carryMask) {
        throw std::invalid_argument("EvalSubKS received an invalid argument");
    }

    auto ab = detail::SynchronizedCall(cc, [&]() {
        return cc->EvalMult(cA, cB);
    });
    auto xorAB = detail::SynchronizedCall(cc, [&]() {
        return cc->EvalSub(cc->EvalAdd(cA, cB), cc->EvalAdd(ab, ab));
    });
    auto G = detail::SynchronizedCall(cc, [&]() {
        return cc->EvalSub(cB, ab);
    });
    auto P = detail::SynchronizedCall(cc, [&]() {
        return cc->EvalSub(masks.activeMask, xorAB);
    });

    for (uint32_t layer = 0; layer < kNumLayers; ++layer) {
        const int32_t rotation =
            kDirectionSign * static_cast<int32_t>(1U << layer);
        const auto Gold = G;
        const auto Pold = P;

        auto Grot = detail::SynchronizedCall(cc, [&]() {
            return cc->EvalRotate(Gold, rotation);
        });
        auto Prot = detail::SynchronizedCall(cc, [&]() {
            return cc->EvalRotate(Pold, rotation);
        });
        auto term = detail::SynchronizedCall(cc, [&]() {
            return cc->EvalMult(Grot, Pold);
        });
        if (layer + 1 < kNumLayers) {
            P = detail::SynchronizedCall(cc, [&]() {
                return cc->EvalMult(Prot, Pold);
            });
        }
        G = detail::SynchronizedCall(cc, [&]() {
            return cc->EvalAdd(Gold, term);
        });
    }

    auto borrowIn = detail::SynchronizedCall(cc, [&]() {
        return cc->EvalRotate(G, kDirectionSign);
    });
    borrowIn = detail::SynchronizedCall(cc, [&]() {
        return cc->EvalMult(borrowIn, masks.carryMask);
    });
    G = detail::SynchronizedCall(cc, [&]() {
        return cc->EvalMult(G, masks.activeMask);
    });

    if (finalBorrowPrefix != nullptr) {
        *finalBorrowPrefix = G;
    }

    return detail::SynchronizedCall(cc, [&]() {
        auto result = cc->EvalSub(cA, cB);
        result = cc->EvalSub(result, borrowIn);
        result = cc->EvalAdd(result, cc->EvalAdd(G, G));
        return result;
    });
}

inline Ciphertext<DCRTPoly> EvalGreaterEqual(
    const CryptoContext<DCRTPoly>& cc,
    const Ciphertext<DCRTPoly>& cA,
    const Ciphertext<DCRTPoly>& cB,
    Masks& masks) {

    if (!cc || !cA || !cB || !masks.activeMask || !masks.resultMask) {
        throw std::invalid_argument(
            "EvalGreaterEqual received an invalid argument");
    }

    auto ab = detail::SynchronizedCall(cc, [&]() {
        return cc->EvalMult(cA, cB);
    });
    auto xorAB = detail::SynchronizedCall(cc, [&]() {
        return cc->EvalSub(cc->EvalAdd(cA, cB), cc->EvalAdd(ab, ab));
    });
    auto G = detail::SynchronizedCall(cc, [&]() {
        return cc->EvalSub(cB, ab);
    });
    auto P = detail::SynchronizedCall(cc, [&]() {
        return cc->EvalSub(masks.activeMask, xorAB);
    });

    for (uint32_t layer = 0; layer < kNumLayers; ++layer) {
        const int32_t step = static_cast<int32_t>(1U << layer);
        const auto Gold = G;
        const auto Pold = P;

        auto Ghigh = detail::SynchronizedCall(cc, [&]() {
            return cc->EvalRotate(Gold, step);
        });
        auto Phigh = detail::SynchronizedCall(cc, [&]() {
            return cc->EvalRotate(Pold, step);
        });
        auto lowerTerm = detail::SynchronizedCall(cc, [&]() {
            return cc->EvalMult(Phigh, Gold);
        });
        if (layer + 1 < kNumLayers) {
            P = detail::SynchronizedCall(cc, [&]() {
                return cc->EvalMult(Phigh, Pold);
            });
        }
        G = detail::SynchronizedCall(cc, [&]() {
            return cc->EvalAdd(Ghigh, lowerTerm);
        });
    }

    auto lessThan = detail::SynchronizedCall(cc, [&]() {
        return cc->EvalMult(G, masks.resultMask);
    });
    return detail::SynchronizedCall(cc, [&]() {
        return cc->EvalSub(masks.resultMask, lessThan);
    });
}

inline Ciphertext<DCRTPoly> BroadcastBit0(
    const CryptoContext<DCRTPoly>& cc,
    const Ciphertext<DCRTPoly>& bit0) {

    if (!cc || !bit0) {
        throw std::invalid_argument("BroadcastBit0 received a null input");
    }

    auto result = bit0;
    for (uint32_t layer = 0; layer < kNumLayers; ++layer) {
        const int32_t rotation =
            -static_cast<int32_t>(1U << layer);
        const auto old = result;
        auto rotated = detail::SynchronizedCall(cc, [&]() {
            return cc->EvalRotate(old, rotation);
        });
        result = detail::SynchronizedCall(cc, [&]() {
            return cc->EvalAdd(old, rotated);
        });
    }
    return result;
}

inline Ciphertext<DCRTPoly> ExtractFinalBorrowBit(
    const CryptoContext<DCRTPoly>& cc,
    const Ciphertext<DCRTPoly>& borrowPrefix,
    Masks& masks) {

    if (!cc || !borrowPrefix || !masks.resultMask) {
        throw std::invalid_argument(
            "ExtractFinalBorrowBit received an invalid argument");
    }

    auto shifted = detail::SynchronizedCall(cc, [&]() {
        return cc->EvalRotate(
            borrowPrefix, static_cast<int32_t>(kWordBits - 1U));
    });
    return detail::SynchronizedCall(cc, [&]() {
        return cc->EvalMult(shifted, masks.resultMask);
    });
}

inline Ciphertext<DCRTPoly> EvalCmux(
    const CryptoContext<DCRTPoly>& cc,
    const Ciphertext<DCRTPoly>& conditionMask,
    const Ciphertext<DCRTPoly>& trueValue,
    const Ciphertext<DCRTPoly>& falseValue) {

    if (!cc || !conditionMask || !trueValue || !falseValue) {
        throw std::invalid_argument("EvalCmux received a null input");
    }
    auto difference = detail::SynchronizedCall(cc, [&]() {
        return cc->EvalSub(trueValue, falseValue);
    });
    auto selectedDifference = detail::SynchronizedCall(cc, [&]() {
        return cc->EvalMult(conditionMask, difference);
    });
    return detail::SynchronizedCall(cc, [&]() {
        return cc->EvalAdd(falseValue, selectedDifference);
    });
}

inline Ciphertext<DCRTPoly> EvalBooleanMultiply(
    const CryptoContext<DCRTPoly>& cc,
    const Ciphertext<DCRTPoly>& value,
    const Ciphertext<DCRTPoly>& conditionMask) {

    if (!cc || !value || !conditionMask) {
        throw std::invalid_argument(
            "EvalBooleanMultiply received a null input");
    }
    return detail::SynchronizedCall(cc, [&]() {
        return cc->EvalMult(value, conditionMask);
    });
}

inline TransferResult EvalTransfer(
    const CryptoContext<DCRTPoly>& cc,
    const Ciphertext<DCRTPoly>& senderBalance,
    const Ciphertext<DCRTPoly>& receiverBalance,
    const Ciphertext<DCRTPoly>& amount,
    Masks& masks,
    Scenario scenario,
    const BootstrapPolicy& bootstrapPolicy,
    TransferTiming* timing = nullptr) {

    if (!cc || !senderBalance || !receiverBalance || !amount) {
        throw std::invalid_argument("EvalTransfer received a null input");
    }

    TransferTiming local;
    TransferTiming& report = timing ? *timing : local;
    report = TransferTiming{};

    detail::Synchronize(cc);
    const auto totalBegin = detail::Clock::now();
    detail::LevelBootstrapManager boot(
        cc, masks, bootstrapPolicy, &report);

    // Local shared-pointer copies allow the manager to replace only the local
    // value with a refreshed ciphertext. The caller's inputs are not mutated.
    auto sender = senderBalance;
    auto receiver = receiverBalance;
    auto transferAmountInput = amount;

    Ciphertext<DCRTPoly> newSender;
    Ciphertext<DCRTPoly> newReceiver;

    if (scenario == Scenario::Whitepaper || scenario == Scenario::NoCmux) {
        sender = boot.EnsureVector(
            sender, kCompareDepthBudget, "sender before greater_equal");
        transferAmountInput = boot.EnsureVector(
            transferAmountInput, kCompareDepthBudget,
            "amount before greater_equal");

        auto hasEnough = detail::TimedCall(
            cc, &report.compareOrOverflowSubMs,
            "greater_equal",
            [&]() {
                return EvalGreaterEqual(
                    cc, sender, transferAmountInput, masks);
            });
        detail::TraceLevel(bootstrapPolicy, "greater_equal output", hasEnough);

        // Broadcast consists only of rotations and additions. Do not bootstrap
        // the comparison result merely because it is Boolean; refresh it only
        // if the next ciphertext multiplication lacks depth.
        hasEnough = boot.EnsureBitZero(
            hasEnough, kSelectDepthBudget,
            "comparison bit before transfer selection");
        auto hasEnoughMask = detail::TimedCall(
            cc, &report.conditionPrepareMs,
            "broadcast comparison bit",
            [&]() { return BroadcastBit0(cc, hasEnough); });

        transferAmountInput = boot.EnsureVector(
            transferAmountInput, kSelectDepthBudget,
            "amount before transfer selection");
        hasEnoughMask = boot.EnsureVector(
            hasEnoughMask, kSelectDepthBudget,
            "condition mask before transfer selection");

        Ciphertext<DCRTPoly> amountToTransfer;
        if (scenario == Scenario::Whitepaper) {
            amountToTransfer = detail::TimedCall(
                cc, &report.transferSelectMs,
                "whitepaper CMUX",
                [&]() {
                    auto encryptedZero = cc->EvalSub(
                        transferAmountInput, transferAmountInput);
                    detail::Synchronize(cc);
                    return EvalCmux(
                        cc, hasEnoughMask,
                        transferAmountInput, encryptedZero);
                });
        }
        else {
            amountToTransfer = detail::TimedCall(
                cc, &report.transferSelectMs,
                "no_cmux Boolean multiply",
                [&]() {
                    return EvalBooleanMultiply(
                        cc, transferAmountInput, hasEnoughMask);
                });
        }
        detail::TraceLevel(
            bootstrapPolicy, "amount-to-transfer", amountToTransfer);

        sender = boot.EnsureVector(
            sender, kKsDepthBudget, "sender before sender subtraction");
        amountToTransfer = boot.EnsureVector(
            amountToTransfer, kKsDepthBudget,
            "amount-to-transfer before sender subtraction");
        newSender = detail::TimedCall(
            cc, &report.senderUpdateMs,
            "sender subtraction",
            [&]() {
                return EvalSubKS(
                    cc, sender, amountToTransfer, masks);
            });

        receiver = boot.EnsureVector(
            receiver, kKsDepthBudget, "receiver before receiver addition");
        amountToTransfer = boot.EnsureVector(
            amountToTransfer, kKsDepthBudget,
            "amount-to-transfer before receiver addition");
        newReceiver = detail::TimedCall(
            cc, &report.receiverUpdateMs,
            "receiver addition",
            [&]() {
                return EvalAddKS(
                    cc, receiver, amountToTransfer, masks);
            });
    }
    else {
        sender = boot.EnsureVector(
            sender, kKsDepthBudget,
            "sender before overflowing subtraction");
        transferAmountInput = boot.EnsureVector(
            transferAmountInput, kKsDepthBudget,
            "amount before overflowing subtraction");

        Ciphertext<DCRTPoly> borrowPrefix;
        auto difference = detail::TimedCall(
            cc, &report.compareOrOverflowSubMs,
            "overflowing subtraction",
            [&]() {
                return EvalSubKS(
                    cc, sender, transferAmountInput, masks, &borrowPrefix);
            });

        borrowPrefix = boot.EnsureVector(
            borrowPrefix, kBorrowExtractDepthBudget,
            "borrow prefix before underflow extraction");

        Ciphertext<DCRTPoly> underflowMask;
        Ciphertext<DCRTPoly> hasEnoughMask;
        std::tie(underflowMask, hasEnoughMask) = detail::TimedCall(
            cc, &report.conditionPrepareMs,
            "extract/broadcast underflow",
            [&]() {
                auto underflowBit =
                    ExtractFinalBorrowBit(cc, borrowPrefix, masks);
                auto underflow = BroadcastBit0(cc, underflowBit);
                auto hasEnough = cc->EvalSub(masks.activeMask, underflow);
                detail::Synchronize(cc);
                return std::make_pair(underflow, hasEnough);
            });

        underflowMask = boot.EnsureVector(
            underflowMask, kSelectDepthBudget,
            "underflow mask before sender CMUX");
        sender = boot.EnsureVector(
            sender, kSelectDepthBudget, "sender before overflow CMUX");
        difference = boot.EnsureVector(
            difference, kSelectDepthBudget,
            "difference before overflow CMUX");
        newSender = detail::TimedCall(
            cc, &report.senderUpdateMs,
            "overflow sender CMUX",
            [&]() {
                return EvalCmux(
                    cc, underflowMask, sender, difference);
            });

        hasEnoughMask = boot.EnsureVector(
            hasEnoughMask, kSelectDepthBudget,
            "has-enough mask before Boolean multiply");
        transferAmountInput = boot.EnsureVector(
            transferAmountInput, kSelectDepthBudget,
            "amount before Boolean multiply");
        auto amountToTransfer = detail::TimedCall(
            cc, &report.transferSelectMs,
            "overflow Boolean multiply",
            [&]() {
                return EvalBooleanMultiply(
                    cc, transferAmountInput, hasEnoughMask);
            });

        receiver = boot.EnsureVector(
            receiver, kKsDepthBudget,
            "receiver before overflow receiver addition");
        amountToTransfer = boot.EnsureVector(
            amountToTransfer, kKsDepthBudget,
            "amount-to-transfer before overflow receiver addition");
        newReceiver = detail::TimedCall(
            cc, &report.receiverUpdateMs,
            "overflow receiver addition",
            [&]() {
                return EvalAddKS(
                    cc, receiver, amountToTransfer, masks);
            });
    }

    // Final state refresh.  The output levels are expected to be 21
    // (whitepaper/no_cmux) or 22 (overflow).  With multiplicativeDepth=25,
    // both are still within the latest legal bootstrap input level 23.
    detail::Synchronize(cc);
    const auto finalBootstrapBegin = detail::Clock::now();

    auto finalBootstrap = [&](Ciphertext<DCRTPoly> value,
                              const char* label) {
        if (!value) {
            throw std::invalid_argument(std::string(label) + " is null");
        }

        const uint32_t before = value->GetLevel();
        if (before > bootstrapPolicy.bootstrapInputMaxLevel) {
            throw std::runtime_error(
                std::string(label) +
                ": final bootstrap requested too late; level=" +
                std::to_string(before) + ", latest=" +
                std::to_string(bootstrapPolicy.bootstrapInputMaxLevel));
        }

        std::cerr << "[ERC7984 final boot] BEGIN " << label
                  << ", input level=" << before << std::endl;

        // Keep the source ciphertext alive until the asynchronous GPU
        // bootstrap has completed.
        auto source = value;
        auto refreshed = cc->EvalBootstrap(value);
        detail::Synchronize(cc);
        source.reset();

        std::cerr << "[ERC7984 final boot] END   " << label
                  << ", output level=" << refreshed->GetLevel()
                  << std::endl;
        return refreshed;
    };

    newSender = finalBootstrap(newSender, "new sender");
    newReceiver = finalBootstrap(newReceiver, "new receiver");

    report.conditionalBootstrapMs += detail::Milliseconds(
        finalBootstrapBegin, detail::Clock::now());
    report.bootstrapCount += 2;

    report.totalMs =
        detail::Milliseconds(totalBegin, detail::Clock::now());

    detail::TraceLevel(bootstrapPolicy, "new sender output", newSender);
    detail::TraceLevel(bootstrapPolicy, "new receiver output", newReceiver);
    return {newSender, newReceiver};
}

}  // namespace erc7984gpu
