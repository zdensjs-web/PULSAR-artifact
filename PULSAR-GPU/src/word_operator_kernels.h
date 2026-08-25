#pragma once

#include <fideslib.hpp>

#include "word_operator_common.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace ksword {

using fideslib::Ciphertext;
using fideslib::CryptoContext;
using fideslib::DCRTPoly;
using fideslib::Plaintext;

enum class OperatorKind {
    ScalarLeftShift,
    ScalarRightShift,
    LeftShift,
    RightShift,
    Xor,
    And,
    Or,
    Not,
};

inline const char* OperatorName(OperatorKind kind) {
    switch (kind) {
        case OperatorKind::ScalarLeftShift:
            return "scalar_left_shift";
        case OperatorKind::ScalarRightShift:
            return "scalar_right_shift";
        case OperatorKind::LeftShift:
            return "left_shift";
        case OperatorKind::RightShift:
            return "right_shift";
        case OperatorKind::Xor:
            return "xor";
        case OperatorKind::And:
            return "and";
        case OperatorKind::Or:
            return "or";
        case OperatorKind::Not:
            return "not";
    }
    return "unknown";
}

inline bool IsScalarShift(OperatorKind kind) {
    return kind == OperatorKind::ScalarLeftShift ||
           kind == OperatorKind::ScalarRightShift;
}

inline bool IsEncryptedShift(OperatorKind kind) {
    return kind == OperatorKind::LeftShift ||
           kind == OperatorKind::RightShift;
}

inline bool IsLeftShift(OperatorKind kind) {
    return kind == OperatorKind::ScalarLeftShift ||
           kind == OperatorKind::LeftShift;
}

inline bool IsUnary(OperatorKind kind) {
    return IsScalarShift(kind) || kind == OperatorKind::Not;
}

inline bool RequiresBootstrap(OperatorKind kind) {
    return kind != OperatorKind::Not;
}

inline bool NeedsEvalMultKey(OperatorKind kind) {
    return IsEncryptedShift(kind) ||
           kind == OperatorKind::Xor ||
           kind == OperatorKind::And ||
           kind == OperatorKind::Or;
}

struct PatternTiming {
    double rotateMs = 0.0;
    double addMs = 0.0;
    double wallMs = 0.0;
};

struct ShiftLayerTiming {
    double controlMaskMs = 0.0;
    double controlAnchorMs = 0.0;
    std::vector<double> broadcastRotateMs;
    std::vector<double> broadcastAddMs;
    double controlWallMs = 0.0;
    double dataRotateMs = 0.0;
    double shiftMaskMs = 0.0;
    double differenceMs = 0.0;
    double selectMultMs = 0.0;
    double selectAddMs = 0.0;
    double wallMs = 0.0;
};

struct TimingReport {
    double inputAddMs = 0.0;
    double productMs = 0.0;
    double doubleProductMs = 0.0;
    double subtractMs = 0.0;
    double rotateMs = 0.0;
    double maskMs = 0.0;
    double zeroMs = 0.0;
    double lowControlMaskMs = 0.0;
    std::vector<PatternTiming> patternLayers;
    std::vector<ShiftLayerTiming> shiftLayers;
    double totalMs = 0.0;
};

struct Masks {
    Plaintext active;
    Plaintext scalarShift;
    Plaintext lowControl;
    std::vector<Plaintext> controlLanes;
    std::vector<Plaintext> shiftStages;
};

namespace detail {

inline void Synchronize(const CryptoContext<DCRTPoly>& cc) {
    cc->Synchronize();
}

template <class Fn>
inline auto TimedCall(const CryptoContext<DCRTPoly>& cc,
                      double* elapsedMs,
                      Fn&& fn) {
    if (elapsedMs == nullptr) {
        return fn();
    }
    Synchronize(cc);
    const auto begin = Clock::now();
    auto result = fn();
    Synchronize(cc);
    *elapsedMs = Milliseconds(begin, Clock::now());
    return result;
}

inline std::vector<double> ActiveValues(const RuntimeConfig& config) {
    std::vector<double> values(config.slots, 0.0);
    for (uint32_t word = 0; word < config.words; ++word) {
        for (uint32_t bit = 0; bit < config.wordBits; ++bit) {
            values[Slot(config, word, bit)] = 1.0;
        }
    }
    return values;
}

inline std::vector<double> ScalarShiftValues(
    const RuntimeConfig& config,
    OperatorKind kind) {
    std::vector<double> values(config.slots, 0.0);
    if (config.scalarShift >= config.wordBits) {
        return values;
    }
    for (uint32_t word = 0; word < config.words; ++word) {
        if (IsLeftShift(kind)) {
            for (uint32_t bit = config.scalarShift;
                 bit < config.wordBits;
                 ++bit) {
                values[Slot(config, word, bit)] = 1.0;
            }
        }
        else {
            for (uint32_t bit = 0;
                 bit < config.wordBits - config.scalarShift;
                 ++bit) {
                values[Slot(config, word, bit)] = 1.0;
            }
        }
    }
    return values;
}

inline std::vector<double> LowControlValues(
    const RuntimeConfig& config) {
    std::vector<double> values(config.slots, 0.0);
    for (uint32_t word = 0; word < config.words; ++word) {
        for (uint32_t bit = 0; bit < config.controlBits(); ++bit) {
            values[Slot(config, word, bit)] = 1.0;
        }
    }
    return values;
}

inline std::vector<double> ControlLaneValues(
    const RuntimeConfig& config,
    uint32_t lane) {
    const uint32_t block = NextPowerOfTwo(config.controlBits());
    std::vector<double> values(config.slots, 0.0);
    for (uint32_t word = 0; word < config.words; ++word) {
        for (uint32_t bit = lane; bit < config.wordBits; bit += block) {
            values[Slot(config, word, bit)] = 1.0;
        }
    }
    return values;
}

inline std::vector<double> ShiftStageValues(
    const RuntimeConfig& config,
    OperatorKind kind,
    uint32_t distance) {
    std::vector<double> values(config.slots, 0.0);
    for (uint32_t word = 0; word < config.words; ++word) {
        if (IsLeftShift(kind)) {
            for (uint32_t bit = distance;
                 bit < config.wordBits;
                 ++bit) {
                values[Slot(config, word, bit)] = 1.0;
            }
        }
        else {
            for (uint32_t bit = 0;
                 bit < config.wordBits - distance;
                 ++bit) {
                values[Slot(config, word, bit)] = 1.0;
            }
        }
    }
    return values;
}

inline Plaintext MakeAndLoad(
    const CryptoContext<DCRTPoly>& cc,
    const RuntimeConfig& config,
    const std::vector<double>& values) {
    auto plaintext = cc->MakeCKKSPackedPlaintext(
        values, 1, 0, nullptr, config.slots);
    plaintext->SetLength(config.slots);
    cc->LoadPlaintext(plaintext);
    return plaintext;
}

}  // namespace detail

inline Masks BuildAndLoadMasks(const CryptoContext<DCRTPoly>& cc,
                               const RuntimeConfig& config,
                               OperatorKind kind) {
    if (!cc) {
        throw std::invalid_argument(
            "word-operator BuildAndLoadMasks received null context");
    }

    Masks masks;
    if (kind == OperatorKind::Not) {
        masks.active = detail::MakeAndLoad(
            cc, config, detail::ActiveValues(config));
    }
    else if (IsScalarShift(kind)) {
        masks.scalarShift = detail::MakeAndLoad(
            cc, config, detail::ScalarShiftValues(config, kind));
    }
    else if (IsEncryptedShift(kind)) {
        masks.lowControl = detail::MakeAndLoad(
            cc, config, detail::LowControlValues(config));
        masks.controlLanes.resize(config.controlBits());
        masks.shiftStages.resize(config.controlBits());
        for (uint32_t layer = 0;
             layer < config.controlBits();
             ++layer) {
            masks.controlLanes[layer] = detail::MakeAndLoad(
                cc, config, detail::ControlLaneValues(config, layer));
            masks.shiftStages[layer] = detail::MakeAndLoad(
                cc, config,
                detail::ShiftStageValues(
                    config, kind, 1U << layer));
        }
    }
    detail::Synchronize(cc);
    return masks;
}

inline std::vector<int32_t> RotationIndices(
    const RuntimeConfig& config,
    OperatorKind kind) {
    std::vector<int32_t> rotations;
    if (IsScalarShift(kind)) {
        if (config.scalarShift > 0 &&
            config.scalarShift < config.wordBits) {
            const int32_t shift =
                static_cast<int32_t>(config.scalarShift);
            rotations.push_back(IsLeftShift(kind) ? -shift : shift);
        }
    }
    else if (IsEncryptedShift(kind)) {
        const uint32_t controlBits = config.controlBits();
        const uint32_t block = NextPowerOfTwo(controlBits);

        for (uint32_t distance = block;
             distance < config.wordBits;
             distance <<= 1U) {
            rotations.push_back(-static_cast<int32_t>(distance));
        }
        for (uint32_t lane = 1; lane < controlBits; ++lane) {
            rotations.push_back(static_cast<int32_t>(lane));
        }
        for (uint32_t distance = 1;
             distance < block;
             distance <<= 1U) {
            rotations.push_back(-static_cast<int32_t>(distance));
        }
        for (uint32_t layer = 0; layer < controlBits; ++layer) {
            const int32_t distance =
                static_cast<int32_t>(1U << layer);
            rotations.push_back(
                IsLeftShift(kind) ? -distance : distance);
        }
    }
    std::sort(rotations.begin(), rotations.end());
    rotations.erase(
        std::unique(rotations.begin(), rotations.end()),
        rotations.end());
    return rotations;
}

inline Ciphertext<DCRTPoly> EvalScalarShift(
    const CryptoContext<DCRTPoly>& cc,
    const Ciphertext<DCRTPoly>& input,
    const RuntimeConfig& config,
    OperatorKind kind,
    Masks& masks,
    TimingReport* report) {
    const bool profile = report != nullptr;
    if (config.scalarShift >= config.wordBits) {
        return detail::TimedCall(
            cc, profile ? &report->zeroMs : nullptr,
            [&]() { return cc->EvalSub(input, input); });
    }
    if (config.scalarShift == 0) {
        return input;
    }

    const int32_t shift =
        static_cast<int32_t>(config.scalarShift);
    auto rotated = detail::TimedCall(
        cc, profile ? &report->rotateMs : nullptr,
        [&]() {
            return cc->EvalRotate(
                input, IsLeftShift(kind) ? -shift : shift);
        });
    return detail::TimedCall(
        cc, profile ? &report->maskMs : nullptr,
        [&]() { return cc->EvalMult(rotated, masks.scalarShift); });
}

inline Ciphertext<DCRTPoly> EvalEncryptedShift(
    const CryptoContext<DCRTPoly>& cc,
    const Ciphertext<DCRTPoly>& input,
    const Ciphertext<DCRTPoly>& encryptedShift,
    const RuntimeConfig& config,
    OperatorKind kind,
    Masks& masks,
    TimingReport* report) {
    const bool profile = report != nullptr;
    const uint32_t controlBits = config.controlBits();
    const uint32_t block = NextPowerOfTwo(controlBits);
    const uint32_t patternLayers =
        IntegerLog2(config.wordBits / block);
    const uint32_t broadcastLayers = IntegerLog2(block);

    if (profile) {
        report->patternLayers.resize(patternLayers);
        report->shiftLayers.resize(controlBits);
        for (auto& layer : report->shiftLayers) {
            layer.broadcastRotateMs.resize(broadcastLayers);
            layer.broadcastAddMs.resize(broadcastLayers);
        }
    }

    auto pattern = detail::TimedCall(
        cc, profile ? &report->lowControlMaskMs : nullptr,
        [&]() {
            return cc->EvalMult(encryptedShift, masks.lowControl);
        });

    for (uint32_t layer = 0; layer < patternLayers; ++layer) {
        auto* item = profile ? &report->patternLayers[layer] : nullptr;
        const auto layerBegin = Clock::now();
        const int32_t distance =
            -static_cast<int32_t>(block << layer);
        const auto oldPattern = pattern;
        auto rotated = detail::TimedCall(
            cc, item ? &item->rotateMs : nullptr,
            [&]() { return cc->EvalRotate(oldPattern, distance); });
        pattern = detail::TimedCall(
            cc, item ? &item->addMs : nullptr,
            [&]() { return cc->EvalAdd(oldPattern, rotated); });
        if (item) {
            detail::Synchronize(cc);
            item->wallMs = Milliseconds(layerBegin, Clock::now());
        }
    }

    auto current = input;
    for (uint32_t layer = 0; layer < controlBits; ++layer) {
        auto* item = profile ? &report->shiftLayers[layer] : nullptr;
        const auto layerBegin = Clock::now();
        const auto controlBegin = layerBegin;

        auto control = detail::TimedCall(
            cc, item ? &item->controlMaskMs : nullptr,
            [&]() {
                return cc->EvalMult(
                    pattern, masks.controlLanes[layer]);
            });
        if (layer != 0) {
            control = detail::TimedCall(
                cc, item ? &item->controlAnchorMs : nullptr,
                [&]() {
                    return cc->EvalRotate(
                        control, static_cast<int32_t>(layer));
                });
        }
        for (uint32_t broadcast = 0;
             broadcast < broadcastLayers;
             ++broadcast) {
            const int32_t distance =
                -static_cast<int32_t>(1U << broadcast);
            const auto oldControl = control;
            auto rotated = detail::TimedCall(
                cc,
                item ? &item->broadcastRotateMs[broadcast] : nullptr,
                [&]() { return cc->EvalRotate(oldControl, distance); });
            control = detail::TimedCall(
                cc,
                item ? &item->broadcastAddMs[broadcast] : nullptr,
                [&]() { return cc->EvalAdd(oldControl, rotated); });
        }
        if (item) {
            detail::Synchronize(cc);
            item->controlWallMs =
                Milliseconds(controlBegin, Clock::now());
        }

        const uint32_t distance = 1U << layer;
        const int32_t rotation =
            IsLeftShift(kind)
                ? -static_cast<int32_t>(distance)
                : static_cast<int32_t>(distance);
        const auto oldCurrent = current;
        auto rotatedData = detail::TimedCall(
            cc, item ? &item->dataRotateMs : nullptr,
            [&]() { return cc->EvalRotate(oldCurrent, rotation); });
        auto candidate = detail::TimedCall(
            cc, item ? &item->shiftMaskMs : nullptr,
            [&]() {
                return cc->EvalMult(
                    rotatedData, masks.shiftStages[layer]);
            });
        auto difference = detail::TimedCall(
            cc, item ? &item->differenceMs : nullptr,
            [&]() { return cc->EvalSub(candidate, oldCurrent); });
        auto selected = detail::TimedCall(
            cc, item ? &item->selectMultMs : nullptr,
            [&]() { return cc->EvalMult(control, difference); });
        current = detail::TimedCall(
            cc, item ? &item->selectAddMs : nullptr,
            [&]() { return cc->EvalAdd(oldCurrent, selected); });

        if (item) {
            detail::Synchronize(cc);
            item->wallMs = Milliseconds(layerBegin, Clock::now());
        }
    }
    return current;
}

inline Ciphertext<DCRTPoly> EvalConfiguredOperator(
    const CryptoContext<DCRTPoly>& cc,
    const Ciphertext<DCRTPoly>& cA,
    const Ciphertext<DCRTPoly>& cB,
    const RuntimeConfig& config,
    OperatorKind kind,
    Masks& masks,
    TimingReport* timing = nullptr) {
    if (!cc || !cA || (!IsUnary(kind) && !cB)) {
        throw std::invalid_argument(
            "EvalConfiguredOperator received invalid input");
    }

    TimingReport local;
    TimingReport& report = timing ? *timing : local;
    const bool profile = timing != nullptr;
    if (profile) {
        report = TimingReport{};
        detail::Synchronize(cc);
    }
    const auto totalBegin = Clock::now();

    Ciphertext<DCRTPoly> result;
    switch (kind) {
        case OperatorKind::ScalarLeftShift:
        case OperatorKind::ScalarRightShift:
            result = EvalScalarShift(
                cc, cA, config, kind, masks,
                profile ? &report : nullptr);
            break;
        case OperatorKind::LeftShift:
        case OperatorKind::RightShift:
            result = EvalEncryptedShift(
                cc, cA, cB, config, kind, masks,
                profile ? &report : nullptr);
            break;
        case OperatorKind::Xor: {
            auto sum = detail::TimedCall(
                cc, profile ? &report.inputAddMs : nullptr,
                [&]() { return cc->EvalAdd(cA, cB); });
            auto product = detail::TimedCall(
                cc, profile ? &report.productMs : nullptr,
                [&]() { return cc->EvalMult(cA, cB); });
            auto twiceProduct = detail::TimedCall(
                cc, profile ? &report.doubleProductMs : nullptr,
                [&]() { return cc->EvalAdd(product, product); });
            result = detail::TimedCall(
                cc, profile ? &report.subtractMs : nullptr,
                [&]() { return cc->EvalSub(sum, twiceProduct); });
            break;
        }
        case OperatorKind::And:
            result = detail::TimedCall(
                cc, profile ? &report.productMs : nullptr,
                [&]() { return cc->EvalMult(cA, cB); });
            break;
        case OperatorKind::Or: {
            auto sum = detail::TimedCall(
                cc, profile ? &report.inputAddMs : nullptr,
                [&]() { return cc->EvalAdd(cA, cB); });
            auto product = detail::TimedCall(
                cc, profile ? &report.productMs : nullptr,
                [&]() { return cc->EvalMult(cA, cB); });
            result = detail::TimedCall(
                cc, profile ? &report.subtractMs : nullptr,
                [&]() { return cc->EvalSub(sum, product); });
            break;
        }
        case OperatorKind::Not:
            result = detail::TimedCall(
                cc, profile ? &report.subtractMs : nullptr,
                [&]() { return cc->EvalSub(masks.active, cA); });
            break;
    }

    if (profile) {
        detail::Synchronize(cc);
        report.totalMs = Milliseconds(totalBegin, Clock::now());
    }
    return result;
}

}  // namespace ksword
