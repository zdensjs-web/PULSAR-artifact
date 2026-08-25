#pragma once

#include "word_operator_common.h"

#include <algorithm>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>

namespace ksdivcipher {

constexpr uint32_t kRadixBits = 4;
constexpr uint32_t kRadix = 1U << kRadixBits;
constexpr uint32_t kDefaultParallelBlocks = 1;
constexpr uint32_t kDefaultDivisionDepth = 36;
constexpr uint32_t kBootstrapEncodingLevelBudget = 4;
constexpr uint32_t kBootstrapDecodingLevelBudget = 4;
// With the {4,4} level budget and UNIFORM_TERNARY keys used by this runtime,
// OpenFHE/FIDES EvalBootstrap returns level 21 at scaleDeg=2. Reusable Boolean
// state is projected with 3x^2-2x^3 after Bootstrap and then lane-masked. The
// projection consumes two levels and the mask consumes one more level.
constexpr uint32_t kBootstrapOutputLevel = 21;
constexpr uint32_t kCanonicalStateLevel = kBootstrapOutputLevel + 3U;
constexpr uint32_t kBooleanPurificationLevels = 3;

struct RuntimeConfig : ksword::RuntimeConfig {
    uint32_t parallelBlocks = kDefaultParallelBlocks;
    bool graphInit = true;
    bool depthExplicit = false;

    uint32_t wideBits() const {
        return 2U * wordBits;
    }

    uint32_t wideLayers() const {
        return ksword::IntegerLog2(wideBits());
    }

    uint32_t radixRounds() const {
        return wordBits / kRadixBits;
    }

    uint32_t effectiveParallelBlocks() const {
        return std::min(parallelBlocks, kRadixBits);
    }

    uint32_t trialLevelCost() const {
        // This is the span from the canonical input to the bottom-level mux,
        // not one uninterrupted multiplicative circuit.  The wide G/P scan
        // is refreshed after its first four layers.
        return expectedTrialOutputLevel() - expectedTrialInputLevel();
    }

    uint32_t expectedTrialOutputLevel() const {
        return multiplicativeDepth - 2U;
    }

    uint32_t expectedTrialInputLevel() const {
        return kCanonicalStateLevel;
    }

    uint32_t prefixLayersBeforeRefresh() const {
        return std::min(uint32_t{4}, wideLayers());
    }

    uint32_t prefixLayersAfterRefresh() const {
        return wideLayers() - prefixLayersBeforeRefresh();
    }

    uint32_t minimumDepth() const {
        // G/P start one level after the canonical input.  Before the
        // midpoint refresh they consume b prefix levels.  After refreshing
        // to level 24, the remaining r levels are followed by one output
        // mask, a three-level borrow projection, and one mux.  Both refresh
        // inputs are padded to depth-2, so the larger side fixes the chain.
        const uint32_t preRefreshEnd =
            kCanonicalStateLevel + 1U + prefixLayersBeforeRefresh();
        const uint32_t selectedStateEnd =
            kCanonicalStateLevel + prefixLayersAfterRefresh() + 5U;
        return std::max(preRefreshEnd, selectedStateEnd) + 2U;
    }
};

inline void FinalizeConfig(RuntimeConfig* config) {
    if (!ksword::IsPowerOfTwo(config->wordBits) ||
        config->wordBits < 8 || config->wordBits > 256) {
        throw std::invalid_argument(
            "--bits must be a power of two between 8 and 256");
    }
    if (config->wordBits % kRadixBits != 0) {
        throw std::invalid_argument(
            "--bits must be divisible by the radix-16 digit width");
    }
    if (config->parallelBlocks == 0 ||
        config->parallelBlocks > kRadixBits) {
        throw std::invalid_argument(
            "--parallel-blocks must be between 1 and 4");
    }

    // A radix-16 recurrence can temporarily reach almost 16*D, so each
    // packed word reserves 2w active bits and an equally sized guard.
    config->guardBits = config->wideBits();
    config->stride = 2U * config->wideBits();
    if (!config->wordsExplicit) {
        config->words = config->maxWords();
    }
    if (config->words == 0 || config->words > config->maxWords()) {
        throw std::invalid_argument(
            "--words exceeds the internal 2*bits lane capacity");
    }

    if (!config->depthExplicit) {
        config->multiplicativeDepth = config->minimumDepth();
    }
    if (config->multiplicativeDepth < config->minimumDepth()) {
        throw std::invalid_argument(
            "--depth is too small for the midpoint-refreshed standard-key "
            "candidate trial; the selected width requires at least " +
            std::to_string(config->minimumDepth()));
    }

    if (config->backend == ksword::Backend::Cpu) {
        config->devices.clear();
        config->graphInit = false;
    }
    else if (config->devices.empty()) {
        throw std::invalid_argument(
            "GPU backend requires at least one logical device");
    }
}

inline RuntimeConfig ParseCommandLine(int argc, char** argv) {
    RuntimeConfig config;
    config.multiplicativeDepth = kDefaultDivisionDepth;
    bool gpuCountExplicit = false;
    bool deviceListExplicit = false;

    for (int index = 1; index < argc; ++index) {
        const std::string option(argv[index]);
        const auto requireValue = [&]() -> std::string {
            if (index + 1 >= argc) {
                throw std::invalid_argument(option + " requires a value");
            }
            return std::string(argv[++index]);
        };

        if (option == "--bits") {
            config.wordBits =
                ksword::ParseU32(requireValue(), "--bits");
        }
        else if (option == "--words") {
            config.words =
                ksword::ParseU32(requireValue(), "--words");
            config.wordsExplicit = true;
        }
        else if (option == "--parallel-blocks") {
            config.parallelBlocks =
                ksword::ParseU32(
                    requireValue(), "--parallel-blocks");
        }
        else if (option == "--backend") {
            config.backend =
                ksword::ParseBackend(requireValue());
        }
        else if (option == "--gpus") {
            if (deviceListExplicit) {
                throw std::invalid_argument(
                    "--gpus and --devices are mutually exclusive");
            }
            config.devices = ksword::SequentialDevices(
                ksword::ParseU32(requireValue(), "--gpus"));
            gpuCountExplicit = true;
        }
        else if (option == "--devices") {
            if (gpuCountExplicit) {
                throw std::invalid_argument(
                    "--gpus and --devices are mutually exclusive");
            }
            config.devices =
                ksword::ParseDeviceList(requireValue());
            deviceListExplicit = true;
        }
        else if (option == "--depth") {
            config.multiplicativeDepth =
                ksword::ParseU32(requireValue(), "--depth");
            config.depthExplicit = true;
        }
        else if (option == "--profile") {
            config.detailedProfile =
                ksword::ParseBool(requireValue(), "--profile");
        }
        else if (option == "--modup-init" ||
                 option == "--graph-init") {
            config.graphInit =
                ksword::ParseBool(requireValue(), "--modup-init");
        }
        else if (option == "--help" || option == "-h") {
            throw std::runtime_error("help");
        }
        else if (option == "--bootstrap") {
            throw std::invalid_argument(
                "ciphertext division always performs required internal "
                "and final Bootstraps");
        }
        else {
            throw std::invalid_argument("unknown option: " + option);
        }
    }

    FinalizeConfig(&config);
    return config;
}

inline std::string Usage(const char* executable) {
    std::ostringstream output;
    output
        << "Usage: " << executable << " [options]\n"
        << "  --bits N             width: 8,16,32,64,128,256 "
           "(default 256)\n"
        << "  --words N            packed encrypted divisions "
           "(default: maximum for the 2w lane)\n"
        << "  --parallel-blocks N  async setup/digit-placement "
           "batch size, 1..4 (default 1)\n"
        << "  --backend B          gpu or cpu (default gpu)\n"
        << "  --gpus N             logical GPUs 0..N-1 (default 2)\n"
        << "  --devices LIST       logical GPU list, for example 0,2,3\n"
        << "  --depth N            CKKS depth (default: width-specific minimum; "
           "36 at 256 bits)\n"
        << "  --modup-init 0|1     initialize level-specific GPU graphs/workspaces "
           "(default 1 on GPU)\n"
        << "  --profile 0|1        synchronized internal timings\n"
        << "\nBoth dividend and divisor are encrypted. Division by zero "
           "returns\n"
        << "quotient=0 and remainder=dividend. All purification and "
           "Bootstrap\n"
        << "calls are included in online latency.\n";
    return output.str();
}

}  // namespace ksdivcipher
