#pragma once

#include "word_operator_common.h"

#include <algorithm>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ksmul {

constexpr uint32_t kDefaultMergeBatch = 1;
constexpr uint32_t kBootstrapOutputLevel = 21;
constexpr uint32_t kBooleanProjectionLevels = 3;
constexpr uint32_t kCanonicalBooleanLevel =
    kBootstrapOutputLevel + kBooleanProjectionLevels;

struct RuntimeConfig : ksword::RuntimeConfig {
    uint32_t mergeBatch = kDefaultMergeBatch;
    bool depthExplicit = false;
    bool graphInit = true;
    bool audit = false;
    uint32_t inputLevel = 0;
    std::vector<uint32_t> daddaPurificationStages;
    uint32_t expectedDaddaPreRefreshLevel = 0;
    uint32_t expectedDaddaOutputLevel = 0;
    uint32_t finalAdderInputLevel = 0;
    uint32_t expectedOutputLevel = 0;
};

inline std::vector<uint32_t> DaddaTargets(uint32_t rows) {
    if (rows < 2) {
        throw std::invalid_argument(
            "Dadda reduction requires at least two input rows");
    }

    std::vector<uint32_t> ascending{2};
    while (ascending.back() < rows) {
        const uint32_t next = (3U * ascending.back()) / 2U;
        if (next <= ascending.back()) {
            throw std::overflow_error("Dadda target generation overflow");
        }
        ascending.push_back(next);
    }

    if (ascending.back() >= rows) {
        ascending.pop_back();
    }
    std::reverse(ascending.begin(), ascending.end());
    return ascending;
}

inline std::vector<uint32_t> DaddaPurificationStages(
    uint32_t wordBits) {
    const uint32_t stages =
        static_cast<uint32_t>(DaddaTargets(wordBits).size());
    if (stages < 3U) {
        return {};
    }

    // Divide the compressor tree into three bounded nonlinear segments.
    // For 256 bits (13 stages), purification follows stages 4 and 8.
    const uint32_t first = ((stages + 2U) / 3U) - 1U;
    const uint32_t second = ((2U * stages + 2U) / 3U) - 1U;
    if (first >= second || second + 1U >= stages) {
        throw std::runtime_error(
            "could not derive two internal Dadda purification boundaries");
    }
    return {first, second};
}

inline uint32_t DaddaPreRefreshLevel(uint32_t wordBits) {
    const uint32_t daddaStages =
        static_cast<uint32_t>(DaddaTargets(wordBits).size());
    const uint32_t projections = static_cast<uint32_t>(
        DaddaPurificationStages(wordBits).size());
    return 1U + 2U * daddaStages +
           kBooleanProjectionLevels * projections;
}

inline uint32_t FinalAdderLevelCost(uint32_t wordBits) {
    return ksword::IntegerLog2(wordBits) + 4U;
}

inline uint32_t MinimumMultiplicativeDepth(uint32_t wordBits) {
    // Both the pre-refresh Dadda path and final adder need two towers at a
    // Bootstrap boundary. The latter dominates for the supported widths,
    // but retain the maximum so the schedule remains mechanically checked.
    return std::max(
        DaddaPreRefreshLevel(wordBits) + 2U,
        kCanonicalBooleanLevel + FinalAdderLevelCost(wordBits) + 2U);
}

inline std::vector<uint32_t> RequiredActiveMaskLevels(
    const RuntimeConfig& config) {
    std::vector<uint32_t> rowLevels(config.wordBits, 1U);
    std::vector<uint32_t> maskLevels;
    const auto targets = DaddaTargets(config.wordBits);

    for (uint32_t stage = 0; stage < targets.size(); ++stage) {
        const uint32_t inputRows =
            static_cast<uint32_t>(rowLevels.size());
        const uint32_t fullAdders = inputRows - targets[stage];
        const uint32_t compressedRows = 3U * fullAdders;
        std::vector<uint32_t> output;
        output.reserve(targets[stage]);
        for (uint32_t offset = 0;
             offset < compressedRows;
             offset += 3U) {
            const uint32_t level =
                std::max({rowLevels[offset],
                          rowLevels[offset + 1U],
                          rowLevels[offset + 2U]}) + 2U;
            output.push_back(level);
            output.push_back(level);
        }
        output.insert(
            output.end(),
            rowLevels.begin() + compressedRows,
            rowLevels.end());

        if (std::find(
                config.daddaPurificationStages.begin(),
                config.daddaPurificationStages.end(),
                stage) != config.daddaPurificationStages.end()) {
            for (uint32_t& level : output) {
                maskLevels.push_back(level + 2U);
                level += kBooleanProjectionLevels;
            }
        }
        rowLevels = std::move(output);
    }

    for (uint32_t level : rowLevels) {
        while (level < config.expectedOutputLevel) {
            maskLevels.push_back(level++);
        }
    }
    // h(x) uses its active mask after the second ciphertext product.
    maskLevels.push_back(kBootstrapOutputLevel + 2U);
    for (uint32_t level = kCanonicalBooleanLevel;
         level < config.finalAdderInputLevel;
         ++level) {
        maskLevels.push_back(level);
    }

    std::sort(maskLevels.begin(), maskLevels.end());
    maskLevels.erase(
        std::unique(maskLevels.begin(), maskLevels.end()),
        maskLevels.end());
    return maskLevels;
}

inline void FinalizeConfig(RuntimeConfig* config) {
    ksword::FinalizeConfig(config);

    if (config->mergeBatch == 0 ||
        config->mergeBatch > config->wordBits) {
        throw std::invalid_argument(
            "--merge-batch must be between 1 and the selected word width");
    }

    config->daddaPurificationStages =
        DaddaPurificationStages(config->wordBits);
    config->expectedDaddaPreRefreshLevel =
        DaddaPreRefreshLevel(config->wordBits);
    const uint32_t minimumDepth =
        MinimumMultiplicativeDepth(config->wordBits);
    if (!config->depthExplicit) {
        config->multiplicativeDepth = minimumDepth;
    }
    if (config->multiplicativeDepth < minimumDepth) {
        throw std::invalid_argument(
            "--depth is too small for the multiplier and the two towers "
            "required by the Bootstrap input (minimum " +
            std::to_string(minimumDepth) + ")");
    }

    config->expectedOutputLevel =
        config->multiplicativeDepth - 2U;
    config->finalAdderInputLevel =
        config->expectedOutputLevel -
        FinalAdderLevelCost(config->wordBits);
    if (config->finalAdderInputLevel < kCanonicalBooleanLevel) {
        throw std::runtime_error(
            "the final-adder input would precede the canonical post-Bootstrap "
            "Boolean level");
    }
    config->inputLevel = 0U;
    config->expectedDaddaOutputLevel =
        config->finalAdderInputLevel;
}

inline RuntimeConfig ParseCommandLine(int argc, char** argv) {
    RuntimeConfig config;
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
        else if (option == "--merge-batch") {
            config.mergeBatch =
                ksword::ParseU32(requireValue(), "--merge-batch");
        }
        else if (option == "--profile") {
            config.detailedProfile =
                ksword::ParseBool(requireValue(), "--profile");
        }
        else if (option == "--audit") {
            config.audit =
                ksword::ParseBool(requireValue(), "--audit");
        }
        else if (option == "--modup-init" ||
                 option == "--graph-init") {
            config.graphInit =
                ksword::ParseBool(requireValue(), option.c_str());
        }
        else if (option == "--help" || option == "-h") {
            throw std::runtime_error("help");
        }
        else if (option == "--bootstrap") {
            throw std::invalid_argument(
                "the multiplier always performs one final Bootstrap");
        }
        else {
            throw std::invalid_argument("unknown option: " + option);
        }
    }

    FinalizeConfig(&config);
    if (config.backend == ksword::Backend::Cpu) {
        config.graphInit = false;
    }
    return config;
}

inline std::string Usage(const char* executable) {
    std::ostringstream output;
    output
        << "Usage: " << executable << " [options]\n"
        << "  --bits N          word width: 8,16,32,64,128,256 "
           "(default 256)\n"
        << "  --words N         packed words per ciphertext "
           "(default: maximum)\n"
        << "  --backend B       gpu or cpu (default gpu)\n"
        << "  --gpus N          use logical GPUs 0..N-1, N in [1,8] "
           "(default 2)\n"
        << "  --devices LIST    explicit logical GPUs, for example 0,2,3\n"
        << "  --merge-batch N   maximum 3:2 row merges between GPU "
           "barriers (default 1)\n"
        << "  --depth N         CKKS depth (default: width-derived)\n"
        << "  --graph-init 0|1  initialize final-adder graph/workspace "
           "(default 1 on GPU)\n"
        << "  --profile 0|1     synchronized internal timing (default 0)\n"
        << "  --audit 0|1       decrypt Dadda rows and raw product checkpoints\n"
        << "\nThe multiplier always returns A*B mod 2^bits. It purifies two\n"
        << "Dadda boundaries, refreshes the final two rows, and performs a\n"
        << "final Bootstrap. CUDA_VISIBLE_DEVICES selects physical GPUs;\n"
        << "--gpus/--devices use logical indices inside the process.\n";
    return output.str();
}

}  // namespace ksmul
