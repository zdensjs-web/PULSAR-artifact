#pragma once

#include "word_operator_common.h"

#include <boost/multiprecision/cpp_int.hpp>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace ksdivscalar {

using boost::multiprecision::cpp_int;

constexpr uint32_t kDefaultParallelBlocks = 1;
constexpr uint32_t kDefaultMergeBatch = 1;
constexpr uint32_t kDefaultDivisionDepth = 40;

struct RuntimeConfig : ksword::RuntimeConfig {
    cpp_int divisor = 3;
    std::string divisorText = "3";
    uint32_t parallelBlocks = kDefaultParallelBlocks;
    uint32_t mergeBatch = kDefaultMergeBatch;
    bool graphInit = true;
    bool depthExplicit = false;

    uint32_t internalBits() const {
        return 2U * wordBits;
    }

    uint32_t internalLayers() const {
        return ksword::IntegerLog2(internalBits());
    }
};

inline cpp_int ParseInteger(const std::string& text,
                            const char* option) {
    if (text.empty()) {
        throw std::invalid_argument(
            std::string(option) + " must not be empty");
    }

    size_t index = 0;
    unsigned radix = 10;
    if (text.size() > 2 && text[0] == '0' &&
        (text[1] == 'x' || text[1] == 'X')) {
        radix = 16;
        index = 2;
    }
    if (index == text.size()) {
        throw std::invalid_argument(
            std::string("invalid value for ") + option + ": " + text);
    }

    cpp_int value = 0;
    for (; index < text.size(); ++index) {
        const unsigned char raw =
            static_cast<unsigned char>(text[index]);
        unsigned digit = 0;
        if (raw >= '0' && raw <= '9') {
            digit = raw - '0';
        }
        else if (raw >= 'a' && raw <= 'f') {
            digit = 10U + raw - 'a';
        }
        else if (raw >= 'A' && raw <= 'F') {
            digit = 10U + raw - 'A';
        }
        else {
            throw std::invalid_argument(
                std::string("invalid value for ") + option + ": " + text);
        }
        if (digit >= radix) {
            throw std::invalid_argument(
                std::string("invalid value for ") + option + ": " + text);
        }
        value *= radix;
        value += digit;
    }
    return value;
}

inline uint32_t BitLength(cpp_int value) {
    if (value < 0) {
        value = -value;
    }
    uint32_t bits = 0;
    while (value != 0) {
        value >>= 1;
        ++bits;
    }
    return bits;
}

inline uint32_t Popcount(cpp_int value) {
    uint32_t count = 0;
    while (value != 0) {
        count += static_cast<uint32_t>((value & 1) != 0);
        value >>= 1;
    }
    return count;
}

inline std::vector<uint8_t> ToBits(cpp_int value,
                                   uint32_t count) {
    std::vector<uint8_t> bits(count, 0);
    for (uint32_t bit = 0; bit < count; ++bit) {
        bits[bit] =
            static_cast<uint8_t>((value & 1) != 0);
        value >>= 1;
    }
    if (value != 0) {
        throw std::overflow_error(
            "public integer does not fit the requested bit width");
    }
    return bits;
}

inline cpp_int Reciprocal(const RuntimeConfig& config) {
    return (cpp_int(1) << config.wordBits) / config.divisor;
}

inline std::vector<uint32_t> DaddaTargets(uint32_t rows) {
    if (rows <= 2) {
        return {};
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

inline uint32_t PublicMultiplyDepth(uint32_t oneBits,
                                    uint32_t outputBits) {
    // The public 0/1 selector mask consumes one level. Two or more
    // rows then enter the 3:2 tree and the latest prefix adder.
    if (oneBits <= 1) {
        return 1U;
    }
    return 1U +
        2U * static_cast<uint32_t>(
            DaddaTargets(oneBits).size()) +
        ksword::IntegerLog2(outputBits) + 3U;
}

inline void FinalizeConfig(RuntimeConfig* config) {
    if (!ksword::IsPowerOfTwo(config->wordBits) ||
        config->wordBits < 8 || config->wordBits > 256) {
        throw std::invalid_argument(
            "--bits must be a power of two between 8 and 256");
    }
    if (config->divisor <= 0 ||
        config->divisor >= (cpp_int(1) << config->wordBits)) {
        throw std::invalid_argument(
            "--divisor must be in [1, 2^bits-1]");
    }
    if (config->parallelBlocks == 0 ||
        config->parallelBlocks > config->wordBits) {
        throw std::invalid_argument(
            "--parallel-blocks must be between 1 and bits");
    }
    if (config->mergeBatch == 0 ||
        config->mergeBatch > config->wordBits) {
        throw std::invalid_argument(
            "--merge-batch must be between 1 and bits");
    }

    // Exact reciprocal multiplication needs a 2w-bit active lane. The
    // equally sized guard prevents rotations from crossing packed words.
    config->guardBits = config->internalBits();
    config->stride = 2U * config->internalBits();
    if (!config->wordsExplicit) {
        config->words = config->maxWords();
    }
    if (config->words == 0 || config->words > config->maxWords()) {
        throw std::invalid_argument(
            "--words exceeds the internal 2*bits lane capacity");
    }

    if (!config->depthExplicit) {
        const cpp_int reciprocal =
            (cpp_int(1) << config->wordBits) / config->divisor;
        const uint32_t reciprocalDepth =
            PublicMultiplyDepth(
                Popcount(reciprocal),
                config->internalBits()) + 3U;
        const uint32_t divisorDepth =
            PublicMultiplyDepth(
                Popcount(config->divisor),
                config->wordBits) + 3U;
        config->multiplicativeDepth = std::max(
            kDefaultDivisionDepth,
            std::max(reciprocalDepth, divisorDepth));
    }
    if (config->multiplicativeDepth < 25) {
        throw std::invalid_argument(
            "--depth must be at least 25 for Bootstrap and the latest "
            "complex-twisted borrow path");
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
        else if (option == "--divisor") {
            config.divisorText = requireValue();
            config.divisor =
                ParseInteger(config.divisorText, "--divisor");
        }
        else if (option == "--parallel-blocks") {
            config.parallelBlocks =
                ksword::ParseU32(
                    requireValue(), "--parallel-blocks");
        }
        else if (option == "--merge-batch") {
            config.mergeBatch =
                ksword::ParseU32(requireValue(), "--merge-batch");
        }
        else if (option == "--backend") {
            config.backend = ksword::ParseBackend(requireValue());
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
                "scalar division performs required internal and final "
                "Bootstraps; --bootstrap is not an option");
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
        << "  --words N            packed words (default: maximum for "
           "the internal 2w lane)\n"
        << "  --divisor N          public nonzero decimal/0x scalar "
           "(default 3)\n"
        << "  --parallel-blocks N  independent partial-product shift "
           "streams (default 1)\n"
        << "  --merge-batch N      3:2 compressors between barriers "
           "(default 1)\n"
        << "  --backend B          gpu or cpu (default gpu)\n"
        << "  --gpus N             logical GPUs 0..N-1 (default 2)\n"
        << "  --devices LIST       logical GPU list, for example 0,2,3\n"
        << "  --depth N            CKKS depth (default 40)\n"
        << "  --modup-init 0|1     initialize twisted-product graphs "
           "(default 1 on GPU)\n"
        << "  --profile 0|1        synchronized internal timings\n"
        << "\nThe divisor is public. The refreshed quotient is returned, "
           "and every\n"
        << "precision refresh is included in online latency. "
           "CUDA_VISIBLE_DEVICES\n"
        << "selects physical GPUs; --gpus/--devices use logical indices.\n";
    return output.str();
}

}  // namespace ksdivscalar
