#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace ksadder {

constexpr uint32_t kDefaultRingDimension = 1U << 17;
constexpr uint32_t kSlotCapacity = kDefaultRingDimension / 2;
constexpr uint32_t kDefaultWordBits = 256;
constexpr uint32_t kDefaultDepth = 25;
constexpr uint32_t kMaximumGpuCount = 8;

enum class Backend {
    Cpu,
    Gpu,
};

enum class Algorithm {
    Baseline,
    ComplexStandard,
    ComplexHoisted,
    ComplexTwisted,
};

enum class LevelPlacement {
    Bottom,
    Top,
};

struct RuntimeConfig {
    uint32_t wordBits = kDefaultWordBits;
    uint32_t guardBits = kDefaultWordBits;
    uint32_t stride = 2 * kDefaultWordBits;
    uint32_t words = kSlotCapacity / (2 * kDefaultWordBits);
    uint32_t slots = kSlotCapacity;
    uint32_t layers = 8;
    uint32_t multiplicativeDepth = kDefaultDepth;
    int directionSign = -1;
    Backend backend = Backend::Gpu;
    Algorithm algorithm = Algorithm::ComplexTwisted;
    LevelPlacement levelPlacement = LevelPlacement::Bottom;
    std::vector<int> devices{0, 1};
    bool bootstrap = true;
    bool detailedProfile = false;
    bool graphInit = true;
    bool projectOutputToReal = true;
    bool wordsExplicit = false;
    bool layersExplicit = false;
    bool algorithmExplicit = false;
    bool guardFreeLayout = false;
    uint32_t inputLevel = 0;
    uint32_t expectedOutputLevel = 0;
    uint32_t circuitLevelCost = 0;

    uint32_t maxWords() const {
        return slots / stride;
    }

    uint32_t usedSlots() const {
        return words * stride;
    }

    uint32_t fullPrefixLayers() const {
        uint32_t result = 0;
        for (uint32_t value = wordBits; value > 1; value >>= 1U) {
            ++result;
        }
        return result;
    }

    bool hasFullPrefix() const {
        return layers == fullPrefixLayers();
    }
};

inline bool IsPowerOfTwo(uint32_t value) {
    return value != 0 && (value & (value - 1U)) == 0;
}

inline uint32_t ParseU32(const std::string& value, const char* option) {
    size_t used = 0;
    const unsigned long parsed = std::stoul(value, &used, 10);
    if (used != value.size() ||
        parsed > std::numeric_limits<uint32_t>::max()) {
        throw std::invalid_argument(
            std::string("invalid value for ") + option + ": " + value);
    }
    return static_cast<uint32_t>(parsed);
}

inline bool ParseBool(const std::string& value, const char* option) {
    if (value == "1" || value == "true" || value == "on") {
        return true;
    }
    if (value == "0" || value == "false" || value == "off") {
        return false;
    }
    throw std::invalid_argument(
        std::string(option) + " expects 0/1, true/false, or on/off");
}

inline Backend ParseBackend(const std::string& value) {
    if (value == "gpu") {
        return Backend::Gpu;
    }
    if (value == "cpu") {
        return Backend::Cpu;
    }
    throw std::invalid_argument("--backend must be cpu or gpu");
}

inline Algorithm ParseAlgorithm(const std::string& value) {
    if (value == "baseline") {
        return Algorithm::Baseline;
    }
    if (value == "complex-standard") {
        return Algorithm::ComplexStandard;
    }
    if (value == "complex-hoisted") {
        return Algorithm::ComplexHoisted;
    }
    if (value == "complex-twisted") {
        return Algorithm::ComplexTwisted;
    }
    throw std::invalid_argument(
        "--mode must be baseline, complex-standard, complex-hoisted, "
        "or complex-twisted");
}

inline LevelPlacement ParseLevelPlacement(const std::string& value) {
    if (value == "bottom") {
        return LevelPlacement::Bottom;
    }
    if (value == "top") {
        return LevelPlacement::Top;
    }
    throw std::invalid_argument(
        "--level-placement must be bottom or top");
}

inline std::vector<int> SequentialDevices(uint32_t count) {
    if (count == 0 || count > kMaximumGpuCount) {
        throw std::invalid_argument("--gpus must be between 1 and 8");
    }
    std::vector<int> result(count);
    for (uint32_t index = 0; index < count; ++index) {
        result[index] = static_cast<int>(index);
    }
    return result;
}

inline std::vector<int> ParseDeviceList(const std::string& value) {
    std::vector<int> result;
    std::stringstream stream(value);
    std::string item;
    while (std::getline(stream, item, ',')) {
        if (item.empty()) {
            throw std::invalid_argument("--devices contains an empty item");
        }
        const uint32_t device = ParseU32(item, "--devices");
        if (device >= kMaximumGpuCount) {
            throw std::invalid_argument(
                "--devices accepts logical device indices 0 through 7");
        }
        result.push_back(static_cast<int>(device));
    }
    if (result.empty()) {
        throw std::invalid_argument("--devices must not be empty");
    }
    std::sort(result.begin(), result.end());
    if (std::adjacent_find(result.begin(), result.end()) != result.end()) {
        throw std::invalid_argument("--devices contains a duplicate device");
    }
    return result;
}

inline const char* BackendName(Backend backend) {
    return backend == Backend::Gpu ? "gpu" : "cpu";
}

inline const char* AlgorithmName(Algorithm algorithm) {
    switch (algorithm) {
        case Algorithm::Baseline:
            return "baseline";
        case Algorithm::ComplexStandard:
            return "complex-standard";
        case Algorithm::ComplexHoisted:
            return "complex-hoisted";
        case Algorithm::ComplexTwisted:
            return "complex-twisted";
    }
    return "unknown";
}

inline const char* LevelPlacementName(LevelPlacement placement) {
    return placement == LevelPlacement::Bottom ? "bottom" : "top";
}

inline uint32_t PrefixCircuitLevelCost(const RuntimeConfig& config) {
    uint32_t cost = config.layers + 2;
    if (config.algorithm != Algorithm::Baseline &&
        config.projectOutputToReal) {
        ++cost;
    }
    return cost;
}

inline uint32_t AdderCircuitLevelCost(const RuntimeConfig& config) {
    return PrefixCircuitLevelCost(config);
}

inline uint32_t UsableMultiplicationLevels(
    uint32_t multiplicativeDepth,
    uint32_t level,
    uint32_t scaleDegree) {
    const uint64_t reserved =
        static_cast<uint64_t>(level) + scaleDegree;
    return reserved < multiplicativeDepth
        ? multiplicativeDepth - static_cast<uint32_t>(reserved)
        : 0;
}

inline std::string DeviceList(const std::vector<int>& devices) {
    if (devices.empty()) {
        return "none";
    }
    std::ostringstream output;
    for (size_t index = 0; index < devices.size(); ++index) {
        if (index != 0) {
            output << ",";
        }
        output << devices[index];
    }
    return output.str();
}

inline void FinalizeConfig(RuntimeConfig* config) {
    if (!IsPowerOfTwo(config->wordBits) ||
        config->wordBits < 8 || config->wordBits > 256) {
        throw std::invalid_argument(
            "--bits must be a power of two between 8 and 256");
    }

    config->guardBits =
        config->guardFreeLayout ? 0 : config->wordBits;
    config->stride = config->wordBits + config->guardBits;

    const uint32_t fullLayers = config->fullPrefixLayers();
    if (!config->layersExplicit) {
        config->layers = fullLayers;
    }
    if (config->layers == 0 || config->layers > fullLayers) {
        throw std::invalid_argument(
            "--layers must be between 1 and log2(bits)");
    }

    if (!config->wordsExplicit) {
        config->words = config->maxWords();
    }
    if (config->words == 0 || config->words > config->maxWords()) {
        throw std::invalid_argument(
            "--words exceeds the selected bit width and CKKS slot capacity");
    }

    if (config->backend == Backend::Cpu) {
        config->devices.clear();
        config->graphInit = false;
        if (config->algorithm == Algorithm::ComplexHoisted) {
            throw std::invalid_argument(
                "complex-hoisted uses the GPU-only shared-hoisting primitive; "
                "use --mode complex-twisted or baseline on CPU");
        }
    }
    else if (config->devices.empty()) {
        throw std::invalid_argument("GPU backend requires at least one device");
    }

    if (config->algorithm == Algorithm::Baseline) {
        config->graphInit = false;
        config->projectOutputToReal = false;
    }

}

inline void FinalizePrefixLevelPlacement(RuntimeConfig* config) {
    config->circuitLevelCost = PrefixCircuitLevelCost(*config);
    constexpr uint32_t outputScaleDegree = 2;
    if (config->multiplicativeDepth <
        config->circuitLevelCost + outputScaleDegree) {
        throw std::invalid_argument(
            "--depth is too small for the selected prefix circuit and the two "
            "towers required by the Bootstrap input");
    }

    if (config->levelPlacement == LevelPlacement::Bottom) {
        config->expectedOutputLevel =
            config->multiplicativeDepth - outputScaleDegree;
        config->inputLevel =
            config->expectedOutputLevel - config->circuitLevelCost;
    }
    else {
        config->inputLevel = 0;
        config->expectedOutputLevel = config->circuitLevelCost;
    }
}

inline void FinalizeAdderLevelPlacement(RuntimeConfig* config) {
    FinalizePrefixLevelPlacement(config);
}

inline RuntimeConfig ParseCommandLine(
    int argc,
    char** argv,
    bool guardFreeLayout = false,
    Algorithm defaultAlgorithm = Algorithm::ComplexTwisted,
    bool allowComplexStandard = false) {
    RuntimeConfig config;
    config.guardFreeLayout = guardFreeLayout;
    config.algorithm = defaultAlgorithm;
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
            config.wordBits = ParseU32(requireValue(), "--bits");
        }
        else if (option == "--words") {
            config.words = ParseU32(requireValue(), "--words");
            config.wordsExplicit = true;
        }
        else if (option == "--layers") {
            config.layers = ParseU32(requireValue(), "--layers");
            config.layersExplicit = true;
        }
        else if (option == "--depth") {
            config.multiplicativeDepth =
                ParseU32(requireValue(), "--depth");
        }
        else if (option == "--backend") {
            config.backend = ParseBackend(requireValue());
        }
        else if (option == "--mode") {
            config.algorithm = ParseAlgorithm(requireValue());
            config.algorithmExplicit = true;
        }
        else if (option == "--level-placement") {
            config.levelPlacement =
                ParseLevelPlacement(requireValue());
        }
        else if (option == "--gpus") {
            if (deviceListExplicit) {
                throw std::invalid_argument(
                    "--gpus and --devices are mutually exclusive");
            }
            config.devices =
                SequentialDevices(ParseU32(requireValue(), "--gpus"));
            gpuCountExplicit = true;
        }
        else if (option == "--devices") {
            if (gpuCountExplicit) {
                throw std::invalid_argument(
                    "--gpus and --devices are mutually exclusive");
            }
            config.devices = ParseDeviceList(requireValue());
            deviceListExplicit = true;
        }
        else if (option == "--bootstrap") {
            config.bootstrap = ParseBool(requireValue(), "--bootstrap");
        }
        else if (option == "--profile") {
            config.detailedProfile = ParseBool(requireValue(), "--profile");
        }
        else if (option == "--graph-init") {
            config.graphInit = ParseBool(requireValue(), "--graph-init");
        }
        else if (option == "--modup-init") {
            config.graphInit = ParseBool(requireValue(), "--modup-init");
        }
        else if (option == "--project-real") {
            config.projectOutputToReal =
                ParseBool(requireValue(), "--project-real");
        }
        else if (option == "--help" || option == "-h") {
            throw std::runtime_error("help");
        }
        else {
            throw std::invalid_argument("unknown option: " + option);
        }
    }

    if (config.algorithm == Algorithm::ComplexStandard &&
        !allowComplexStandard) {
        throw std::invalid_argument(
            "complex-standard is currently implemented only by the "
            "guard-free prefix executables");
    }
    FinalizeConfig(&config);
    return config;
}

inline std::string Usage(
    const char* executable,
    Algorithm defaultAlgorithm = Algorithm::ComplexTwisted,
    bool allowComplexStandard = false) {
    std::ostringstream output;
    output
        << "Usage: " << executable << " [options]\n"
        << "  --bits N          word width: 8,16,32,64,128,256 (default 256)\n"
        << "  --words N         packed words per ciphertext (default: maximum)\n"
        << "  --layers N        prefix layers (default log2(bits))\n"
        << "  --backend B       gpu or cpu (default gpu)\n"
        << "  --mode M          baseline, "
        << (allowComplexStandard ? "complex-standard, " : "")
        << "complex-hoisted, complex-twisted\n"
        << "                    (default "
        << AlgorithmName(defaultAlgorithm) << ")\n"
        << "  --level-placement bottom|top (default bottom)\n"
        << "                    bottom ends at the lowest legal Bootstrap level\n"
        << "  --gpus N          use logical GPUs 0..N-1, N in [1,8] (default 2)\n"
        << "  --devices LIST    explicit logical GPUs, for example 0,2,3\n"
        << "  --depth N         CKKS multiplicative depth (default 25)\n"
        << "  --bootstrap 0|1   final bootstrap (default 1)\n"
        << "  --profile 0|1     synchronized per-operation timing (default 0)\n"
        << "  --modup-init 0|1  initialize level-specific ModUp state (default 1)\n"
        << "  --graph-init 0|1  compatibility alias for --modup-init\n"
        << "  --project-real 0|1 project hoisted output to real slots\n"
        << "\nCUDA_VISIBLE_DEVICES chooses physical GPUs; --gpus/--devices use\n"
        << "the logical indices visible inside the process.\n";
    return output.str();
}

inline uint32_t Slot(const RuntimeConfig& config,
                     uint32_t word,
                     uint32_t position) {
    return word * config.stride + position;
}

inline std::vector<int32_t> RotationIndices(
    const RuntimeConfig& config) {
    std::vector<int32_t> rotations;
    rotations.reserve(config.layers);
    for (uint32_t layer = 0; layer < config.layers; ++layer) {
        rotations.push_back(
            config.directionSign *
            static_cast<int32_t>(1U << layer));
    }
    return rotations;
}

inline std::vector<int32_t> OrdinaryRotationIndices(
    const RuntimeConfig& config) {
    if (config.algorithm == Algorithm::ComplexTwisted) {
        // Twisted prefix layers use raw automorphisms plus mixed-basis KSKs.
        // Only the final carry extraction is an ordinary slot rotation.
        return {config.directionSign};
    }
    return RotationIndices(config);
}

inline std::string BitsToHex(const std::vector<uint8_t>& bits,
                             const RuntimeConfig& config,
                             uint32_t word) {
    constexpr char kHex[] = "0123456789abcdef";
    std::string output(config.wordBits / 4, '0');
    for (uint32_t nibble = 0; nibble < config.wordBits / 4; ++nibble) {
        uint8_t value = 0;
        for (uint32_t offset = 0; offset < 4; ++offset) {
            const uint32_t bit = nibble * 4 + offset;
            value |= static_cast<uint8_t>(
                bits[Slot(config, word, bit)] << offset);
        }
        output[output.size() - 1 - nibble] = kHex[value];
    }
    return "0x" + output;
}

inline void BuildExpectedSum(const RuntimeConfig& config,
                             const std::vector<uint8_t>& a,
                             const std::vector<uint8_t>& b,
                             std::vector<uint8_t>* expected) {
    expected->assign(config.slots, 0);
    for (uint32_t word = 0; word < config.words; ++word) {
        uint32_t carry = 0;
        for (uint32_t bit = 0; bit < config.wordBits; ++bit) {
            const uint32_t slot = Slot(config, word, bit);
            const uint32_t total = a[slot] + b[slot] + carry;
            (*expected)[slot] = static_cast<uint8_t>(total & 1U);
            carry = total >> 1U;
        }
    }
}

inline double PrecisionBits(double error) {
    return error <= 0.0
        ? std::numeric_limits<double>::infinity()
        : -std::log2(error);
}

using Clock = std::chrono::steady_clock;

inline double Milliseconds(const Clock::time_point& begin,
                           const Clock::time_point& end) {
    return std::chrono::duration<double, std::milli>(end - begin).count();
}

inline void ConfigureStableTwistedRuntime(const RuntimeConfig& config) {
    if (config.backend != Backend::Gpu ||
        config.algorithm != Algorithm::ComplexTwisted ||
        std::getenv("FIDESLIB_USE_MODUP_GRAPH_CAPTURE") != nullptr) {
        return;
    }
#if defined(_WIN32)
    _putenv_s("FIDESLIB_USE_MODUP_GRAPH_CAPTURE", "0");
#else
    setenv("FIDESLIB_USE_MODUP_GRAPH_CAPTURE", "0", 0);
#endif
}

}  // namespace ksadder
