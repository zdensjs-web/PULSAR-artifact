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

namespace ksword {

constexpr uint32_t kRingDimension = 1U << 17;
constexpr uint32_t kSlotCapacity = kRingDimension / 2U;
constexpr uint32_t kDefaultWordBits = 256;
constexpr uint32_t kDefaultDepth = 25;
constexpr uint32_t kMaximumGpuCount = 8;
constexpr uint32_t kDefaultScalarShift = 13;

enum class Backend {
    Cpu,
    Gpu,
};

struct RuntimeConfig {
    uint32_t wordBits = kDefaultWordBits;
    uint32_t guardBits = kDefaultWordBits;
    uint32_t stride = 2U * kDefaultWordBits;
    uint32_t words = kSlotCapacity / (2U * kDefaultWordBits);
    uint32_t slots = kSlotCapacity;
    uint32_t multiplicativeDepth = kDefaultDepth;
    uint32_t scalarShift = kDefaultScalarShift;
    Backend backend = Backend::Gpu;
    std::vector<int> devices{0, 1};
    bool detailedProfile = false;
    bool wordsExplicit = false;
    bool shiftExplicit = false;

    uint32_t maxWords() const {
        return slots / stride;
    }

    uint32_t usedSlots() const {
        return words * stride;
    }

    uint32_t controlBits() const {
        uint32_t result = 0;
        for (uint32_t value = wordBits; value > 1U; value >>= 1U) {
            ++result;
        }
        return result;
    }
};

inline bool IsPowerOfTwo(uint32_t value) {
    return value != 0 && (value & (value - 1U)) == 0;
}

inline uint32_t NextPowerOfTwo(uint32_t value) {
    if (value == 0) {
        return 1;
    }
    uint32_t result = 1;
    while (result < value) {
        result <<= 1U;
    }
    return result;
}

inline uint32_t IntegerLog2(uint32_t value) {
    if (!IsPowerOfTwo(value)) {
        throw std::invalid_argument(
            "IntegerLog2 requires a power-of-two value");
    }
    uint32_t result = 0;
    while (value > 1U) {
        value >>= 1U;
        ++result;
    }
    return result;
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
    config->guardBits = config->wordBits;
    config->stride = 2U * config->wordBits;

    if (!config->wordsExplicit) {
        config->words = config->maxWords();
    }
    if (config->words == 0 || config->words > config->maxWords()) {
        throw std::invalid_argument(
            "--words exceeds the selected bit width and CKKS slot capacity");
    }

    if (config->backend == Backend::Cpu) {
        config->devices.clear();
    }
    else if (config->devices.empty()) {
        throw std::invalid_argument("GPU backend requires at least one device");
    }
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
            config.wordBits = ParseU32(requireValue(), "--bits");
        }
        else if (option == "--words") {
            config.words = ParseU32(requireValue(), "--words");
            config.wordsExplicit = true;
        }
        else if (option == "--backend") {
            config.backend = ParseBackend(requireValue());
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
        else if (option == "--depth") {
            config.multiplicativeDepth =
                ParseU32(requireValue(), "--depth");
        }
        else if (option == "--shift") {
            config.scalarShift = ParseU32(requireValue(), "--shift");
            config.shiftExplicit = true;
        }
        else if (option == "--profile") {
            config.detailedProfile =
                ParseBool(requireValue(), "--profile");
        }
        else if (option == "--help" || option == "-h") {
            throw std::runtime_error("help");
        }
        else if (option == "--bootstrap") {
            throw std::invalid_argument(
                "--bootstrap is not a runtime option for these operators");
        }
        else {
            throw std::invalid_argument("unknown option: " + option);
        }
    }

    FinalizeConfig(&config);
    return config;
}

inline std::string Usage(const char* executable, bool scalarShift) {
    std::ostringstream output;
    output
        << "Usage: " << executable << " [options]\n"
        << "  --bits N          word width: 8,16,32,64,128,256 (default 256)\n"
        << "  --words N         packed words per ciphertext (default: maximum)\n"
        << "  --backend B       gpu or cpu (default gpu)\n"
        << "  --gpus N          use logical GPUs 0..N-1, N in [1,8] (default 2)\n"
        << "  --devices LIST    explicit logical GPUs, for example 0,2,3\n"
        << "  --depth N         CKKS multiplicative depth (default 25)\n";
    if (scalarShift) {
        output
            << "  --shift N         public shift amount (default 13)\n";
    }
    output
        << "  --profile 0|1     synchronized per-operation timing (default 0)\n"
        << "\nCUDA_VISIBLE_DEVICES chooses physical GPUs; --gpus/--devices use\n"
        << "the logical indices visible inside the process.\n";
    return output.str();
}

inline uint32_t Slot(const RuntimeConfig& config,
                     uint32_t word,
                     uint32_t position) {
    return word * config.stride + position;
}

inline std::string BitsToHex(const std::vector<uint8_t>& bits,
                             const RuntimeConfig& config,
                             uint32_t word) {
    constexpr char kHex[] = "0123456789abcdef";
    std::string output(config.wordBits / 4U, '0');
    for (uint32_t nibble = 0; nibble < config.wordBits / 4U; ++nibble) {
        uint8_t value = 0;
        for (uint32_t offset = 0; offset < 4U; ++offset) {
            const uint32_t bit = nibble * 4U + offset;
            value |= static_cast<uint8_t>(
                bits[Slot(config, word, bit)] << offset);
        }
        output[output.size() - 1U - nibble] = kHex[value];
    }
    return "0x" + output;
}

using Clock = std::chrono::steady_clock;

inline double Milliseconds(const Clock::time_point& begin,
                           const Clock::time_point& end) {
    return std::chrono::duration<double, std::milli>(end - begin).count();
}

inline double PrecisionBits(double error) {
    return error <= 0.0
        ? std::numeric_limits<double>::infinity()
        : -std::log2(error);
}

}  // namespace ksword
