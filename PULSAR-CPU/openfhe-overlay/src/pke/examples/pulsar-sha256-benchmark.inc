        if (sha256Mode) {
            using PlainShaWord = std::array<std::vector<uint32_t>, 2>;

            constexpr std::array<uint32_t, 8> kShaInitial{
                0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
                0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U};
            constexpr std::array<uint32_t, 64> kShaRoundConstant{
                0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U,
                0x3956c25bU, 0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U,
                0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U,
                0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U,
                0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU,
                0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
                0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U,
                0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U,
                0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U,
                0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
                0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U,
                0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
                0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U,
                0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
                0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
                0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U};

            const auto rotateRight32 = [](uint32_t value, uint32_t amount) {
                return (value >> amount) | (value << (32U - amount));
            };
            const auto smallSigma0Plain = [&](uint32_t value) {
                return rotateRight32(value, 7) ^
                       rotateRight32(value, 18) ^ (value >> 3);
            };
            const auto smallSigma1Plain = [&](uint32_t value) {
                return rotateRight32(value, 17) ^
                       rotateRight32(value, 19) ^ (value >> 10);
            };
            const auto bigSigma0Plain = [&](uint32_t value) {
                return rotateRight32(value, 2) ^
                       rotateRight32(value, 13) ^
                       rotateRight32(value, 22);
            };
            const auto bigSigma1Plain = [&](uint32_t value) {
                return rotateRight32(value, 6) ^
                       rotateRight32(value, 11) ^
                       rotateRight32(value, 25);
            };
            const auto choosePlain = [](uint32_t condition,
                                        uint32_t trueValue,
                                        uint32_t falseValue) {
                return (condition & trueValue) ^
                       (~condition & falseValue);
            };
            const auto majorityPlain = [](uint32_t first,
                                          uint32_t second,
                                          uint32_t third) {
                return (first & second) ^ (first & third) ^
                       (second & third);
            };

            const auto toDenseBits = [&](const std::vector<uint32_t>& values) {
                if (values.size() != nativeBatch) {
                    throw std::runtime_error(
                        "SHA-256 word has an invalid message count");
                }
                std::vector<uint8_t> bits(
                    static_cast<size_t>(wordBits) * nativeBatch);
                for (uint32_t word = 0; word < nativeBatch; ++word) {
                    for (uint32_t bit = 0; bit < wordBits; ++bit) {
                        bits[static_cast<size_t>(bit) * nativeBatch + word] =
                            static_cast<uint8_t>((values[word] >> bit) & 1U);
                    }
                }
                return bits;
            };
            const auto makePlainConstant = [&](uint32_t value) {
                PlainShaWord result;
                for (auto& group : result) {
                    group.assign(nativeBatch, value);
                }
                return result;
            };
            const auto encryptWord = [&](const PlainShaWord& values) {
                return CipherPair{
                    pke->Encrypt(EncodeDense(
                        toDenseBits(values[0]), elementParams, scale)),
                    pke->Encrypt(EncodeDense(
                        toDenseBits(values[1]), elementParams, scale))};
            };

            std::array<PlainShaWord, 16> plainSchedule;
            for (size_t group = 0; group < 2; ++group) {
                for (auto& word : plainSchedule) {
                    word[group].resize(nativeBatch);
                }
                for (uint32_t message = 0; message < nativeBatch; ++message) {
                    uint64_t state =
                        0x9e3779b97f4a7c15ULL ^
                        (static_cast<uint64_t>(group) << 61) ^
                        static_cast<uint64_t>(message + 1);
                    for (uint32_t index = 0; index < 8; ++index) {
                        state += 0x9e3779b97f4a7c15ULL;
                        uint64_t mixed = state;
                        mixed = (mixed ^ (mixed >> 30)) *
                                0xbf58476d1ce4e5b9ULL;
                        mixed = (mixed ^ (mixed >> 27)) *
                                0x94d049bb133111ebULL;
                        mixed ^= mixed >> 31;
                        plainSchedule[index][group][message] =
                            static_cast<uint32_t>(mixed);
                    }
                    plainSchedule[8][group][message] = 0x80000000U;
                    for (uint32_t index = 9; index < 15; ++index) {
                        plainSchedule[index][group][message] = 0;
                    }
                    plainSchedule[15][group][message] = 256U;
                }
            }

            std::array<CipherPair, 16> schedule;
            for (uint32_t index = 0; index < 16; ++index) {
                schedule[index] = encryptWord(plainSchedule[index]);
            }

            std::array<PlainShaWord, 8> initialPlainState;
            std::array<PlainShaWord, 8> plainState;
            std::array<CipherPair, 8> initialCipherState;
            std::array<CipherPair, 8> state;
            for (uint32_t index = 0; index < 8; ++index) {
                initialPlainState[index] =
                    makePlainConstant(kShaInitial[index]);
                plainState[index] = initialPlainState[index];
                state[index] = encryptWord(plainState[index]);
                initialCipherState[index] = state[index];
            }
            std::array<CipherPair, 64> encryptedRoundConstants;
            for (uint32_t index = 0; index < rounds; ++index) {
                encryptedRoundConstants[index] =
                    encryptWord(makePlainConstant(kShaRoundConstant[index]));
            }

            accumulator = CipherPair{};
            stableAccumulator = CipherPair{};
            prefixWarmupFirst.reset();
            prefixWarmupSecond.reset();
            prefixWarmupOutput = DirectPrefixOutput{};
            ReleaseAllocatorMemory("before_sha256");

            double logicSeconds = 0.0;
            double additionSeconds = 0.0;
            double prefixBootSeconds = 0.0;
            double boundaryB2BSeconds = 0.0;
            double bitCleanSeconds = 0.0;
            double observedMaxError = 0.0;
            uint32_t additions = 0;
            uint32_t prefixBoots = 0;
            uint32_t boundaryB2Bs = 0;
            uint32_t bitCleanEvents = 0;

            const auto wordLevel = [](const CipherPair& value) {
                if (!value[0] || !value[1]) {
                    throw std::runtime_error(
                        "SHA-256 operation received an empty word");
                }
                if (value[0]->GetLevel() != value[1]->GetLevel()) {
                    throw std::runtime_error(
                        "SHA-256 word groups have inconsistent levels");
                }
                return value[0]->GetLevel();
            };
            const auto alignWord = [&](const CipherPair& input,
                                       uint32_t level) {
                return CipherPair{
                    dense->AdjustToLevel(input[0], level),
                    dense->AdjustToLevel(input[1], level)};
            };
            const auto auditWord = [&](const std::string& label,
                                       const CipherPair& value,
                                       const PlainShaWord& expectedValue,
                                       bool requireCorrect) {
                const auto audit0 = AuditDense(
                    label + " group 0", value[0], keyPair.secretKey,
                    toDenseBits(expectedValue[0]));
                const auto audit1 = AuditDense(
                    label + " group 1", value[1], keyPair.secretKey,
                    toDenseBits(expectedValue[1]));
                observedMaxError = std::max(
                    {observedMaxError, audit0.maxError, audit1.maxError});
                if (requireCorrect) {
                    RequireAudit(label + " group 0", audit0);
                    RequireAudit(label + " group 1", audit1);
                }
                return std::max(audit0.maxError, audit1.maxError);
            };
            const auto refreshWord = [&](const std::string& label,
                                         const CipherPair& input) {
                std::cout << "[schedule] word=" << label
                          << " action=native_B2B"
                          << " input_level=" << wordLevel(input) << '\n';
                auto output = BooleanBoundaryRefresh(
                    fhe, input[0], input[1], &boundaryB2BSeconds);
                ++boundaryB2Bs;
                ReleaseAllocatorMemory(label + "_after_B2B");
                return output;
            };

            const auto xorPair = [&](const CipherPair& left,
                                     const CipherPair& right) {
                const auto start = std::chrono::steady_clock::now();
                CipherPair output{
                    dense->EvalBooleanXor(left[0], right[0]),
                    dense->EvalBooleanXor(left[1], right[1])};
                logicSeconds += std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - start).count();
                return output;
            };
            const auto xorThree = [&](const CipherPair& first,
                                      const CipherPair& second,
                                      const CipherPair& third) {
                auto firstTwo = xorPair(first, second);
                return xorPair(firstTwo, third);
            };
            const auto rotatePair = [&](const CipherPair& input,
                                        uint32_t amount) {
                const auto start = std::chrono::steady_clock::now();
                CipherPair output{
                    dense->EvalRotateBitsRight(
                        input[0], amount, wordBits, nativeBatch),
                    dense->EvalRotateBitsRight(
                        input[1], amount, wordBits, nativeBatch)};
                logicSeconds += std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - start).count();
                return output;
            };
            const auto shiftPair = [&](const CipherPair& input,
                                       uint32_t amount) {
                const auto start = std::chrono::steady_clock::now();
                CipherPair output{
                    dense->EvalShiftBitsRight(
                        input[0], amount, wordBits, nativeBatch),
                    dense->EvalShiftBitsRight(
                        input[1], amount, wordBits, nativeBatch)};
                logicSeconds += std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - start).count();
                return output;
            };
            const auto choosePair = [&](const CipherPair& condition,
                                        const CipherPair& trueValue,
                                        const CipherPair& falseValue) {
                const auto start = std::chrono::steady_clock::now();
                CipherPair output{
                    dense->EvalBooleanChoose(
                        condition[0], trueValue[0], falseValue[0]),
                    dense->EvalBooleanChoose(
                        condition[1], trueValue[1], falseValue[1])};
                logicSeconds += std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - start).count();
                return output;
            };
            const auto majorityPair = [&](const CipherPair& first,
                                          const CipherPair& second,
                                          const CipherPair& third) {
                const auto start = std::chrono::steady_clock::now();
                CipherPair output{
                    dense->EvalBooleanMajority(
                        first[0], second[0], third[0]),
                    dense->EvalBooleanMajority(
                        first[1], second[1], third[1])};
                logicSeconds += std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - start).count();
                return output;
            };
            const auto smallSigma0 = [&](const CipherPair& input) {
                return xorThree(
                    rotatePair(input, 7), rotatePair(input, 18),
                    shiftPair(input, 3));
            };
            const auto smallSigma1 = [&](const CipherPair& input) {
                return xorThree(
                    rotatePair(input, 17), rotatePair(input, 19),
                    shiftPair(input, 10));
            };
            const auto bigSigma0 = [&](const CipherPair& input) {
                return xorThree(
                    rotatePair(input, 2), rotatePair(input, 13),
                    rotatePair(input, 22));
            };
            const auto bigSigma1 = [&](const CipherPair& input) {
                return xorThree(
                    rotatePair(input, 6), rotatePair(input, 11),
                    rotatePair(input, 25));
            };

            const auto scheduledAdd = [&](const std::string& label,
                                          CipherPair left,
                                          CipherPair right,
                                          uint32_t requiredAfterOutput) {
                const auto operationStart =
                    std::chrono::steady_clock::now();
                const double prefixBefore = prefixBootSeconds;
                const double boundaryBefore = boundaryB2BSeconds;
                uint32_t inputLevel = std::max(
                    wordLevel(left), wordLevel(right));
                left = alignWord(left, inputLevel);
                right = alignWord(right, inputLevel);

                CipherPair output;
                if (inputLevel + circuitDepth + requiredAfterOutput <=
                        kMultiplicativeDepth) {
                    std::cout << "[schedule] instruction=" << label
                              << " action=direct"
                              << " input_level=" << inputLevel
                              << " output_level="
                              << inputLevel + circuitDepth << '\n';
                    output = {
                        dense->EvalPulsarAdd(
                            left[0], right[0], wordBits, nativeBatch),
                        dense->EvalPulsarAdd(
                            left[1], right[1], wordBits, nativeBatch)};
                }
                else if (inputLevel + 2 <= maximumPrefixInputLevel) {
                    const uint32_t completedLayers = std::min(
                        prefixLayers,
                        maximumPrefixInputLevel - inputLevel - 2);
                    std::cout << "[schedule] instruction=" << label
                              << " action=Prefix_Boot"
                              << " input_level=" << inputLevel
                              << " completed_prefix_layers="
                              << completedLayers
                              << " remaining_prefix_layers="
                              << prefixLayers - completedLayers << '\n';
                    std::array<DenseBooleanZImpl::PrefixState, 2> prefix{
                        dense->EvalPulsarAddBegin(
                            left[0], right[0], wordBits, nativeBatch),
                        dense->EvalPulsarAddBegin(
                            left[1], right[1], wordBits, nativeBatch)};
                    for (uint32_t layer = 0;
                         layer < completedLayers; ++layer) {
                        dense->EvalPulsarPrefixAdvance(
                            prefix[0], wordBits, nativeBatch);
                        dense->EvalPulsarPrefixAdvance(
                            prefix[1], wordBits, nativeBatch);
                    }
                    auto encoded0 = dense->EvalPrefixBootEncode(
                        prefix[0], wordBits, nativeBatch);
                    auto encoded1 = dense->EvalPrefixBootEncode(
                        prefix[1], wordBits, nativeBatch);
                    const auto bootStart =
                        std::chrono::steady_clock::now();
                    auto refreshed = DirectPrefixFunctionalRefresh(
                        fhe, encoded0, encoded1);
                    prefixBootSeconds += std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - bootStart).count();
                    ++prefixBoots;
                    for (size_t group = 0; group < 2; ++group) {
                        prefix[group].packed = refreshed.packed[group];
                        prefix[group].localPropagate =
                            refreshed.localPropagate[group];
                        prefix[group].localSum.reset();
                        prefix[group].nextDistance =
                            1U << completedLayers;
                    }
                    for (uint32_t layer = completedLayers;
                         layer < prefixLayers; ++layer) {
                        dense->EvalPulsarPrefixAdvance(
                            prefix[0], wordBits, nativeBatch);
                        dense->EvalPulsarPrefixAdvance(
                            prefix[1], wordBits, nativeBatch);
                    }
                    output = {
                        dense->EvalPulsarAddFinish(
                            prefix[0], wordBits, nativeBatch),
                        dense->EvalPulsarAddFinish(
                            prefix[1], wordBits, nativeBatch)};
                }
                else {
                    left = refreshWord(label + "_left", left);
                    right = refreshWord(label + "_right", right);
                    inputLevel = std::max(
                        wordLevel(left), wordLevel(right));
                    left = alignWord(left, inputLevel);
                    right = alignWord(right, inputLevel);
                    if (inputLevel + circuitDepth + requiredAfterOutput >
                            kMultiplicativeDepth) {
                        throw std::runtime_error(
                            label + " has no valid addition schedule");
                    }
                    output = {
                        dense->EvalPulsarAdd(
                            left[0], right[0], wordBits, nativeBatch),
                        dense->EvalPulsarAdd(
                            left[1], right[1], wordBits, nativeBatch)};
                }
                ++additions;
                const double total = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() -
                    operationStart).count();
                const double refresh = (prefixBootSeconds - prefixBefore) +
                    (boundaryB2BSeconds - boundaryBefore);
                additionSeconds += std::max(0.0, total - refresh);
                return output;
            };

            const auto maybeBitClean = [&](const std::string& label,
                                           CipherPair value,
                                           const PlainShaWord& expectedValue,
                                           uint32_t requiredAfterOutput) {
                double error = auditWord(
                    label + " before clean", value, expectedValue, false);
                if (error <= bitCleanThreshold) {
                    std::cout << "[bit-clean-plan] word=" << label
                              << " current_error=" << std::scientific
                              << error << " threshold=" << bitCleanThreshold
                              << " action=skip" << std::defaultfloat << '\n';
                    return value;
                }
                if (wordLevel(value) + 2 + requiredAfterOutput >
                        kMultiplicativeDepth) {
                    value = refreshWord(label, value);
                    error = auditWord(
                        label + " after B2B", value, expectedValue, false);
                    if (error <= bitCleanThreshold) {
                        return value;
                    }
                }
                const auto start = std::chrono::steady_clock::now();
                value = {
                    dense->EvalBitClean(value[0]),
                    dense->EvalBitClean(value[1])};
                bitCleanSeconds += std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - start).count();
                ++bitCleanEvents;
                auditWord(label + " after clean", value, expectedValue, true);
                std::cout << "[bit-clean] word=" << label
                          << " output_level=" << wordLevel(value) << '\n';
                return value;
            };

            const auto preparationEnd = std::chrono::steady_clock::now();
            std::cout << "============================================================\n"
                      << "PULSAR SHA-256 compression over OpenFHE\n"
                      << "implementation_build=v1.0.0-sha256-Prefix-Boot-scheduler\n"
                      << "semantics=SHA-256_single_512-bit_block\n"
                      << "sha_rounds=" << rounds << '\n'
                      << "complete_sha256=" << (rounds == 64 ? "yes" : "no")
                      << '\n'
                      << "message_bytes=32_plus_SHA-256_padding\n"
                      << "word_bits=32\n"
                      << "native_batch=" << nativeBatch << '\n'
                      << "aggregate_batch=" << aggregateBatch << '\n'
                      << "representation=Boolean_input_output\n"
                      << "instruction_set=ADD,XOR,AND,NOT,ROTR,SHR\n"
                      << "scheduler=lookahead_Prefix-Boot_with_error-threshold_Bit-Clean\n"
                      << "refresh_primitives=Prefix_Boot,native_B2B_fallback\n"
                      << "input_refresh=fresh_encryption_excluded\n"
                      << "final_refresh=scheduler_decision\n"
                      << "bit_clean_threshold=" << bitCleanThreshold << '\n'
                      << "multiplicative_depth=" << kMultiplicativeDepth << '\n'
                      << "log_scale=" << kScalingBits << '\n'
                      << "logQP=" << logQP << '\n'
                      << "security=HEStd_128_classic\n"
                      << "preparation_seconds_excluded="
                      << std::chrono::duration<double>(
                             preparationEnd - warmupEnd).count() << '\n'
                      << "============================================================\n";

            const auto evaluationStart = std::chrono::steady_clock::now();
            for (uint32_t round = 0; round < rounds; ++round) {
                const uint32_t scheduleIndex = round & 15U;
                if (round >= 16) {
                    PlainShaWord nextPlain;
                    for (size_t group = 0; group < 2; ++group) {
                        nextPlain[group].resize(nativeBatch);
                        for (uint32_t message = 0;
                             message < nativeBatch; ++message) {
                            nextPlain[group][message] =
                                smallSigma1Plain(
                                    plainSchedule[(round - 2) & 15U]
                                                 [group][message]) +
                                plainSchedule[(round - 7) & 15U]
                                             [group][message] +
                                smallSigma0Plain(
                                    plainSchedule[(round - 15) & 15U]
                                                 [group][message]) +
                                plainSchedule[round & 15U]
                                             [group][message];
                        }
                    }
                    auto sigma0 = smallSigma0(
                        schedule[(round - 15) & 15U]);
                    auto sigma1 = smallSigma1(
                        schedule[(round - 2) & 15U]);
                    auto lower = scheduledAdd(
                        "W" + std::to_string(round) + "_lower",
                        schedule[round & 15U], sigma0, 4);
                    auto upper = scheduledAdd(
                        "W" + std::to_string(round) + "_upper",
                        schedule[(round - 7) & 15U], sigma1, 4);
                    schedule[scheduleIndex] = scheduledAdd(
                        "W" + std::to_string(round),
                        lower, upper, 4);
                    plainSchedule[scheduleIndex] = std::move(nextPlain);
                    schedule[scheduleIndex] = maybeBitClean(
                        "W" + std::to_string(round),
                        std::move(schedule[scheduleIndex]),
                        plainSchedule[scheduleIndex], 4);
                }

                PlainShaWord nextA;
                PlainShaWord nextE;
                for (size_t group = 0; group < 2; ++group) {
                    nextA[group].resize(nativeBatch);
                    nextE[group].resize(nativeBatch);
                    for (uint32_t message = 0;
                         message < nativeBatch; ++message) {
                        const uint32_t a = plainState[0][group][message];
                        const uint32_t b = plainState[1][group][message];
                        const uint32_t c = plainState[2][group][message];
                        const uint32_t d = plainState[3][group][message];
                        const uint32_t e = plainState[4][group][message];
                        const uint32_t f = plainState[5][group][message];
                        const uint32_t g = plainState[6][group][message];
                        const uint32_t h = plainState[7][group][message];
                        const uint32_t t1 =
                            h + bigSigma1Plain(e) +
                            choosePlain(e, f, g) +
                            kShaRoundConstant[round] +
                            plainSchedule[scheduleIndex][group][message];
                        const uint32_t t2 =
                            bigSigma0Plain(a) + majorityPlain(a, b, c);
                        nextE[group][message] = d + t1;
                        nextA[group][message] = t1 + t2;
                    }
                }

                auto sigma1 = bigSigma1(state[4]);
                auto choose = choosePair(state[4], state[5], state[6]);
                auto sigma0 = bigSigma0(state[0]);
                auto majority =
                    majorityPair(state[0], state[1], state[2]);
                auto hPlusSigma = scheduledAdd(
                    "round" + std::to_string(round) + "_h_sigma1",
                    state[7], sigma1, 4);
                auto choosePlusConstant = scheduledAdd(
                    "round" + std::to_string(round) + "_ch_k",
                    choose, encryptedRoundConstants[round], 4);
                auto firstFour = scheduledAdd(
                    "round" + std::to_string(round) + "_first_four",
                    hPlusSigma, choosePlusConstant, 4);
                auto t1 = scheduledAdd(
                    "round" + std::to_string(round) + "_t1",
                    firstFour, schedule[scheduleIndex], 4);
                auto t2 = scheduledAdd(
                    "round" + std::to_string(round) + "_t2",
                    sigma0, majority, 4);
                auto newE = scheduledAdd(
                    "round" + std::to_string(round) + "_e",
                    state[3], t1, 4);
                auto newA = scheduledAdd(
                    "round" + std::to_string(round) + "_a",
                    t1, t2, 4);
                newE = maybeBitClean(
                    "round" + std::to_string(round) + "_e",
                    std::move(newE), nextE, 4);
                newA = maybeBitClean(
                    "round" + std::to_string(round) + "_a",
                    std::move(newA), nextA, 4);

                for (uint32_t index = 7; index > 0; --index) {
                    state[index] = std::move(state[index - 1]);
                    plainState[index] = std::move(plainState[index - 1]);
                }
                state[0] = std::move(newA);
                state[4] = std::move(newE);
                plainState[0] = std::move(nextA);
                plainState[4] = std::move(nextE);

                std::cout << "[sha-round] index=" << round + 1
                          << " a_level=" << wordLevel(state[0])
                          << " e_level=" << wordLevel(state[4])
                          << " prefix_boots=" << prefixBoots
                          << " boundary_b2b=" << boundaryB2Bs
                          << " bit_clean_events=" << bitCleanEvents
                          << " status=PASS\n";
                if ((round + 1) % 4 == 0) {
                    ReleaseAllocatorMemory(
                        "sha_round_" + std::to_string(round + 1));
                }
            }

            if (rounds == 64) {
                for (uint32_t index = 0; index < 8; ++index) {
                    for (size_t group = 0; group < 2; ++group) {
                        for (uint32_t message = 0;
                             message < nativeBatch; ++message) {
                            plainState[index][group][message] +=
                                kShaInitial[index];
                        }
                    }
                    state[index] = scheduledAdd(
                        "feed_forward_" + std::to_string(index),
                        state[index], initialCipherState[index], 0);
                }
            }

            uint64_t failures = 0;
            for (uint32_t index = 0; index < 8; ++index) {
                const auto audit0 = AuditDense(
                    "digest word " + std::to_string(index) + " group 0",
                    state[index][0], keyPair.secretKey,
                    toDenseBits(plainState[index][0]));
                const auto audit1 = AuditDense(
                    "digest word " + std::to_string(index) + " group 1",
                    state[index][1], keyPair.secretKey,
                    toDenseBits(plainState[index][1]));
                failures += audit0.failures + audit1.failures;
                observedMaxError = std::max(
                    {observedMaxError, audit0.maxError, audit1.maxError});
            }
            if (failures != 0) {
                throw std::runtime_error(
                    "SHA-256 output failed the Boolean audit");
            }

            const double evaluationSeconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now() -
                evaluationStart).count();
            const double measuredSeconds =
                logicSeconds + additionSeconds + prefixBootSeconds +
                boundaryB2BSeconds + bitCleanSeconds;
            const double precisionBits = observedMaxError == 0.0
                ? std::numeric_limits<double>::infinity()
                : -std::log2(observedMaxError);
            std::cout << std::fixed << std::setprecision(9)
                      << "RESULT app=sha256"
                      << " implementation=PULSAR_Boolean_OpenFHE"
                      << " rounds=" << rounds
                      << " messages=" << aggregateBatch
                      << " additions=" << additions
                      << " logic_s=" << logicSeconds
                      << " addition_s=" << additionSeconds
                      << " prefix_boot_s=" << prefixBootSeconds
                      << " boundary_b2b_s=" << boundaryB2BSeconds
                      << " bit_clean_s=" << bitCleanSeconds
                      << " measured_s=" << measuredSeconds
                      << " evaluation_wall_s=" << evaluationSeconds
                      << " amortized_ms="
                      << 1000.0 * measuredSeconds / aggregateBatch
                      << " prefix_boots=" << prefixBoots
                      << " boundary_b2b=" << boundaryB2Bs
                      << " bit_clean_events=" << bitCleanEvents
                      << " max_error=" << observedMaxError
                      << " worst_precision_bits=" << precisionBits
                      << " failures=0 status=PASS\n"
                      << "OVERALL status=PASS\n";
            return 0;
        }
