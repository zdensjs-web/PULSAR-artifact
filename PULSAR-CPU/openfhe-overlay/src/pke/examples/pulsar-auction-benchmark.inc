        if (auctionMode) {
            constexpr uint32_t selectionDepth = 1;

            accumulator = CipherPair{};
            stableAccumulator = CipherPair{};
            prefixWarmupFirst.reset();
            prefixWarmupSecond.reset();
            prefixWarmupOutput = DirectPrefixOutput{};
            ReleaseAllocatorMemory("before_auction_input");

            auto bidsPlain = MakeAuctionBids(wordBits, nativeBatch);
            CipherPair current{
                pke->Encrypt(EncodeDense(bidsPlain[0], elementParams, scale)),
                pke->Encrypt(EncodeDense(bidsPlain[1], elementParams, scale))};

            double inputB2BSeconds = 0.0;
            std::cout << "[input-refresh] auction bids native B2B excluded\n";
            current = BooleanBoundaryRefresh(
                fhe, current[0], current[1], &inputB2BSeconds);
            ReleaseAllocatorMemory("after_auction_input_B2B");
            boundaryRefreshOutputLevel = current[0]->GetLevel();
            RequireAudit(
                "refreshed auction bids group 0",
                AuditDense("refreshed auction bids group 0", current[0],
                           keyPair.secretKey, bidsPlain[0]));
            RequireAudit(
                "refreshed auction bids group 1",
                AuditDense("refreshed auction bids group 1", current[1],
                           keyPair.secretKey, bidsPlain[1]));

            const uint32_t reductionRounds =
                PrefixLayerCount(nativeBatch) + 1;
            const double setupSeconds = std::chrono::duration<double>(
                setupEnd - setupStart).count();
            const double warmupSeconds = std::chrono::duration<double>(
                warmupEnd - warmupStart).count();

            std::cout << "============================================================\n"
                      << "PULSAR Boolean auction over OpenFHE\n"
                      << "implementation_build=v3.4.0-auction-B2B-first-scheduler\n"
                      << "semantics=winner=max(bid[0],...,bid[B-1])\n"
                      << "word_bits=" << wordBits << '\n'
                      << "native_batch=" << nativeBatch << '\n'
                      << "aggregate_batch=" << aggregateBatch << '\n'
                      << "bidders=" << aggregateBatch << '\n'
                      << "reduction_rounds=" << reductionRounds << '\n'
                      << "layout=bit_major_dense_no_guard_slots\n"
                      << "input_domain=[0,2^(w-1))\n"
                      << "instruction_sequence=rotate,GE,select\n"
                      << "reduction_layout=noncyclic_tree_one_valid_output_per_group\n"
                      << "scheduler=direct-until-full-round-does-not-fit_then-native-B2B\n"
                      << "refresh_primitives=native_B2B\n"
                      << "Prefix_Boot=disabled_for_auction\n"
                      << "b2b_transform_plaintext_cache=cleared_before_each_call\n"
                      << "bit_clean=3x^2-2x^3\n"
                      << "bit_clean_depth=2\n"
                      << "bit_clean_threshold=" << bitCleanThreshold << '\n'
                      << "error_source=profiling_decryption_excluded_from_latency\n"
                      << "input_refresh=one_native_B2B_call_excluded\n"
                      << "terminal_output_refresh=scheduler_decision\n"
                      << "input_b2b_s_excluded=" << inputB2BSeconds << '\n'
                      << "input_level=" << boundaryRefreshOutputLevel << '\n'
                      << "comparison_depth=" << circuitDepth << '\n'
                      << "selection_depth=" << selectionDepth << '\n'
                      << "multiplicative_depth=" << kMultiplicativeDepth << '\n'
                      << "log_scale=" << kScalingBits << '\n'
                      << "logQP=" << logQP << '\n'
                      << "security=HEStd_128_classic\n"
                      << "correctness_claim=empirical_threshold_only\n"
                      << "setup_seconds_excluded=" << setupSeconds << '\n'
                      << "warmup_seconds_excluded=" << warmupSeconds << '\n'
                      << "============================================================\n";

            double rotationSeconds = 0.0;
            double comparisonSeconds = 0.0;
            double selectionSeconds = 0.0;
            double bitCleanSeconds = 0.0;
            double onlineBoundaryB2BSeconds = 0.0;
            double observedMaxError = 0.0;
            uint32_t bitCleanEvents = 0;
            uint32_t bitCleanCiphertexts = 0;
            uint32_t onlineBoundaryB2Bs = 0;
            uint32_t activeWords = nativeBatch;

            auto auditPair = [&] (
                const std::string& label,
                const CipherPair& values,
                const std::array<std::vector<uint8_t>, 2>& expectedValues) {
                const auto audit0 = AuditDenseActiveWords(
                    label + " group 0", values[0], keyPair.secretKey,
                    expectedValues[0], wordBits, nativeBatch, activeWords);
                const auto audit1 = AuditDenseActiveWords(
                    label + " group 1", values[1], keyPair.secretKey,
                    expectedValues[1], wordBits, nativeBatch, activeWords);
                RequireAudit(label + " group 0", audit0);
                RequireAudit(label + " group 1", audit1);
                observedMaxError = std::max(
                    {observedMaxError, audit0.maxError, audit1.maxError});
                return std::max(audit0.maxError, audit1.maxError);
            };

            auto maybeBitClean = [&] (
                const std::string& boundary,
                CipherPair values,
                const std::array<std::vector<uint8_t>, 2>& expectedValues,
                uint32_t requiredLevelsAfterClean) {
                double currentError = auditPair(
                    boundary + " before clean", values, expectedValues);
                bool apply = currentError > bitCleanThreshold;
                std::cout << "[bit-clean-plan] boundary=" << boundary
                          << " current_error=" << std::scientific
                          << currentError
                          << " threshold=" << bitCleanThreshold
                          << " action=" << (apply ? "apply" : "skip")
                          << std::defaultfloat << '\n';
                if (!apply) {
                    return values;
                }

                const uint32_t cleanLimit = requiredLevelsAfterClean == 0
                    ? kMultiplicativeDepth
                    : maximumPrefixInputLevel;
                if (values[0]->GetLevel() + 2 + requiredLevelsAfterClean >
                        cleanLimit ||
                    values[1]->GetLevel() + 2 + requiredLevelsAfterClean >
                        cleanLimit) {
                    std::cout << "[schedule] boundary=" << boundary
                              << " action=native_B2B_before_Bit_Clean"
                              << " input_level=" << values[0]->GetLevel()
                              << " required_levels_after_clean="
                              << requiredLevelsAfterClean << '\n';
                    values = BooleanBoundaryRefresh(
                        fhe, values[0], values[1],
                        &onlineBoundaryB2BSeconds);
                    ++onlineBoundaryB2Bs;
                    ReleaseAllocatorMemory(boundary + "_after_B2B");
                    currentError = auditPair(
                        boundary + " after B2B", values, expectedValues);
                    apply = currentError > bitCleanThreshold;
                    std::cout << "[bit-clean-plan] boundary=" << boundary
                              << " post_b2b_error=" << std::scientific
                              << currentError
                              << " threshold=" << bitCleanThreshold
                              << " action=" << (apply ? "apply" : "skip")
                              << std::defaultfloat << '\n';
                    if (!apply) {
                        return values;
                    }
                }
                if (values[0]->GetLevel() + 2 + requiredLevelsAfterClean >
                        cleanLimit ||
                    values[1]->GetLevel() + 2 + requiredLevelsAfterClean >
                        cleanLimit) {
                    throw std::runtime_error(
                        "B2B did not leave enough levels for Bit Clean and its successor");
                }

                const auto start = std::chrono::steady_clock::now();
                CipherPair cleaned{
                    dense->EvalBitClean(values[0]),
                    dense->EvalBitClean(values[1])};
                const auto end = std::chrono::steady_clock::now();
                const double seconds = std::chrono::duration<double>(
                    end - start).count();
                bitCleanSeconds += seconds;
                ++bitCleanEvents;
                bitCleanCiphertexts += 2;
                auditPair(boundary + " after clean", cleaned, expectedValues);
                std::cout << "[bit-clean] boundary=" << boundary
                          << " input_level=" << values[0]->GetLevel()
                          << " output_level=" << cleaned[0]->GetLevel()
                          << " seconds=" << seconds << '\n';
                return cleaned;
            };

            auto prepareStateForComparison = [&] (
                const std::string& boundary,
                CipherPair values,
                const std::array<std::vector<uint8_t>, 2>& expectedValues,
                bool requireReusableOutput) {
                const uint32_t inputLevel = values[0]->GetLevel();
                if (values[1]->GetLevel() != inputLevel) {
                    throw std::runtime_error(
                        boundary + " state groups have inconsistent levels");
                }
                const uint32_t directLimit = requireReusableOutput
                    ? maximumBoundaryInputLevel
                    : kMultiplicativeDepth;
                const bool direct = inputLevel + circuitDepth +
                    selectionDepth <= directLimit;
                if (direct) {
                    return values;
                }
                std::cout << "[schedule] boundary=" << boundary
                          << " action=native_B2B"
                          << " input_level=" << inputLevel
                          << " reason=next_GE_and_select_do_not_fit\n";
                values = BooleanBoundaryRefresh(
                    fhe, values[0], values[1],
                    &onlineBoundaryB2BSeconds);
                ++onlineBoundaryB2Bs;
                ReleaseAllocatorMemory(boundary + "_after_B2B");
                auditPair(boundary + " after B2B", values, expectedValues);
                if (values[0]->GetLevel() + circuitDepth + selectionDepth >
                        directLimit) {
                    throw std::runtime_error(
                        "native B2B did not leave enough levels for an auction round");
                }
                return values;
            };

            auto evaluateComparison = [&] (
                const std::string& label,
                const CipherPair& left,
                const CipherPair& right,
                bool requireReusableOutput) {
                const uint32_t inputLevel = left[0]->GetLevel();
                if (left[1]->GetLevel() != inputLevel ||
                    right[0]->GetLevel() != inputLevel ||
                    right[1]->GetLevel() != inputLevel) {
                    throw std::runtime_error(
                        label + " comparison inputs have inconsistent levels");
                }

                const uint32_t directLimit = requireReusableOutput
                    ? maximumBoundaryInputLevel
                    : kMultiplicativeDepth;
                if (inputLevel + circuitDepth + selectionDepth >
                        directLimit) {
                    throw std::runtime_error(
                        label + " comparison reached the evaluator without B2B preparation");
                }
                std::cout << "[schedule] stage=" << label
                          << " instruction=GE action=direct"
                          << " input_level=" << inputLevel
                          << " projected_comparison_level="
                          << (inputLevel + circuitDepth) << '\n';

                const auto start = std::chrono::steady_clock::now();
                CipherPair output{
                    dense->EvalPulsarGreaterEqual(
                        left[0], right[0], wordBits, nativeBatch),
                    dense->EvalPulsarGreaterEqual(
                        left[1], right[1], wordBits, nativeBatch)};
                const auto end = std::chrono::steady_clock::now();
                comparisonSeconds += std::chrono::duration<double>(
                    end - start).count();
                return output;
            };

            auto evaluateMaximum = [&] (
                const std::string& label,
                const CipherPair& left,
                const CipherPair& right,
                const std::array<std::vector<uint8_t>, 2>& leftPlain,
                const std::array<std::vector<uint8_t>, 2>& rightPlain,
                uint32_t levelsRequiredAfterStateClean) {
                std::array<std::vector<uint8_t>, 2> conditionPlain{
                    PlainGreaterEqual(
                        leftPlain[0], rightPlain[0], wordBits, nativeBatch),
                    PlainGreaterEqual(
                        leftPlain[1], rightPlain[1], wordBits, nativeBatch)};
                CipherPair condition = evaluateComparison(
                    label, left, right, levelsRequiredAfterStateClean != 0);
                condition = maybeBitClean(
                    label + "_condition", std::move(condition),
                    conditionPlain, selectionDepth);

                CipherPair selected;
                const auto selectionStart =
                    std::chrono::steady_clock::now();
                for (size_t group = 0; group < selected.size(); ++group) {
                    const uint32_t targetLevel = std::max(
                        {left[group]->GetLevel(), right[group]->GetLevel(),
                         condition[group]->GetLevel()});
                    auto falseValue = dense->AdjustToLevel(
                        right[group], targetLevel);
                    auto trueValue = dense->AdjustToLevel(
                        left[group], targetLevel);
                    auto mask = dense->AdjustToLevel(
                        condition[group], targetLevel);
                    selected[group] = dense->EvalBooleanSelect(
                        falseValue, trueValue, mask);
                }
                const auto selectionEnd =
                    std::chrono::steady_clock::now();
                selectionSeconds += std::chrono::duration<double>(
                    selectionEnd - selectionStart).count();

                std::array<std::vector<uint8_t>, 2> selectedPlain{
                    PlainMaximum(
                        leftPlain[0], rightPlain[0], wordBits, nativeBatch),
                    PlainMaximum(
                        leftPlain[1], rightPlain[1], wordBits, nativeBatch)};
                selected = maybeBitClean(
                    label + "_state", std::move(selected), selectedPlain,
                    levelsRequiredAfterStateClean);
                std::cout << "[auction] stage=" << label
                          << " output_level=" << selected[0]->GetLevel()
                          << " status=PASS\n";
                return std::make_pair(std::move(selected),
                                      std::move(selectedPlain));
            };

            for (uint32_t offset = 1; offset < nativeBatch; offset <<= 1) {
                const std::string label =
                    "reduce_" + std::to_string(offset);
                current = prepareStateForComparison(
                    label, std::move(current), bidsPlain, true);
                const auto rotationStart =
                    std::chrono::steady_clock::now();
                CipherPair shifted{
                    dense->EvalRotateWords(
                        current[0], static_cast<int32_t>(offset)),
                    dense->EvalRotateWords(
                        current[1], static_cast<int32_t>(offset))};
                const auto rotationEnd =
                    std::chrono::steady_clock::now();
                rotationSeconds += std::chrono::duration<double>(
                    rotationEnd - rotationStart).count();
                std::array<std::vector<uint8_t>, 2> shiftedPlain{
                    PlainRotateWords(
                        bidsPlain[0], wordBits, nativeBatch, offset),
                    PlainRotateWords(
                        bidsPlain[1], wordBits, nativeBatch, offset)};
                if (activeWords <= offset) {
                    throw std::runtime_error(label + " has no valid words after rotation");
                }
                activeWords -= offset;
                auto next = evaluateMaximum(
                    label, current, shifted, bidsPlain, shiftedPlain,
                    2);
                current = std::move(next.first);
                bidsPlain = std::move(next.second);
            }

            current = prepareStateForComparison(
                "merge_groups", std::move(current), bidsPlain, false);
            CipherPair opposite{current[1], current[0]};
            std::array<std::vector<uint8_t>, 2> oppositePlain{
                bidsPlain[1], bidsPlain[0]};
            auto merged = evaluateMaximum(
                "merge_groups", current, opposite, bidsPlain,
                oppositePlain, 0);
            current = std::move(merged.first);
            bidsPlain = std::move(merged.second);

            const double circuitSeconds = rotationSeconds +
                comparisonSeconds + selectionSeconds;
            const double totalSeconds = circuitSeconds + bitCleanSeconds +
                onlineBoundaryB2BSeconds;
            const double precisionBits = observedMaxError == 0.0
                ? std::numeric_limits<double>::infinity()
                : -std::log2(observedMaxError);
            std::cout << std::fixed << std::setprecision(9)
                      << "RESULT app=auction implementation=PULSAR_Boolean_OpenFHE"
                      << " rotation_s=" << rotationSeconds
                      << " comparison_s=" << comparisonSeconds
                      << " selection_s=" << selectionSeconds
                      << " circuit_s=" << circuitSeconds
                      << " bit_clean_s=" << bitCleanSeconds
                      << " prefix_boot_s=0.000000000"
                      << " boundary_b2b_s=" << onlineBoundaryB2BSeconds
                      << " total_s=" << totalSeconds
                      << " bidders=" << aggregateBatch
                      << " reduction_rounds=" << reductionRounds
                      << " valid_output_words_per_group=" << activeWords
                      << " bit_clean_events=" << bitCleanEvents
                      << " bit_clean_ciphertexts=" << bitCleanCiphertexts
                      << " prefix_boots=0"
                      << " online_boundary_b2b=" << onlineBoundaryB2Bs
                      << " input_b2b_excluded=1"
                      << " max_error=" << observedMaxError
                      << " worst_precision_bits=" << precisionBits
                      << " output_level=" << current[0]->GetLevel()
                      << " failures=0 status=PASS\n"
                      << "OVERALL status=PASS\n";
            return 0;
        }
