cmake_minimum_required(VERSION 3.25.2)

project(PULSAR_GPU VERSION 1.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

find_package(fideslib REQUIRED CONFIG)
find_package(OpenMP REQUIRED COMPONENTS CXX)

function(pulsar_gpu_add_operator target source)
    add_executable(${target} "src/${source}")
    target_include_directories(${target} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_link_libraries(${target} PRIVATE fideslib::fideslib OpenMP::OpenMP_CXX)
endfunction()

pulsar_gpu_add_operator(pulsar_gpu_add 256adder_ks_complex_hoisted_gpu.cpp)
pulsar_gpu_add_operator(pulsar_gpu_sub 256subtractor_ks_complex_hoisted_gpu.cpp)

pulsar_gpu_add_operator(pulsar_gpu_gt 256greaterthan_ks_complex_hoisted_gpu.cpp)
pulsar_gpu_add_operator(pulsar_gpu_ge 256greaterequal_ks_complex_hoisted_gpu.cpp)
pulsar_gpu_add_operator(pulsar_gpu_lt 256lessthan_ks_complex_hoisted_gpu.cpp)
pulsar_gpu_add_operator(pulsar_gpu_le 256lessequal_ks_complex_hoisted_gpu.cpp)
pulsar_gpu_add_operator(pulsar_gpu_eq 256equal_ks_complex_hoisted_gpu.cpp)
pulsar_gpu_add_operator(pulsar_gpu_ne 256notequal_ks_complex_hoisted_gpu.cpp)

pulsar_gpu_add_operator(pulsar_gpu_scalar_shl 256scalar_left_shift_runtime.cpp)
pulsar_gpu_add_operator(pulsar_gpu_scalar_shr 256scalar_right_shift_runtime.cpp)
pulsar_gpu_add_operator(pulsar_gpu_shl 256left_shift_runtime.cpp)
pulsar_gpu_add_operator(pulsar_gpu_shr 256right_shift_runtime.cpp)

pulsar_gpu_add_operator(pulsar_gpu_xor 256xor_runtime.cpp)
pulsar_gpu_add_operator(pulsar_gpu_and 256and_runtime.cpp)
pulsar_gpu_add_operator(pulsar_gpu_or 256or_runtime.cpp)
pulsar_gpu_add_operator(pulsar_gpu_not 256not_runtime.cpp)

pulsar_gpu_add_operator(pulsar_gpu_mul 256multiplier_ks_runtime.cpp)
pulsar_gpu_add_operator(pulsar_gpu_scalar_mul 256scalar_mul_public_pc_ks_gpu.cpp)
pulsar_gpu_add_operator(pulsar_gpu_scalar_div 256divider_public_scalar_runtime.cpp)
pulsar_gpu_add_operator(pulsar_gpu_div 256divider_cipher_runtime.cpp)

pulsar_gpu_add_operator(pulsar_gpu_transfer 256erc7984_transfer_gpu.cpp)
pulsar_gpu_add_operator(pulsar_gpu_auction 256auction_max_128_gpu.cpp)
pulsar_gpu_add_operator(pulsar_gpu_vault 256vault_claim_shares_public_pc_ks_gpu.cpp)

install(
    TARGETS
        pulsar_gpu_add pulsar_gpu_sub
        pulsar_gpu_gt pulsar_gpu_ge pulsar_gpu_lt pulsar_gpu_le
        pulsar_gpu_eq pulsar_gpu_ne
        pulsar_gpu_scalar_shl pulsar_gpu_scalar_shr pulsar_gpu_shl pulsar_gpu_shr
        pulsar_gpu_xor pulsar_gpu_and pulsar_gpu_or pulsar_gpu_not
        pulsar_gpu_mul pulsar_gpu_scalar_mul pulsar_gpu_scalar_div pulsar_gpu_div
        pulsar_gpu_transfer pulsar_gpu_auction pulsar_gpu_vault
    RUNTIME DESTINATION bin
)
