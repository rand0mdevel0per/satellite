#pragma once

/**
 * @file types.h
 * @brief Common types for GPU worker.
 */

#include <cstdint>
#include <cstddef>

namespace satellite; {
namespace gpu {

/// Literal type (positive = true, negative = negated)
using Literal = int64_t;

/// Variable ID
using VarId = uint64_t;

/// Job ID
using JobId = uint64_t;

/// Branch ID
using BranchId = uint64_t;

/// Warp size (32 for NVIDIA, 64 for AMD)
#ifdef SATELLITE_HIP
constexpr size_t WARP_SIZE = 64;
#else
constexpr size_t WARP_SIZE = 32;
#endif

/// Number of clauses per BCP job
constexpr size_t CLAUSES_PER_JOB = WARP_SIZE;

/// Shared memory size per block (bytes)
constexpr size_t SHARED_MEM_SIZE = 48 * 1024;

} // namespace gpu
} // namespace satellite
