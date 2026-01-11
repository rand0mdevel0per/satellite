#pragma once

#include <cstdint>
#include <vector>

namespace satellite {

struct SatInput {
    const int8_t* assignments;
    size_t num_vars;
    const uint8_t* batch_data;
    size_t batch_len;
    const uint8_t* vec_data;
    size_t vec_len;
};

struct SatOutput {
    int32_t satisfied; // 1=true, 0=false, -1=unknown
    int64_t* propagated;
    size_t propagated_count;
    int64_t* conflict;
    size_t conflict_len;
};

// Base class for constraints
class Constraint {
public:
    virtual ~Constraint() = default;
    
    // The main evaluation function to be implemented by users
    virtual bool evaluate(const SatInput* input, SatOutput* output) = 0;
};

// Macros to simplify constraint definition
#define SATELLITE_CONSTRAINT(Name) \
    class Name : public satellite::Constraint { \
    public: \
        bool evaluate(const satellite::SatInput* input, satellite::SatOutput* output) override; \
    }; \
    extern "C" bool Name##_wrapper(const satellite::SatInput* input, satellite::SatOutput* output) { \
        Name constraint; \
        return constraint.evaluate(input, output); \
    } \
    bool Name::evaluate(const satellite::SatInput* input, satellite::SatOutput* output)

} // namespace satellite
