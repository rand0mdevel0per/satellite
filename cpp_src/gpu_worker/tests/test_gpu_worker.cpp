/**
 * @file test_gpu_worker.cpp
 * @brief Tests for GPU worker.
 */

#include "gpu_worker.h"
#include <gtest/gtest.h>
#include <vector>

class GpuWorkerTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize GPU worker
        int result = gpu_worker_init();
        // May fail if no GPU available, that's OK for tests
        has_gpu_ = (result == GPU_OK);
    }

    void TearDown() override {
        gpu_worker_shutdown();
    }

    bool has_gpu_ = false;
};

TEST_F(GpuWorkerTest, InitShutdown) {
    // Init/shutdown already done in SetUp/TearDown
    SUCCEED();
}

TEST_F(GpuWorkerTest, IsAvailable) {
    int available = gpu_worker_is_available();
    EXPECT_EQ(available, has_gpu_ ? 1 : 0);
}

TEST_F(GpuWorkerTest, DeviceCount) {
    int count = gpu_worker_device_count();
    if (has_gpu_) {
        EXPECT_GE(count, 1);
    } else {
        EXPECT_EQ(count, 0);
    }
}

TEST_F(GpuWorkerTest, SubmitBcp) {
    if (!has_gpu_) {
        GTEST_SKIP() << "No GPU available";
    }

    // Simple test: 2 clauses, 3 variables
    // Clause 1: (x1 OR NOT x2) = [1, -2, 0]
    // Clause 2: (x2 OR x3) = [2, 3, 0]
    std::vector<int64_t> clause_data = {1, -2, 0, 2, 3, 0};
    
    // Assignments: x1=true, x2=unassigned, x3=unassigned
    std::vector<int8_t> assignments = {1, 0, 0};

    int result = gpu_worker_submit_bcp(
        clause_data.data(),
        2, // num_clauses
        assignments.data(),
        3  // num_vars
    );

    EXPECT_EQ(result, GPU_OK);

    // Wait for result
    gpu_worker_sync();

    GpuBcpResult bcp_result;
    int poll_result = gpu_worker_poll_result(&bcp_result);
    EXPECT_EQ(poll_result, 0); // Result available
    EXPECT_EQ(bcp_result.status, 0); // No conflict
}

TEST_F(GpuWorkerTest, DetectConflict) {
    if (!has_gpu_) {
        GTEST_SKIP() << "No GPU available";
    }

    // Conflicting clauses:
    // Clause 1: (x1) = [1, 0]
    // Clause 2: (NOT x1) = [-1, 0]
    // With x1=true, clause 2 is a conflict
    std::vector<int64_t> clause_data = {1, 0, -1, 0};
    std::vector<int8_t> assignments = {1}; // x1=true

    int result = gpu_worker_submit_bcp(
        clause_data.data(),
        2,
        assignments.data(),
        1
    );

    EXPECT_EQ(result, GPU_OK);

    gpu_worker_sync();

    GpuBcpResult bcp_result;
    int poll_result = gpu_worker_poll_result(&bcp_result);
    EXPECT_EQ(poll_result, 0);
    EXPECT_EQ(bcp_result.status, 1); // Conflict detected
    EXPECT_EQ(bcp_result.conflict_clause, 1); // Second clause
}

TEST_F(GpuWorkerTest, MemoryInfo) {
    if (!has_gpu_) {
        GTEST_SKIP() << "No GPU available";
    }

    size_t used, total;
    int result = gpu_worker_memory_info(&used, &total);
    
    EXPECT_EQ(result, GPU_OK);
    EXPECT_GT(total, 0);
    EXPECT_LE(used, total);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
