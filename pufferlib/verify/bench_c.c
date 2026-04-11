#include "math_verify.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

int main(int argc, char** argv) {
    int n = 1000000;
    if (argc > 1) n = atoi(argv[1]);

    // Pre-generate test data (simple but representative)
    const char* templates[] = {
        "The answer is 42.",
        "After calculating, I get 3.14",
        "Step 1: 10\nStep 2: 20\nThe final answer is 30.",
        "\\boxed{42}",
        "#### 42",
        "I calculated 99 first, but then corrected to 42.",
        "Therefore, the result is -7.5.",
        "The answer is 42,000.",
    };
    int num_templates = sizeof(templates) / sizeof(templates[0]);
    const char* ground_truths[] = {"42", "3.14", "30", "42", "42", "42", "-7.5", "42000"};

    // Allocate batch
    VerifyResult* results = calloc(n, sizeof(VerifyResult));
    for (int i = 0; i < n; i++) {
        int idx = i % num_templates;
        results[i].response = templates[idx];
        results[i].ground_truth = ground_truths[idx];
    }

    // Time the batch
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    verify_math_batch(results, n, 1e-5);
    clock_gettime(CLOCK_MONOTONIC, &t1);

    double elapsed = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
    double throughput = n / elapsed;

    // Check results
    int correct = 0;
    for (int i = 0; i < n; i++) {
        if (results[i].reward == 1.0f) correct++;
    }

    printf("Pure C benchmark: %d verifications\n", n);
    printf("  Time:       %.4f s\n", elapsed);
    printf("  Throughput: %.0f verifications/sec\n", throughput);
    printf("  Correct:    %d/%d\n", correct, n);

    free(results);
    return 0;
}
