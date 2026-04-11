#ifndef PUFFER_MATH_VERIFY_H
#define PUFFER_MATH_VERIFY_H

#include <stddef.h>

typedef struct {
    const char* response;
    const char* ground_truth;
    float reward;
    double parsed_value;
    double expected_value;
    int parse_success;
} VerifyResult;

// Verify a single math response against ground truth.
// tolerance: absolute difference allowed (default 1e-5)
VerifyResult verify_math(const char* response, const char* ground_truth, double tolerance);

// Verify a batch of responses. Results array must be pre-allocated with n entries.
// response and ground_truth fields must be set before calling.
void verify_math_batch(VerifyResult* results, int n, double tolerance);

// Extract a numerical answer from a string. Returns 1 on success, 0 on failure.
// Handles: integers, decimals, negatives, commas, fractions, percentages,
// LaTeX \boxed{}, GSM8K #### format.
int extract_number(const char* text, double* out);

// Bulk buffer API: verify N responses packed into contiguous buffers.
// responses_buf: concatenated null-terminated response strings
// gt_buf: concatenated null-terminated ground truth strings
// offsets_resp: byte offset of each response in responses_buf (length n)
// offsets_gt: byte offset of each ground truth in gt_buf (length n)
// rewards_out: pre-allocated float array of length n (output)
void verify_math_bulk(const char* responses_buf, const char* gt_buf,
                      const int* offsets_resp, const int* offsets_gt,
                      float* rewards_out, int n, double tolerance);

#endif
