#include "math_verify.h"
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <ctype.h>

// Skip whitespace forward
static const char* skip_ws(const char* s) {
    while (*s && isspace((unsigned char)*s)) s++;
    return s;
}

// Try to parse a number starting at *s. On success, set *out and return
// pointer past the number. On failure, return NULL.
static const char* parse_number_at(const char* s, double* out) {
    s = skip_ws(s);
    if (!*s) return NULL;

    int negative = 0;
    if (*s == '-') { negative = 1; s++; }
    else if (*s == '+') { s++; }

    if (!isdigit((unsigned char)*s) && *s != '.') return NULL;

    // Parse integer part, skipping commas (e.g. 42,000)
    double integer_part = 0;
    int has_digits = 0;
    while (isdigit((unsigned char)*s) || *s == ',') {
        if (*s == ',') { s++; continue; }
        integer_part = integer_part * 10 + (*s - '0');
        has_digits = 1;
        s++;
    }

    double frac_part = 0;
    if (*s == '.') {
        s++;
        double place = 0.1;
        while (isdigit((unsigned char)*s)) {
            frac_part += (*s - '0') * place;
            place *= 0.1;
            has_digits = 1;
            s++;
        }
    }

    if (!has_digits) return NULL;

    double value = integer_part + frac_part;
    if (negative) value = -value;

    // Handle percentage: 42% → 42 (keep as-is, not /100)
    // Most math benchmarks treat "42%" as the answer "42" when ground truth is "42"
    if (*s == '%') s++;

    *out = value;
    return s;
}

// Try to parse a fraction like "3/4" starting at position s.
// Returns pointer past fraction on success, NULL on failure.
static const char* parse_fraction_at(const char* s, double* out) {
    double num;
    const char* after_num = parse_number_at(s, &num);
    if (!after_num) return NULL;

    const char* p = skip_ws(after_num);
    if (*p != '/') return NULL;
    p++;

    double den;
    const char* after_den = parse_number_at(p, &den);
    if (!after_den || den == 0.0) return NULL;

    *out = num / den;
    return after_den;
}

// Find content inside \boxed{...}, handling nested braces.
// Returns pointer to content start, sets *len to content length.
static const char* find_boxed(const char* text, int* len) {
    const char* p = text;
    while ((p = strstr(p, "\\boxed{")) != NULL) {
        const char* start = p + 7; // skip \boxed{
        int depth = 1;
        const char* end = start;
        while (*end && depth > 0) {
            if (*end == '{') depth++;
            else if (*end == '}') depth--;
            if (depth > 0) end++;
        }
        if (depth == 0) {
            *len = (int)(end - start);
            return start;
        }
        p = start;
    }
    return NULL;
}

// Find content after #### (GSM8K format)
static const char* find_gsm8k_answer(const char* text, int* len) {
    const char* p = strstr(text, "####");
    if (!p) return NULL;
    p += 4;
    p = skip_ws(p);
    const char* start = p;
    while (*p && *p != '\n' && *p != '\r') p++;
    // trim trailing whitespace
    while (p > start && isspace((unsigned char)*(p-1))) p--;
    *len = (int)(p - start);
    return start;
}

// Max input length to prevent DoS with pathological inputs
#define MAX_INPUT_LEN (1 << 20)

int extract_number(const char* text, double* out) {
    if (!text || !*text || !out) return 0;

    int len;

    // Priority 1: Try \boxed{...} (LaTeX format)
    const char* boxed = find_boxed(text, &len);
    if (boxed && len > 0 && len < 256) {
        char buf[256];
        memcpy(buf, boxed, len);
        buf[len] = '\0';
        // Try fraction first, then plain number
        if (parse_fraction_at(buf, out)) return 1;
        if (parse_number_at(buf, out)) return 1;
    }

    // Priority 2: Try #### answer (GSM8K format)
    const char* gsm = find_gsm8k_answer(text, &len);
    if (gsm && len > 0 && len < 256) {
        char buf[256];
        memcpy(buf, gsm, len);
        buf[len] = '\0';
        if (parse_fraction_at(buf, out)) return 1;
        if (parse_number_at(buf, out)) return 1;
    }

    // Priority 3: Find the LAST number in the text
    double last_value = 0;
    int found = 0;

    const char* p = text;
    while (*p) {
        // Try fraction first (e.g. "3/4")
        double val;
        const char* after = parse_fraction_at(p, &val);
        if (after) {
            last_value = val;
            found = 1;
            p = after;
            continue;
        }

        // Try plain number
        after = parse_number_at(p, &val);
        if (after) {
            last_value = val;
            found = 1;
            p = after;
            continue;
        }

        p++;
    }

    if (found) {
        *out = last_value;
        return 1;
    }

    return 0;
}

VerifyResult verify_math(const char* response, const char* ground_truth, double tolerance) {
    VerifyResult result;
    result.response = response;
    result.ground_truth = ground_truth;
    result.reward = 0.0f;
    result.parsed_value = 0.0;
    result.expected_value = 0.0;
    result.parse_success = 0;

    if (!response || !ground_truth) return result;

    // Parse ground truth
    double expected;
    if (!extract_number(ground_truth, &expected)) return result;
    result.expected_value = expected;

    // Parse response
    double parsed;
    if (!extract_number(response, &parsed)) return result;
    result.parsed_value = parsed;
    result.parse_success = 1;

    // Compare with tolerance
    if (fabs(parsed - expected) <= tolerance) {
        result.reward = 1.0f;
    }

    return result;
}

void verify_math_batch(VerifyResult* results, int n, double tolerance) {
    if (!results || n <= 0) return;
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int i = 0; i < n; i++) {
        VerifyResult r = verify_math(results[i].response, results[i].ground_truth, tolerance);
        results[i].reward = r.reward;
        results[i].parsed_value = r.parsed_value;
        results[i].expected_value = r.expected_value;
        results[i].parse_success = r.parse_success;
    }
}

void verify_math_bulk(const char* responses_buf, const char* gt_buf,
                      const int* offsets_resp, const int* offsets_gt,
                      float* rewards_out, int n, double tolerance) {
    if (!responses_buf || !gt_buf || !offsets_resp || !offsets_gt || !rewards_out || n <= 0)
        return;
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int i = 0; i < n; i++) {
        if (offsets_resp[i] < 0 || offsets_gt[i] < 0) {
            rewards_out[i] = 0.0f;
            continue;
        }
        const char* resp = responses_buf + offsets_resp[i];
        const char* gt = gt_buf + offsets_gt[i];
        VerifyResult r = verify_math(resp, gt, tolerance);
        rewards_out[i] = r.reward;
    }
}
