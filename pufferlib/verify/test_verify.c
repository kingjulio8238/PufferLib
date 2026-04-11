#include "math_verify.h"
#include <stdio.h>
#include <math.h>
#include <string.h>

static int tests_run = 0;
static int tests_passed = 0;

#define ASSERT_EXTRACT(input, expected_val) do { \
    double _v; \
    int _ok = extract_number(input, &_v); \
    tests_run++; \
    if (!_ok) { \
        printf("FAIL [%d]: extract_number(\"%s\") -> parse failed (expected %.6f)\n", \
               tests_run, input, (double)(expected_val)); \
    } else if (fabs(_v - (expected_val)) > 1e-5) { \
        printf("FAIL [%d]: extract_number(\"%s\") -> %.6f (expected %.6f)\n", \
               tests_run, input, _v, (double)(expected_val)); \
    } else { \
        tests_passed++; \
    } \
} while(0)

#define ASSERT_EXTRACT_FAIL(input) do { \
    double _v; \
    int _ok = extract_number(input, &_v); \
    tests_run++; \
    if (_ok) { \
        printf("FAIL [%d]: extract_number(\"%s\") -> %.6f (expected parse failure)\n", \
               tests_run, input, _v); \
    } else { \
        tests_passed++; \
    } \
} while(0)

#define ASSERT_VERIFY(resp, gt, expected_reward) do { \
    VerifyResult _r = verify_math(resp, gt, 1e-5); \
    tests_run++; \
    if (fabs(_r.reward - (expected_reward)) > 1e-6) { \
        printf("FAIL [%d]: verify(\"%s\", \"%s\") -> reward=%.1f (expected %.1f) parsed=%.6f\n", \
               tests_run, resp, gt, _r.reward, (float)(expected_reward), _r.parsed_value); \
    } else { \
        tests_passed++; \
    } \
} while(0)

int main(void) {
    printf("=== PufferVerify Math Verifier Tests ===\n\n");

    // --- Basic integer extraction ---
    ASSERT_EXTRACT("42", 42.0);
    ASSERT_EXTRACT("The answer is 42.", 42.0);
    ASSERT_EXTRACT("The answer is 42", 42.0);
    ASSERT_EXTRACT("I got 7 and then 42", 42.0); // last number
    ASSERT_EXTRACT("  42  ", 42.0);
    ASSERT_EXTRACT("0", 0.0);
    ASSERT_EXTRACT("100", 100.0);

    // --- Negative numbers ---
    ASSERT_EXTRACT("-5", -5.0);
    ASSERT_EXTRACT("The result is -3.14", -3.14);
    ASSERT_EXTRACT("-0.5", -0.5);

    // --- Decimals ---
    ASSERT_EXTRACT("3.14159", 3.14159);
    ASSERT_EXTRACT("0.001", 0.001);
    ASSERT_EXTRACT(".5", 0.5);
    ASSERT_EXTRACT("The answer is 2.5 meters", 2.5);

    // --- Commas in numbers ---
    ASSERT_EXTRACT("42,000", 42000.0);
    ASSERT_EXTRACT("1,234,567", 1234567.0);
    ASSERT_EXTRACT("The population is 1,000,000.", 1000000.0);

    // --- Fractions ---
    ASSERT_EXTRACT("3/4", 0.75);
    ASSERT_EXTRACT("The answer is 1/3", 1.0/3.0);
    ASSERT_EXTRACT("7/2", 3.5);
    ASSERT_EXTRACT("-1/4", -0.25);

    // --- Percentages ---
    ASSERT_EXTRACT("42%", 42.0);
    ASSERT_EXTRACT("The rate is 3.5%", 3.5);

    // --- LaTeX \boxed{} ---
    ASSERT_EXTRACT("\\boxed{42}", 42.0);
    ASSERT_EXTRACT("Therefore $\\boxed{3.14}$", 3.14);
    ASSERT_EXTRACT("\\boxed{-7}", -7.0);
    ASSERT_EXTRACT("\\boxed{1/2}", 0.5);
    ASSERT_EXTRACT("The answer is \\boxed{100}.", 100.0);
    ASSERT_EXTRACT("\\boxed{42,000}", 42000.0);

    // --- Nested braces in boxed ---
    ASSERT_EXTRACT("\\boxed{2^{3}}", 2.0); // extracts first number from boxed content "2^{3}"

    // --- GSM8K #### format ---
    ASSERT_EXTRACT("Some work...\n#### 42", 42.0);
    ASSERT_EXTRACT("Step 1: 10\nStep 2: 20\n#### 30", 30.0);
    ASSERT_EXTRACT("#### 3.5", 3.5);
    ASSERT_EXTRACT("####42", 42.0);
    ASSERT_EXTRACT("#### -12", -12.0);
    ASSERT_EXTRACT("#### 1,500", 1500.0);

    // --- Multiple numbers, pick last ---
    ASSERT_EXTRACT("First 10, then 20, finally 30", 30.0);
    ASSERT_EXTRACT("Step 1 gives 5. Step 2 gives 10. The answer is 15.", 15.0);

    // --- Edge cases ---
    ASSERT_EXTRACT_FAIL("");
    ASSERT_EXTRACT_FAIL("no numbers here");
    ASSERT_EXTRACT_FAIL("abc");
    ASSERT_EXTRACT("+5", 5.0);

    // --- Priority: boxed > #### > last number ---
    ASSERT_EXTRACT("I computed 99 but \\boxed{42}", 42.0);
    ASSERT_EXTRACT("Step: 10\n#### 42\nExtra text 99", 42.0);

    // --- Verify correctness ---
    ASSERT_VERIFY("The answer is 42", "42", 1.0f);
    ASSERT_VERIFY("The answer is 42", "43", 0.0f);
    ASSERT_VERIFY("#### 42", "42", 1.0f);
    ASSERT_VERIFY("\\boxed{42}", "42", 1.0f);
    ASSERT_VERIFY("The answer is 3.14", "3.14", 1.0f);
    ASSERT_VERIFY("The answer is 3.14159", "3.14159", 1.0f);
    ASSERT_VERIFY("I think it's 41", "42", 0.0f);
    ASSERT_VERIFY("3/4", "0.75", 1.0f);
    ASSERT_VERIFY("The result is 1/2", "0.5", 1.0f);
    ASSERT_VERIFY("#### 42,000", "42000", 1.0f);

    // --- Verify with tolerance ---
    {
        VerifyResult r = verify_math("3.141592", "3.14159", 1e-4);
        tests_run++;
        if (r.reward == 1.0f) tests_passed++;
        else printf("FAIL [%d]: tolerance test\n", tests_run);
    }

    // --- Verify failures ---
    ASSERT_VERIFY("no answer", "42", 0.0f);
    ASSERT_VERIFY("", "42", 0.0f);
    ASSERT_VERIFY("42", "", 0.0f);

    // --- Batch verify ---
    {
        VerifyResult batch[4];
        batch[0].response = "The answer is 42";
        batch[0].ground_truth = "42";
        batch[1].response = "\\boxed{7}";
        batch[1].ground_truth = "7";
        batch[2].response = "I got 10";
        batch[2].ground_truth = "11";
        batch[3].response = "#### 3.5";
        batch[3].ground_truth = "3.5";

        verify_math_batch(batch, 4, 1e-5);

        tests_run += 4;
        if (batch[0].reward == 1.0f) tests_passed++; else printf("FAIL: batch[0]\n");
        if (batch[1].reward == 1.0f) tests_passed++; else printf("FAIL: batch[1]\n");
        if (batch[2].reward == 0.0f) tests_passed++; else printf("FAIL: batch[2]\n");
        if (batch[3].reward == 1.0f) tests_passed++; else printf("FAIL: batch[3]\n");
    }

    printf("\n=== Results: %d/%d passed ===\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
