// G1 standalone demo. B2: headless random-policy loop (build/link check).
// B3 will add the raylib viewer (define G1_HAS_RENDER + real c_render).
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "g1.h"

int main(void) {
    G1 env = {0};
    float obs[G1_OBS_SIZE], act[G1_NUM_JOINTS], rew = 0, term = 0;
    env.observations = obs;
    env.actions = act;
    env.rewards = &rew;
    env.terminals = &term;
    env.rng = 0;
    g1_set_default_config(&env);
    g1_init(&env);
    c_reset(&env);

    unsigned int arng = 7u;
    const int n = 2000;
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < G1_NUM_JOINTS; j++)
            act[j] = 0.3f * (2.0f * ((float)rand_r(&arng) / (float)RAND_MAX) - 1.0f);
        c_step(&env);
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double secs = (double)(t1.tv_sec - t0.tv_sec) + 1e-9 * (double)(t1.tv_nsec - t0.tv_nsec);
    printf("g1 standalone: %d control steps, %.0f ctrl-steps/s, pelvis z=%.3f, return-ish r=%.4f\n",
           n, n / secs, env.d->qpos[2], rew);
    c_close(&env);
    return 0;
}
