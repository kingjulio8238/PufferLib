// Originally made by Sam Turner and Finlay Sanders, 2025.
// Included in pufferlib under the original project's MIT license.
// https://github.com/tensaur/drone

#pragma once

#include <limits.h>
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>

#include "dronelib.h"
#include "tasks.h"

#define HORIZON 1024


typedef struct Client Client;
typedef struct DroneEnv DroneEnv;

struct DroneEnv {
    Log log;
    float* observations;
    float* actions;
    float* rewards;
    float* terminals;
    int num_agents;
    unsigned int rng;

    int tick;
    DroneTask task;
    Drone* agents;

    int max_rings;
    Target* ring_buffer;

    Client* client;

    // reward scaling
    float alpha_dist;
    float alpha_hover;
    float alpha_shaping;
    float alpha_omega;

    // hover task parameters
    float hover_target_dist;
    float hover_dist;
    float hover_omega;
    float hover_vel;

    // chase task parameters
    int num_chasers;
    float capture_radius;
    float alpha_chase;
    float alpha_capture;
    float alpha_survive;
    float alpha_evade;
    int captures;
    int evaders_remaining;
};

void init(DroneEnv* env) {
    env->agents = (Drone*)calloc(env->num_agents, sizeof(Drone));
    env->ring_buffer = (Target*)calloc(env->max_rings, sizeof(Target));

    for (int i = 0; i < env->num_agents; i++) {
        env->agents[i].target = (Target*)calloc(1, sizeof(Target));
        env->agents[i].buffer_idx = 0;
    }

    env->log = (Log){0};
    env->tick = 0;
}

void add_log(DroneEnv* env, int idx, bool oob, bool timeout) {
    Drone* agent = &env->agents[idx];

    env->log.episode_return += agent->episode_return;
    env->log.episode_length += agent->episode_length;
    env->log.collisions += agent->collisions;

    if (oob) env->log.oob += 1.0f;
    if (timeout) env->log.timeout += 1.0f;

    env->log.score += agent->hover_score;
    env->log.perf += agent->hover_ema;
    env->log.rings_passed += agent->rings_passed;
    env->log.ema_dist += agent->ema_dist;
    env->log.ema_vel += agent->ema_vel;
    env->log.ema_omega += agent->ema_omega;

    env->log.n += 1.0f;

    agent->episode_length = 0;
    agent->episode_return = 0.0f;
    agent->collisions = 0.0f;
    agent->score = 0.0f;
    agent->rings_passed = 0.0f;
    agent->prev_chase_dist = 0.0f;
}

void compute_observations(DroneEnv* env) {
    for (int i = 0; i < env->num_agents; i++) {
        float* obs = env->observations + i * 30;
        // Base 23 observations (velocity, orientation, target, RPMs)
        compute_drone_observations(&env->agents[i], obs);

        // Chase-specific observations (indices 23-29)
        if (env->task == CHASE) {
            bool is_chaser = (i < env->num_chasers);
            int search_start = is_chaser ? env->num_chasers : 0;
            int search_end = is_chaser ? env->num_agents : env->num_chasers;
            nearest_two_opponent_obs(&env->agents[i], env->agents,
                                     search_start, search_end, obs + 23);
            obs[29] = is_chaser ? 0.0f : 1.0f;
        } else {
            // Zero-fill for non-chase tasks
            for (int j = 23; j < 30; j++) obs[j] = 0.0f;
        }
    }
}

void reset_agent(DroneEnv* env, Drone* agent, int idx) {
    agent->episode_return = 0.0f;
    agent->episode_length = 0;
    agent->collisions = 0.0f;
    agent->rings_passed = 0;
    agent->score = 0.0f;
    agent->hover_score = 0.0f;
    agent->hover_ema = 0.0f;
    agent->ema_dist = 0.0f;
    agent->ema_vel = 0.0f;
    agent->ema_omega = 0.0f;
    agent->prev_chase_dist = 0.0f;
    agent->prev_nearest_opponent = -1;

    agent->buffer = env->ring_buffer;
    agent->buffer_size = env->max_rings;

    init_drone(agent, &env->rng, 0.05f);

    if (env->task == CHASE) {
        // Spawn chasers and evaders close together in small arena
        bool is_chaser = (idx < env->num_chasers);
        if (is_chaser) {
            agent->state.pos = (Vec3){
                rndf(-6.0f, -1.0f, &env->rng),
                rndf(-5.0f, 5.0f, &env->rng),
                rndf(-3.0f, 3.0f, &env->rng)};
        } else {
            agent->state.pos = (Vec3){
                rndf(1.0f, 6.0f, &env->rng),
                rndf(-5.0f, 5.0f, &env->rng),
                rndf(-3.0f, 3.0f, &env->rng)};
        }
    } else {
        agent->state.pos =
            (Vec3){rndf(-MARGIN_X, MARGIN_X, &env->rng), rndf(-MARGIN_Y, MARGIN_Y, &env->rng), rndf(-MARGIN_Z, MARGIN_Z, &env->rng)};
    }

    if (env->task == RACE) {
        while (norm3(sub3(agent->state.pos, env->ring_buffer[0].pos)) < 2.0f * RING_RADIUS) {
            agent->state.pos = (Vec3){rndf(-MARGIN_X, MARGIN_X, &env->rng), rndf(-MARGIN_Y, MARGIN_Y, &env->rng),
                                      rndf(-MARGIN_Z, MARGIN_Z, &env->rng)};
        }
    }

    agent->prev_pos = agent->state.pos;
    agent->prev_potential = hover_potential(agent, env->hover_dist, env->hover_omega, env->hover_vel);
}

void c_reset(DroneEnv* env) {
    if (env->task == RACE) {
        reset_rings(&env->rng, env->ring_buffer, env->max_rings);
    }

    if (env->task == CHASE) {
        env->captures = 0;
        env->evaders_remaining = env->num_agents - env->num_chasers;
    }

    for (int i = 0; i < env->num_agents; i++) {
        Drone* agent = &env->agents[i];
        reset_agent(env, agent, i);
        set_target(&env->rng, env->task, env->agents, i, env->num_agents,
                   env->hover_target_dist, env->num_chasers);
    }

    compute_observations(env);
}

void c_step(DroneEnv* env) {
    env->tick = (env->tick + 1) % HORIZON;

    // Chase pre-pass: detect captures ONCE per evader to avoid double counting
    bool evader_captured[256] = {false}; // max agents
    if (env->task == CHASE) {
        for (int e = env->num_chasers; e < env->num_agents; e++) {
            float min_dist = nearest_opponent_dist(
                &env->agents[e], env->agents, 0, env->num_chasers, NULL);
            if (min_dist < env->capture_radius) {
                evader_captured[e] = true;
                env->captures++;
                env->log.captures += 1.0f;
            }
        }
    }

    for (int i = 0; i < env->num_agents; i++) {
        Drone* agent = &env->agents[i];

        agent->prev_pos = agent->state.pos;
        move_drone(agent, &env->actions[4 * i]);
        agent->episode_length++;

        float omega = norm3(agent->state.omega);
        float reward;
        bool oob;
        bool timeout = (agent->episode_length >= HORIZON);

        if (env->task == CHASE) {
            // Update chase targets dynamically every step
            set_target_chase(env->agents, i, env->num_agents, env->num_chasers);

            bool is_chaser = (i < env->num_chasers);
            bool captured = false;

            // OOB: clamp position to CHASE arena bounds (no termination)
            oob = fabsf(agent->state.pos.x) > CHASE_X ||
                  fabsf(agent->state.pos.y) > CHASE_Y ||
                  fabsf(agent->state.pos.z) > CHASE_Z;
            if (oob) {
                agent->state.pos.x = clampf(agent->state.pos.x, -CHASE_X, CHASE_X);
                agent->state.pos.y = clampf(agent->state.pos.y, -CHASE_Y, CHASE_Y);
                agent->state.pos.z = clampf(agent->state.pos.z, -CHASE_Z, CHASE_Z);
                agent->state.vel = (Vec3){0, 0, 0};
            }

            if (is_chaser) {
                // Chaser reward: distance shaping toward nearest evader
                int nearest_idx = -1;
                float min_evader_dist = nearest_opponent_dist(
                    agent, env->agents, env->num_chasers, env->num_agents, &nearest_idx);

                // Always apply distance shaping (don't gate on same opponent)
                if (agent->prev_chase_dist > 0.0f) {
                    reward = env->alpha_chase * (agent->prev_chase_dist - min_evader_dist);
                } else {
                    reward = 0.0f;
                }
                agent->prev_chase_dist = min_evader_dist;
                agent->prev_nearest_opponent = nearest_idx;

                // Capture bonus: team reward if ANY evader was captured this step
                for (int e = env->num_chasers; e < env->num_agents; e++) {
                    if (evader_captured[e]) {
                        reward += env->alpha_capture;
                        break; // one bonus per chaser per step
                    }
                }

                // Track ema_dist for monitoring
                agent->ema_dist = 0.99f * agent->ema_dist + 0.01f * min_evader_dist;
            } else {
                // Evader reward: delta-based distance shaping + survival
                int nearest_idx = -1;
                float min_chaser_dist = nearest_opponent_dist(
                    agent, env->agents, 0, env->num_chasers, &nearest_idx);

                reward = env->alpha_survive;
                // Always apply distance shaping
                if (agent->prev_chase_dist > 0.0f) {
                    reward += env->alpha_evade * (min_chaser_dist - agent->prev_chase_dist);
                }
                agent->prev_chase_dist = min_chaser_dist;
                agent->prev_nearest_opponent = nearest_idx;
                env->log.evader_survival += 1.0f;

                // Captured = forced reset (detected in pre-pass)
                captured = evader_captured[i];

                if (captured) {
                    reward = -5.0f; // strong penalty for being caught
                }

                // Track ema_dist for monitoring
                agent->ema_dist = 0.99f * agent->ema_dist + 0.01f * min_chaser_dist;
            }

            reward -= env->alpha_omega * omega;
            // Only terminate on capture (not OOB)
            oob = captured;
        } else {
            // Original reward logic for all other tasks
            oob = norm3(sub3(agent->target->pos, agent->state.pos)) > (env->hover_target_dist + 1.0f);

            float curr = hover_potential(agent, env->hover_dist, env->hover_omega, env->hover_vel);
            float prev_dist = norm3(sub3(agent->target->pos, agent->prev_pos));
            float curr_dist = norm3(sub3(agent->target->pos, agent->state.pos));

            reward = env->alpha_dist * (prev_dist - curr_dist)
                   + env->alpha_hover * curr
                   + env->alpha_shaping * (curr - agent->prev_potential)
                   - env->alpha_omega * omega;

            agent->prev_potential = curr;

            float h = check_hover(agent, env->hover_dist, env->hover_omega, env->hover_vel);
            agent->hover_score += h;
            agent->hover_ema = (1.0f - 0.02f) * agent->hover_ema + 0.02f * h;
            float curr_dist2 = norm3(sub3(agent->target->pos, agent->state.pos));
            agent->ema_dist = 0.99f * agent->ema_dist + 0.01f * curr_dist2;
        }

        agent->ema_vel = 0.99f * agent->ema_vel + 0.01f * norm3(agent->state.vel);
        agent->ema_omega = 0.99f * agent->ema_omega + 0.01f * omega;
        agent->episode_return += reward;
        env->rewards[i] = reward;

        bool reset = oob || timeout;
        env->terminals[i] = reset ? 1.0f : 0.0f;

        if (reset) {
            add_log(env, i, oob, timeout);
            reset_agent(env, agent, i);
            set_target(&env->rng, env->task, env->agents, i, env->num_agents,
                       env->hover_target_dist, env->num_chasers);
        }
    }

    compute_observations(env);
}

void c_close_client(Client* client);

void c_close(DroneEnv* env) {
    for (int i = 0; i < env->num_agents; i++) {
        free(env->agents[i].target);
    }

    free(env->agents);
    free(env->ring_buffer);

    if (env->client != NULL) {
        c_close_client(env->client);
    }
}
