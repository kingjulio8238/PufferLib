# Phase 1: Pursuit-Evasion

Multi-agent adversarial task — chasers vs evaders with emergent coordination.

## Overview

Split N drones into two teams: chasers and evaders. Chasers are team-rewarded for capturing evaders; evaders are rewarded for survival. A shared policy with role conditioning allows all agents to train simultaneously.

## Sources

- CrazyRL Catch: https://github.com/ffelten/CrazyRL
- thu-uav Multi-UAV pursuit-evasion: https://github.com/thu-uav/Multi-UAV-pursuit-evasion
- NTU-ICG AMS-DRL: https://github.com/NTU-ICG/AMS-DRL-for-Pursuit-Evasion

## Task Design

- **Agents:** 16 drones total — 12 chasers + 4 evaders (configurable)
- **Chasers:** Rewarded for closing distance to nearest evader; team reward on capture
- **Evaders:** Rewarded for maintaining distance from nearest chaser; survival bonus per step
- **Capture:** Distance between chaser and evader < threshold (e.g., 0.5m)
- **Episode termination:** All evaders captured OR timeout (HORIZON steps)
- **Role conditioning:** 1-bit flag in observations (0 = chaser, 1 = evader)

## Observation Space

Current (23 floats):
1. Linear velocity in body frame (3)
2. Angular velocity in world frame (3)
3. Orientation quaternion (4)
4. Relative target position — coarse (3) + fine (3)
5. Target ring normal in body frame (3)
6. Motor RPMs (4)

Additions for Phase 1:
7. Relative position to nearest agent of OPPOSITE team (3)
8. Relative position to 2nd nearest agent of opposite team (3)
9. Role flag: 0 = chaser, 1 = evader (1)

New total: **30 floats**

Note: "target" for chasers = nearest evader position. "target" for evaders = position away from nearest chaser (or a safe waypoint).

## Action Space

Unchanged: 4 continuous actions (per-motor RPM targets)

## Reward Design

### Chasers
```
reward = alpha_chase * (prev_dist_to_nearest_evader - curr_dist_to_nearest_evader)
       + alpha_capture * capture_event                    # sparse, team-wide
       - alpha_collision * inter_chaser_collision_penalty
       - alpha_omega * angular_velocity
```

### Evaders
```
reward = alpha_survive * per_step_survival_bonus
       + alpha_evade * (curr_min_dist_to_chaser - prev_min_dist_to_chaser)
       - alpha_oob * out_of_bounds_penalty
       - alpha_omega * angular_velocity
```

### Team reward (from thu-uav)
When ANY chaser captures an evader, ALL chasers receive the capture bonus. This encourages cooperation over individual pursuit.

## Code Changes Required

### tasks.h
- Add `CHASE` to `DroneTask` enum (after RACE)
- Add "chase" to `TASK_NAMES` array
- Implement `set_target_chase()` — sets chaser targets to nearest evader position, evader targets to flee direction
- Add case in `set_target()` dispatcher

### dronelib.h
- Add helper: `nearest_opponent_drone()` — finds nearest drone of opposite team
- Modify `compute_drone_observations()` to include neighbor relative positions and role flag
- Update `OBS_SIZE` computation

### drone.h
- Add chase-specific fields to `DroneEnv`: `num_chasers`, `num_evaders`, `capture_radius`, `alpha_chase`, `alpha_capture`, `alpha_survive`, `alpha_evade`
- Add `is_captured[]` array to track evader state
- Modify `c_step()` to compute chase/evade rewards based on role
- Modify `c_reset()` to reset capture state, randomize spawn positions with team separation

### binding.c
- Update `OBS_SIZE` from 23 to 30
- Add new config params to `my_init()`: `num_chasers`, `capture_radius`, chase reward alphas
- Add new metrics to `my_log()`: `captures`, `survival_time`, `chase_dist_ema`

### config/drone.ini
- Add `[env]` entries: `num_chasers`, `capture_radius`, chase-specific alpha values
- Set `task = 8` (CHASE)

## Rendering

### render.h
- Color chasers red, evaders blue (override COLORS array based on role)
- Draw capture radius sphere around evaders (wireframe)
- Flash effect on capture event
- HUD: show captures count, evaders remaining, time

## Training Strategy

- **Shared policy** with role conditioning (1-bit in obs)
- All 16 agents use the same neural network
- Role determines reward computation, not action space
- PufferLib's existing single-policy training loop works as-is

### Alternative (future): Self-play
- Train chaser policy against frozen evader checkpoints and vice versa
- Periodically swap frozen opponents from checkpoint pool
- More complex but produces more robust strategies

## Expected Emergent Behaviors

Based on thu-uav results (Nature Scientific Reports 2025):
- Pincer movements (chasers converge from opposite sides)
- Lazy pursuit (energy-efficient chasers let others do active chasing)
- Formation encirclement (spread out to cut off escape routes)
- Evasive spiraling and altitude changes from evaders
- Chaser specialization (some cut off escape paths, others pursue directly)

## Eval Metrics

| Metric | Description |
|---|---|
| capture_rate | % of episodes where all evaders captured |
| time_to_capture | Mean steps to first/last capture |
| chaser_coordination | Spatial spread of chasers around evader |
| evader_survival_time | Mean steps before capture |
| chaser_collision_rate | Inter-chaser collisions per episode |
| chase_dist_ema | EMA of chaser-to-evader distance |

## Curriculum (Optional)

1. Start with 1 evader, 4 chasers (easy)
2. Scale to 2 evaders, 8 chasers
3. Full 4 evaders, 12 chasers
4. Add obstacles (future phase)

## Dependencies

- No new physics (uses existing 6-DOF quadrotor dynamics)
- No new external libraries
- Requires recompile after OBS_SIZE change: `./build.sh drone` (or `--cpu` / `--fast`)

---

## Precise Integration Points (from codebase audit)

### tasks.h — Exact Changes

**Line 13-23: Add CHASE before TASK_N**
```c
typedef enum {
    IDLE,
    HOVER,
    ORBIT,
    FOLLOW,
    CUBE,
    CONGO,
    FLAG,
    RACE,
    CHASE,       // <-- NEW (value = 8)
    TASK_N
} DroneTask;
```

**Line 25-26: Add "chase" to TASK_NAMES**
```c
static char const* TASK_NAMES[TASK_N] = {"idle", "hover", "orbit", "follow",
                                         "cube", "congo", "flag",  "race", "chase"};
```

**After line 144: Add set_target_chase()**
```c
void set_target_chase(Drone* agents, int idx, int num_agents, int num_chasers) {
    Drone* agent = &agents[idx];
    bool is_chaser = (idx < num_chasers);

    if (is_chaser) {
        // Chaser: target = nearest evader position
        float min_dist = FLT_MAX;
        for (int i = num_chasers; i < num_agents; i++) {
            float d = norm3(sub3(agent->state.pos, agents[i].state.pos));
            if (d < min_dist) {
                min_dist = d;
                agent->target->pos = agents[i].state.pos;
            }
        }
    } else {
        // Evader: target = position away from nearest chaser
        float min_dist = FLT_MAX;
        Vec3 nearest_chaser_pos = {0};
        for (int i = 0; i < num_chasers; i++) {
            float d = norm3(sub3(agent->state.pos, agents[i].state.pos));
            if (d < min_dist) {
                min_dist = d;
                nearest_chaser_pos = agents[i].state.pos;
            }
        }
        // Flee direction: away from nearest chaser, clamped to arena
        Vec3 flee_dir = sub3(agent->state.pos, nearest_chaser_pos);
        float flee_norm = norm3(flee_dir);
        if (flee_norm > 0.01f) {
            flee_dir = scalmul3(flee_dir, 5.0f / flee_norm);
        }
        Vec3 flee_pos = add3(agent->state.pos, flee_dir);
        agent->target->pos = (Vec3){
            clampf(flee_pos.x, -MARGIN_X, MARGIN_X),
            clampf(flee_pos.y, -MARGIN_Y, MARGIN_Y),
            clampf(flee_pos.z, -MARGIN_Z, MARGIN_Z)
        };
    }
    agent->target->vel = (Vec3){0, 0, 0};
}
```

**Line 146-157: Add CHASE case to set_target() dispatcher**
The dispatcher uses if/else if chains (not switch). Add after line 156:
```c
else if (task == CHASE) {} // targets set dynamically in c_step, not here
```

### drone.h — Exact Changes

**Line 20-49: Add fields to DroneEnv struct**
After `float hover_vel;` (line 48), add:
```c
    // chase task parameters
    int num_chasers;
    float capture_radius;
    float alpha_chase;
    float alpha_capture;
    float alpha_survive;
    float alpha_evade;
    int captures;              // count of captures this episode
    int evaders_remaining;
```

**Line 64-88: Add chase metrics to Log struct (dronelib.h:57-71)**
Add to the Log struct:
```c
    float captures;
    float evader_survival;
```

**Line 90-94: Modify compute_observations()**
Currently hardcodes `i*23`. Must change to use OBS_SIZE or the new size:
```c
void compute_observations(DroneEnv* env) {
    for (int i = 0; i < env->num_agents; i++) {
        compute_drone_observations(&env->agents[i], env->observations + i*OBS_SIZE);
    }
}
```
Note: OBS_SIZE is defined in binding.c. drone.h will need a forward reference or we pass the size.

**Line 96-125: Modify reset_agent() for team spawn separation**
For CHASE task, spawn chasers on one side, evaders on the other:
```c
if (env->task == CHASE) {
    bool is_chaser = (idx < env->num_chasers);
    if (is_chaser) {
        agent->state.pos = (Vec3){
            rndf(-MARGIN_X, 0, &env->rng),  // chasers spawn in negative-x half
            rndf(-MARGIN_Y, MARGIN_Y, &env->rng),
            rndf(-MARGIN_Z, MARGIN_Z, &env->rng)};
    } else {
        agent->state.pos = (Vec3){
            rndf(0, MARGIN_X, &env->rng),    // evaders spawn in positive-x half
            rndf(-MARGIN_Y, MARGIN_Y, &env->rng),
            rndf(-MARGIN_Z, MARGIN_Z, &env->rng)};
    }
}
```

**Line 127-139: Modify c_reset() for CHASE**
After the RACE ring reset block, add:
```c
if (env->task == CHASE) {
    env->captures = 0;
    env->evaders_remaining = env->num_agents - env->num_chasers;
}
```

**Line 141-186: Modify c_step() for CHASE rewards**
The core reward loop (lines 159-162) currently computes hover-style rewards. For CHASE, we need role-based rewards. After line 162, add:
```c
if (env->task == CHASE) {
    bool is_chaser = (i < env->num_chasers);

    // Update chase targets dynamically
    set_target_chase(env->agents, i, env->num_agents, env->num_chasers);

    if (is_chaser) {
        // Find nearest evader distance
        float min_evader_dist = FLT_MAX;
        for (int j = env->num_chasers; j < env->num_agents; j++) {
            float d = norm3(sub3(agent->state.pos, env->agents[j].state.pos));
            if (d < min_evader_dist) min_evader_dist = d;
        }
        float prev_evader_dist = /* need to store prev */;
        reward = env->alpha_chase * (prev_evader_dist - min_evader_dist);

        // Check capture
        if (min_evader_dist < env->capture_radius) {
            reward += env->alpha_capture;
            env->captures++;
        }
    } else {
        // Evader: reward for staying alive and far from chasers
        float min_chaser_dist = FLT_MAX;
        for (int j = 0; j < env->num_chasers; j++) {
            float d = norm3(sub3(agent->state.pos, env->agents[j].state.pos));
            if (d < min_chaser_dist) min_chaser_dist = d;
        }
        reward = env->alpha_survive + env->alpha_evade * min_chaser_dist;

        // Check if captured
        if (min_chaser_dist < env->capture_radius) {
            oob = true; // force reset for captured evader
        }
    }
    reward -= env->alpha_omega * omega;
}
```

### dronelib.h — Exact Changes

**Line 476-493: nearest_drone() already exists**
Can be extended to `nearest_opponent_drone()` filtering by team:
```c
static inline Drone* nearest_opponent_drone(Drone* agent, Drone* others,
                                            int start, int end) {
    float min_dist = FLT_MAX;
    Drone* nearest = NULL;
    for (int i = start; i < end; i++) {
        Drone* other = &others[i];
        if (other == agent) continue;
        float dist = norm3(sub3(agent->state.pos, other->state.pos));
        if (dist < min_dist) {
            min_dist = dist;
            nearest = other;
        }
    }
    return nearest;
}
```

**Line 559-607: Expand compute_drone_observations()**
Currently writes indices 0-22 (23 floats). Insert new obs before RPMs (which the comment says "should always be last"). Insert at index 19 (before RPMs at 19-22), shifting RPMs to 26-29:

```c
// --- NEW: nearest opponent relative position (body frame) ---
// For chasers (idx < num_chasers): nearest evader
// For evaders: nearest chaser
// This requires passing env info; may need to change signature to:
// compute_drone_observations(Drone* agent, float* obs, Drone* all, int num_agents, int num_chasers)
Vec3 to_opponent_world = sub3(nearest_opponent->state.pos, agent->state.pos);
Vec3 to_opponent = quat_rotate(q_inv, to_opponent_world);
observations[idx++] = tanhf(to_opponent.x * 0.1f);
observations[idx++] = tanhf(to_opponent.y * 0.1f);
observations[idx++] = tanhf(to_opponent.z * 0.1f);

// 2nd nearest opponent
observations[idx++] = tanhf(to_opponent2.x * 0.1f);
observations[idx++] = tanhf(to_opponent2.y * 0.1f);
observations[idx++] = tanhf(to_opponent2.z * 0.1f);

// Role flag
observations[idx++] = is_chaser ? 0.0f : 1.0f;

// RPMs (always last)
observations[idx++] = agent->state.rpms[0] / agent->params.max_rpm;
observations[idx++] = agent->state.rpms[1] / agent->params.max_rpm;
observations[idx++] = agent->state.rpms[2] / agent->params.max_rpm;
observations[idx++] = agent->state.rpms[3] / agent->params.max_rpm;
// Total: 30 floats
```

### binding.c — Exact Changes

**Line 4: Change OBS_SIZE**
```c
#define OBS_SIZE 30  // was 23
```

**Line 12-24: Add new params to my_init()**
After line 23 (`env->hover_vel = ...`), add:
```c
    env->num_chasers = (int)dict_get(kwargs, "num_chasers")->value;
    env->capture_radius = dict_get(kwargs, "capture_radius")->value;
    env->alpha_chase = dict_get(kwargs, "alpha_chase")->value;
    env->alpha_capture = dict_get(kwargs, "alpha_capture")->value;
    env->alpha_survive = dict_get(kwargs, "alpha_survive")->value;
    env->alpha_evade = dict_get(kwargs, "alpha_evade")->value;
```

**Line 27-40: Add chase metrics to my_log()**
After line 39 (`dict_set(out, "ema_omega", ...)`), add:
```c
    dict_set(out, "captures", log->captures);
    dict_set(out, "evader_survival", log->evader_survival);
```

### config/drone.ini — New Entries

Under `[env]`:
```ini
num_chasers = 12
capture_radius = 0.5
alpha_chase = 1.0
alpha_capture = 10.0
alpha_survive = 0.01
alpha_evade = 0.1
```

Set task to CHASE:
```ini
task = 8
```

### render.h — Exact Changes

**Line 14-19: Team-based coloring**
In the drone drawing loop (line 524-541), replace color assignment at line 527:
```c
// Was: Color body_color = (inspect_mode && is_selected) ? PUFF_GREEN : COLORS[i % 64];
Color body_color;
if (env->task == CHASE) {
    bool is_chaser = (i < env->num_chasers);
    body_color = (inspect_mode && is_selected) ? PUFF_GREEN
               : is_chaser ? (Color){255, 60, 60, 255}   // red for chasers
                           : (Color){60, 120, 255, 255};  // blue for evaders
} else {
    body_color = (inspect_mode && is_selected) ? PUFF_GREEN : COLORS[i % 64];
}
```

**Line 577-582: Draw capture radius in CHASE mode**
After the RACE ring drawing block, add:
```c
if (env->task == CHASE) {
    for (int i = env->num_chasers; i < env->num_agents; i++) {
        Vec3 p = env->agents[i].state.pos;
        DrawSphereWires((Vector3){p.x, p.y, p.z}, env->capture_radius,
                        8, 8, ColorAlpha(BLUE, 0.3f));
    }
}
```

**Line 611-613: Add chase stats to HUD**
After the task name display, add:
```c
if (env->task == CHASE) {
    DrawText(TextFormat("Captures: %d | Evaders: %d", env->captures, env->evaders_remaining),
             10, y, 18, YELLOW);
    y += 22;
}
```

### Key Design Decisions

1. **Observation signature change**: `compute_drone_observations()` currently takes `(Drone*, float*)`. For chase, it needs access to all agents and team info. Options:
   - (a) Change signature to include `Drone* all_agents, int num_agents, int num_chasers`
   - (b) Store chase-relevant obs in the Drone struct during c_step, read in obs function
   - Recommend (a) — cleaner, obs computed from current state

2. **The `i*23` hardcode in drone.h:92**: Must change to `i*OBS_SIZE` or a variable. Since OBS_SIZE is in binding.c, either pass it through or define a shared constant.

3. **Target updates**: For CHASE, targets must update every step (not just on reset), since the nearest opponent moves. Call `set_target_chase()` inside c_step() before reward computation.

4. **Backward compatibility**: Non-CHASE tasks still use 23 obs but OBS_SIZE is now 30. Fill extra 7 slots with zeros for non-chase tasks, or conditionally compute. Recommend: always fill 30, set chase-specific obs to 0.0 for other tasks. This keeps the network architecture fixed.
