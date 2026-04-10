# Extending PufferLib Drone Env with Real-World Tasks

Full research synthesis and phased implementation roadmap.

**Goal:** Multi-agent behavior → Aerial manipulation

**Date:** 2026-04-10

---

## The Landscape

We surveyed **11 open-source drone RL environments**, **6 real-world competitions**, and **dozens of papers** on aerial manipulation to find existing tasks and implementations we can adapt for PufferLib's drone environment.

### Environments Surveyed

| Environment | Source | Key Tasks | Multi-Agent | PufferLib Fit |
|---|---|---|---|---|
| gym-pybullet-drones | utiasDSL | Hover, Flock, LeaderFollower, Meetup | Yes | HIGH |
| Flightmare | UZH RPG | High-speed gate racing, vision nav | Limited | MEDIUM |
| safe-control-gym | utiasDSL | Constrained hover/tracking | No | HIGH (constraints) |
| CrazyRL | ffelten | Circle, Surround, Escort, **Catch (pursuit-evasion)** | Yes (core) | HIGH |
| AirGym | emNavi | Hover, Balloon (target dash), **Avoid (dodge obstacles)** | No | MEDIUM-HIGH |
| OmniDrones | btx0424 | **InvPendulum, PayloadTrack, TransportTrack, Formation** | Yes | HIGH (task designs) |
| Aerial Gym | ntnu-arl | Depth-based nav, obstacle avoidance | Parallel only | MEDIUM |
| quad-swarm-rl | Zhehui-Huang | Swarm nav with collision avoidance | Yes (swarm) | MEDIUM-HIGH |
| learning-to-fly | ARP Lab | Attitude control, hover (18s training on M1!) | No | Already shared physics |
| lsy_drone_racing | utiasDSL | Gate racing with difficulty levels | No | Covered by RACE |
| VolleyBots | thu-uav | **Drone volleyball** (NeurIPS 2025) | Yes (competitive) | MODERATE |

### Competitions Surveyed

| Competition | Task | Relevance |
|---|---|---|
| **Swift** (Nature 2023) | 7 gates, 3 laps, beat human world champion | Gate racing benchmark |
| **AlphaPilot** (2019) | Autonomous gate racing, $1M prize | Same paradigm |
| **A2RL x DCL** (2025) | 170m track, beat 3 FPV world champions | Scaled racing |
| **IROS Drone Racing** (2016-2019) | Autonomous gate navigation | Academic benchmark |
| **Game of Drones** (NeurIPS 2019) | Racing + opponent avoidance | Multi-agent racing |
| **Anduril AI Grand Prix** (2026) | $500K autonomous drone racing | Active competition |

### Key Papers

| Paper | Year | Relevance |
|---|---|---|
| thu-uav Multi-UAV Pursuit-Evasion | 2025 | 99.9% capture, emergent flanking, real Crazyflie transfer |
| UPenn Agile Flight from Competitive Racing | 2024 | Sparse rewards → emergent aggressive flight |
| Swooper: High-Speed Aerial Grasping | 2026 | 1D gripper action, 84% real-world success, <60 min training |
| Sreenath/Lee/Kumar Geometric Control | 2013 | Cable-suspended payload equations |
| OmniDrones Platform | 2023 | Multi-task drone benchmark (ICLR) |
| OpenAI Hide and Seek | 2020 | Emergent tool use from self-play |

---

## Current PufferLib Drone Environment

### Architecture

| File | Role | Lines |
|---|---|---|
| `ocean/drone/dronelib.h` | Core physics, state, observations | 607 |
| `ocean/drone/drone.h` | Environment struct, init, reset, step | 202 |
| `ocean/drone/tasks.h` | 8 task definitions, target dispatch | 158 |
| `ocean/drone/render.h` | Raylib 3D visualization | 701 |
| `ocean/drone/drone.c` | Standalone executable | 91 |
| `ocean/drone/binding.c` | Python/gym C interface | 41 |

### Physics Model
- **Platform:** Crazyflie 2.0 (27g, 39.6mm arm length)
- **Dynamics:** Full 6-DOF quadrotor, RK4 integration at 500Hz
- **Forces:** Gravity, per-motor thrust (T = k_thrust * rpm^2), aerodynamic drag, angular damping, gyroscopic coupling
- **Motor dynamics:** First-order response with k_mot = 0.15s time constant
- **Domain randomization:** All physical parameters randomized +/-5% per episode
- **Arena:** 30x30x10m

### Observation Space (23 floats)
1. Linear velocity in body frame (3)
2. Angular velocity in world frame (3)
3. Orientation quaternion (4)
4. Relative target position - coarse (3) + fine (3)
5. Target ring normal in body frame (3)
6. Motor RPMs (4)

### Action Space (4 continuous)
- Per-motor RPM targets, range [-1, 1]

### Existing Tasks (8)
| Task | Description | Type |
|---|---|---|
| IDLE | Random moving target bouncing off walls | Single-agent |
| **HOVER** | Stationary target at random location (DEFAULT) | Single-agent |
| ORBIT | Fibonacci sphere of fixed target points | Multi-drone formation |
| FOLLOW | All drones follow agent 0's target | Leader-follower |
| CUBE | 4x4x4 grid of fixed waypoints | Static formation |
| CONGO | Conga line — each drone follows previous drone (40-step lag) | Formation tracking |
| FLAG | Linear grid arrangement | Static formation |
| RACE | Sequential ring navigation (up to 10 rings) | Single-agent sequential |

### Reward Structure
```
reward = alpha_dist * (prev_dist - curr_dist)       # 0.782192
       + alpha_hover * hover_potential               # 0.071445
       + alpha_shaping * (curr - prev_potential)     # 3.9754
       - alpha_omega * angular_velocity              # 0.00135588
```

### What Requires Recompilation
- OBS_SIZE, NUM_ATNS (observation/action dimensions)
- Environment physics, collision detection
- Reward calculation logic
- New metrics in my_log()

### What's Config-Only (drone.ini)
- All reward coefficients (alpha_*)
- Network architecture (hidden_size, num_layers)
- Training hyperparameters (lr, gamma, clip_coef, etc.)
- Task selection (task = 0-7)
- Number of drones, max rings

---

## Phased Implementation Roadmap

### Phase 1: Pursuit-Evasion (Multi-Agent Adversarial)

**Priority:** FIRST — fastest to implement, most visually impressive, validates multi-agent infra

**Sources:**
- CrazyRL Catch task: [github.com/ffelten/CrazyRL](https://github.com/ffelten/CrazyRL)
- thu-uav Multi-UAV pursuit-evasion: [github.com/thu-uav/Multi-UAV-pursuit-evasion](https://github.com/thu-uav/Multi-UAV-pursuit-evasion)
- NTU-ICG AMS-DRL: [github.com/NTU-ICG/AMS-DRL-for-Pursuit-Evasion](https://github.com/NTU-ICG/AMS-DRL-for-Pursuit-Evasion)

**Design:**
- Split N drones into chasers and evaders (e.g., 12 chasers + 4 evaders)
- Chasers rewarded for closing distance to nearest evader; team reward on capture
- Evaders rewarded for maintaining distance from nearest chaser; survival bonus
- Capture = distance < threshold (e.g., 0.5m)
- Shared policy with 1-bit role conditioning (am I a chaser or evader?)

**Observation Changes:**
- Current: 23 floats
- Add: relative position to nearest 2 agents (6 floats) + role bit (1 float)
- New total: ~30 floats
- Requires recompile (OBS_SIZE change)

**Code Changes (~50-80 lines):**
- `tasks.h`: Add `CHASE` task enum + `set_target_chase()` function
- `dronelib.h`: Add `nearest_drone()` helper for inter-agent distance (already partially exists)
- `binding.c`: Update OBS_SIZE
- `drone.h`: Modify `compute_drone_observations()` to include neighbor state
- `drone.h`: Add chase-specific reward terms in `c_step()`

**Reward Design (from literature):**
```
Chaser: alpha_chase * (prev_dist_to_evader - curr_dist_to_evader)
      + alpha_capture * capture_event          # sparse team reward
      - alpha_collision * inter_chaser_collision

Evader: alpha_survive * time_alive
      + alpha_evade * min_dist_to_chaser
      - alpha_oob * out_of_bounds
```

**Training:**
- Shared policy with role conditioning (PufferLib-native, single policy for all agents)
- Role encoded as 1-bit in observation: 0 = chaser, 1 = evader
- Expected training time: ~30 min on RTX 4090

**Expected Emergent Behaviors (from thu-uav):**
- Pincer movements (chasers flank from opposite sides)
- Lazy pursuit (one chaser minimizes effort while others chase)
- Evasive spiraling and altitude changes
- Formation-based encirclement
- 99.9% capture rate after convergence

**Eval Metrics:**
- Capture rate (% of episodes with successful capture)
- Time to capture (steps)
- Chaser coordination score (spread around evader)
- Evader survival time

---

### Phase 2: Competitive Racing (Multi-Agent Head-to-Head)

**Priority:** SECOND — already 90% built (RACE task exists), adds competitive pressure

**Sources:**
- UPenn Agile Flight: [github.com/Jirl-upenn/AgileFlight_MultiAgent](https://github.com/Jirl-upenn/AgileFlight_MultiAgent)
- Swift (Nature 2023): UZH RPG Lab
- Game of Drones (NeurIPS 2019): [github.com/microsoft/AirSim-NeurIPS2019-Drone-Racing](https://github.com/microsoft/AirSim-NeurIPS2019-Drone-Racing)
- SPIRAL (IEEE 2025): Self-Play Incremental Racing Algorithm

**Design:**
- 2-4 drones racing through shared ring sequence
- Reward for passing gates while leading (10 pts) vs trailing (5 pts)
- Bonus for lap completion (50 pts), penalty for crash
- Key insight from UPenn: sparse rewards only (no distance shaping) → emergent aggressive flight

**Observation Changes:**
- Add: opponent gate progress (1 float per opponent), opponent relative position (3 floats per opponent)
- New total: ~30-33 floats

**Code Changes (~30-50 lines):**
- `tasks.h`: Modify RACE to track per-drone gate progress
- `drone.h`: Add opponent state to observations
- `drone.h`: Replace distance-shaping reward with sparse gate-passing rewards
- Optional: add collision detection between racing drones

**Reward Design:**
```
Sparse: +10 for passing gate while in lead position
        +5 for passing gate while trailing
        +50 for completing all gates (lap)
        -20 for crashing into opponent or wall
```

**Training Strategy:**
- Option A: Shared policy (all racers use same policy) — simplest
- Option B: Self-play with frozen opponent pool (SPIRAL approach) — more robust

**Expected Emergent Behaviors (from UPenn):**
- Blocking (flying defensive lines through gates)
- Overtaking (aggressive inside-line maneuvers)
- Collision avoidance emerges WITHOUT explicit reward
- Agents fly 9.9 m/s against competitors vs 8.7 m/s against crashed drones
- Better sim-to-real transfer than single-agent training

**Eval Metrics:**
- Lap time
- Gates passed per episode
- Win rate (head-to-head)
- Average speed
- Collision rate

---

### Phase 3: Cable-Suspended Payload (Bridge to Manipulation)

**Priority:** THIRD — introduces manipulation physics, stepping stone to cooperative transport

**Sources:**
- OmniDrones PayloadTrack: [github.com/btx0424/OmniDrones](https://github.com/btx0424/OmniDrones)
- Sreenath/Lee/Kumar geometric control: [arxiv.org/abs/1309.6717](https://arxiv.org/abs/1309.6717)
- MATLAB reference: [github.com/wuyou33/Quadrotor_Suspended](https://github.com/wuyou33/Quadrotor_Suspended)

**Design:**
- Rigid massless rod of length L connecting drone CoM to point-mass payload
- Spherical pendulum dynamics (2 swing angles + 2 angular velocities)
- Task: transport payload to target position while minimizing swing
- Natural curriculum: hover with payload → transport → minimize swing → transport through gates

**Physics Model:**
```
Cable direction: q = (sin(phi)*cos(theta), sin(phi)*sin(theta), -cos(phi))
Payload position: p_load = p_quad + L * q
Cable tension: T = m_load * (g*cos(phi) + L*phi_dot^2 + L*theta_dot^2*sin(phi)^2)
Quadrotor force modification: F_quad += T * q  (cable pulls on drone)
Pendulum angular acceleration:
  phi_ddot = -g/L * sin(phi) + theta_dot^2 * sin(phi)*cos(phi) + a_quad_perp / L
  theta_ddot = -2*phi_dot*theta_dot*cos(phi)/sin(phi) + a_quad_tangent / (L*sin(phi))
```

**State Changes:**
- Add to `State` struct: `cable_phi`, `cable_theta`, `cable_phi_dot`, `cable_theta_dot` (4 floats)
- Add to `Params` struct: `cable_length`, `payload_mass` (2 floats)

**Observation Changes:**
- Add: payload position relative to target (3), cable angles (2), cable angular velocity (2)
- New total: 23 + 7 = 30 floats (or 37 if building on Phase 1's obs)

**Code Changes (~60-80 lines):**
- `dronelib.h`: Add cable state to `State`, add params, modify `compute_derivatives()` to include pendulum dynamics and cable tension force
- `tasks.h`: Add `TRANSPORT` task — fly payload to target with swing penalty
- `drone.h`: Add payload obs to `compute_drone_observations()`
- `drone.h`: Add swing penalty to reward in `c_step()`

**Reward Design:**
```
reward = alpha_dist * (prev_payload_dist - curr_payload_dist)  # payload-to-target distance
       + alpha_hover * payload_hover_potential                  # payload proximity + stability
       - alpha_swing * (phi^2 + theta^2)                       # swing angle penalty
       - alpha_swing_vel * (phi_dot^2 + theta_dot^2)           # swing velocity penalty
       - alpha_omega * angular_velocity                         # drone stability
```

**Curriculum Progression:**
1. Hover with payload (stationary target, learn to compensate for extra mass)
2. Transport payload (move to target, accept some swing)
3. Smooth transport (minimize swing during transport)
4. Transport through gates (combine RACE + payload)

**Eval Metrics:**
- Payload distance to target
- RMS swing angle
- Transport time
- Payload stability (EMA of swing)

---

### Phase 4: Cooperative Transport (Multi-Agent + Manipulation)

**Priority:** FOURTH — combines multi-agent coordination with physical coupling

**Sources:**
- OmniDrones TransportTrack/TransportFlyThrough: [github.com/btx0424/OmniDrones](https://github.com/btx0424/OmniDrones)
- Cooperative payload transport with MADDPG (Robotics and Autonomous Systems, 2023)

**Design:**
- 2-4 drones connected to a shared rigid platform/box via rigid links
- Platform has its own rigid body state (position, orientation, angular velocity)
- Drones must coordinate thrust to transport platform to target
- Coupling: each drone's thrust affects the shared platform → affects all other drones

**Physics Model:**
```
Platform state: position, velocity, quaternion, angular_velocity
Connection: rigid links from drone[i] to attachment_point[i] on platform
Force balance: platform accel = sum(drone_forces[i]) + gravity
Torque balance: platform ang_accel = sum(cross(r[i], F[i])) / I_platform
Constraint: drone[i] position = platform_pos + rotate(attachment_offset[i], platform_quat)
```

**Observation Changes (per drone):**
- Platform state relative to target (6: position + orientation error)
- Platform velocity (3)
- Other drones' relative positions (3 * (N-1))
- Role: attachment point index

**Code Changes (~100-150 lines):**
- `dronelib.h`: Add `Platform` struct with rigid body state
- `dronelib.h`: Add constraint force computation between drones and platform
- `tasks.h`: Add `COOP_TRANSPORT` task
- `drone.h`: Multi-agent reward based on shared platform position

**Reward Design:**
```
Team reward (shared by all transport drones):
  alpha_platform_dist * (prev_platform_dist - curr_platform_dist)
  + alpha_platform_stable * platform_stability
  - alpha_tilt * platform_tilt_angle
  - alpha_individual_effort * sum(motor_effort)
```

**Training:**
- Shared policy — all transport drones use same policy, differentiated by attachment position
- Can start with 2 drones (simplest) then scale to 4

**Eval Metrics:**
- Platform position error
- Platform tilt angle
- Transport time
- Coordination score (thrust balance across drones)

---

### Phase 5: Active Grasping (Full Aerial Manipulation)

**Priority:** FIFTH — full manipulation capability, builds on all prior phases

**Sources:**
- Swooper (2026): [arxiv.org/abs/2603.05935](https://arxiv.org/abs/2603.05935)
- CatchIt aerial manipulator: [github.com/skywoodsz/CatchIt](https://github.com/skywoodsz/CatchIt)
- SoAG soft gripper drone: [github.com/UCR-Robotics/SoAG](https://github.com/UCR-Robotics/SoAG)

**Design:**
- Add 1 continuous action: gripper width (normalized 0-1, from closed to open)
- Object spawned at random position in arena
- Drone must: fly to object → position above → close gripper → transport to target → release
- Proximity-based grasp: if gripper_width < threshold AND distance_to_object < grasp_radius → attach
- Once attached: object moves with drone (added mass), flight dynamics change

**Action Space Change:**
- Current: 4 (motor RPMs)
- New: 5 (4 motor RPMs + 1 gripper width)
- Requires recompile (NUM_ATNS changes from 4 to 5)

**Observation Changes:**
- Add: object position relative to drone (3), object velocity (3), gripper state (1), is_grasped flag (1)
- New total: 23 + 8 = 31 floats (or more if building on prior phases)

**Physics:**
- Gripper modeled as 1D continuous variable (width)
- Grasp detection: proximity + gripper width threshold
- Once grasped: object position locked to drone with offset, drone mass increases by object mass
- On release: object enters free-fall from current position

**Code Changes (~80-120 lines):**
- `dronelib.h`: Add `Object` struct (position, velocity, mass, is_grasped)
- `dronelib.h`: Add gripper state, grasp detection, mass modification on grasp
- `tasks.h`: Add `GRASP` task — pick up object, transport to target, release
- `binding.c`: Update NUM_ATNS to 5, update ACT_SIZES
- `drone.h`: Add object obs, grasp reward terms

**Reward Design (from Swooper, 2-stage):**
```
Stage 1 (approach):
  alpha_approach * (prev_dist_to_object - curr_dist_to_object)
  + alpha_align * alignment_score  # drone above object

Stage 2 (transport, after grasp):
  alpha_transport * (prev_dist_to_target - curr_dist_to_target)
  + alpha_grasp * grasp_success    # sparse bonus on grasp
  + alpha_deliver * delivery_success  # sparse bonus on delivery
  - alpha_drop * drop_penalty
```

**Training (from Swooper):**
- Two-stage: pre-train flight (Phase 1-3 policies), then fine-tune with grasp
- Or curriculum: start with object near drone, gradually increase approach distance
- Expected training: <60 min on RTX 4090 (Swooper benchmark)

**Eval Metrics:**
- Grasp success rate
- Transport success rate (grasp + deliver)
- Time to grasp
- Time to deliver
- Drop rate

---

## Extension Points (Beyond Phase 5)

### Additional Tasks (Low Effort, After Core Phases)
- **SURROUND** (from CrazyRL): Drones cooperatively encircle a target point
- **ESCORT** (from CrazyRL): Formation around a moving target
- **BALLOON** (from AirGym): High-speed dash to distant target
- **AVOID** (from AirGym): Dodge thrown objects while hovering
- **LANDING** (from TornadoDrone): Precision landing on moving platform
- **INV_PENDULUM** (from OmniDrones): Balance inverted pendulum while hovering

### Environment Enhancements
- **Wind/turbulence fields:** Spatially varying wind forces (moderate C change)
- **Static obstacles:** Pillars/walls for slalom navigation (add to arena)
- **Terrain:** Ground heightmap with collision (major rendering change)
- **Vision observations:** Depth buffer from Raylib camera (major obs change)

### Multi-Agent Enhancements
- **Self-play training:** Periodic checkpoint freezing for opponent pool
- **Heterogeneous policies:** Separate networks for different roles
- **Communication channels:** Learned inter-agent messaging
- **Drone volleyball** (from VolleyBots): Ball physics + cooperative-competitive play

---

## Technical Notes

### Adding a New Task (Checklist)
1. Add enum entry to `DroneTask` in `tasks.h`
2. Add string name to `TASK_NAMES` array in `tasks.h`
3. Implement `set_target_TASKNAME()` function in `tasks.h`
4. Add case branch in `set_target()` dispatcher in `tasks.h`
5. (Optional) Add per-task reset logic in `c_reset()` in `drone.h`
6. (If obs changes) Update `OBS_SIZE` in `binding.c`, rebuild
7. (If action changes) Update `NUM_ATNS` and `ACT_SIZES` in `binding.c`, rebuild
8. Add task-specific reward terms in `c_step()` in `drone.h`
9. Add task-specific metrics in `my_log()` in `binding.c`

### Build Commands
```bash
# Standalone viewer (macOS)
CC=clang ./build.sh drone --fast

# CPU training backend (macOS)
CC=clang CXX=clang++ ./build.sh drone --cpu

# GPU training backend (RunPod)
./build.sh drone

# Float32 for PyTorch backend
./build.sh drone --float
```

### Training Commands
```bash
# Train (change task in config/drone.ini)
puffer train drone

# Eval with trained checkpoint
puffer eval drone --load-model-path ./checkpoint.bin
```

### Config Location
- Default: `config/default.ini`
- Drone-specific: `config/drone.ini`
- Key setting: `task = N` (0=IDLE, 1=HOVER, ..., 7=RACE, 8+=new tasks)

---

## References

### GitHub Repositories
- gym-pybullet-drones: https://github.com/utiasDSL/gym-pybullet-drones
- CrazyRL: https://github.com/ffelten/CrazyRL
- OmniDrones: https://github.com/btx0424/OmniDrones
- Flightmare: https://github.com/uzh-rpg/flightmare
- AirGym: https://github.com/emNavi/AirGym
- Aerial Gym: https://github.com/ntnu-arl/aerial_gym_simulator
- quad-swarm-rl: https://github.com/Zhehui-Huang/quad-swarm-rl
- learning-to-fly: https://github.com/arplaboratory/learning-to-fly
- lsy_drone_racing: https://github.com/utiasDSL/lsy_drone_racing
- safe-control-gym: https://github.com/utiasDSL/safe-control-gym
- VolleyBots: https://github.com/thu-uav/VolleyBots
- Multi-UAV pursuit-evasion: https://github.com/thu-uav/Multi-UAV-pursuit-evasion
- AgileFlight MultiAgent: https://github.com/Jirl-upenn/AgileFlight_MultiAgent
- Quadrotor Suspended: https://github.com/wuyou33/Quadrotor_Suspended
- OpenAI Multi-Agent Emergence: https://github.com/openai/multi-agent-emergence-environments
- Game of Drones: https://github.com/microsoft/AirSim-NeurIPS2019-Drone-Racing
- MATE: https://github.com/XuehaiPan/mate
- JaxMARL: https://github.com/FLAIROx/JaxMARL

### Papers
- Swift (Nature 2023): Champion-level drone racing via deep RL
- Swooper (2026): High-speed aerial grasping, arxiv.org/abs/2603.05935
- Sreenath/Lee/Kumar (2013): Geometric control of quadrotor with suspended load, arxiv.org/abs/1309.6717
- OmniDrones (ICLR 2024): Multi-task drone platform, arxiv.org/abs/2309.12825
- thu-uav Pursuit-Evasion (Nature Sci Reports 2025): Multi-UAV pursuit with MAPPO
- UPenn Agile Flight (2024): Emergent agile flight from competitive racing
- VolleyBots (NeurIPS 2025): Drone volleyball testbed
- Non-Prehensile Aerial Manipulation (2024): arxiv.org/abs/2407.00889
- OpenAI Hide and Seek (ICLR 2020): Emergent tool use from self-play
