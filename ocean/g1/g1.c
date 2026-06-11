// G1 standalone raylib viewer (and Phase-4 WASM seed).
//
// Renders the collision primitives from mjData (the visual meshes are skipped
// in v1 — the robot appears as its capsule/sphere/box collision skeleton).
// Camera follows the pelvis; mujoco is z-up so camera.up = +z.
//
// Controls: arrows = vx / yaw command, A/D = vy, Z = zero command,
//           SPACE = shove the robot, R = reset.
// Policy: zero actions (in-model servos hold the home pose) until trained
// PufferNet weights exist (D2).
//
// Self-check mode: G1_VIEW_FRAMES=N runs N frames headlessly-ish, saves
// g1_view.png via TakeScreenshot, and exits — used to verify rendering.

#include <stdio.h>
#include <stdlib.h>

#include "raylib.h"
#include "raymath.h"
#include "rlgl.h"

#define G1_HAS_RENDER
#include "g1.h"

static Matrix mj_to_rl_matrix(const mjtNum* xmat, const mjtNum* xpos) {
    Matrix r = {
        (float)xmat[0], (float)xmat[1], (float)xmat[2], (float)xpos[0],
        (float)xmat[3], (float)xmat[4], (float)xmat[5], (float)xpos[1],
        (float)xmat[6], (float)xmat[7], (float)xmat[8], (float)xpos[2],
        0.0f, 0.0f, 0.0f, 1.0f,
    };
    return r;
}

void c_render(G1* env) {
    mjData* d = env->d;
    const mjModel* m = g1_model;

    Camera3D cam = {0};
    cam.position = (Vector3){(float)d->qpos[0] - 2.2f, (float)d->qpos[1] - 2.2f, 1.4f};
    cam.target = (Vector3){(float)d->qpos[0], (float)d->qpos[1], 0.7f};
    cam.up = (Vector3){0.0f, 0.0f, 1.0f};
    cam.fovy = 45.0f;
    cam.projection = CAMERA_PERSPECTIVE;

    BeginDrawing();
    ClearBackground((Color){24, 24, 28, 255});
    BeginMode3D(cam);

    // floor: thin slab + grid lines (mujoco is z-up; raylib DrawGrid is y-up)
    DrawCube((Vector3){(float)d->qpos[0], (float)d->qpos[1], -0.012f}, 40, 40, 0.02f,
             (Color){56, 60, 66, 255});
    for (int i = -20; i <= 20; i++) {
        DrawLine3D((Vector3){(float)i, -20, 0.001f}, (Vector3){(float)i, 20, 0.001f},
                   (Color){80, 84, 92, 255});
        DrawLine3D((Vector3){-20, (float)i, 0.001f}, (Vector3){20, (float)i, 0.001f},
                   (Color){80, 84, 92, 255});
    }

    for (int g = 0; g < m->ngeom; g++) {
        int type = m->geom_type[g];
        if (type == mjGEOM_PLANE || type == mjGEOM_MESH) continue;  // floor drawn above; meshes skipped v1
        const mjtNum* size = m->geom_size + 3 * g;
        const mjtNum* pos = d->geom_xpos + 3 * g;
        const mjtNum* mat = d->geom_xmat + 9 * g;
        Color col = (Color){200, 205, 215, 255};
        const char* name = mj_id2name(m, mjOBJ_GEOM, g);
        if (name && (strstr(name, "foot") != NULL)) col = (Color){90, 170, 255, 255};

        if (type == mjGEOM_SPHERE) {
            DrawSphere((Vector3){(float)pos[0], (float)pos[1], (float)pos[2]},
                       (float)size[0], col);
        } else if (type == mjGEOM_CAPSULE || type == mjGEOM_CYLINDER) {
            // local z axis = third column of xmat (row-major)
            Vector3 axis = {(float)mat[2], (float)mat[5], (float)mat[8]};
            float hl = (float)size[1];
            Vector3 a = {(float)pos[0] - axis.x * hl, (float)pos[1] - axis.y * hl,
                         (float)pos[2] - axis.z * hl};
            Vector3 b = {(float)pos[0] + axis.x * hl, (float)pos[1] + axis.y * hl,
                         (float)pos[2] + axis.z * hl};
            if (type == mjGEOM_CAPSULE) DrawCapsule(a, b, (float)size[0], 10, 4, col);
            else DrawCylinderEx(a, b, (float)size[0], (float)size[0], 12, col);
        } else if (type == mjGEOM_BOX || type == mjGEOM_ELLIPSOID) {
            rlPushMatrix();
            Matrix tr = mj_to_rl_matrix(mat, pos);
            rlMultMatrixf(MatrixToFloatV(tr).v);
            if (type == mjGEOM_BOX) {
                DrawCube((Vector3){0, 0, 0}, 2.0f * (float)size[0],
                         2.0f * (float)size[1], 2.0f * (float)size[2], col);
            } else {
                DrawSphere((Vector3){0, 0, 0}, (float)size[0], col);  // approx
            }
            rlPopMatrix();
        }
    }

    // command arrow (world frame, from pelvis)
    Vector3 p = {(float)d->qpos[0], (float)d->qpos[1], 1.1f};
    Vector3 c = {p.x + env->cmd[0], p.y + env->cmd[1], 1.1f};
    DrawLine3D(p, c, (Color){255, 200, 60, 255});
    DrawSphere(c, 0.03f, (Color){255, 200, 60, 255});

    EndMode3D();

    DrawText(TextFormat("cmd  vx %+.2f  vy %+.2f  wyaw %+.2f", env->cmd[0],
                        env->cmd[1], env->cmd[2]), 12, 12, 20, RAYWHITE);
    DrawText(TextFormat("pelvis z %.2f   tick %d   r %+.4f", (float)d->qpos[2],
                        env->tick, env->rewards[0]), 12, 38, 20, RAYWHITE);
    DrawText("arrows vx/yaw  A/D vy  Z zero  SPACE shove  R reset", 12, 64, 16, GRAY);
    EndDrawing();
}

int main(void) {
    G1 env = {0};
    float obs[G1_OBS_SIZE], act[G1_NUM_JOINTS] = {0}, rew = 0, term = 0;
    env.observations = obs;
    env.actions = act;
    env.rewards = &rew;
    env.terminals = &term;
    env.rng = 0;
    g1_set_default_config(&env);
    g1_init(&env);
    c_reset(&env);
    env.cmd[0] = env.cmd[1] = env.cmd[2] = 0.0f;  // viewer: user drives commands
    env.cmd_resample_interval = 0;
    env.max_episode_len = 1 << 30;

    const char* frames_env = getenv("G1_VIEW_FRAMES");
    int auto_frames = frames_env ? atoi(frames_env) : 0;

    InitWindow(1280, 720, "G1 — mujoco-ultra-fast (Phase 1 viewer)");
    SetTargetFPS(50);  // one 20 ms control step per frame = real time

    // viewer owns the command (env resets resample env.cmd; we re-apply ours)
    float vcmd[3] = {0, 0, 0};
    int frame = 0;
    while (!WindowShouldClose()) {
        if (IsKeyDown(KEY_UP)) vcmd[0] = 0.8f;
        else if (IsKeyDown(KEY_DOWN)) vcmd[0] = -0.5f;
        if (IsKeyDown(KEY_LEFT)) vcmd[2] = 1.0f;
        else if (IsKeyDown(KEY_RIGHT)) vcmd[2] = -1.0f;
        if (IsKeyDown(KEY_A)) vcmd[1] = 0.4f;
        else if (IsKeyDown(KEY_D)) vcmd[1] = -0.4f;
        if (IsKeyPressed(KEY_Z)) vcmd[0] = vcmd[1] = vcmd[2] = 0.0f;
        if (IsKeyPressed(KEY_SPACE)) { env.d->qvel[0] += 1.0; env.d->qvel[1] += 0.5; }
        if (IsKeyPressed(KEY_R)) c_reset(&env);
        env.cmd[0] = vcmd[0]; env.cmd[1] = vcmd[1]; env.cmd[2] = vcmd[2];

        // policy: zero actions until trained weights exist (servos hold pose;
        // note: the G1's standing pose is only marginally stable — it tips in
        // ~1.2 s without active balance. That's the policy's job to learn.)
        for (int j = 0; j < G1_NUM_JOINTS; j++) act[j] = 0.0f;

        c_step(&env);
        c_render(&env);

        frame++;
        if (auto_frames > 0 && frame >= auto_frames) {
            TakeScreenshot("g1_view.png");
            break;
        }
    }
    CloseWindow();
    c_close(&env);
    return 0;
}
