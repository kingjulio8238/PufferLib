// G1 standalone raylib viewer (and Phase-4 WASM seed).
//
// Renders the FULL visual meshes from the mjModel (built into raylib meshes at
// startup, per-geom material tints, baked directional shading). Collision
// primitives are a debug overlay (key C). Light aesthetic to match the
// reference Unitree look.
//
// Controls: arrows = vx / yaw command, A/D = vy, Z = zero command,
//           SPACE = shove the robot, R = reset, C = collision overlay.
// Policy: zero actions (in-model servos hold the home pose; the stance is
// marginally stable and tips in ~1.2 s — balance is the policy's job) until
// trained PufferNet weights exist (D2).
//
// Self-check mode: G1_VIEW_FRAMES=N runs N frames, saves g1_view.png, exits.

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

// ---------------------------------------------------------------------------
// MuJoCo meshes -> raylib Models (unindexed, flat normals, shading baked into
// grayscale vertex colors so no lighting shader is needed; per-geom material
// color applied as DrawModel tint).
// ---------------------------------------------------------------------------
static Model* g_models = NULL;
static int g_show_collision = 0;

static void build_meshes(const mjModel* m) {
    if (g_models != NULL) return;
    g_models = (Model*)calloc(m->nmesh, sizeof(Model));
    Vector3 L = Vector3Normalize((Vector3){0.35f, 0.25f, 0.90f});

    for (int i = 0; i < m->nmesh; i++) {
        int vadr = m->mesh_vertadr[i];
        int fadr = m->mesh_faceadr[i];
        int nf = m->mesh_facenum[i];

        Mesh mesh = {0};
        mesh.triangleCount = nf;
        mesh.vertexCount = nf * 3;
        mesh.vertices = (float*)RL_MALLOC((size_t)nf * 9 * sizeof(float));
        mesh.normals = (float*)RL_MALLOC((size_t)nf * 9 * sizeof(float));
        mesh.colors = (unsigned char*)RL_MALLOC((size_t)nf * 12);

        for (int f = 0; f < nf; f++) {
            Vector3 v[3];
            for (int k = 0; k < 3; k++) {
                int vi = m->mesh_face[3 * (fadr + f) + k];  // local index
                v[k].x = m->mesh_vert[3 * (vadr + vi) + 0];
                v[k].y = m->mesh_vert[3 * (vadr + vi) + 1];
                v[k].z = m->mesh_vert[3 * (vadr + vi) + 2];
            }
            Vector3 n = Vector3Normalize(Vector3CrossProduct(
                Vector3Subtract(v[1], v[0]), Vector3Subtract(v[2], v[0])));
            float lambert = Vector3DotProduct(n, L);
            if (lambert < 0.0f) lambert = 0.0f;
            unsigned char shade = (unsigned char)(255.0f * (0.45f + 0.55f * lambert));
            for (int k = 0; k < 3; k++) {
                int o = 9 * f + 3 * k;
                mesh.vertices[o + 0] = v[k].x;
                mesh.vertices[o + 1] = v[k].y;
                mesh.vertices[o + 2] = v[k].z;
                mesh.normals[o + 0] = n.x;
                mesh.normals[o + 1] = n.y;
                mesh.normals[o + 2] = n.z;
                int c = 12 * f + 4 * k;
                mesh.colors[c + 0] = shade;
                mesh.colors[c + 1] = shade;
                mesh.colors[c + 2] = shade;
                mesh.colors[c + 3] = 255;
            }
        }
        UploadMesh(&mesh, false);
        g_models[i] = LoadModelFromMesh(mesh);
    }
}

static Color geom_color(const mjModel* m, int g) {
    const float* rgba;
    int mid = m->geom_matid[g];
    rgba = (mid >= 0) ? (m->mat_rgba + 4 * mid) : (m->geom_rgba + 4 * g);
    return (Color){(unsigned char)(255 * rgba[0]), (unsigned char)(255 * rgba[1]),
                   (unsigned char)(255 * rgba[2]), (unsigned char)(255 * rgba[3])};
}

void c_render(G1* env) {
    mjData* d = env->d;
    const mjModel* m = g1_model;
    build_meshes(m);

    Camera3D cam = {0};
    cam.position = (Vector3){(float)d->qpos[0] - 1.9f, (float)d->qpos[1] - 1.9f, 1.35f};
    cam.target = (Vector3){(float)d->qpos[0], (float)d->qpos[1], 0.72f};
    cam.up = (Vector3){0.0f, 0.0f, 1.0f};
    cam.fovy = 42.0f;
    cam.projection = CAMERA_PERSPECTIVE;

    BeginDrawing();
    ClearBackground((Color){235, 238, 242, 255});
    BeginMode3D(cam);

    // light floor + grid (mujoco is z-up)
    DrawCube((Vector3){(float)d->qpos[0], (float)d->qpos[1], -0.012f}, 60, 60, 0.02f,
             (Color){250, 250, 251, 255});
    for (int i = -25; i <= 25; i++) {
        Color gl = (i % 5 == 0) ? (Color){190, 194, 200, 255} : (Color){214, 218, 224, 255};
        DrawLine3D((Vector3){(float)i, -25, 0.001f}, (Vector3){(float)i, 25, 0.001f}, gl);
        DrawLine3D((Vector3){-25, (float)i, 0.001f}, (Vector3){25, (float)i, 0.001f}, gl);
    }

    for (int g = 0; g < m->ngeom; g++) {
        int type = m->geom_type[g];
        const mjtNum* pos = d->geom_xpos + 3 * g;
        const mjtNum* mat = d->geom_xmat + 9 * g;

        if (type == mjGEOM_MESH) {
            int mid = m->geom_dataid[g];
            g_models[mid].transform = mj_to_rl_matrix(mat, pos);
            DrawModel(g_models[mid], (Vector3){0, 0, 0}, 1.0f, geom_color(m, g));
            continue;
        }
        if (type == mjGEOM_PLANE || !g_show_collision) continue;

        // debug overlay: collision primitives
        const mjtNum* size = m->geom_size + 3 * g;
        Color col = (Color){80, 160, 255, 120};
        if (type == mjGEOM_SPHERE) {
            DrawSphereWires((Vector3){(float)pos[0], (float)pos[1], (float)pos[2]},
                            (float)size[0], 8, 8, col);
        } else if (type == mjGEOM_CAPSULE || type == mjGEOM_CYLINDER) {
            Vector3 axis = {(float)mat[2], (float)mat[5], (float)mat[8]};
            float hl = (float)size[1];
            Vector3 a = {(float)pos[0] - axis.x * hl, (float)pos[1] - axis.y * hl,
                         (float)pos[2] - axis.z * hl};
            Vector3 b = {(float)pos[0] + axis.x * hl, (float)pos[1] + axis.y * hl,
                         (float)pos[2] + axis.z * hl};
            DrawCapsuleWires(a, b, (float)size[0], 8, 4, col);
        } else if (type == mjGEOM_BOX) {
            rlPushMatrix();
            Matrix tr = mj_to_rl_matrix(mat, pos);
            rlMultMatrixf(MatrixToFloatV(tr).v);
            DrawCubeWires((Vector3){0, 0, 0}, 2.0f * (float)size[0],
                          2.0f * (float)size[1], 2.0f * (float)size[2], col);
            rlPopMatrix();
        }
    }

    // command arrow (world frame, above the robot)
    if (env->cmd[0] != 0 || env->cmd[1] != 0) {
        Vector3 p = {(float)d->qpos[0], (float)d->qpos[1], 1.25f};
        Vector3 c = {p.x + env->cmd[0], p.y + env->cmd[1], 1.25f};
        DrawLine3D(p, c, (Color){255, 160, 30, 255});
        DrawSphere(c, 0.03f, (Color){255, 160, 30, 255});
    }

    EndMode3D();

    Color ink = (Color){40, 44, 52, 255};
    DrawText(TextFormat("cmd  vx %+.2f  vy %+.2f  wyaw %+.2f", env->cmd[0],
                        env->cmd[1], env->cmd[2]), 12, 12, 20, ink);
    DrawText(TextFormat("pelvis z %.2f   tick %d   r %+.4f", (float)d->qpos[2],
                        env->tick, env->rewards[0]), 12, 38, 20, ink);
    DrawText("arrows vx/yaw  A/D vy  Z zero  SPACE shove  R reset  C collision",
             12, 64, 16, (Color){120, 126, 136, 255});
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
        if (IsKeyPressed(KEY_C)) g_show_collision = !g_show_collision;
        env.cmd[0] = vcmd[0]; env.cmd[1] = vcmd[1]; env.cmd[2] = vcmd[2];

        // policy: zero actions until trained weights exist
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
