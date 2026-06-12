// G1 standalone viewer + checkpoint evaluator (and Phase-4 WASM seed).
//
// Usage:
//   ./g1                      zero-action policy, 1 robot
//   ./g1 weights.bin          run a trained checkpoint, 1 robot
//   ./g1 weights.bin 9        N robots in a grid (1..16), same policy, each
//                             with its own episode + commands ("population view")
//   ./g1 - 9                  9 robots, zero policy
//
// Checkpoints come from training (Modal volume):
//   modal volume get ultra-g1-checkpoints <run>/<step>.bin weights.bin
//
// The policy is PufferNet (pure C) running the EXACT native-trainer
// architecture: Linear(96->128) -> 3x MinGRU(128) -> Linear(128->30, value
// fused) + logstd — weight file is the trainer's raw fp32 master_weights dump.
// Eval is deterministic (Gaussian mean). MinGRU state is zeroed per robot on
// its episode reset.
//
// Controls: arrows = vx/yaw command (robot 0), A/D = vy, Z = zero command,
//           SPACE = shove robot 0, R = reset all, C = collision overlay.
// Self-check: G1_VIEW_FRAMES=N renders N frames, saves g1_view.png, exits.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "raylib.h"
#include "raymath.h"
#include "rlgl.h"

#define G1_HAS_RENDER
#include "g1.h"
#include "puffernet.h"

#define MAX_ROBOTS 16

typedef struct {
    G1 env;
    float rew, term;
    long resets;
    long ticks_alive_accum;
} Robot;

// ---------------------------------------------------------------------------
// MuJoCo meshes -> raylib Models (see previous revision for details)
// ---------------------------------------------------------------------------
static Model* g_models = NULL;
static int g_show_collision = 0;

static Matrix mj_to_rl_matrix(const mjtNum* xmat, const mjtNum* xpos, Vector3 off) {
    Matrix r = {
        (float)xmat[0], (float)xmat[1], (float)xmat[2], (float)xpos[0] + off.x,
        (float)xmat[3], (float)xmat[4], (float)xmat[5], (float)xpos[1] + off.y,
        (float)xmat[6], (float)xmat[7], (float)xmat[8], (float)xpos[2] + off.z,
        0.0f, 0.0f, 0.0f, 1.0f,
    };
    return r;
}

static void build_meshes(const mjModel* m) {
    if (g_models != NULL) return;
    g_models = (Model*)calloc(m->nmesh, sizeof(Model));
    Vector3 L = Vector3Normalize((Vector3){0.35f, 0.25f, 0.90f});
    for (int i = 0; i < m->nmesh; i++) {
        int vadr = m->mesh_vertadr[i], fadr = m->mesh_faceadr[i], nf = m->mesh_facenum[i];
        Mesh mesh = {0};
        mesh.triangleCount = nf;
        mesh.vertexCount = nf * 3;
        mesh.vertices = (float*)RL_MALLOC((size_t)nf * 9 * sizeof(float));
        mesh.normals = (float*)RL_MALLOC((size_t)nf * 9 * sizeof(float));
        mesh.colors = (unsigned char*)RL_MALLOC((size_t)nf * 12);
        for (int f = 0; f < nf; f++) {
            Vector3 v[3];
            for (int k = 0; k < 3; k++) {
                int vi = m->mesh_face[3 * (fadr + f) + k];
                v[k].x = m->mesh_vert[3 * (vadr + vi) + 0];
                v[k].y = m->mesh_vert[3 * (vadr + vi) + 1];
                v[k].z = m->mesh_vert[3 * (vadr + vi) + 2];
            }
            Vector3 n = Vector3Normalize(Vector3CrossProduct(
                Vector3Subtract(v[1], v[0]), Vector3Subtract(v[2], v[0])));
            float lam = Vector3DotProduct(n, L);
            if (lam < 0.0f) lam = 0.0f;
            unsigned char shade = (unsigned char)(255.0f * (0.45f + 0.55f * lam));
            for (int k = 0; k < 3; k++) {
                int o = 9 * f + 3 * k;
                mesh.vertices[o] = v[k].x; mesh.vertices[o + 1] = v[k].y; mesh.vertices[o + 2] = v[k].z;
                mesh.normals[o] = n.x; mesh.normals[o + 1] = n.y; mesh.normals[o + 2] = n.z;
                int c = 12 * f + 4 * k;
                mesh.colors[c] = shade; mesh.colors[c + 1] = shade;
                mesh.colors[c + 2] = shade; mesh.colors[c + 3] = 255;
            }
        }
        UploadMesh(&mesh, false);
        g_models[i] = LoadModelFromMesh(mesh);
    }
}

static Color geom_color(const mjModel* m, int g) {
    int mid = m->geom_matid[g];
    const float* rgba = (mid >= 0) ? (m->mat_rgba + 4 * mid) : (m->geom_rgba + 4 * g);
    return (Color){(unsigned char)(255 * rgba[0]), (unsigned char)(255 * rgba[1]),
                   (unsigned char)(255 * rgba[2]), (unsigned char)(255 * rgba[3])};
}

static void render_robot(G1* env, Vector3 off) {
    mjData* d = env->d;
    const mjModel* m = g1_model;
    for (int g = 0; g < m->ngeom; g++) {
        int type = m->geom_type[g];
        const mjtNum* pos = d->geom_xpos + 3 * g;
        const mjtNum* mat = d->geom_xmat + 9 * g;
        if (type == mjGEOM_MESH) {
            int mid = m->geom_dataid[g];
            g_models[mid].transform = mj_to_rl_matrix(mat, pos, off);
            DrawModel(g_models[mid], (Vector3){0, 0, 0}, 1.0f, geom_color(m, g));
            continue;
        }
        if (type == mjGEOM_PLANE || !g_show_collision) continue;
        const mjtNum* size = m->geom_size + 3 * g;
        Color col = (Color){80, 160, 255, 120};
        Vector3 p = {(float)pos[0] + off.x, (float)pos[1] + off.y, (float)pos[2] + off.z};
        if (type == mjGEOM_SPHERE) {
            DrawSphereWires(p, (float)size[0], 8, 8, col);
        } else if (type == mjGEOM_CAPSULE || type == mjGEOM_CYLINDER) {
            Vector3 ax = {(float)mat[2], (float)mat[5], (float)mat[8]};
            float hl = (float)size[1];
            Vector3 a = {p.x - ax.x * hl, p.y - ax.y * hl, p.z - ax.z * hl};
            Vector3 b = {p.x + ax.x * hl, p.y + ax.y * hl, p.z + ax.z * hl};
            DrawCapsuleWires(a, b, (float)size[0], 8, 4, col);
        } else if (type == mjGEOM_BOX) {
            rlPushMatrix();
            Matrix tr = mj_to_rl_matrix(mat, pos, off);
            rlMultMatrixf(MatrixToFloatV(tr).v);
            DrawCubeWires((Vector3){0, 0, 0}, 2.0f * (float)size[0],
                          2.0f * (float)size[1], 2.0f * (float)size[2], col);
            rlPopMatrix();
        }
    }
}

static Vector3 robot_offset(int i) {
    int k = (i < 4) ? 2 : (i < 9 ? 3 : 4);  // grid side for up to 16
    return (Vector3){(float)(i % k) * 2.2f, (float)(i / k) * 2.2f, 0.0f};
}

static void zero_rnn_state(PufferNet* net, int robot) {
    if (net == NULL) return;
    MinGRU* mg = net->mingru;
    for (int l = 0; l < mg->num_layers; l++)
        memset(mg->state + ((size_t)l * mg->batch_size + robot) * mg->hidden_size,
               0, (size_t)mg->hidden_size * sizeof(float));
}

int main(int argc, char** argv) {
    // --- args: [weights.bin | -] [num_robots] ---
    const char* wpath = (argc > 1 && strcmp(argv[1], "-") != 0) ? argv[1] : NULL;
    int n = (argc > 2) ? atoi(argv[2]) : 1;
    if (n < 1) n = 1;
    if (n > MAX_ROBOTS) n = MAX_ROBOTS;

    PufferNet* net = NULL;
    if (wpath != NULL) {
        Weights* w = load_weights(wpath);
        if (w == NULL) { fprintf(stderr, "failed to load %s\n", wpath); return 1; }
        int logit_sizes[G1_NUM_JOINTS];
        for (int j = 0; j < G1_NUM_JOINTS; j++) logit_sizes[j] = 1;
        net = make_puffernet(w, n, G1_OBS_SIZE, 128, 3, logit_sizes, G1_NUM_JOINTS);
        printf("loaded policy %s (%d weights) for %d robot(s)\n", wpath, w->size, n);
    }

    // --- batched IO buffers (env slices point into them, vecenv-style) ---
    static Robot robots[MAX_ROBOTS];
    float* obs_all = (float*)calloc((size_t)n * G1_OBS_SIZE, sizeof(float));
    float* act_all = (float*)calloc((size_t)n * G1_NUM_JOINTS, sizeof(float));
    for (int i = 0; i < n; i++) {
        Robot* r = &robots[i];
        memset(r, 0, sizeof(*r));
        r->env.observations = obs_all + (size_t)i * G1_OBS_SIZE;
        r->env.actions = act_all + (size_t)i * G1_NUM_JOINTS;
        r->env.rewards = &r->rew;
        r->env.terminals = &r->term;
        r->env.rng = (unsigned int)(i + 1);
        g1_set_default_config(&r->env);
        // checkpoints are tied to the action_scale they trained with
        // (task v1.2 = 0.25); override via G1_ACTION_SCALE
        const char* as = getenv("G1_ACTION_SCALE");
        r->env.action_scale = as ? (float)atof(as) : 0.25f;
        g1_init(&r->env);
        c_reset(&r->env);
    }
    // robot 0 is user-driven: no auto command resample, no timeout
    robots[0].env.cmd_resample_interval = 0;
    robots[0].env.max_episode_len = 1 << 30;

    const char* frames_env = getenv("G1_VIEW_FRAMES");
    int auto_frames = frames_env ? atoi(frames_env) : 0;

    InitWindow(1280, 720, "G1 — mujoco-ultra-fast (checkpoint viewer)");
    SetTargetFPS(50);

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
        if (IsKeyPressed(KEY_SPACE)) { robots[0].env.d->qvel[0] += 1.0; robots[0].env.d->qvel[1] += 0.5; }
        if (IsKeyPressed(KEY_C)) g_show_collision = !g_show_collision;
        if (IsKeyPressed(KEY_R))
            for (int i = 0; i < n; i++) { c_reset(&robots[i].env); zero_rnn_state(net, i); }
        robots[0].env.cmd[0] = vcmd[0];
        robots[0].env.cmd[1] = vcmd[1];
        robots[0].env.cmd[2] = vcmd[2];

        // --- policy: batched PufferNet forward (deterministic mean), or zero ---
        if (net != NULL) forward_puffernet(net, obs_all, act_all);
        else memset(act_all, 0, (size_t)n * G1_NUM_JOINTS * sizeof(float));

        long alive = 0;
        for (int i = 0; i < n; i++) {
            c_step(&robots[i].env);
            if (robots[i].term > 0.5f) { robots[i].resets++; zero_rnn_state(net, i); }
            else alive++;
            robots[i].ticks_alive_accum += 1;
        }

        // --- render ---
        build_meshes(g1_model);
        mjData* d0 = robots[0].env.d;
        Vector3 center = {(float)d0->qpos[0], (float)d0->qpos[1], 0.0f};
        if (n > 1) { Vector3 o = robot_offset(n - 1); center.x += o.x * 0.5f; center.y += o.y * 0.5f; }
        float dist = 1.9f + 0.6f * (float)(n > 1 ? 3 : 0);
        Camera3D cam = {0};
        cam.position = (Vector3){center.x - dist, center.y - dist, 1.35f + 0.35f * (n > 1 ? 3 : 0)};
        cam.target = (Vector3){center.x, center.y, 0.72f};
        cam.up = (Vector3){0, 0, 1};
        cam.fovy = 42.0f;
        cam.projection = CAMERA_PERSPECTIVE;

        BeginDrawing();
        ClearBackground((Color){235, 238, 242, 255});
        BeginMode3D(cam);
        DrawCube((Vector3){center.x, center.y, -0.012f}, 80, 80, 0.02f, (Color){250, 250, 251, 255});
        for (int i = -25; i <= 25; i++) {
            Color gl = (i % 5 == 0) ? (Color){190, 194, 200, 255} : (Color){214, 218, 224, 255};
            DrawLine3D((Vector3){(float)i, -25, 0.001f}, (Vector3){(float)i, 25, 0.001f}, gl);
            DrawLine3D((Vector3){-25, (float)i, 0.001f}, (Vector3){25, (float)i, 0.001f}, gl);
        }
        for (int i = 0; i < n; i++) render_robot(&robots[i].env, robot_offset(i));
        if (vcmd[0] != 0 || vcmd[1] != 0) {
            mjData* d = robots[0].env.d;
            Vector3 p = {(float)d->qpos[0], (float)d->qpos[1], 1.25f};
            Vector3 c = {p.x + vcmd[0], p.y + vcmd[1], 1.25f};
            DrawLine3D(p, c, (Color){255, 160, 30, 255});
            DrawSphere(c, 0.03f, (Color){255, 160, 30, 255});
        }
        EndMode3D();

        long total_resets = 0;
        for (int i = 0; i < n; i++) total_resets += robots[i].resets;
        Color ink = (Color){40, 44, 52, 255};
        DrawText(TextFormat("policy: %s   robots: %d   alive: %ld/%d   resets: %ld",
                            wpath ? wpath : "ZERO (untrained)", n, alive, n, total_resets),
                 12, 12, 20, ink);
        DrawText(TextFormat("cmd[0]  vx %+.2f  vy %+.2f  wyaw %+.2f    pelvis z %.2f   tick %d",
                            vcmd[0], vcmd[1], vcmd[2], (float)d0->qpos[2], robots[0].env.tick),
                 12, 38, 20, ink);
        DrawText("arrows vx/yaw  A/D vy  Z zero  SPACE shove  R reset  C collision",
                 12, 64, 16, (Color){120, 126, 136, 255});
        EndDrawing();

        frame++;
        if (auto_frames > 0 && frame >= auto_frames) { TakeScreenshot("g1_view.png"); break; }
    }
    CloseWindow();
    for (int i = 0; i < n; i++) c_close(&robots[i].env);
    return 0;
}
