#include <nccl.h>

__global__ void muon_norm_reduce(float* __restrict__ out, const float* __restrict__ partials, int num_blocks) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    sdata[tid] = (tid < num_blocks) ? partials[tid] : 0.0f;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        *out = sdata[0];
    }
}

__global__ void muon_norm_partials(float* __restrict__ partials, const precision_t* __restrict__ src, int n) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    float sum = 0.0f;
    for (int i = blockIdx.x * blockDim.x + tid; i < n; i += blockDim.x * gridDim.x) {
        float v = to_float(src[i]);
        sum += v * v;
    }
    sdata[tid] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        partials[blockIdx.x] = sdata[0];
    }
}

__global__ void muon_norm_apply(precision_t* __restrict__ dst, const float* __restrict__ norm_ptr, float eps, int n) {
    float inv_norm = 1.0f / fmaxf(sqrtf(*norm_ptr), eps);
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = from_float(to_float(dst[idx]) * inv_norm);
    }
}

// Nesterov with f32 momentum accumulator and precision_t gradients
__global__ void muon_nesterov(float* __restrict__ mb, precision_t* __restrict__ gc, float mu, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float m = mu * mb[idx] + to_float(gc[idx]);
        mb[idx] = m;
        gc[idx] = from_float(to_float(gc[idx]) + mu * m);
    }
}

// Fused weight update: wb = wb * (1 - lr*wd) - lr * scale * update
// FUSE_MUON_BF16_CAST: also emit the bf16 param copy here (when bf16_out != null),
// eliminating the separate full-param cast<<<>>> in the train loop (graph audit #3).
// Bit-exact: identical fp32 result, same from_float round.
__global__ void muon_weight_update(float* __restrict__ wb, const precision_t* __restrict__ update,
        const float* __restrict__ lr_ptr, float wd, float scale, int n,
        precision_t* __restrict__ bf16_out = nullptr) {
    float lr = *lr_ptr;
    float wd_scale = 1.0f - lr * wd;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float v = wb[idx] * wd_scale - lr * scale * to_float(update[idx]);
        wb[idx] = v;
        if (bf16_out) bf16_out[idx] = from_float(v);
    }
}

__global__ void muon_clip_norm(precision_t* __restrict__ dst,
        const float* __restrict__ sum_sq_ptr, float max_norm, float eps, int n) {
    float clip_coef = fminf(max_norm / (sqrtf(*sum_sq_ptr) + eps), 1.0f);
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = from_float(to_float(dst[idx]) * clip_coef);
    }
}

#ifdef MUON_BATCHED_NS
#define MAX_NS_BATCH 4   // upper bound on a contiguous same-shape weight-matrix group
#endif

static constexpr double ns_coeffs[5][3] = {
    {4.0848, -6.8946, 2.9270},
    {3.9505, -6.3029, 2.6377},
    {3.7418, -5.5913, 2.3037},
    {2.8769, -3.1427, 1.2046},
    {2.8366, -3.0525, 1.2012},
};

struct Muon {
    double momentum, weight_decay, eps;
    float lr_val_init;
    float* lr_ptr;
    float* lr_derived_ptr;
    float* norm_ptr;
    float* grad_norm_ptr;
    FloatTensor lr_puf, lr_derived_puf, ns_norm_puf, grad_norm_puf;
    FloatTensor mb_puf;
    PrecisionTensor gram, gram_buf, x_buf;
    FloatTensor norm_partials;
    long max_M, max_N;
    Allocator* param_alloc;  // params allocator — shapes used by muon_step
    ncclComm_t nccl_comm;
    int world_size;
};

void muon_init(Muon* m, Allocator* param_alloc, double lr_val,
        double momentum, double eps, double weight_decay,
        Allocator* alloc) {
    m->momentum = momentum;
    m->weight_decay = weight_decay;
    m->eps = eps;
    m->lr_val_init = (float)lr_val;
    m->lr_ptr = nullptr;
    m->lr_derived_ptr = nullptr;
    m->param_alloc = param_alloc;
    m->nccl_comm = nullptr;
    m->world_size = 1;
    m->max_M = 0; m->max_N = 0;
    long n = param_alloc->total_elems;
    m->lr_puf =         {.shape = {1}};
    m->lr_derived_puf = {.shape = {2}};
    m->mb_puf =         {.shape = {n}};
    m->norm_partials =  {.shape = {256}};
    m->grad_norm_puf =  {.shape = {1}};
    alloc_register(alloc, &m->lr_puf);
    alloc_register(alloc, &m->lr_derived_puf);
    alloc_register(alloc, &m->mb_puf);
    alloc_register(alloc, &m->norm_partials);
    alloc_register(alloc, &m->grad_norm_puf);
    long max_M = 0, max_N = 0;
    for (int _i = 0; _i < param_alloc->num_regs; _i++) {
        AllocEntry& e = param_alloc->regs[_i];
        if (ndim(e.shape) >= 2) {
            long R = e.shape[0], C = numel(e.shape) / R;
            max_M = max(max_M, min(R, C));
            max_N = max(max_N, max(R, C));
        }
    }
    if (max_M > 0) {
        m->max_M = max_M; m->max_N = max_N;
#ifdef MUON_BATCHED_NS
        // scratch holds up to MAX_NS_BATCH matrices' Newton-Schulz buffers so the
        // contiguous same-shape group (3 MinGRU mats) can run as one strided batch.
        const long NB = MAX_NS_BATCH;
#else
        const long NB = 1;
#endif
        m->gram =        {.shape = {NB * max_M, max_M}};
        m->gram_buf =    {.shape = {NB * max_M, max_M}};
        m->x_buf =       {.shape = {NB * max_M, max_N}};
        m->ns_norm_puf = {.shape = {1}};
        alloc_register(alloc, &m->gram);
        alloc_register(alloc, &m->gram_buf);
        alloc_register(alloc, &m->x_buf);
        alloc_register(alloc, &m->ns_norm_puf);
    }
}

void muon_post_create(Muon* m) {
    m->lr_ptr = m->lr_puf.data;
    m->lr_derived_ptr = m->lr_derived_puf.data;
    m->grad_norm_ptr = m->grad_norm_puf.data;
    if (m->ns_norm_puf.data) m->norm_ptr = m->ns_norm_puf.data;
    cudaMemcpy(m->lr_ptr, &m->lr_val_init, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(m->lr_derived_ptr, 0, 2 * sizeof(float));
    cudaMemset(m->mb_puf.data, 0, numel(m->mb_puf.shape) * sizeof(float));
}

void muon_step(Muon* m, FloatTensor weights, PrecisionTensor grads, float max_grad_norm,
        cudaStream_t stream = 0, precision_t* bf16_out = nullptr) {
    // Multi-GPU support: simple all-reduce over a contiguous grad buffer
    if (m->nccl_comm != nullptr && m->world_size > 1) {
        ncclAllReduce(grads.data, grads.data, numel(grads.shape),
            NCCL_PRECISION, ncclAvg, m->nccl_comm, stream);
    }

    // Clip gradients by norm
    int clip_blocks = min((int)grid_size(numel(grads.shape)), 256);
    muon_norm_partials<<<clip_blocks, 256, 0, stream>>>(
        m->norm_partials.data, grads.data, numel(grads.shape));
    muon_norm_reduce<<<1, 256, 0, stream>>>(m->grad_norm_ptr, m->norm_partials.data, clip_blocks);
    muon_clip_norm<<<grid_size(numel(grads.shape)), BLOCK_SIZE, 0, stream>>>(
        grads.data, m->grad_norm_ptr, max_grad_norm, 1e-6f, numel(grads.shape));

    // Nesterov momentum
    muon_nesterov<<<grid_size(numel(m->mb_puf.shape)), BLOCK_SIZE, 0, stream>>>(
        m->mb_puf.data, grads.data, (float)m->momentum, numel(m->mb_puf.shape));

    long offset = 0;
    for (int _i = 0; _i < m->param_alloc->num_regs; _i++) {
        AllocEntry& e = m->param_alloc->regs[_i];
        precision_t* gc_ptr = grads.data + offset;
        float* wb_ptr = weights.data + offset;
        long ne = numel(e.shape);
        const precision_t* update_ptr = gc_ptr;
        float scale = 1.0f;

#ifdef MUON_BATCHED_NS
        // Batch the Newton-Schulz of a run of consecutive identical-shape 2D weight
        // matrices (the 3 MinGRU {3H,H} mats, contiguous in the flat buffer) into ONE
        // strided batch -> collapses 3 serial ~28-kernel chains to 1. Algorithmically
        // equivalent (StridedBatched may differ in accum order from per-mat GemmEx ->
        // NOT bit-identical; gait A/B gates it). idiot_learner.md / muon-auditor design.
        if (ndim(e.shape) >= 2) {
            int G = 1;
            while (_i + G < m->param_alloc->num_regs &&
                   ndim(m->param_alloc->regs[_i + G].shape) >= 2 &&
                   m->param_alloc->regs[_i + G].shape[0] == e.shape[0] &&
                   numel(m->param_alloc->regs[_i + G].shape) == ne &&
                   G < MAX_NS_BATCH) G++;
            if (G > 1) {
                long R = e.shape[0], C = ne / R, M = min(R, C), N = max(R, C);
                bool tall = R > C; long RC = R * C, MM = M * M;
                precision_t* xb = m->x_buf.data, *gr = m->gram.data, *grb = m->gram_buf.data;
                for (int g = 0; g < G; g++) {       // per-matrix RMS-normalize (serial, cheap)
                    precision_t* xg = gc_ptr + (long)g * RC;
                    int nb = min((int)grid_size(RC), 256);
                    muon_norm_partials<<<nb, 256, 0, stream>>>(m->norm_partials.data, xg, RC);
                    muon_norm_reduce<<<1, 256, 0, stream>>>(m->norm_ptr, m->norm_partials.data, nb);
                    muon_norm_apply<<<grid_size(RC), BLOCK_SIZE, 0, stream>>>(xg, m->norm_ptr, 1e-7f, RC);
                }
                cublasOperation_t ga = tall ? CUBLAS_OP_T : CUBLAS_OP_N;
                cublasOperation_t gb = tall ? CUBLAS_OP_N : CUBLAS_OP_T;
                for (int i = 0; i < 5; ++i) {
                    precision_t* src = (i % 2 == 0) ? gc_ptr : xb;
                    precision_t* dst = (i % 2 == 0) ? xb : gc_ptr;
                    cublasGemmStridedBatchedExDense(ga, gb, (int)M, (int)M, (int)N,
                        src, RC, src, RC, gr, MM, G, stream);
                    cudaMemcpyAsync(grb, gr, (size_t)G * MM * sizeof(precision_t),
                        cudaMemcpyDeviceToDevice, stream);
                    // serial: puf_addmm_nn(gram, gram, gram_buf, c2, c1) -> out=gram_buf,
                    // a=gram, b=gram (gram_buf holds the c1*beta base via the memcpy above).
                    cublasGemmStridedBatchedExDense(CUBLAS_OP_N, CUBLAS_OP_N, (int)M, (int)M, (int)M,
                        gr, MM, gr, MM, grb, MM, G, stream, ns_coeffs[i][2], ns_coeffs[i][1]);
                    cudaMemcpyAsync(dst, src, (size_t)G * RC * sizeof(precision_t),
                        cudaMemcpyDeviceToDevice, stream);
                    cublasGemmStridedBatchedExDense(CUBLAS_OP_N, CUBLAS_OP_N, (int)R, (int)C, (int)M,
                        tall ? src : grb, tall ? RC : MM, tall ? grb : src, tall ? MM : RC,
                        dst, RC, G, stream, 1.0f, ns_coeffs[i][0]);
                }
                float bscale = sqrtf(fmaxf(1.0f, (float)R / (float)C));  // result in xb (i=4 even)
                muon_weight_update<<<grid_size(G * ne), BLOCK_SIZE, 0, stream>>>(
                    wb_ptr, xb, m->lr_ptr, (float)m->weight_decay, bscale, (int)(G * ne),
                    bf16_out ? bf16_out + offset : nullptr);
                offset += (long)G * ne;
                _i += G - 1;
                continue;
            }
        }
#endif

        // Orthogonalize the update
        // E1 PROBE (-DMUON_SGD_PROBE): skip the whole Newton-Schulz orthogonalization
        // (norm kernels + 5 iters × [2 GEMM + 2 copy + 1 addmm]) -> plain SGD update with
        // the raw grad. Measures Muon's share of the learner FORWARD via the SPS delta
        // (timing probe; weights are garbage, ignore quality). idiot_learner.md E1.
#ifndef MUON_SGD_PROBE
        if (ndim(e.shape) >= 2) {
            long R = e.shape[0], C = ne / R;
            long M = min(R, C), N = max(R, C);
            bool tall = R > C;
            PrecisionTensor x = {.data = gc_ptr, .shape = {R, C}};
            PrecisionTensor x_buf = {.data = m->x_buf.data, .shape = {R, C}};
            PrecisionTensor gram = {.data = m->gram.data, .shape = {M, M}};
            PrecisionTensor gram_buf = {.data = m->gram_buf.data, .shape = {M, M}};

            int nblk = min((int)grid_size(numel(x.shape)), 256);
            muon_norm_partials<<<nblk, 256, 0, stream>>>(
                m->norm_partials.data, x.data, numel(x.shape));
            muon_norm_reduce<<<1, 256, 0, stream>>>(m->norm_ptr, m->norm_partials.data, nblk);
            muon_norm_apply<<<grid_size(numel(x.shape)), BLOCK_SIZE, 0, stream>>>(
                x.data, m->norm_ptr, 1e-7f, numel(x.shape));

            cublasOperation_t gram_op_a = tall ? CUBLAS_OP_T : CUBLAS_OP_N;
            cublasOperation_t gram_op_b = tall ? CUBLAS_OP_N : CUBLAS_OP_T;
            for (int i = 0; i < 5; ++i) {
                PrecisionTensor& src = (i % 2 == 0) ? x : x_buf;
                PrecisionTensor& dst = (i % 2 == 0) ? x_buf : x;
                cublasGemmExDense(gram_op_a, gram_op_b, (int)M, (int)M, (int)N,
                    src.data, src.data, gram.data, stream);
                puf_copy(&gram_buf, &gram, stream);
                puf_addmm_nn(&gram, &gram, &gram_buf, ns_coeffs[i][2], ns_coeffs[i][1], stream);
                puf_copy(&dst, &src, stream);
                cublasGemmExDense(CUBLAS_OP_N, CUBLAS_OP_N, (int)R, (int)C, (int)M,
                    tall ? src.data : gram_buf.data, tall ? gram_buf.data : src.data, dst.data,
                    stream, 1.0f, ns_coeffs[i][0]);
            }

            update_ptr = x_buf.data;
            scale = sqrtf(fmaxf(1.0f, (float)R / (float)C));
        }
#endif  // MUON_SGD_PROBE

        muon_weight_update<<<grid_size(ne), BLOCK_SIZE, 0, stream>>>(
            wb_ptr, update_ptr, m->lr_ptr, (float)m->weight_decay, scale, (int)ne,
            bf16_out ? bf16_out + offset : nullptr);
        offset += ne;
    }
}
