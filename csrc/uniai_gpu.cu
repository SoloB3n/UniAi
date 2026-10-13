/*
 * UniAi — GPU backend (CUDA).
 *
 * The same model, training loop and commands as the CPU engine (driver.h);
 * only the kernels differ.  Every kernel is hand-written: a shared-memory
 * tiled GEMM, block-level reductions for LayerNorm and softmax, and an
 * attention backward split in two passes so no two threads ever write the
 * same memory (no atomics, deterministic).
 *
 *   make gpu        -> ./uniai_gpu  (needs nvcc and an NVIDIA GPU)
 *   make gpu-emu    -> ./uniai_gpu_emu: the same kernels run on the CPU
 *                      through csrc/cuda_emu.h, for testing without a GPU
 */

#ifdef UNIAI_EMULATE
#include "cuda_emu.h"
#define BACKEND_NAME "GPU (emulated on CPU)"
#else
#include <cuda_runtime.h>
#define LAUNCH(kernel, grid, block, ...) kernel<<<(grid), (block)>>>(__VA_ARGS__)
#define BACKEND_NAME "CUDA"
#endif

#include "common.h"

#define CK(x)                                                                    \
    do {                                                                         \
        cudaError_t e_ = (x);                                                    \
        if (e_ != cudaSuccess) {                                                 \
            fprintf(stderr, "CUDA error: %s (%s:%d)\n", cudaGetErrorString(e_),  \
                    __FILE__, __LINE__);                                         \
            exit(1);                                                             \
        }                                                                        \
    } while (0)
#define KCHECK() CK(cudaGetLastError())

#define BS 256 /* threads per block for every kernel */

static unsigned nblocks(size_t n) { return (unsigned)((n + BS - 1) / BS); }

/* ---------------------------------------------------------- reductions
 * Must be called by every thread of the block (they contain barriers). */
__device__ float block_sum(float v) {
    __shared__ float buf[BS];
    __syncthreads(); /* buf may still be read by a previous call */
    buf[threadIdx.x] = v;
    __syncthreads();
    for (int s = BS / 2; s > 0; s >>= 1) {
        if ((int)threadIdx.x < s) buf[threadIdx.x] += buf[threadIdx.x + s];
        __syncthreads();
    }
    float r = buf[0];
    __syncthreads();
    return r;
}

__device__ float block_max(float v) {
    __shared__ float buf[BS];
    __syncthreads();
    buf[threadIdx.x] = v;
    __syncthreads();
    for (int s = BS / 2; s > 0; s >>= 1) {
        if ((int)threadIdx.x < s) buf[threadIdx.x] = fmaxf(buf[threadIdx.x], buf[threadIdx.x + s]);
        __syncthreads();
    }
    float r = buf[0];
    __syncthreads();
    return r;
}

/* ------------------------------------------------------------ elementwise */

__global__ void encoder_forward_k(float *out, const int *inp, const float *wte,
                                  const float *wpe, int T, int C, size_t total) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) {
        size_t n = i / C;
        int c = (int)(i % C), t = (int)(n % T);
        out[i] = wte[(size_t)inp[n] * C + c] + wpe[(size_t)t * C + c];
    }
}

__global__ void encoder_backward_k(float *dwte, float *dwpe, const float *dout,
                                   const int *inp, int T, int C, size_t total) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) {
        size_t n = i / C;
        int c = (int)(i % C), t = (int)(n % T);
        atomicAdd(&dwte[(size_t)inp[n] * C + c], dout[i]);
        atomicAdd(&dwpe[(size_t)t * C + c], dout[i]);
    }
}

__global__ void dropout_k(float *x, size_t n, uint64_t seed, float p, uint32_t site) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= drop_hash(seed, p, site, i);
}

__device__ float tanh_approx(float u) {
    u = fminf(fmaxf(u, -15.0f), 15.0f);
    return 1.0f - 2.0f / (1.0f + expf(2.0f * u));
}

__global__ void gelu_forward_k(float *out, const float *inp, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = inp[i];
        out[i] = 0.5f * x * (1.0f + tanh_approx(GELU_K * (x + 0.044715f * x * x * x)));
    }
}

__global__ void gelu_backward_k(float *dinp, const float *inp, const float *dout, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = inp[i], t = tanh_approx(GELU_K * (x + 0.044715f * x * x * x));
        float dinner = GELU_K * (1.0f + 3.0f * 0.044715f * x * x);
        dinp[i] += dout[i] * (0.5f * (1.0f + t) + 0.5f * x * (1.0f - t * t) * dinner);
    }
}

__global__ void add_k(float *out, const float *a, const float *b, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}

__global__ void accumulate_k(float *dst, const float *src, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] += src[i];
}

__global__ void scale_k(float *x, size_t n, float s) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= s;
}

/* out (N,OC) = bias broadcast over rows, or zero */
__global__ void bias_init_k(float *out, const float *bias, int OC, size_t total) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) out[i] = bias ? bias[i % OC] : 0.0f;
}

/* db[j] += sum_n dout[n, j]  — one thread per column, coalesced across j */
__global__ void colsum_k(float *db, const float *dout, int N, int OC) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < OC) {
        float s = 0.0f;
        for (int n = 0; n < N; n++) s += dout[(size_t)n * OC + j];
        db[j] += s;
    }
}

/* ------------------------------------------------------------- layernorm */

__global__ void layernorm_forward_k(float *out, float *mean, float *rstd, const float *inp,
                                    const float *w, const float *b, int C) {
    size_t n = blockIdx.x;
    const float *x = inp + n * C;
    float s = 0.0f;
    for (int c = threadIdx.x; c < C; c += BS) s += x[c];
    float m = block_sum(s) / C;
    float v = 0.0f;
    for (int c = threadIdx.x; c < C; c += BS) { float d = x[c] - m; v += d * d; }
    float r = 1.0f / sqrtf(block_sum(v) / C + 1e-5f);
    for (int c = threadIdx.x; c < C; c += BS) out[n * C + c] = (x[c] - m) * r * w[c] + b[c];
    if (threadIdx.x == 0) { mean[n] = m; rstd[n] = r; }
}

__global__ void layernorm_backward_k(float *dinp, const float *dout, const float *inp,
                                     const float *w, const float *mean, const float *rstd,
                                     int C) {
    size_t n = blockIdx.x;
    const float *d = dout + n * C, *x = inp + n * C;
    float m = mean[n], s = rstd[n], dm = 0.0f, dn = 0.0f;
    for (int c = threadIdx.x; c < C; c += BS) {
        float g = w[c] * d[c];
        dm += g;
        dn += g * (x[c] - m) * s;
    }
    dm = block_sum(dm) / C;
    dn = block_sum(dn) / C;
    for (int c = threadIdx.x; c < C; c += BS) {
        float xh = (x[c] - m) * s;
        dinp[n * C + c] += (w[c] * d[c] - dm - xh * dn) * s;
    }
}

__global__ void layernorm_param_grad_k(float *dw, float *db, const float *dout,
                                       const float *inp, const float *mean,
                                       const float *rstd, int N, int C) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < C) {
        float gw = 0.0f, gb = 0.0f;
        for (int n = 0; n < N; n++) {
            float d = dout[(size_t)n * C + c];
            gw += d * (inp[(size_t)n * C + c] - mean[n]) * rstd[n];
            gb += d;
        }
        dw[c] += gw;
        db[c] += gb;
    }
}

/* ------------------------------------------------------------------ GEMM
 * Cm (M,N) += op(A) (M,K) @ op(B) (K,N), Cm row-major.  Element (i,k) of
 * op(A) is A[i*sai + k*sak] and (k,j) of op(B) is B[k*sbk + j*sbj], so one
 * kernel covers every transpose combination the model needs.
 * Each 256-thread block computes a 64x64 tile; each thread a 4x4 sub-tile
 * (rows ty+16r, cols tx+16c) accumulated in registers. */
#define TM 64
#define TN 64
#define TK 16

__global__ void gemm_k(int M, int N, int K, const float *A, long long sai, long long sak,
                       const float *B, long long sbk, long long sbj, float *Cm) {
    __shared__ float As[TK][TM];
    __shared__ float Bs[TK][TN];
    int tid = threadIdx.x, tx = tid % 16, ty = tid / 16;
    int row0 = blockIdx.y * TM, col0 = blockIdx.x * TN;
    float acc[4][4];
    for (int r = 0; r < 4; r++)
        for (int c = 0; c < 4; c++) acc[r][c] = 0.0f;

    for (int k0 = 0; k0 < K; k0 += TK) {
        /* stage tiles; consecutive threads read consecutive addresses */
        for (int e = tid; e < TK * TM; e += BS) {
            int kk, ii;
            if (sak == 1) { kk = e % TK; ii = e / TK; } else { ii = e % TM; kk = e / TM; }
            int gi = row0 + ii, gk = k0 + kk;
            As[kk][ii] = (gi < M && gk < K) ? A[gi * sai + gk * sak] : 0.0f;
        }
        for (int e = tid; e < TK * TN; e += BS) {
            int kk, jj;
            if (sbj == 1) { jj = e % TN; kk = e / TN; } else { kk = e % TK; jj = e / TK; }
            int gk = k0 + kk, gj = col0 + jj;
            Bs[kk][jj] = (gk < K && gj < N) ? B[gk * sbk + gj * sbj] : 0.0f;
        }
        __syncthreads();
        for (int kk = 0; kk < TK; kk++) {
            float a[4], b[4];
            for (int r = 0; r < 4; r++) a[r] = As[kk][ty + 16 * r];
            for (int c = 0; c < 4; c++) b[c] = Bs[kk][tx + 16 * c];
            for (int r = 0; r < 4; r++)
                for (int c = 0; c < 4; c++) acc[r][c] += a[r] * b[c];
        }
        __syncthreads();
    }
    for (int r = 0; r < 4; r++)
        for (int c = 0; c < 4; c++) {
            int gi = row0 + ty + 16 * r, gj = col0 + tx + 16 * c;
            if (gi < M && gj < N) Cm[(size_t)gi * N + gj] += acc[r][c];
        }
}

static void gemm(int M, int N, int K, const float *A, long long sai, long long sak,
                 const float *B, long long sbk, long long sbj, float *Cm) {
    dim3 grid((N + TN - 1) / TN, (M + TM - 1) / TM);
    LAUNCH(gemm_k, grid, BS, M, N, K, A, sai, sak, B, sbk, sbj, Cm);
    KCHECK();
}

/* ------------------------------------------------------------- attention */

/* one thread per (b, h, t): scores, softmax, then the weighted sum of V */
__global__ void attention_forward_k(float *out, float *att, const float *qkv, int B, int T,
                                    int C, int H, uint64_t seed, float p, uint32_t site) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < (size_t)B * H * T) {
        int t = (int)(idx % T), h = (int)((idx / T) % H), b = (int)(idx / ((size_t)T * H));
        int D = C / H, C3 = 3 * C;
        float scale = 1.0f / sqrtf((float)D);
        size_t row = (((size_t)b * H + h) * T + t) * T;
        float *a = att + row;
        const float *q = qkv + ((size_t)b * T + t) * C3 + h * D;
        float mx = -1e30f;
        for (int t2 = 0; t2 <= t; t2++) {
            const float *k = qkv + ((size_t)b * T + t2) * C3 + C + h * D;
            float s = 0.0f;
            for (int i = 0; i < D; i++) s += q[i] * k[i];
            s *= scale;
            a[t2] = s;
            mx = fmaxf(mx, s);
        }
        float sum = 0.0f;
        for (int t2 = 0; t2 <= t; t2++) { a[t2] = expf(a[t2] - mx); sum += a[t2]; }
        float inv = 1.0f / sum;
        for (int t2 = 0; t2 < T; t2++) a[t2] = t2 <= t ? a[t2] * inv : 0.0f;
        float *o = out + ((size_t)b * T + t) * C + h * D;
        for (int i = 0; i < D; i++) o[i] = 0.0f;
        for (int t2 = 0; t2 <= t; t2++) {
            const float *v = qkv + ((size_t)b * T + t2) * C3 + 2 * C + h * D;
            float w = a[t2];
            if (p > 0.0f) w *= drop_hash(seed, p, site, row + t2);
            for (int i = 0; i < D; i++) o[i] += w * v[i];
        }
    }
}

/* pass 1, one thread per query row (b, h, t): dpre = softmax backward of
 * dL/datt, and dq for that row */
__global__ void attention_backward_rows_k(float *dqkv, float *dpre, const float *dout,
                                          const float *qkv, const float *att, int B, int T,
                                          int C, int H, uint64_t seed, float p, uint32_t site) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < (size_t)B * H * T) {
        int t = (int)(idx % T), h = (int)((idx / T) % H), b = (int)(idx / ((size_t)T * H));
        int D = C / H, C3 = 3 * C;
        float scale = 1.0f / sqrtf((float)D);
        size_t row = (((size_t)b * H + h) * T + t) * T;
        const float *a = att + row;
        float *dp = dpre + row;
        const float *d = dout + ((size_t)b * T + t) * C + h * D;
        float dot = 0.0f;
        for (int t2 = 0; t2 <= t; t2++) {
            const float *v = qkv + ((size_t)b * T + t2) * C3 + 2 * C + h * D;
            float s = 0.0f;
            for (int i = 0; i < D; i++) s += v[i] * d[i];
            if (p > 0.0f) s *= drop_hash(seed, p, site, row + t2);
            dp[t2] = s;
            dot += s * a[t2];
        }
        float *dq = dqkv + ((size_t)b * T + t) * C3 + h * D;
        for (int t2 = 0; t2 <= t; t2++) {
            float g = a[t2] * (dp[t2] - dot) * scale;
            dp[t2] = g;
            const float *k = qkv + ((size_t)b * T + t2) * C3 + C + h * D;
            for (int i = 0; i < D; i++) dq[i] += g * k[i];
        }
    }
}

/* pass 2, one thread per key/value position (b, h, t2): dk and dv gather
 * over all later queries t >= t2, so each thread owns its outputs */
__global__ void attention_backward_cols_k(float *dqkv, const float *dpre, const float *dout,
                                          const float *qkv, const float *att, int B, int T,
                                          int C, int H, uint64_t seed, float p, uint32_t site) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < (size_t)B * H * T) {
        int t2 = (int)(idx % T), h = (int)((idx / T) % H), b = (int)(idx / ((size_t)T * H));
        int D = C / H, C3 = 3 * C;
        float *dk = dqkv + ((size_t)b * T + t2) * C3 + C + h * D;
        float *dv = dqkv + ((size_t)b * T + t2) * C3 + 2 * C + h * D;
        for (int t = t2; t < T; t++) {
            size_t row = (((size_t)b * H + h) * T + t) * T;
            float g = dpre[row + t2], w = att[row + t2];
            if (p > 0.0f) w *= drop_hash(seed, p, site, row + t2);
            const float *q = qkv + ((size_t)b * T + t) * C3 + h * D;
            const float *d = dout + ((size_t)b * T + t) * C + h * D;
            for (int i = 0; i < D; i++) { dk[i] += g * q[i]; dv[i] += w * d[i]; }
        }
    }
}

/* ------------------------------------------------------------------ head */

/* one block per row: softmax over V in place, and the row's loss */
__global__ void softmax_loss_k(float *probs, const int *targets, float *losses, int V) {
    size_t n = blockIdx.x;
    float *p = probs + n * V;
    float mx = -1e30f;
    for (int v = threadIdx.x; v < V; v += BS) mx = fmaxf(mx, p[v]);
    mx = block_max(mx);
    float s = 0.0f;
    for (int v = threadIdx.x; v < V; v += BS) { float e = expf(p[v] - mx); p[v] = e; s += e; }
    float inv = 1.0f / block_sum(s);
    for (int v = threadIdx.x; v < V; v += BS) p[v] *= inv;
    __syncthreads();
    if (targets && threadIdx.x == 0) losses[n] = -logf(p[targets[n]] + 1e-30f);
}

/* probs -> dlogits = (probs - onehot) / N, in place */
__global__ void dlogits_k(float *probs, const int *targets, int V, float invN, size_t total) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) {
        size_t n = i / V;
        float g = probs[i];
        if ((int)(i % V) == targets[n]) g -= 1.0f;
        probs[i] = g * invN;
    }
}

/* ------------------------------------------------------------ optimizer */

__global__ void sumsq_k(float *partial, const float *g, size_t n) {
    float s = 0.0f;
    for (size_t i = (size_t)blockIdx.x * BS + threadIdx.x; i < n; i += (size_t)gridDim.x * BS)
        s += g[i] * g[i];
    s = block_sum(s);
    if (threadIdx.x == 0) partial[blockIdx.x] = s;
}

__global__ void adamw_k(float *p, const float *g, float *m, float *v, size_t n, float lr,
                        float wd, int decay, float c1, float c2) {
    size_t j = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (j < n) {
        const float b1 = 0.9f, b2 = 0.95f, eps = 1e-8f;
        float gj = g[j];
        m[j] = b1 * m[j] + (1 - b1) * gj;
        v[j] = b2 * v[j] + (1 - b2) * gj * gj;
        float pj = p[j];
        if (decay) pj -= lr * wd * pj;
        p[j] = pj - lr * (m[j] / c1) / (sqrtf(v[j] / c2) + eps);
    }
}

/* ========================================================== backend API */

static void *bk_alloc(size_t bytes) {
    void *p;
    CK(cudaMalloc(&p, bytes ? bytes : 1));
    CK(cudaMemset(p, 0, bytes));
    return p;
}
static void bk_zero(void *p, size_t bytes) { CK(cudaMemset(p, 0, bytes)); }
static void bk_upload(void *dst, const void *src, size_t bytes) {
    CK(cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice));
}
static void bk_download(void *dst, const void *src, size_t bytes) {
    CK(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost));
}

static void bk_describe(char *buf, size_t n) {
#ifdef UNIAI_EMULATE
    snprintf(buf, n, "kernels emulated on CPU (slow, for testing)");
#else
    cudaDeviceProp prop;
    int dev = 0;
    CK(cudaGetDevice(&dev));
    CK(cudaGetDeviceProperties(&prop, dev));
    snprintf(buf, n, "%s, %.1f GB, sm_%d%d", prop.name, prop.totalGlobalMem / 1073741824.0,
             prop.major, prop.minor);
#endif
}

/* device scratch, grown on demand */
static float *dev_scratch(size_t n) {
    static float *buf = NULL;
    static size_t cap = 0;
    if (n > cap) {
        if (buf) CK(cudaFree(buf));
        buf = (float *)bk_alloc(n * sizeof(float));
        cap = n;
    }
    return buf;
}

static float bk_grad_norm_clip(float *g, size_t n, float max_norm) {
    const int G = 1024;
    static float h_part[G];
    float *part = (float *)dev_scratch(G);
    unsigned blocks = nblocks(n) < (unsigned)G ? nblocks(n) : (unsigned)G;
    LAUNCH(sumsq_k, blocks, BS, part, g, n);
    KCHECK();
    bk_download(h_part, part, blocks * sizeof(float));
    double ss = 0.0;
    for (unsigned i = 0; i < blocks; i++) ss += h_part[i];
    float norm = (float)sqrt(ss);
    if (max_norm > 0 && norm > max_norm) {
        LAUNCH(scale_k, nblocks(n), BS, g, n, max_norm / (norm + 1e-6f));
        KCHECK();
    }
    return norm;
}

static void bk_adamw(float *p, const float *g, float *m, float *v, size_t n,
                     float lr, float wd, int decay, float c1, float c2) {
    LAUNCH(adamw_k, nblocks(n), BS, p, g, m, v, n, lr, wd, decay, c1, c2);
    KCHECK();
}

/* --------------------------------------- layer ops called by driver.h */

static void encoder_forward(float *out, const int *inp, const float *wte, const float *wpe,
                            int B, int T, int C) {
    size_t total = (size_t)B * T * C;
    LAUNCH(encoder_forward_k, nblocks(total), BS, out, inp, wte, wpe, T, C, total);
    KCHECK();
}

static void encoder_backward(float *dwte, float *dwpe, const float *dout, const int *inp,
                             int B, int T, int C) {
    size_t total = (size_t)B * T * C;
    LAUNCH(encoder_backward_k, nblocks(total), BS, dwte, dwpe, dout, inp, T, C, total);
    KCHECK();
}

static void dropout_apply(float *x, size_t n, uint32_t site) {
    if (drop_p <= 0.0f) return;
    LAUNCH(dropout_k, nblocks(n), BS, x, n, drop_seed, drop_p, site);
    KCHECK();
}

static void layernorm_forward(float *out, float *mean, float *rstd, const float *inp,
                              const float *w, const float *b, int N, int C) {
    LAUNCH(layernorm_forward_k, N, BS, out, mean, rstd, inp, w, b, C);
    KCHECK();
}

static void layernorm_backward(float *dinp, float *dw, float *db, const float *dout,
                               const float *inp, const float *w, const float *mean,
                               const float *rstd, int N, int C) {
    LAUNCH(layernorm_backward_k, N, BS, dinp, dout, inp, w, mean, rstd, C);
    KCHECK();
    LAUNCH(layernorm_param_grad_k, nblocks(C), BS, dw, db, dout, inp, mean, rstd, N, C);
    KCHECK();
}

/* out (N,OC) = inp (N,C) @ W (C,OC) + bias */
static void matmul_forward(float *out, const float *inp, const float *W, const float *bias,
                           int N, int C, int OC) {
    size_t total = (size_t)N * OC;
    LAUNCH(bias_init_k, nblocks(total), BS, out, bias, OC, total);
    KCHECK();
    gemm(N, OC, C, inp, C, 1, W, OC, 1, out);
}

static void matmul_backward(float *dinp, float *dW, float *db, const float *dout,
                            const float *inp, const float *W, int N, int C, int OC) {
    gemm(N, C, OC, dout, OC, 1, W, 1, OC, dinp);   /* dinp += dout @ W^T */
    gemm(C, OC, N, inp, 1, C, dout, OC, 1, dW);    /* dW   += inp^T @ dout */
    if (db) {
        LAUNCH(colsum_k, nblocks(OC), BS, db, dout, N, OC);
        KCHECK();
    }
}

static void attention_forward(float *out, float *att, const float *qkv, int B, int T, int C,
                              int H, uint32_t site) {
    size_t rows = (size_t)B * H * T;
    LAUNCH(attention_forward_k, nblocks(rows), BS, out, att, qkv, B, T, C, H, drop_seed,
           drop_p, site);
    KCHECK();
}

static void attention_backward(float *dqkv, const float *dout, const float *qkv,
                               const float *att, int B, int T, int C, int H, uint32_t site) {
    size_t rows = (size_t)B * H * T;
    float *dpre = dev_scratch(rows * T);
    LAUNCH(attention_backward_rows_k, nblocks(rows), BS, dqkv, dpre, dout, qkv, att, B, T, C,
           H, drop_seed, drop_p, site);
    KCHECK();
    LAUNCH(attention_backward_cols_k, nblocks(rows), BS, dqkv, dpre, dout, qkv, att, B, T, C,
           H, drop_seed, drop_p, site);
    KCHECK();
}

static void gelu_forward(float *out, const float *inp, size_t n) {
    LAUNCH(gelu_forward_k, nblocks(n), BS, out, inp, n);
    KCHECK();
}

static void gelu_backward(float *dinp, const float *inp, const float *dout, size_t n) {
    LAUNCH(gelu_backward_k, nblocks(n), BS, dinp, inp, dout, n);
    KCHECK();
}

static void residual_forward(float *out, const float *a, const float *b, size_t n) {
    LAUNCH(add_k, nblocks(n), BS, out, a, b, n);
    KCHECK();
}

static void accumulate(float *dst, const float *src, size_t n) {
    LAUNCH(accumulate_k, nblocks(n), BS, dst, src, n);
    KCHECK();
}

/* logits (N,V) = x (N,C) @ wte^T, softmax; returns the mean loss if targets */
static double head_forward(float *probs, const float *x, const float *wte, const int *targets,
                           float *losses, int N, int C, int V) {
    bk_zero(probs, sizeof(float) * (size_t)N * V);
    gemm(N, V, C, x, C, 1, wte, 1, C, probs);
    LAUNCH(softmax_loss_k, N, BS, probs, targets, losses, V);
    KCHECK();
    if (!targets) return 0.0;
    static float *h_loss = NULL;
    static int cap = 0;
    if (cap < N) { free(h_loss); h_loss = (float *)xmalloc(sizeof(float) * N); cap = N; }
    bk_download(h_loss, losses, sizeof(float) * N);
    double total = 0.0;
    for (int n = 0; n < N; n++) total += h_loss[n];
    return total / N;
}

static void head_backward(float *dx, float *dwte, float *probs, const float *x,
                          const float *wte, const int *targets, int N, int C, int V) {
    size_t total = (size_t)N * V;
    LAUNCH(dlogits_k, nblocks(total), BS, probs, targets, V, 1.0f / N, total);
    KCHECK();
    gemm(N, C, V, probs, V, 1, wte, C, 1, dx);     /* dx   += dlogits @ wte */
    gemm(V, C, N, probs, 1, V, x, C, 1, dwte);     /* dwte += dlogits^T @ x */
}

#include "driver.h"
