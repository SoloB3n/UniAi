/*
 * UniAi — GPT training engine, CPU backend in pure C.
 *
 * Same model as uniai/nn.py (pre-norm transformer, tied embeddings, GELU,
 * AdamW), with every forward and backward kernel written by hand, a
 * register-tiled GEMM, and loops parallelised with OpenMP.  The model,
 * training loop and commands live in driver.h and are shared with the GPU
 * backend (uniai_gpu.cu).
 *
 *   uniai_c train  -i data/shakespeare -o model.bin [options]
 *   uniai_c sample -m model.bin -v data/shakespeare_vocab.bin
 *   uniai_c check  model.bin batch.bin out.bin     (parity test vs Python)
 *
 * Weight layout matches the Python Linear: W is (in, out), y = x @ W + b.
 */

#include "common.h"
#ifdef _OPENMP
#include <omp.h>
#endif

#define BACKEND_NAME "CPU"

/* ---------------------------------------------------------------- kernels */

static void encoder_forward(float *out, const int *inp, const float *wte,
                            const float *wpe, int B, int T, int C) {
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < B; b++)
        for (int t = 0; t < T; t++) {
            float *o = out + ((size_t)b * T + t) * C;
            const float *we = wte + (size_t)inp[b * T + t] * C, *pe = wpe + (size_t)t * C;
            for (int c = 0; c < C; c++) o[c] = we[c] + pe[c];
        }
}

static void encoder_backward(float *dwte, float *dwpe, const float *dout,
                             const int *inp, int B, int T, int C) {
    /* serial: several positions may hit the same token row */
    for (int b = 0; b < B; b++)
        for (int t = 0; t < T; t++) {
            const float *d = dout + ((size_t)b * T + t) * C;
            float *we = dwte + (size_t)inp[b * T + t] * C, *pe = dwpe + (size_t)t * C;
            for (int c = 0; c < C; c++) { we[c] += d[c]; pe[c] += d[c]; }
        }
}

static void layernorm_forward(float *out, float *mean, float *rstd, const float *inp,
                              const float *w, const float *bias, int N, int C) {
    #pragma omp parallel for
    for (int n = 0; n < N; n++) {
        const float *x = inp + (size_t)n * C;
        float m = 0.0f, v = 0.0f;
        for (int c = 0; c < C; c++) m += x[c];
        m /= C;
        for (int c = 0; c < C; c++) { float d = x[c] - m; v += d * d; }
        v /= C;
        float s = 1.0f / sqrtf(v + 1e-5f);
        float *o = out + (size_t)n * C;
        for (int c = 0; c < C; c++) o[c] = (x[c] - m) * s * w[c] + bias[c];
        mean[n] = m;
        rstd[n] = s;
    }
}

static void layernorm_backward(float *dinp, float *dw, float *db, const float *dout,
                               const float *inp, const float *w, const float *mean,
                               const float *rstd, int N, int C) {
    #pragma omp parallel for
    for (int n = 0; n < N; n++) {
        const float *d = dout + (size_t)n * C, *x = inp + (size_t)n * C;
        float m = mean[n], s = rstd[n], dmean = 0.0f, dnorm = 0.0f;
        for (int c = 0; c < C; c++) {
            float g = w[c] * d[c];
            dmean += g;
            dnorm += g * (x[c] - m) * s;
        }
        dmean /= C;
        dnorm /= C;
        float *di = dinp + (size_t)n * C;
        for (int c = 0; c < C; c++) {
            float xh = (x[c] - m) * s;
            di[c] += (w[c] * d[c] - dmean - xh * dnorm) * s;
        }
    }
    for (int n = 0; n < N; n++) {   /* row-wise: contiguous and vectorized */
        const float *d = dout + (size_t)n * C, *x = inp + (size_t)n * C;
        float m = mean[n], s = rstd[n];
        for (int c = 0; c < C; c++) {
            dw[c] += d[c] * (x[c] - m) * s;
            db[c] += d[c];
        }
    }
}

/* ----------------------------------------------------------------- GEMM
 * C (M,N) += op(A) (M,K) @ B (K,N), row-major, where element (i,k) of op(A)
 * lives at A[i*sai + k*sak]: (sai,sak) = (K,1) is A itself, (1,M) is A^T,
 * so dW = X^T dY needs no explicit transpose.  The core is an MR x NR
 * register tile: MR rows of A are broadcast against NR columns of B and
 * accumulated in registers across the whole K loop, so each loaded element
 * of B is reused MR times and C is touched only once.  With -march=native
 * the compiler turns the NR loop into AVX/AVX-512/NEON vector FMAs.
 */
#define MR 6
#if defined(__AVX512F__)
#define NR 32
#else
#define NR 16
#endif
#define KC 256 /* K blocking keeps the B panel (KC x NR) resident in L1 */

static inline void gemm_tile(int K, const float *A, size_t sai, size_t sak, const float *B,
                             int ldb, float *C, int ldc) {
    float acc[MR][NR] = {{0}};
    for (int k = 0; k < K; k++) {
        const float *b = B + (size_t)k * ldb;
        for (int i = 0; i < MR; i++) {
            float a = A[i * sai + k * sak];
            for (int j = 0; j < NR; j++) acc[i][j] += a * b[j];
        }
    }
    for (int i = 0; i < MR; i++)
        for (int j = 0; j < NR; j++) C[(size_t)i * ldc + j] += acc[i][j];
}

static void gemm_edge(int m, int n, int K, const float *A, size_t sai, size_t sak,
                      const float *B, int ldb, float *C, int ldc) {
    for (int i = 0; i < m; i++)
        for (int k = 0; k < K; k++) {
            float a = A[i * sai + k * sak];
            const float *b = B + (size_t)k * ldb;
            float *c = C + (size_t)i * ldc;
            for (int j = 0; j < n; j++) c[j] += a * b[j];
        }
}

static inline void gemm_body(int M, int N, int K, const float *A, size_t sai, size_t sak,
                             const float *B, float *C) {
    int mb = (M + MR - 1) / MR;
    #pragma omp parallel for schedule(static)
    for (int ib = 0; ib < mb; ib++) {
        int i = ib * MR, m = M - i < MR ? M - i : MR;
        for (int k0 = 0; k0 < K; k0 += KC) {
            int kc = K - k0 < KC ? K - k0 : KC;
            const float *a = A + i * sai + k0 * sak;
            int j = 0;
            if (m == MR)
                for (; j + NR <= N; j += NR)
                    gemm_tile(kc, a, sai, sak, B + (size_t)k0 * N + j, N, C + (size_t)i * N + j, N);
            if (j < N || m < MR)
                gemm_edge(m, N - j, kc, a, sai, sak, B + (size_t)k0 * N + j, N,
                          C + (size_t)i * N + j, N);
        }
    }
}

/* separate entry points so each gets a tile specialised for its strides */
static void gemm(int M, int N, int K, const float *A, const float *B, float *C) {
    gemm_body(M, N, K, A, (size_t)K, 1, B, C);
}
static void gemm_tn(int M, int N, int K, const float *A, const float *B, float *C) {
    gemm_body(M, N, K, A, 1, (size_t)M, B, C);   /* A stored as (K,M) */
}

/* scratch space for transposes; grown on demand, never shrunk */
static float *scratch_buf = NULL;
static size_t scratch_cap = 0;
static float *scratch(size_t n) {
    if (n > scratch_cap) {
        free(scratch_buf);
        scratch_buf = xmalloc(n * sizeof(float));
        scratch_cap = n;
    }
    return scratch_buf;
}

/* dst (C,R) = src (R,C)^T, in 32x32 blocks for cache friendliness */
static void transpose(float *dst, const float *src, int R, int C) {
    #pragma omp parallel for collapse(2)
    for (int r0 = 0; r0 < R; r0 += 32)
        for (int c0 = 0; c0 < C; c0 += 32)
            for (int r = r0; r < R && r < r0 + 32; r++)
                for (int c = c0; c < C && c < c0 + 32; c++)
                    dst[(size_t)c * R + r] = src[(size_t)r * C + c];
}

/* out (N,OC) = inp (N,C) @ W (C,OC) + bias */
static void matmul_forward(float *out, const float *inp, const float *W,
                           const float *bias, int N, int C, int OC) {
    #pragma omp parallel for
    for (int n = 0; n < N; n++) {
        float *o = out + (size_t)n * OC;
        for (int j = 0; j < OC; j++) o[j] = bias ? bias[j] : 0.0f;
    }
    gemm(N, OC, C, inp, W, out);
}

static void matmul_backward(float *dinp, float *dW, float *db, const float *dout,
                            const float *inp, const float *W, int N, int C, int OC) {
    float *t = scratch((size_t)C * OC);
    /* dinp (N,C) += dout (N,OC) @ W^T (OC,C) */
    transpose(t, W, C, OC);
    gemm(N, C, OC, dout, t, dinp);
    /* dW (C,OC) += inp^T (C,N) @ dout (N,OC) */
    gemm_tn(C, OC, N, inp, dout, dW);
    if (db)
        for (int n = 0; n < N; n++) {
            const float *d = dout + (size_t)n * OC;
            for (int j = 0; j < OC; j++) db[j] += d[j];
        }
}

/* --------------------------------------------------------------- dropout */

static inline float drop_scale(uint32_t site, size_t i) {
    return drop_hash(drop_seed, drop_p, site, i);
}

/* x *= mask, in place (forward on an output, or backward on its gradient) */
static void dropout_apply(float *x, size_t N, uint32_t site) {
    if (drop_p <= 0.0f) return;
    #pragma omp parallel for
    for (size_t i = 0; i < N; i++) x[i] *= drop_scale(site, i);
}

/* qkv (B,T,3C) -> out (B,T,C); att (B,H,T,T) keeps the softmax for backward.
 * Work is split per (batch, head).  K is transposed to (D,T) first so every
 * inner loop runs over contiguous memory as an axpy, which vectorizes
 * without reassociating floating-point sums. */
static void attention_forward(float *out, float *att, const float *qkv,
                              int B, int T, int C, int H, uint32_t site) {
    int D = C / H, C3 = 3 * C;
    float scale = 1.0f / sqrtf((float)D);
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < B; b++)
        for (int h = 0; h < H; h++) {
            float *kt = xmalloc(sizeof(float) * D * T);
            for (int t = 0; t < T; t++)
                for (int i = 0; i < D; i++)
                    kt[(size_t)i * T + t] = qkv[((size_t)b * T + t) * C3 + C + h * D + i];
            for (int t = 0; t < T; t++) {
                const float *q = qkv + ((size_t)b * T + t) * C3 + h * D;
                float *a = att + (((size_t)b * H + h) * T + t) * T;
                for (int t2 = 0; t2 < T; t2++) a[t2] = 0.0f;
                for (int i = 0; i < D; i++) {
                    float qi = q[i] * scale;
                    const float *k = kt + (size_t)i * T;
                    for (int t2 = 0; t2 <= t; t2++) a[t2] += qi * k[t2];
                }
                float mx = -1e30f, sum = 0.0f;
                for (int t2 = 0; t2 <= t; t2++) if (a[t2] > mx) mx = a[t2];
                for (int t2 = 0; t2 <= t; t2++) { a[t2] = expf(a[t2] - mx); sum += a[t2]; }
                float inv = 1.0f / sum;
                for (int t2 = 0; t2 <= t; t2++) a[t2] *= inv;
                float *o = out + ((size_t)b * T + t) * C + h * D;
                for (int i = 0; i < D; i++) o[i] = 0.0f;
                for (int t2 = 0; t2 <= t; t2++) {
                    const float *v = qkv + ((size_t)b * T + t2) * C3 + 2 * C + h * D;
                    float w = a[t2];
                    if (drop_p > 0.0f) w *= drop_scale(site, ((((size_t)b * H + h) * T + t) * T) + t2);
                    for (int i = 0; i < D; i++) o[i] += w * v[i];
                }
            }
            free(kt);
        }
}

static void attention_backward(float *dqkv, const float *dout, const float *qkv,
                               const float *att, int B, int T, int C, int H, uint32_t site) {
    int D = C / H, C3 = 3 * C;
    float scale = 1.0f / sqrtf((float)D);
    /* each (b,h) touches only its own head's slice of dqkv: no races */
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < B; b++)
        for (int h = 0; h < H; h++) {
            float *vt = xmalloc(sizeof(float) * (D * T + T));
            float *datt = vt + (size_t)D * T;
            for (int t = 0; t < T; t++)
                for (int i = 0; i < D; i++)
                    vt[(size_t)i * T + t] = qkv[((size_t)b * T + t) * C3 + 2 * C + h * D + i];
            for (int t = 0; t < T; t++) {
                const float *a = att + (((size_t)b * H + h) * T + t) * T;
                const float *d = dout + ((size_t)b * T + t) * C + h * D;
                const float *q = qkv + ((size_t)b * T + t) * C3 + h * D;
                float *dq = dqkv + ((size_t)b * T + t) * C3 + h * D;
                /* datt = V d ;  dV += a d^T */
                for (int t2 = 0; t2 <= t; t2++) datt[t2] = 0.0f;
                for (int i = 0; i < D; i++) {
                    float di = d[i];
                    const float *v = vt + (size_t)i * T;
                    for (int t2 = 0; t2 <= t; t2++) datt[t2] += di * v[t2];
                }
                size_t row = (((size_t)b * H + h) * T + t) * T;
                if (drop_p > 0.0f)
                    for (int t2 = 0; t2 <= t; t2++) datt[t2] *= drop_scale(site, row + t2);
                float dot = 0.0f;
                for (int t2 = 0; t2 <= t; t2++) dot += datt[t2] * a[t2];
                for (int t2 = 0; t2 <= t; t2++) {
                    float *dv = dqkv + ((size_t)b * T + t2) * C3 + 2 * C + h * D;
                    float w = a[t2];
                    float wd = drop_p > 0.0f ? w * drop_scale(site, row + t2) : w;
                    for (int i = 0; i < D; i++) dv[i] += wd * d[i];
                    /* softmax backward (on the undropped probabilities) */
                    float dpre = w * (datt[t2] - dot) * scale;
                    const float *k = qkv + ((size_t)b * T + t2) * C3 + C + h * D;
                    float *dk = dqkv + ((size_t)b * T + t2) * C3 + C + h * D;
                    for (int i = 0; i < D; i++) { dq[i] += dpre * k[i]; dk[i] += dpre * q[i]; }
                }
            }
            free(vt);
        }
}

/* GELU, tanh approximation.  tanh(u) = 1 - 2/(1+e^{2u}) lets the compiler
 * use its vectorized expf (with -fno-math-errno). */
static inline float fast_tanh(float u) {
    u = u > 15.0f ? 15.0f : (u < -15.0f ? -15.0f : u);
    return 1.0f - 2.0f / (1.0f + expf(2.0f * u));
}

static void gelu_forward(float *out, const float *inp, size_t N) {
    #pragma omp parallel for simd
    for (size_t i = 0; i < N; i++) {
        float x = inp[i];
        out[i] = 0.5f * x * (1.0f + fast_tanh(GELU_K * (x + 0.044715f * x * x * x)));
    }
}

static void gelu_backward(float *dinp, const float *inp, const float *dout, size_t N) {
    #pragma omp parallel for simd
    for (size_t i = 0; i < N; i++) {
        float x = inp[i], t = fast_tanh(GELU_K * (x + 0.044715f * x * x * x));
        float dinner = GELU_K * (1.0f + 3.0f * 0.044715f * x * x);
        dinp[i] += dout[i] * (0.5f * (1.0f + t) + 0.5f * x * (1.0f - t * t) * dinner);
    }
}

static void residual_forward(float *out, const float *a, const float *b, size_t N) {
    #pragma omp parallel for
    for (size_t i = 0; i < N; i++) out[i] = a[i] + b[i];
}

static void accumulate(float *dst, const float *src, size_t N) {
    #pragma omp parallel for
    for (size_t i = 0; i < N; i++) dst[i] += src[i];
}

/* logits (N,V) = x (N,C) @ wte^T, then softmax; returns mean loss if targets */
static double head_forward(float *probs, const float *x, const float *wte,
                          const int *targets, float *losses, int N, int C, int V) {
    float *wt = scratch((size_t)C * V);
    transpose(wt, wte, V, C);
    memset(probs, 0, sizeof(float) * (size_t)N * V);
    gemm(N, V, C, x, wt, probs);
    #pragma omp parallel for
    for (int n = 0; n < N; n++) {
        float *p = probs + (size_t)n * V, mx = -1e30f, sum = 0.0f;
        for (int v = 0; v < V; v++) if (p[v] > mx) mx = p[v];
        for (int v = 0; v < V; v++) { p[v] = expf(p[v] - mx); sum += p[v]; }
        float inv = 1.0f / sum;
        for (int v = 0; v < V; v++) p[v] *= inv;
        if (targets) losses[n] = -logf(p[targets[n]] + 1e-30f);
    }
    if (!targets) return 0.0;
    double total = 0.0;
    for (int n = 0; n < N; n++) total += losses[n];
    return total / N;
}

/* turns probs into dlogits in place, then backprops through the tied head */
static void head_backward(float *dx, float *dwte, float *probs, const float *x,
                          const float *wte, const int *targets, int N, int C, int V) {
    float inv = 1.0f / N;
    #pragma omp parallel for
    for (int n = 0; n < N; n++) {
        float *p = probs + (size_t)n * V;
        p[targets[n]] -= 1.0f;
        for (int v = 0; v < V; v++) p[v] *= inv;
    }
    gemm(N, C, V, probs, wte, dx);              /* dx (N,C) += dlogits @ wte */
    gemm_tn(V, C, N, probs, x, dwte);           /* dwte (V,C) += dlogits^T @ x */
}

/* ---------------------------------------------------------- backend API */

static void *bk_alloc(size_t bytes) { return xmalloc(bytes); }
static void bk_zero(void *p, size_t bytes) { memset(p, 0, bytes); }
static void bk_upload(void *dst, const void *src, size_t bytes) { memcpy(dst, src, bytes); }
static void bk_download(void *dst, const void *src, size_t bytes) { memcpy(dst, src, bytes); }

static void bk_describe(char *buf, size_t n) {
    int threads = 1;
#ifdef _OPENMP
    threads = omp_get_max_threads();
#endif
    snprintf(buf, n, "%d threads", threads);
}

static float bk_grad_norm_clip(float *g, size_t n, float max_norm) {
    double ss = 0.0;
    #pragma omp parallel for reduction(+ : ss)
    for (size_t i = 0; i < n; i++) ss += (double)g[i] * g[i];
    float norm = (float)sqrt(ss);
    if (max_norm > 0 && norm > max_norm) {
        float s = max_norm / (norm + 1e-6f);
        #pragma omp parallel for
        for (size_t i = 0; i < n; i++) g[i] *= s;
    }
    return norm;
}

static void bk_adamw(float *p, const float *g, float *m, float *v, size_t n,
                     float lr, float wd, int decay, float c1, float c2) {
    const float b1 = 0.9f, b2 = 0.95f, eps = 1e-8f;
    #pragma omp parallel for
    for (size_t j = 0; j < n; j++) {
        m[j] = b1 * m[j] + (1 - b1) * g[j];
        v[j] = b2 * v[j] + (1 - b2) * g[j] * g[j];
        if (decay) p[j] -= lr * wd * p[j];
        p[j] -= lr * (m[j] / c1) / (sqrtf(v[j] / c2) + eps);
    }
}

#include "driver.h"
