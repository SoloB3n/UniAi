/*
 * UniAi — GPT training engine in pure C.
 *
 * Same model as uniai/nn.py (pre-norm transformer, tied embeddings, GELU,
 * AdamW), but with every forward and backward kernel written by hand,
 * all memory allocated once up front, and loops parallelised with OpenMP.
 *
 *   uniai train  -i data/shakespeare -o model.bin [options]
 *   uniai sample -m model.bin -v data/shakespeare_vocab.bin [-n 300 -t 0.8]
 *   uniai check  model.bin batch.bin out.bin     (parity test vs Python)
 *
 * Weight layout matches the Python Linear: W is (in, out), y = x @ W + b.
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#define MAGIC 0x41494E55 /* "UNIA" */
#define VERSION 1

/* ------------------------------------------------------------------ utils */

static void *xmalloc(size_t n) {
    void *p = calloc(1, n);
    if (!p) { fprintf(stderr, "out of memory (%zu bytes)\n", n); exit(1); }
    return p;
}

static FILE *xfopen(const char *path, const char *mode) {
    FILE *f = fopen(path, mode);
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    return f;
}

static void xfread(void *p, size_t sz, size_t n, FILE *f) {
    if (fread(p, sz, n, f) != n) { fprintf(stderr, "short read\n"); exit(1); }
}

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* xorshift64* + Box-Muller: small, fast, reproducible */
static uint64_t rng_state = 1337;
static uint32_t rand_u32(void) {
    rng_state ^= rng_state >> 12;
    rng_state ^= rng_state << 25;
    rng_state ^= rng_state >> 27;
    return (uint32_t)((rng_state * 0x2545F4914F6CDD1DULL) >> 32);
}
static float rand_f32(void) { return (rand_u32() >> 8) / 16777216.0f; }
static float rand_normal(void) {
    float u1 = rand_f32() + 1e-7f, u2 = rand_f32();
    return sqrtf(-2.0f * logf(u1)) * cosf(6.2831853f * u2);
}

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

/* qkv (B,T,3C) -> out (B,T,C); att (B,H,T,T) keeps the softmax for backward.
 * Work is split per (batch, head).  K is transposed to (D,T) first so every
 * inner loop runs over contiguous memory as an axpy, which vectorizes
 * without reassociating floating-point sums. */
static void attention_forward(float *out, float *att, const float *qkv,
                              int B, int T, int C, int H) {
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
                    for (int i = 0; i < D; i++) o[i] += w * v[i];
                }
            }
            free(kt);
        }
}

static void attention_backward(float *dqkv, const float *dout, const float *qkv,
                               const float *att, int B, int T, int C, int H) {
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
                float dot = 0.0f;
                for (int t2 = 0; t2 <= t; t2++) dot += datt[t2] * a[t2];
                for (int t2 = 0; t2 <= t; t2++) {
                    float *dv = dqkv + ((size_t)b * T + t2) * C3 + 2 * C + h * D;
                    float w = a[t2];
                    for (int i = 0; i < D; i++) dv[i] += w * d[i];
                    /* softmax backward, then into q and k */
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
#define GELU_K 0.7978845608f /* sqrt(2/pi) */
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
static float head_forward(float *probs, const float *x, const float *wte,
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
    if (!targets) return 0.0f;
    double total = 0.0;
    for (int n = 0; n < N; n++) total += losses[n];
    return (float)(total / N);
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

/* ------------------------------------------------------------------ model */

typedef struct { int V, T, L, H, C; } Config;

enum { WTE, WPE, LN1W, LN1B, QKVW, QKVB, APW, APB, LN2W, LN2B,
       FCW, FCB, FPW, FPB, LNFW, LNFB, NPARAMS };

/* activations; each per-layer tensor is stacked over L */
enum { ENC, LN1, LN1M, LN1R, QKV, ATTY, ATT, APROJ, RES2, LN2, LN2M, LN2R,
       FCH, FCHG, FPROJ, RES3, LNF, LNFM, LNFR, PROBS, LOSSES, NACTS };

typedef struct {
    Config cfg;
    size_t psize[NPARAMS], nparams;
    float *params, *grads, *m, *v, *p[NPARAMS], *g[NPARAMS];
    int B, T; /* activation capacity */
    size_t asize[NACTS];
    size_t acts_total;
    float *acts, *dacts, *a[NACTS], *da[NACTS];
    int *inputs, *targets;
    int step;
    float val_loss;
} GPT;

static void param_sizes(Config c, size_t *s) {
    size_t C = c.C, L = c.L;
    s[WTE] = (size_t)c.V * C; s[WPE] = (size_t)c.T * C;
    s[LN1W] = s[LN1B] = s[LN2W] = s[LN2B] = L * C;
    s[QKVW] = L * C * 3 * C; s[QKVB] = L * 3 * C;
    s[APW] = L * C * C; s[APB] = L * C;
    s[FCW] = L * C * 4 * C; s[FCB] = L * 4 * C;
    s[FPW] = L * 4 * C * C; s[FPB] = L * C;
    s[LNFW] = s[LNFB] = C;
}

static void gpt_alloc_params(GPT *m, Config c) {
    m->cfg = c;
    param_sizes(c, m->psize);
    m->nparams = 0;
    for (int i = 0; i < NPARAMS; i++) m->nparams += m->psize[i];
    m->params = xmalloc(m->nparams * sizeof(float));
    m->grads = xmalloc(m->nparams * sizeof(float));
    m->m = xmalloc(m->nparams * sizeof(float));
    m->v = xmalloc(m->nparams * sizeof(float));
    size_t off = 0;
    for (int i = 0; i < NPARAMS; i++) {
        m->p[i] = m->params + off;
        m->g[i] = m->grads + off;
        off += m->psize[i];
    }
}

static void gpt_init(GPT *m) {
    Config c = m->cfg;
    float proj_std = 0.02f / sqrtf(2.0f * c.L);
    for (int i = 0; i < NPARAMS; i++) {
        float *p = m->p[i];
        size_t n = m->psize[i];
        float std = 0.0f, fill = 0.0f;
        switch (i) {
            case WTE: case QKVW: case FCW: std = 0.02f; break;
            case WPE: std = 0.01f; break;
            case APW: case FPW: std = proj_std; break;
            case LN1W: case LN2W: case LNFW: fill = 1.0f; break;
            default: break;
        }
        for (size_t j = 0; j < n; j++) p[j] = std > 0 ? std * rand_normal() : fill;
    }
}

static void gpt_alloc_acts(GPT *m, int B, int T) {
    Config c = m->cfg;
    size_t L = c.L, N = (size_t)B * T, C = c.C, *s = m->asize;
    s[ENC] = N * C;
    s[LN1] = L * N * C; s[LN1M] = s[LN1R] = L * N;
    s[QKV] = L * N * 3 * C; s[ATTY] = L * N * C;
    s[ATT] = L * B * c.H * (size_t)T * T;
    s[APROJ] = s[RES2] = s[LN2] = L * N * C; s[LN2M] = s[LN2R] = L * N;
    s[FCH] = s[FCHG] = L * N * 4 * C;
    s[FPROJ] = s[RES3] = L * N * C;
    s[LNF] = N * C; s[LNFM] = s[LNFR] = N;
    s[PROBS] = N * c.V; s[LOSSES] = N;
    size_t total = 0;
    for (int i = 0; i < NACTS; i++) total += s[i];
    m->acts_total = total;
    m->acts = xmalloc(total * sizeof(float));
    m->dacts = xmalloc(total * sizeof(float));
    size_t off = 0;
    for (int i = 0; i < NACTS; i++) {
        m->a[i] = m->acts + off;
        m->da[i] = m->dacts + off;
        off += s[i];
    }
    m->inputs = xmalloc(N * sizeof(int));
    m->targets = xmalloc(N * sizeof(int));
    m->B = B;
    m->T = T;
}

/* forward over (B,T) tokens in m->inputs; B*T must fit the allocation.
 * Per-layer slices are offset by the *current* B*T, so smaller batches
 * (e.g. B=1 during sampling) reuse the same buffers. */
static float gpt_forward(GPT *m, int B, int T, int with_targets) {
    Config c = m->cfg;
    int C = c.C, H = c.H;
    size_t N = (size_t)B * T, NC = N * C, att_sz = (size_t)B * H * T * T;
    float **p = m->p, **a = m->a;
    encoder_forward(a[ENC], m->inputs, p[WTE], p[WPE], B, T, C);
    for (int l = 0; l < c.L; l++) {
        float *res = l == 0 ? a[ENC] : a[RES3] + (l - 1) * NC;
        float *ln1 = a[LN1] + l * NC, *qkv = a[QKV] + l * 3 * NC, *atty = a[ATTY] + l * NC;
        float *att = a[ATT] + l * att_sz, *aproj = a[APROJ] + l * NC, *res2 = a[RES2] + l * NC;
        float *ln2 = a[LN2] + l * NC, *fch = a[FCH] + l * 4 * NC, *fchg = a[FCHG] + l * 4 * NC;
        float *fproj = a[FPROJ] + l * NC, *res3 = a[RES3] + l * NC;
        layernorm_forward(ln1, a[LN1M] + l * N, a[LN1R] + l * N, res,
                          p[LN1W] + l * C, p[LN1B] + l * C, N, C);
        matmul_forward(qkv, ln1, p[QKVW] + (size_t)l * C * 3 * C, p[QKVB] + l * 3 * C, N, C, 3 * C);
        attention_forward(atty, att, qkv, B, T, C, H);
        matmul_forward(aproj, atty, p[APW] + (size_t)l * C * C, p[APB] + l * C, N, C, C);
        residual_forward(res2, res, aproj, NC);
        layernorm_forward(ln2, a[LN2M] + l * N, a[LN2R] + l * N, res2,
                          p[LN2W] + l * C, p[LN2B] + l * C, N, C);
        matmul_forward(fch, ln2, p[FCW] + (size_t)l * C * 4 * C, p[FCB] + l * 4 * C, N, C, 4 * C);
        gelu_forward(fchg, fch, 4 * NC);
        matmul_forward(fproj, fchg, p[FPW] + (size_t)l * 4 * C * C, p[FPB] + l * C, N, 4 * C, C);
        residual_forward(res3, res2, fproj, NC);
    }
    float *last = a[RES3] + (c.L - 1) * NC;
    layernorm_forward(a[LNF], a[LNFM], a[LNFR], last, p[LNFW], p[LNFB], N, C);
    return head_forward(a[PROBS], a[LNF], p[WTE], with_targets ? m->targets : NULL,
                        a[LOSSES], N, C, c.V);
}

static void gpt_backward(GPT *m, int B, int T) {
    Config c = m->cfg;
    int C = c.C, H = c.H;
    size_t N = (size_t)B * T, NC = N * C, att_sz = (size_t)B * H * T * T;
    float **p = m->p, **g = m->g, **a = m->a, **da = m->da;
    memset(m->dacts, 0, sizeof(float) * m->acts_total);
    float *last = a[RES3] + (c.L - 1) * NC;
    head_backward(da[LNF], g[WTE], a[PROBS], a[LNF], p[WTE], m->targets, N, C, c.V);
    layernorm_backward(da[RES3] + (c.L - 1) * NC, g[LNFW], g[LNFB], da[LNF], last,
                       p[LNFW], a[LNFM], a[LNFR], N, C);
    for (int l = c.L - 1; l >= 0; l--) {
        float *res = l == 0 ? a[ENC] : a[RES3] + (l - 1) * NC;
        float *dres = l == 0 ? da[ENC] : da[RES3] + (l - 1) * NC;
        float *ln1 = a[LN1] + l * NC, *qkv = a[QKV] + l * 3 * NC, *atty = a[ATTY] + l * NC;
        float *att = a[ATT] + l * att_sz, *res2 = a[RES2] + l * NC;
        float *ln2 = a[LN2] + l * NC, *fch = a[FCH] + l * 4 * NC, *fchg = a[FCHG] + l * 4 * NC;
        float *dln1 = da[LN1] + l * NC, *dqkv = da[QKV] + l * 3 * NC, *datty = da[ATTY] + l * NC;
        float *daproj = da[APROJ] + l * NC, *dres2 = da[RES2] + l * NC, *dln2 = da[LN2] + l * NC;
        float *dfch = da[FCH] + l * 4 * NC, *dfchg = da[FCHG] + l * 4 * NC;
        float *dfproj = da[FPROJ] + l * NC, *dres3 = da[RES3] + l * NC;
        /* res3 = res2 + fproj */
        accumulate(dres2, dres3, NC);
        accumulate(dfproj, dres3, NC);
        matmul_backward(dfchg, g[FPW] + (size_t)l * 4 * C * C, g[FPB] + l * C, dfproj, fchg,
                        p[FPW] + (size_t)l * 4 * C * C, N, 4 * C, C);
        gelu_backward(dfch, fch, dfchg, 4 * NC);
        matmul_backward(dln2, g[FCW] + (size_t)l * C * 4 * C, g[FCB] + l * 4 * C, dfch, ln2,
                        p[FCW] + (size_t)l * C * 4 * C, N, C, 4 * C);
        layernorm_backward(dres2, g[LN2W] + l * C, g[LN2B] + l * C, dln2, res2,
                           p[LN2W] + l * C, a[LN2M] + l * N, a[LN2R] + l * N, N, C);
        /* res2 = res + aproj */
        accumulate(dres, dres2, NC);
        accumulate(daproj, dres2, NC);
        matmul_backward(datty, g[APW] + (size_t)l * C * C, g[APB] + l * C, daproj, atty,
                        p[APW] + (size_t)l * C * C, N, C, C);
        attention_backward(dqkv, datty, qkv, att, B, T, C, H);
        matmul_backward(dln1, g[QKVW] + (size_t)l * C * 3 * C, g[QKVB] + l * 3 * C, dqkv, ln1,
                        p[QKVW] + (size_t)l * C * 3 * C, N, C, 3 * C);
        layernorm_backward(dres, g[LN1W] + l * C, g[LN1B] + l * C, dln1, res,
                           p[LN1W] + l * C, a[LN1M] + l * N, a[LN1R] + l * N, N, C);
    }
    encoder_backward(g[WTE], g[WPE], da[ENC], m->inputs, B, T, C);
}

static float clip_grad_norm(GPT *m, float max_norm) {
    double ss = 0.0;
    #pragma omp parallel for reduction(+ : ss)
    for (size_t i = 0; i < m->nparams; i++) ss += (double)m->grads[i] * m->grads[i];
    float norm = (float)sqrt(ss);
    if (max_norm > 0 && norm > max_norm) {
        float s = max_norm / (norm + 1e-6f);
        #pragma omp parallel for
        for (size_t i = 0; i < m->nparams; i++) m->grads[i] *= s;
    }
    return norm;
}

/* identical to uniai/optim.py AdamW: decoupled decay on matrices only */
static void adamw_step(GPT *m, float lr, float wd, int t) {
    const float b1 = 0.9f, b2 = 0.95f, eps = 1e-8f;
    float c1 = 1.0f - powf(b1, t), c2 = 1.0f - powf(b2, t);
    for (int i = 0; i < NPARAMS; i++) {
        int decay = i == WTE || i == WPE || i == QKVW || i == APW || i == FCW || i == FPW;
        float *p = m->p[i], *g = m->g[i];
        float *mm = m->m + (p - m->params), *vv = m->v + (p - m->params);
        size_t n = m->psize[i];
        #pragma omp parallel for
        for (size_t j = 0; j < n; j++) {
            mm[j] = b1 * mm[j] + (1 - b1) * g[j];
            vv[j] = b2 * vv[j] + (1 - b2) * g[j] * g[j];
            if (decay) p[j] -= lr * wd * p[j];
            p[j] -= lr * (mm[j] / c1) / (sqrtf(vv[j] / c2) + eps);
        }
    }
}

static float cosine_lr(int step, float max_lr, int warmup, int total) {
    if (step < warmup) return max_lr * (step + 1) / warmup;
    float pr = (float)(step - warmup) / (total - warmup > 1 ? total - warmup : 1);
    if (pr > 1) pr = 1;
    return max_lr * (0.1f + 0.9f * 0.5f * (1.0f + cosf(3.14159265f * pr)));
}

/* --------------------------------------------------------- checkpoint i/o */

/* header: int32[16] = magic, version, V, T, L, H, C, step, has_opt, (float)val_loss */
static void gpt_save(GPT *m, const char *path, int with_opt) {
    int hdr[16] = {MAGIC, VERSION, m->cfg.V, m->cfg.T, m->cfg.L, m->cfg.H, m->cfg.C,
                   m->step, with_opt};
    memcpy(&hdr[9], &m->val_loss, sizeof(float));
    char tmp[1024];
    snprintf(tmp, sizeof tmp, "%s.tmp", path);
    FILE *f = xfopen(tmp, "wb");
    fwrite(hdr, sizeof(int), 16, f);
    fwrite(m->params, sizeof(float), m->nparams, f);
    if (with_opt) {
        fwrite(m->m, sizeof(float), m->nparams, f);
        fwrite(m->v, sizeof(float), m->nparams, f);
    }
    fclose(f);
    rename(tmp, path);
}

static void gpt_load(GPT *m, const char *path) {
    FILE *f = xfopen(path, "rb");
    int hdr[16];
    xfread(hdr, sizeof(int), 16, f);
    if (hdr[0] != MAGIC || hdr[1] != VERSION) { fprintf(stderr, "%s: not a UniAi model\n", path); exit(1); }
    Config c = {hdr[2], hdr[3], hdr[4], hdr[5], hdr[6]};
    gpt_alloc_params(m, c);
    m->step = hdr[7];
    memcpy(&m->val_loss, &hdr[9], sizeof(float));
    xfread(m->params, sizeof(float), m->nparams, f);
    if (hdr[8]) {
        xfread(m->m, sizeof(float), m->nparams, f);
        xfread(m->v, sizeof(float), m->nparams, f);
    }
    fclose(f);
}

/* ------------------------------------------------------------------- data */

typedef struct { uint16_t *tok; size_t n; } Tokens;

static Tokens load_tokens(const char *path) {
    FILE *f = xfopen(path, "rb");
    fseek(f, 0, SEEK_END);
    size_t bytes = ftell(f);
    fseek(f, 0, SEEK_SET);
    Tokens t = {xmalloc(bytes), bytes / 2};
    xfread(t.tok, 2, t.n, f);
    fclose(f);
    return t;
}

static void get_batch(GPT *m, Tokens d, int B, int T) {
    for (int b = 0; b < B; b++) {
        size_t i = ((size_t)rand_u32() << 16 ^ rand_u32()) % (d.n - T - 1);
        for (int t = 0; t < T; t++) {
            m->inputs[b * T + t] = d.tok[i + t];
            m->targets[b * T + t] = d.tok[i + t + 1];
        }
    }
}

/* vocab.bin: int32 V, then per token: uint8 len, bytes */
typedef struct { int V; unsigned char **s; unsigned char *len; } Vocab;

static Vocab load_vocab(const char *path) {
    FILE *f = xfopen(path, "rb");
    Vocab v;
    xfread(&v.V, sizeof(int), 1, f);
    v.s = xmalloc(sizeof(char *) * v.V);
    v.len = xmalloc(v.V);
    for (int i = 0; i < v.V; i++) {
        xfread(&v.len[i], 1, 1, f);
        v.s[i] = xmalloc(v.len[i] + 1);
        xfread(v.s[i], 1, v.len[i], f);
    }
    fclose(f);
    return v;
}

static int sample_probs(const float *p, int V, float temperature) {
    /* probs are softmax(logits); temperature re-weights them as p^(1/temp) */
    double total = 0.0;
    static double *w = NULL;
    static int cap = 0;
    if (cap < V) { free(w); w = xmalloc(sizeof(double) * V); cap = V; }
    for (int i = 0; i < V; i++) { w[i] = pow(p[i], 1.0 / temperature); total += w[i]; }
    double r = rand_f32() * total, acc = 0.0;
    for (int i = 0; i < V; i++) { acc += w[i]; if (r < acc) return i; }
    return V - 1;
}

static void generate(GPT *m, Vocab *voc, int ntok, float temperature, FILE *out) {
    int T = m->cfg.T, *ctx = xmalloc(sizeof(int) * (ntok + 1)), n = 1;
    ctx[0] = '\n';
    for (int k = 0; k < ntok; k++) {
        int len = n < T ? n : T, start = n - len;
        if (len > m->T) { len = m->T; start = n - len; }
        memcpy(m->inputs, ctx + start, sizeof(int) * len);
        gpt_forward(m, 1, len, 0);
        int nxt = sample_probs(m->a[PROBS] + (size_t)(len - 1) * m->cfg.V, m->cfg.V, temperature);
        ctx[n++] = nxt;
        if (voc) fwrite(voc->s[nxt], 1, voc->len[nxt], out);
    }
    fflush(out);
    free(ctx);
}

/* ------------------------------------------------------------------- main */

static void usage(void) {
    fprintf(stderr,
        "UniAi C engine\n"
        "  uniai train  -i PREFIX -o OUT.bin [-r RESUME.bin] [-L layers] [-H heads]\n"
        "               [-C embd] [-T block] [-B batch] [-n steps] [-l lr] [-w warmup]\n"
        "               [-d weight_decay] [-e eval_every] [-s seed]\n"
        "               reads PREFIX_train.bin, PREFIX_val.bin, PREFIX_vocab.bin\n"
        "  uniai sample -m MODEL.bin -v VOCAB.bin [-n tokens] [-t temperature] [-s seed]\n"
        "  uniai check  MODEL.bin BATCH.bin OUT.bin\n");
    exit(1);
}

static float eval_loss(GPT *m, Tokens d, int B, int T, int iters) {
    double s = 0.0;
    for (int i = 0; i < iters; i++) {
        get_batch(m, d, B, T);
        s += gpt_forward(m, B, T, 1);
    }
    return (float)(s / iters);
}

static int cmd_train(int argc, char **argv) {
    const char *prefix = NULL, *out = "model.bin", *resume = NULL;
    Config c = {0, 128, 4, 4, 128};
    int B = 16, steps = 3000, warmup = 100, eval_every = 500, eval_iters = 20;
    float lr = 1e-3f, wd = 0.1f;
    for (int i = 2; i + 1 < argc; i += 2) {
        char f = argv[i][1];
        const char *v = argv[i + 1];
        if (f == 'i') prefix = v; else if (f == 'o') out = v; else if (f == 'r') resume = v;
        else if (f == 'L') c.L = atoi(v); else if (f == 'H') c.H = atoi(v);
        else if (f == 'C') c.C = atoi(v); else if (f == 'T') c.T = atoi(v);
        else if (f == 'B') B = atoi(v); else if (f == 'n') steps = atoi(v);
        else if (f == 'l') lr = atof(v); else if (f == 'w') warmup = atoi(v);
        else if (f == 'd') wd = atof(v); else if (f == 'e') eval_every = atoi(v);
        else if (f == 's') rng_state = strtoull(v, NULL, 10) * 2654435761ULL + 1;
        else usage();
    }
    if (!prefix) usage();
    char path[1024];
    snprintf(path, sizeof path, "%s_train.bin", prefix);
    Tokens train = load_tokens(path);
    snprintf(path, sizeof path, "%s_val.bin", prefix);
    Tokens val = load_tokens(path);
    snprintf(path, sizeof path, "%s_vocab.bin", prefix);
    Vocab voc = load_vocab(path);

    GPT m = {0};
    if (resume) {
        gpt_load(&m, resume);
        printf("resuming %s at step %d\n", resume, m.step);
    } else {
        c.V = voc.V;
        if (c.C % c.H) { fprintf(stderr, "embd must be divisible by heads\n"); return 1; }
        gpt_alloc_params(&m, c);
        gpt_init(&m);
        m.val_loss = 1e30f;
    }
    c = m.cfg;
    int T = c.T;
    gpt_alloc_acts(&m, B, T);
    int threads = 1;
#ifdef _OPENMP
    threads = omp_get_max_threads();
#endif
    printf("UniAi C engine | %zu parameters | V=%d T=%d L=%d H=%d C=%d | batch %d | %d threads\n",
           m.nparams, c.V, c.T, c.L, c.H, c.C, B, threads);
    printf("train %zu tokens, val %zu tokens\n", train.n, val.n);

    double t0 = now(), tlast = t0;
    for (int step = m.step; step < steps; step++) {
        float cur_lr = cosine_lr(step, lr, warmup, steps);
        get_batch(&m, train, B, T);
        float loss = gpt_forward(&m, B, T, 1);
        memset(m.grads, 0, m.nparams * sizeof(float));
        gpt_backward(&m, B, T);
        float norm = clip_grad_norm(&m, 1.0f);
        adamw_step(&m, cur_lr, wd, step + 1);
        m.step = step + 1;
        if (step % 50 == 0) {
            double t = now();
            printf("step %5d | loss %.4f | lr %.2e | grad %.2f | %.0f ms/step\n", step, loss,
                   cur_lr, norm, (t - tlast) * 1000.0 / (step % 50 == 0 && step > 0 ? 50 : 1));
            tlast = t;
            fflush(stdout);
        }
        if ((step + 1) % eval_every == 0 || step + 1 == steps) {
            float tl = eval_loss(&m, train, B, T, eval_iters);
            float vl = eval_loss(&m, val, B, T, eval_iters);
            printf("---- eval @ %d: train %.4f | val %.4f | val ppl %.1f | %.0fs elapsed\n",
                   step + 1, tl, vl, expf(vl), now() - t0);
            if (vl < m.val_loss) {
                m.val_loss = vl;
                gpt_save(&m, out, 1);
                printf("     saved %s\n", out);
            }
            printf("     sample: ");
            generate(&m, &voc, 80, 0.8f, stdout);
            printf("\n");
            fflush(stdout);
        }
    }
    printf("done in %.1fs | best val loss %.4f\n", now() - t0, m.val_loss);
    return 0;
}

static int cmd_sample(int argc, char **argv) {
    const char *model = NULL, *vocab = NULL;
    int n = 300;
    float temp = 0.8f;
    rng_state = (uint64_t)time(NULL);
    for (int i = 2; i + 1 < argc; i += 2) {
        char f = argv[i][1];
        const char *v = argv[i + 1];
        if (f == 'm') model = v; else if (f == 'v') vocab = v;
        else if (f == 'n') n = atoi(v); else if (f == 't') temp = atof(v);
        else if (f == 's') rng_state = strtoull(v, NULL, 10) * 2654435761ULL + 1;
        else usage();
    }
    if (!model || !vocab) usage();
    GPT m = {0};
    gpt_load(&m, model);
    Vocab voc = load_vocab(vocab);
    gpt_alloc_acts(&m, 1, m.cfg.T);
    generate(&m, &voc, n, temp, stdout);
    printf("\n");
    return 0;
}

/* forward + backward + one AdamW step on a fixed batch; dumps everything */
static int cmd_check(int argc, char **argv) {
    if (argc != 5) usage();
    GPT m = {0};
    gpt_load(&m, argv[2]);
    FILE *f = xfopen(argv[3], "rb");
    int hdr[2];
    xfread(hdr, sizeof(int), 2, f);
    int B = hdr[0], T = hdr[1];
    gpt_alloc_acts(&m, B, T);
    xfread(m.inputs, sizeof(int), (size_t)B * T, f);
    xfread(m.targets, sizeof(int), (size_t)B * T, f);
    fclose(f);
    float loss = gpt_forward(&m, B, T, 1);
    gpt_backward(&m, B, T);
    f = xfopen(argv[4], "wb");
    fwrite(&loss, sizeof(float), 1, f);
    fwrite(m.grads, sizeof(float), m.nparams, f);
    float norm = clip_grad_norm(&m, 1.0f);
    adamw_step(&m, 1e-3f, 0.1f, 1);
    fwrite(&norm, sizeof(float), 1, f);
    fwrite(m.params, sizeof(float), m.nparams, f);
    fclose(f);
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 2) usage();
    if (!strcmp(argv[1], "train")) return cmd_train(argc, argv);
    if (!strcmp(argv[1], "sample")) return cmd_sample(argc, argv);
    if (!strcmp(argv[1], "check")) return cmd_check(argc, argv);
    usage();
    return 1;
}
