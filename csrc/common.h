/*
 * UniAi — pieces shared by every backend (CPU: uniai.c, GPU: uniai_gpu.cu).
 * Valid C and C++; functions marked HD also compile for the GPU.
 */
#ifndef UNIAI_COMMON_H
#define UNIAI_COMMON_H

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(__CUDACC__) || defined(UNIAI_EMULATE)
#define HD __host__ __device__
#else
#define HD
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

/* --------------------------------------------------------------- dropout
 * Masks are never stored: keep/drop for element i of site s is a pure hash
 * of (seed, s, i), so the backward pass regenerates exactly the same mask,
 * on any backend.  drop_p == 0 disables dropout. */
static float drop_p = 0.0f;
static uint64_t drop_seed = 0;

static inline HD float drop_hash(uint64_t seed, float p, uint32_t site, size_t i) {
    uint64_t x = seed ^ ((uint64_t)site << 44) ^ ((uint64_t)i * 0x9E3779B97F4A7C15ULL);
    x ^= x >> 30; x *= 0xBF58476D1CE4E5B9ULL;
    x ^= x >> 27; x *= 0x94D049BB133111EBULL;
    x ^= x >> 31;
    return (float)(x >> 40) * (1.0f / 16777216.0f) >= p ? 1.0f / (1.0f - p) : 0.0f;
}

enum { SITE_EMB = 0, SITE_ATT = 1, SITE_APROJ = 2, SITE_FPROJ = 3 };
#define SITE(l, kind) ((uint32_t)(1 + (l)) * 4 + (kind))

#define GELU_K 0.7978845608f /* sqrt(2/pi) */

#endif
