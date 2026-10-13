/*
 * A tiny CUDA emulator: runs the *same* kernel source on the CPU so the GPU
 * backend can be tested on machines without a GPU.
 *
 * Every GPU thread of a block becomes a coroutine (ucontext) on one OS
 * thread.  __syncthreads() suspends the coroutine; the scheduler resumes
 * the block's threads round-robin, so all of them reach each barrier before
 * any passes it — the real CUDA semantics.  Blocks run one after another,
 * so `__shared__` can simply be a static local: it is shared by all threads
 * of the running block.
 *
 * Supported: __global__/__device__/__shared__, threadIdx/blockIdx/blockDim/
 * gridDim, __syncthreads, atomicAdd(float), cudaMalloc/Free/Memcpy/Memset,
 * kernels launched with LAUNCH(kernel, grid, block, args...).
 * Not supported (and not used by UniAi): warp intrinsics, dynamic shared
 * memory, streams.  It is slow; it exists for correctness, not speed.
 *
 * Divergent barriers (some threads of a block exit or skip a barrier others
 * wait at) are undefined behaviour in CUDA; the emulator detects them and
 * aborts, so they are caught here rather than as a hang on a real GPU.
 */
#ifndef UNIAI_CUDA_EMU_H
#define UNIAI_CUDA_EMU_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ucontext.h>

#include <functional>
#include <vector>

struct dim3 {
    unsigned x, y, z;
    dim3(unsigned x_ = 1, unsigned y_ = 1, unsigned z_ = 1) : x(x_), y(y_), z(z_) {}
};

static dim3 threadIdx, blockIdx, blockDim, gridDim;

#define __global__
#define __device__
#define __host__
#define __shared__ static
#define __restrict__

typedef int cudaError_t;
enum { cudaSuccess = 0 };
enum cudaMemcpyKind { cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost, cudaMemcpyDeviceToDevice };

template <class T> static cudaError_t cudaMalloc(T **p, size_t n) {
    *p = (T *)calloc(1, n ? n : 1);
    return *p ? cudaSuccess : 2;
}
static cudaError_t cudaFree(void *p) { free(p); return cudaSuccess; }
static cudaError_t cudaMemcpy(void *d, const void *s, size_t n, cudaMemcpyKind) {
    memmove(d, s, n);
    return cudaSuccess;
}
static cudaError_t cudaMemset(void *p, int v, size_t n) { memset(p, v, n); return cudaSuccess; }
static cudaError_t cudaDeviceSynchronize(void) { return cudaSuccess; }
static cudaError_t cudaGetLastError(void) { return cudaSuccess; }
static const char *cudaGetErrorString(cudaError_t) { return "emulator error"; }

/* one OS thread runs everything, so plain arithmetic is atomic */
static inline float atomicAdd(float *a, float v) { float old = *a; *a += v; return old; }

namespace emu {

static const size_t STACK = 64 * 1024;
static ucontext_t sched_ctx;
static std::vector<ucontext_t> ctx;
static std::vector<char *> stacks;
static std::vector<int> state; /* 0 running, 1 at barrier, 2 finished */
static int cur;
static const std::function<void()> *body;

static void trampoline() {
    (*body)();
    state[cur] = 2;
    swapcontext(&ctx[cur], &sched_ctx);
}

static inline void sync() {
    state[cur] = 1;
    swapcontext(&ctx[cur], &sched_ctx);
}

static void run_block(int nthreads) {
    for (int t = 0; t < nthreads; t++) {
        getcontext(&ctx[t]);
        ctx[t].uc_stack.ss_sp = stacks[t];
        ctx[t].uc_stack.ss_size = STACK;
        ctx[t].uc_link = &sched_ctx;
        makecontext(&ctx[t], trampoline, 0);
        state[t] = 0;
    }
    for (;;) {
        for (int t = 0; t < nthreads; t++) {
            if (state[t] == 2) continue;
            cur = t;
            threadIdx = dim3(t % blockDim.x, (t / blockDim.x) % blockDim.y,
                             t / (blockDim.x * blockDim.y));
            state[t] = 0;
            swapcontext(&sched_ctx, &ctx[t]);
        }
        int finished = 0, waiting = 0;
        for (int t = 0; t < nthreads; t++) {
            finished += state[t] == 2;
            waiting += state[t] == 1;
        }
        if (finished == nthreads) return;
        if (finished > 0 && waiting > 0) {
            fprintf(stderr, "cuda_emu: divergent __syncthreads (%d threads exited, %d waiting)\n",
                    finished, waiting);
            abort();
        }
    }
}

template <class F> static void launch(dim3 grid, dim3 block, F f) {
    int nthreads = block.x * block.y * block.z;
    if (nthreads > 1024) { fprintf(stderr, "cuda_emu: block too large\n"); abort(); }
    while ((int)stacks.size() < nthreads) stacks.push_back((char *)malloc(STACK));
    if ((int)ctx.size() < nthreads) { ctx.resize(nthreads); state.resize(nthreads); }
    std::function<void()> fn(f);
    body = &fn;
    gridDim = grid;
    blockDim = block;
    for (unsigned z = 0; z < grid.z; z++)
        for (unsigned y = 0; y < grid.y; y++)
            for (unsigned x = 0; x < grid.x; x++) {
                blockIdx = dim3(x, y, z);
                run_block(nthreads);
            }
    body = nullptr;
}

} // namespace emu

#define __syncthreads() emu::sync()
#define LAUNCH(kernel, grid, block, ...) \
    emu::launch(dim3(grid), dim3(block), [&]() { kernel(__VA_ARGS__); })

#endif
