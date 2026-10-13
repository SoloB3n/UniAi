CC ?= gcc
CXX ?= g++
NVCC ?= nvcc
CFLAGS ?= -O3 -march=native -fopenmp -fno-math-errno -Wall -Wextra -Wno-unused-parameter
# -arch=native builds for the GPU in this machine; override for others,
# e.g. make gpu NVCC_ARCH="-gencode arch=compute_80,code=sm_80"
NVCC_ARCH ?= -arch=native
HEADERS = csrc/common.h csrc/driver.h

uniai_c: csrc/uniai.c $(HEADERS)
	$(CC) $(CFLAGS) -o $@ $< -lm

gpu: uniai_gpu
uniai_gpu: csrc/uniai_gpu.cu $(HEADERS)
	$(NVCC) -O3 -std=c++17 $(NVCC_ARCH) -o $@ $<

# the GPU kernels, run on the CPU by csrc/cuda_emu.h (for testing, slow)
gpu-emu: uniai_gpu_emu
uniai_gpu_emu: csrc/uniai_gpu.cu csrc/cuda_emu.h $(HEADERS)
	$(CXX) -O2 -std=c++17 -DUNIAI_EMULATE -x c++ -o $@ $< -lm

test: uniai_c uniai_gpu_emu
	python3 tests/test_autograd.py
	python3 tests/test_c_engine.py

clean:
	rm -f uniai_c uniai_gpu uniai_gpu_emu

.PHONY: gpu gpu-emu test clean
