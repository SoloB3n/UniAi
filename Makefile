CC ?= gcc
CFLAGS ?= -O3 -march=native -fopenmp -fno-math-errno -Wall -Wextra -Wno-unused-parameter

uniai_c: csrc/uniai.c
	$(CC) $(CFLAGS) -o $@ $< -lm

test: uniai_c
	python3 tests/test_autograd.py
	python3 tests/test_c_engine.py

clean:
	rm -f uniai_c

.PHONY: test clean
