MPICXX ?= mpicxx

DEBUG ?= 0
GPU_ARCH ?= gfx942
CC          ?= clang
CXX         ?= clang++

CXXFLAGS_COMMON = -fopenmp -foffload-lto --offload-arch=$(GPU_ARCH) -I../core
LDFLAGS ?= -L../core -lcore_s -fopenmp -foffload-lto --offload-arch=$(GPU_ARCH) -ldl

ifeq ($(strip $(DEBUG)),0)
	CXXFLAGS_COMMON += -O3
else
	CXXFLAGS_COMMON += -O0 -ggdb
endif

include ../core/make_blas.mk

BIN := forall target_gpu
OBJ := $(BIN:%=%.o)

.PHONY: all
all: $(BIN)

# Compila cada .cc em .o com Clang++ e offload para GPU
%.o: %.cc
	$(CXX) $(CXXFLAGS_COMMON) -c $< -o $@

# Linka com MPI usando mpicxx
$(BIN): %: %.o
	$(MPICXX) -cxx=$(CXX) -o $@ $< $(LDFLAGS)

.PHONY: clean
clean:
	rm -f *.o $(BIN)
