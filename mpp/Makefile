DEBUG ?= 0
CUDA ?= 0

## -fopenmp-targets=x86_64-pc-linux-mpi mpirun -np 2
## gdb ./main lldb

ifeq ($(strip $(CUDA)),1)
CXXFLAGS = -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda
LDFLAGS  = -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda
else
CXXFLAGS = -fopenmp -fopenmp-targets=x86_64-pc-linux-gnu
LDFLAGS  = -fopenmp -fopenmp-targets=x86_64-pc-linux-gnu
endif

ifeq ($(strip $(DEBUG)),1)
CXXFLAGS += -g -O0
LDFLAGS  += -g -O0
else
CXXFLAGS += -O3
LDFLAGS  += -O3
endif

# Include directories
INC        = -I../core
INC_EXT    =

# Location of the libraries.
LIB        = -L../core -lcore_s
LIB_EXT    =

INC := $(INC) $(INC_EXT)
LIB := $(LIB) $(LIB_EXT)

CXXFLAGS += $(INC)

include ../core/make_blas.mk

TARGET = main
all: $(TARGET)

.PRECIOUS: %.cc %.o

main.o: main.cc ../core/timer.h
	$(CXX) -c $(CXXFLAGS) $<

main: main.o
	$(CXX) $^ $(LIB) $(LDFLAGS) -o $@

clean:
	rm -f *.o
	rm -f $(TARGET)

.PHONY: all clean
