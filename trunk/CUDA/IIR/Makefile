# Check CUDA Version
CUDA60_DIR = /usr/local/cuda-6.0
CUDA65_DIR = /usr/local/cuda-6.5

CUDA60 = 0
CUDA65 = 0
ifneq "$(wildcard $(CUDA60_DIR) )" ""
  CUDA60 = 1
endif

ifneq "$(wildcard $(CUDA65_DIR) )" ""
  CUDA65 = 1
endif

CUDA_DIR =

ifeq ($(CUDA60), 1)
  ifeq ($(CUDA65),1)
    # use higher version
    CUDA_DIR = $(CUDA65_DIR)
  else
  	# use 60 when no 65
    CUDA_DIR = $(CUDA60_DIR)
  endif
else
  ifeq ($(CUDA65),1)
	# use 65
    CUDA_DIR = $(CUDA65_DIR)
  else
    # error msg: no cuda 60 and 65
    $(error CUDA Version >= 6.0 is required for Benchmark Suites!)
  endif
endif

# 32-bit or 64-bit  
OS_TYPE = $(shell uname -m | sed -e "s/i.86/32/" -e "s/x86_64/64/" -e "s/armv7l/32/")

# Compilers
GCC ?= g++
NVC = $(CUDA_DIR)/bin/nvcc  -ccbin $(GCC)

NVCC_FLAGS   :=	
NVCC_FLAGS   += -O2 
NVCC_FLAGS   += -m${OS_TYPE}

# Directories for Header Files 
NV_INC = -I$(CUDA_DIR)/include/ -I$(CUDA_DIR)/samples/common/inc

# Directories for Libraries
ifeq ($(OS_TYPE), 64)
  NV_LIB = -L$(CUDA_DIR)/lib64
else
  NV_LIB = -L$(CUDA_DIR)/lib
endif

LIB_LINKR = -lcudart

SMS ?= 30 35 37 50

ifeq ($(GENCODE_FLAGS),)
  $(foreach sm,$(SMS),$(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))
endif

#-------------------------------------------------------------------------------------------------#
# Build Targets
#-------------------------------------------------------------------------------------------------#
all: build 

build: parIIR 

parIIR: parIIR.cu
	$(NVC) $(NV_INC) $(NV_LIB) $(NVCC_FLAGS) $(GENCODE_FLAGS) -o $@ $^ $(LIB_LINKER)

run: build
	./parIIR

.PHONY: clean
clean:
	rm -rf parIIR 
