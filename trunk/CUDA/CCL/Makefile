CPP = g++

NVCC = /usr/local/cuda-6.5/bin/nvcc

SOURCEDIR = .


CPP_SOURCES = $(SOURCEDIR)/accl_main.cpp
CU_SOURCES = $(SOURCEDIR)/accl.cu

CFLAGS = 

NVCCFLAGS = -pg 

ARCH = -gencode arch=compute_35,code=sm_35 

EXE = $(SOURCEDIR)/accl

DEVICE_OBJS = $(SOURCEDIR)/accl_cu.o

LINK_OBJS = $(SOURCEDIR)/accl_cu_dlink.o

all: accl

accl: accl_cu_dlink.o
	$(CPP) -o $(EXE) $(DEVICE_OBJS) $(LINK_OBJS) $(CPP_SOURCES) -L/usr/local/cuda-6.5/lib64 -lcudart -lcuda -lcudadevrt -lpthread -fopenmp

accl_cu_dlink.o: accl_cu.o
	$(NVCC) $(NVCCFLAGS) $(ARCH) -Xcompiler '-fPIC' -dlink $(DEVICE_OBJS) -o $(LINK_OBJS) -lcudadevrt

accl_cu.o: 
	$(NVCC) $(NVCCFLAGS) $(ARCH) -Xcompiler '-fPIC' -dc -m64 $(CU_SOURCES) -o $(DEVICE_OBJS)

clean:
	rm -f $(DEVICE_OBJS) 
	rm -f $(LINK_OBJS) 
	rm -f $(EXE)
