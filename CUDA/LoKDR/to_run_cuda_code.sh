#!/bin/bash
export PATH=$PATH:/usr/local/cuda/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib

gcc -m32 -c src/*.c
nvcc -m32 -o lokdr -I ./src args.o parsing.o vector.o lokdr.cu -lcuda -lcublas -D_CRT_SECURE_NO_DEPRECATE
