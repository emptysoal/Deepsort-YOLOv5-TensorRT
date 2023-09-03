CUDA_PATH       = /usr/local/cuda
TRT_INC_PATH    = /usr/include/aarch64-linux-gnu
TRT_LIB_PATH    = /usr/lib/aarch64-linux-gnu
INCLUDE         = -I$(CUDA_PATH)/include -I$(TRT_INC_PATH)
LDFLAG          = -L$(CUDA_PATH)/lib64 -lcudart -L$(TRT_LIB_PATH) -lnvinfer -lnvonnxparser

CC = g++

all: trt_export

trt_export: trt_export.cpp
	$(CC) trt_export.cpp -o trt_export $(INCLUDE) $(LDFLAG) -lz

clean:
	rm -rf ./trt_export ./*.plan
