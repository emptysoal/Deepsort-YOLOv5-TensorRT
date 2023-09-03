CUDA_PATH       = /usr/local/cuda
TRT_INC_PATH    = /usr/include/aarch64-linux-gnu
TRT_LIB_PATH    = /usr/lib/aarch64-linux-gnu
OPENCV_INC_PATH = /usr/include/opencv4
OPENCV_LIB_PATH = /usr/lib/aarch64-linux-gnu
INCLUDE         = -I$(CUDA_PATH)/include -I$(TRT_INC_PATH) -I$(OPENCV_INC_PATH) -I./include
LDFLAG          = -L$(CUDA_PATH)/lib64 -lcudart -L$(TRT_LIB_PATH) -lnvinfer
LDFLAG         += -L$(OPENCV_LIB_PATH) -lopencv_core -lopencv_imgcodecs -lopencv_imgproc

CC = nvcc

all: libyolo_infer.so main

libyolo_infer.so: ./src/yolo_lib.cpp ./src/preprocess.cu ./src/yololayer.cu
	mkdir lib
	$(CC) -std=c++11 -Xcompiler -fPIC -shared -o lib/libyolo_infer.so ./src/yolo_lib.cpp ./src/preprocess.cu ./src/yololayer.cu $(INCLUDE) $(LDFLAG) -lz

main: main.cpp libyolo_infer.so
	g++ -o main main.cpp -L./lib -lyolo_infer $(INCLUDE) $(LDFLAG)

clean:
	rm -rf ./main ./lib ./*.jpg ./*.jpeg
