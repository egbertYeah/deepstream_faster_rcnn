CUDA_VER=11.4
ifeq ($(CUDA_VER),)
  $(error "CUDA_VER is not set")
endif
CC:= g++
NVCC:=/usr/local/cuda-$(CUDA_VER)/bin/nvcc
NVDS_VERSION:=6.0
CFLAGS:= -Wall -std=c++11 -shared -fPIC -Wno-error=deprecated-declarations

CFLAGS+= -I/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/sources/includes \
		 -I/usr/local/cuda-$(CUDA_VER)/include \
		 -I/usr/local/cuda-$(CUDA_VER)/targets/x86_64-linux/include -g -O0

LIBS:= -L/home/yangye/local/TensorRT-8.2.4.2/lib -lnvinfer_plugin -lnvinfer -lnvparsers \
		-L/usr/local/cuda-$(CUDA_VER)/lib64 -lcudart -lcublas -lstdc++fs

PLUGIN_LIB_DIR := /home/yangye/workspace/deepstream_faster_rcnn/nvinfer_faster_rcnn_impl
LIBS += -L$(PLUGIN_LIB_DIR) -lmmdeploy_tensorrt_ops -Wl,-rpath,$(PLUGIN_LIB_DIR)


LFLAGS:= -shared -Wl,--start-group $(LIBS) -Wl,--end-group

INCS:= $(wildcard *.h)
SRCFILES:= nvdsparsebbox.cpp

TARGET_LIB:= libfaster_rcnn.so

TARGET_OBJS:= $(SRCFILES:.cpp=.o)
TARGET_OBJS:= $(TARGET_OBJS:.cu=.o)

all: $(TARGET_LIB)

%.o: %.cpp $(INCS) Makefile
	$(CC) -c -o $@ $(CFLAGS) $<

%.o: %.cu $(INCS) Makefile
	$(NVCC) -c -o $@ --compiler-options '-fPIC' $<

$(TARGET_LIB) : $(TARGET_OBJS)
	$(CC) -o $@  $(TARGET_OBJS) $(LFLAGS)

clean:
	rm -rf $(TARGET_LIB)
	rm -rf $(TARGET_OBJS)
