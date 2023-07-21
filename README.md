# deepstream_faster_rcnn

# 项目介绍
  在测试mmdeploy转换模型为tensorrt engine过程中，尝试将生成的faster r-cnn engine使用deepstream部署

# 环境依赖
- tensorrt 8.2.3
- deepstream 6.0
- cuda 11.4
- cudnn 8
- mmdeploy 1.2

# 运行过程
  1. 使用mmdeploy转换faster r-cnn模型生成static的tensorrt engine
  2. 将engine文件和mmdeploy实现的tensorrt op的动态库拷贝到当前项目中
  3. 编译程序
  4. 运行程序
## 编译程序
编译后处理源文件
```
cd nvinfer_faster_rcnn_impl
make
```
编译完成之后会生成nvinfer_faster_rcnn_impl/libfaster_rcnn.so动态库文件

## 运行程序
运行程序之前需要需要加载 tensorrt op的动态库，才能正确加载运行tensorrt engine, 执行如下命令即可。
```
LD_PRELOAD=./nvinfer_faster_rcnn_impl/libmmdeploy_tensorrt_ops.so deepstream-app -c ds_faster_rcnn.cfg
```
运行完成之后，会在当前文件夹下生成out-source0.mp4结果文件
