# Deepsort-YOLOv5-TensorRT

## 项目简介

- 基于 `TensorRT-v8.2` ，加速`YOLOv5-v5.0` + `deepsort` 的目标跟踪；
- 在 `Jetson nano` 上进行部署；
- 在 `Linux x86_64` 系统上也是可行的，需要修改本项目中所有的 `Makefile` 和 `CMakeLists.txt` ，把这 2 种文件里 `TensorRT` 和 `OpenCV` 头文件、库文件的路径换成对应系统的即可，`CUDA`的路径一般不用修改，最好也确认下。

## 项目效果

- 下面是从一段测试视频当中选取的两帧跟踪的结果

![result_01](samples/result_01.jpg)

![result_02](samples/result_02.jpg)

## 环境配置

-  `Jetson nano` 上烧录的系统镜像为 `Jetpack 4.6.1`，该`jetpack` 原装环境如下：

| CUDA | cuDNN | TensorRT | OpenCV |
| ---- | ----- | -------- | ------ |
| 10.2 | 8.2   | 8.2.1    | 4.1.1  |

关于如何在 `Jetson nano` 上烧录镜像，网上资料还是很多的，这里就不赘述了，注意下载 `Jetpack`镜像时选择 4.6.1 版本，该版本对应的 TensorRT v8 版本

- 安装`Eigen`库

```bash
apt install libeigen3-dev
```

## 模型转换

把 `YOLO`检测模型，和`ReID`特征提取模型，转换成`TensorRT`的序列化文件，后缀 `.plan`（作者的习惯，也可以是`.engine`或其他）

### 原模型下载

- 链接：https://pan.baidu.com/s/1YG-A8dXL4zWvecsD6mW2ug 
- 提取码：y2oz

下载并解压后，

目录中的文件说明如下：

```bash
Deepsort模型文件
    |
    ├── ReID模型  # 该目录中存放的是 ReID 特征提取网络的模型
    │   ├── ckpt.t7  # 官方 PyTorch 格式的模型文件
    │   └── deepsort.onnx  # 根据 ckpt.t7 导出的 onnx 格式模型文件
    |
    └── YOLOv5-v5.0  # 该目录中存放的是 YOLOv5 目标检测网络的模型
        ├── yolov5s.pt  # 官方 PyTorch 格式的模型文件
        └── para.wts  # 根据 yolov5s.pt 导出的 wts 格式模型文件
```

### YOLO模型转换

- 将上述 `yolov5s.pt` 转为 `model.plan`，或 `para.wts`转为 `model.plan`
- 具体转换方法参考下面链接，也是作者自己发布的一个项目

https://github.com/emptysoal/TensorRT-v8-YOLOv5-v5.0/tree/main

**注意**：使用时，记得把 `Makefile` 中`TensorRT` 和 `OpenCV` 头文件、库文件所在的路径，换成自己 `Jetson nano` 上的

完成之后便可得到 `model.plan` ，为检测网络的 `TensorRT` 序列化模型文件。

### ReID网络模型转换

- 将上述 `ckpt.t7` 转为 `deepsort.plan`，或 `deepsort.onnx` 转为 `deepsort.plan`
- 按如下步骤运行

```bash
# 进入本项目的 reid_torch2trt 目录
cd reid_torch2trt

# 执行命令
# 如果jetson nano上没有安装pytorch，这一步随便在任何有pytorch上的环境运行都可以
python onnx_export.py
# 此命令运行后，ckpt.t7 转为 deepsort.onnx
# 不过作者已给出 deepsort.onnx，因此可以跳过这步，从下面的开始

# 依次执行
make
./trt_export
# 此命令运行后，deepsort.onnx 转为 deepsort.plan
```

完成之后便可得到 `deepsort.plan` ，为特征提取网络的 `TensorRT` 序列化模型文件。

## 运行项目

- 开始编译并运行目标跟踪的代码
- 按如下步骤运行

```bash
# 进入本项目的 yolov5-deepsort-tensorrt 目录
cd yolov5-deepsort-tensorrt
mkdir resources
# 把上面转换得到的 2 个 plan 文件复制到目录resources中
cp {TensorRT-v8-YOLOv5-v5.0}/model.plan ./resources
cp ../reid_torch2trt/deepsort.plan ./resources

mkdir test_videos  # 向其中放入测试的视频文件，并命名为 demo.mp4

vim src/main.cpp  # 可以根据自己的需求修改一些配置信息

mkdir build
cd build
cmake ..
make
./yolosort  # 运行后即看到跟踪效果
```

## 项目参考

### 参考链接

主要参考了下面的项目：

- https://github.com/RichardoMrMu/yolov5-deepsort-tensorrt
- https://blog.csdn.net/weixin_42264234/article/details/120152117

### 本项目改进

- 本项目的代码大部分参考了上方项目的内容
- **但作者自己也做了不少变更**，具体如下：

1. 所参考项目`ReID`网络模型输出为 `512` 维度的向量，但代码中是 `256` 维，作者做了修正；

2. 对于`YOLOv5-v5.0`目标检测网络：

   2.1 TensorRT 模型转换使用的是作者自己的项目，进行了**预处理的 CUDA 加速**；

   2.2 对 YOLOv5 的 TensorRT 推理做了**类的封装**，并编译为动态链接库，在其他项目中使用也非常方便；

3. 实现了类别筛选功能，可以在 main.cpp 中设置想要跟踪的类别，忽略掉那些不关心的种类；

4. 为了适配 TensorRT 8 版本，对模型推理的地方做了一定的改动。

