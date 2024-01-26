oneflow-lite是一个轻量化的模型部署框架，支持在MCU、移动端、边缘端和服务端等各种应用场景下执行高效的推理任务。

oneflow-lite由compiler和runtime两部分组成。compiler负责对oneflow训练保存的checkpoint进行编译优化，最终转换成oneflow-lite runtime所支持的二进制文件格式，runtime提供该二进制文件的运行时环境，驱动硬件执行计算过程，并向上提供合适的用户接口。

Compiler支持对计算图进行一序列优化，包括计算融合、冗余消除、Layout变换等中端图优化，也包括内存规划、算子放置、计算图切分和后端代码生成等优化。

Runtime由纯C语言开发，核心框架二进制大小只有50KB左右，包含硬件抽象层、执行的上下文和支撑库，支持以非常简洁的方式集成各类硬件模块，比如X86 CPU、英伟达CUDA GPU、华为Ascend NPU等。



### 编译

- Compiler

  1. 下载/更新子模块

     ```shell
     git submodule init && git submodule update
     ```

  2. 编译

     - 目标硬件为CUDA

       ```shell
       cd compiler && mkdir build
       cmake .. -DLITE_USE_CUDA=ON && make
       ```

     - 目标硬件为Ascend NPU

       安装华为Ascend Toolkit，到[华为Ascend网站](https://www.hiascend.com/software/cann/commercial)上下载Ascend-cann-toolkit安装包，完成安装之后执行下面的命令更新一下环境变量。

       ```shell
       source /Ascend-Toolkit-Install-Path/ascend-toolkit/set_env.sh
       ```

       接着执行下面命令完成编译。

       ```shell
       cd compiler && mkdir build && cd build
       cmake .. -DLITE_USE_ASCEND_NPU=ON && make
       ```

  3. 检查编译结果

     在bin目录下检查是否生成了编译工具oneflow-lite-compile。

  注意：Compiler组件不需要在目标硬件平台上就可以编译。

- Runtime

  - 在目标硬件平台上编译

    比如目标平台为X86 CPU + Ascend NPU(同样适用于华为昇腾910)，编译命令如下，

    ```shell
    cd runtime && mkdir build
    cmake .. -DBUILD_X86=ON -DBUILD_ASCEND_NPU=ON
    ```

  - 交叉编译

    暂时不支持

### 使用教程

- 优化和转换模型

  1. 使用graph模式导出mlir模型

      ```shell
      pip install flowvision
      ```
     
     ```python
        from flowvision.models import ModelCreator
        
        model=ModelCreator.create_model("resnet50",pretrained=True)

        class MyGraph(nn.Graph):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def build(self, *input):
                return self.model(*input)

        if __name__ == "__main__":
             
            model=ModelCreator.create_model("resnet50",pretrained=True)
            model.eval()
            graph = MyGraph(model)
            input = flow.rand(1,3,224,224)
            outs = graph(input)
            flow.save(graph, "./esnet50_model/")
      ```

  2. 使用`oneflow-lite-compile`工具将模型编译成`oneflow-lite`的模型文件格式（以华为Ascend NPU为例）

     ```shell
     oneflow-lite/compiler/build/bin/oneflow-lite-compile ./resnet50_model -o resnet50_lite.bin --targets=ascend
     ```

- 在目标硬件平台上执行（以华为Ascend NPU为例）

  ```shell
  source /Ascend-Toolkit-Install-Path/ascend-toolkit/set_env.sh
  oneflow-lite/runtime/build/oneflow-lite/tests/test-resnet50 resnet50_lite.bin
  ```
