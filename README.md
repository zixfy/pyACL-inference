# pyACL-inference

开坑

当前系统中华为.om模型在python中的推理是在华为异步推理的官方样例代码上改的

为了

1.去除官方代码冗余 

2.统一暴露对外的接口，接受任意batchsize数据，方便调用处进行并行推理

对这部分代码重构一下，并和.onnx对比精度

\* pyACL doc: [link](https://support.huawei.com/enterprise/zh/doc/EDOC1100164876/5bda6391)

\* 注意pyacl直接对ndarray底层数组进行操作
