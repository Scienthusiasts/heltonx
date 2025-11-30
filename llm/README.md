<div align='center'>
    <h1>HeltonX♾️</h1>
    <img src="https://github.com/Scienthusiasts/heltonx/blob/main/demo/logo.png"/>
    <p><em>致力于 0~1 实现通用深度学习框架，基于 Pytorch，支持各类下游任务，不断完善中 ~</em></p>
</div>


## ✒️`llm` 设计逻辑

```
detection:
├─configs  (自定义模型配置参数文件)
├─datasets (自定义Dataset和数据增强)
├─losses   (自定义损失函数)
├─models         (自定义网络组件)
│  ├─base_model_configs (基础模型的配置超参数, 继承transformer.PretrainedConfig)
│  ├─base_model         (基础模型包含的各个模块)
│  └─training_model     (在基础模型之上，套壳各种训练方法(pretrain, sft, rl等))
├─tokenizer_configs (各种模型的分词器的配置)
├─demos    (测试用图像)
└─tools    (同pretrain)
```
