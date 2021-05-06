# pix2pix 
## 说明
pytorch复现《Image-to-Image Translation with Conditional Adversarial Networks》
[Arxiv](https://arxiv.org/abs/1611.07004)  
这里大量的代码是拷贝自https://hub.fastgit.org/junyanz/pytorch-CycleGAN-and-pix2pix
。这里用yaml配置工程，同时使用tensorboard展现训练过程
## 使用
以maps数据集为例
```python
python datasetPrepare\donwloadDataset.py --name maps
```
