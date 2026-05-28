# Deep Learning Notes

## Chapter 1: 深度学习简介

- 1.1 神经网络：一个可学习的函数
- 1.3 前向传播、反向传播与计算图

## Chapter 2: PyTorch 入门

- 2.1 PyTorch 中的自动微分
- 2.2 PyTorch 中的梯度记录与控制
- 2.3 PyTorch 中的 Dataset 与 DataLoader
- 2.4 PyTorch 中的 nn.Module
- 2.5 PyTorch 中的优化器
- 2.6 PyTorch 中的训练循环

## Chapter 8: Attention 与 Transformer：从动态检索到序列建模

- 8.1 Bahdanau Attention：从信息压缩到动态检索
- 8.2 Cross-Attention：一个序列查询另一个序列
- 8.3 Self-Attention：序列内部的信息交互
- 8.4 Multi-Head Attention：从单一视角到多重视角
- 8.5 位置编码：给 Attention 补上位置信息
- 8.6 Transformer Encoder：把 Self-Attention 堆起来
- 8.7 Transformer Decoder：Masked Self-Attention 与 Cross-Attention
- 8.8 Encoder-Decoder Transformer：把 Encoder 和 Decoder 连接起来
- 8.9 KV Cache：为什么推理时不用重复算过去
- 8.10 Transformer 的三种不同架构：理解、生成与输入输出转换
- 8.11 Hugging Face Transformers API：从结构到调用

## Chapter 10: 高效 Attention 实现：从 Memory-Efficient Attention 到 FlashAttention

- 10.1 为什么 Attention 是 IO-Bound
- 10.2 Flash Attention v1：消除 Attention 的 IO 瓶颈

## Chapter 11: Vision Transformer：从图像分类到视觉序列建模

- 11.1 从 CNN 到 Vision Transformer：把图像当成序列
- 11.2 Patch Embedding：把图像切成 Token
- 11.3 Class Token 与 Positional Embedding：让序列表示整张图
- 11.4 ViT Encoder：让 Patch Token 之间交换信息
- 11.5 ViT Backbone：预训练与微调

## Chapter 12: GAN：在对抗中学习生成

- 12.1 GAN 基础：生成对抗网络的核心思想与训练流程

## Chapter 13: VAE：从压缩重建到概率生成

- 13.1 AutoEncoder：从压缩与重建开始
- 13.2 VAE：概率建模与重参数化技巧
- 13.3 ELBO：VAE 的目标函数从哪里来
- 13.4 VAE 的训练现象与潜空间直觉
- 13.5 VAE 的优点、局限与后续发展

## Chapter 14: Diffusion Models：从去噪到生成

- 14.1 DDPM：从去噪到生成
- 14.2 DDPM 的前向加噪过程
- 14.3 DDPM 的反向去噪过程与训练目标
- 14.4 DDPM 的网络结构与采样过程
- 14.5 从变分推导看 DDPM：ELBO 从哪里来

## Chapter 15: 视觉语言模型：从图文对齐到多模态对话

- 15.1 CLIP：把图像和文本映射到同一个语义空间
