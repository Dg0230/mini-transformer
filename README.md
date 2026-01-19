# Mini Transformer

一个从零实现的 Rust Transformer 模型，专注于学习和理解 Transformer 架构。

## 项目概述

这是一个教育性的 Transformer 实现，使用纯 Rust 和 ndarray 库构建。项目展示了 Transformer 的核心组件，每个组件都有详细的注释，适合学习原理。

## 架构

```text
Input → Embedding → Positional Encoding →
    [Transformer Encoder Layer × N] → Output
    ├── Multi-Head Self-Attention
    ├── Add & Norm
    ├── Feed Forward Network
    └── Add & Norm
```

## 项目结构

```
src/
├── lib.rs              # 模块导出和预设配置
├── tensor.rs           # 张量操作扩展
├── embedding.rs        # 词嵌入和位置编码
├── attention.rs        # Multi-Head Self-Attention
├── layers.rs           # FFN 和 Layer Norm
├── transformer.rs      # 完整 Transformer 模型
└── main.rs             # 示例程序
```

## 核心组件

### 1. Multi-Head Self-Attention
- Query, Key, Value 计算
- 缩放点积注意力
- 多头并行处理

### 2. Feed-Forward Network
- 两层全连接网络
- GELU 激活函数

### 3. Layer Normalization
- 特征维度归一化
- 可学习的缩放和平移参数

### 4. 位置编码
- 正弦/余弦位置编码
- 允许模型学习序列顺序

## 快速开始

### 安装依赖

```bash
cd mini-transformer
cargo build
```

### 运行示例

```bash
cargo run
```

### 运行测试

```bash
cargo test
```

## 使用示例

```rust
use mini_transformer::{
    TransformerEncoder, TransformerClassifier, TransformerConfig, configs,
};

// 使用预设配置
let config = configs::mini();  // 或 configs::base()
let encoder = TransformerEncoder::new(config);

// 前向传播
let input = vec![5, 12, 23, 45, 67];  // token IDs
let output = encoder.forward(&input, None);

// 分类任务
let classifier = TransformerClassifier::new(config, 5);  // 5 个类别
let class = classifier.predict(&input);
```

## 模型配置

### Mini 配置（快速测试）
- vocab_size: 1000
- d_model: 128
- n_heads: 4
- n_layers: 2
- 参数量: ~530K

### Base 配置（平衡性能）
- vocab_size: 10000
- d_model: 512
- n_heads: 8
- n_layers: 6
- 参数量: ~24M

## 自定义配置

```rust
let config = TransformerConfig {
    vocab_size: 5000,
    d_model: 256,
    n_heads: 8,
    n_layers: 4,
    d_ff: 512,
    max_seq_len: 128,
    dropout: 0.1,
};
```

## 学习路径

1. **tensor.rs**: 理解张量操作和数学函数
2. **embedding.rs**: 学习词嵌入和位置编码
3. **attention.rs**: 深入注意力机制（核心）
4. **layers.rs**: 理解 FFN 和层归一化
5. **transformer.rs**: 查看完整架构组装

## 关键概念

### Multi-Head Attention
```
Attention(Q, K, V) = softmax(QK^T / √d_k) × V
```

### Layer Normalization
```
y = γ × ((x - μ) / √(σ² + ε)) + β
```

### GELU Activation
```
GELU(x) ≈ 0.5 × x × (1 + tanh(√(2/π) × (x + 0.044715 × x³)))
```

## 下一步扩展

- [ ] 实现训练循环和反向传播
- [ ] 添加优化器（Adam、SGD）
- [ ] 实现损失函数（交叉熵）
- [ ] 添加 Decoder 层（实现完整 Seq2Seq）
- [ ] 支持 GPU 加速（通过 candle 或 tch-rs）
- [ ] 添加模型保存/加载功能

## 技术栈

- **Rust** 2021 Edition
- **ndarray** - 张量操作
- **num-traits** - 数值类型 traits

## 参考资料

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - 原始 Transformer 论文
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) - 可视化解释
- [Harvard NLP - The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html) - 注释版实现

## License

MIT License - 仅用于学习目的
