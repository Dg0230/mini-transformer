# Mini Transformer

一个从零实现的 Rust Transformer 模型，专注于学习和理解 Transformer 架构。

## 项目概述

这是一个教育性的 Transformer 实现，使用纯 Rust 和 ndarray 库构建。项目展示了 Transformer 的核心组件，每个组件都有详细的注释，适合学习原理。

## 架构

### Encoder 架构

```text
Input → Embedding → Positional Encoding →
    [Transformer Encoder Layer × N] → Output
    ├── Multi-Head Self-Attention
    ├── Add & Norm
    ├── Feed Forward Network
    └── Add & Norm
```

### Seq2Seq 架构

```text
Encoder → Decoder → Output
           ↓
    ├── Masked Self-Attention (因果掩码)
    ├── Cross-Attention (关注 Encoder)
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
├── autograd.rs         # 自动求导和梯度计算
├── loss.rs             # 损失函数
├── optimizer.rs        # 优化器（SGD、Adam、AdamW）
├── lr_scheduler.rs     # 学习率调度器
├── trainer.rs          # 训练器和数据加载
└── main.rs             # 示例程序

examples/
└── training.rs         # 训练示例
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

### 基础推理

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

### 情感分析训练

```bash
# 运行情感分析示例
cargo run --example sentiment_analysis
```

```rust
use mini_transformer::{
    TrainableTransformer, create_sentiment_dataset,
};

// 加载内置数据集
let dataset = create_sentiment_dataset();

// 创建模型
let mut model = TrainableTransformer::new(
    dataset.vocab.vocab_size(),  // 词汇表大小
    128,                          // 模型维度
    4,                            // 注意力头数
    2,                            # 层数
    256,                          # FFN 维度
    10,                           # 最大序列长度
    dataset.n_classes,            // 类别数
);

// 训练（带早停）
let (train_inputs, train_targets) = dataset.encode_train(10);
let (test_inputs, test_targets) = dataset.encode_test(10);

model.train(
    &train_inputs,
    &train_targets,
    &test_inputs,
    &test_targets,
    50,    // 最大 epochs
    0.01,  // 学习率
    10,    // 早停 patience
);
```

### 批处理训练

```bash
# 对比单样本 vs 批处理训练
cargo run --example batch_training
```

### 优化训练

```bash
# 使用学习率调度器
cargo run --example optimized_training
```

### 文本生成

```bash
# Seq2Seq 文本生成
cargo run --example text_generation
```

```rust
use mini_transformer::Seq2SeqTransformer;

// 创建 Seq2Seq 模型
let mut model = Seq2SeqTransformer::new(
    1000,  // vocab_size
    128,   // d_model
    4,     // n_heads
    2,     // n_layers
    256,   // d_ff
    50,    // max_seq_len
);

// 贪婪解码生成
let output = model.generate_greedy(
    &source,    // 源序列
    20,          // 最大生成长度
    0,           // 起始 token
);

// 束搜索生成
let output = model.generate_beam(
    &source,
    20,          // 最大生成长度
    3,           // 束宽度
    0,           // 起始 token
);
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

## 已实现功能 ✓

### 基础架构
- [x] Multi-Head Self-Attention
- [x] Feed-Forward Network
- [x] Layer Normalization
- [x] 位置编码
- [x] 残差连接

### 训练功能
- [x] 自动求导系统（框架）
- [x] 损失函数（交叉熵、MSE、BCE）
- [x] 优化器（SGD、Adam、AdamW）
- [x] 学习率调度器
- [x] 数据加载器（批处理、shuffle）
- [x] 训练循环框架
- [x] 评估功能

### 完整训练
- [x] **反向传播** - 支持端到端训练
- [x] **模型序列化** - 保存/加载检查点
- [x] **早停机制** - 防止过拟合
- [x] **文本数据集** - 情感分类数据集
- [x] **实战示例** - 情感分析训练
- [x] **批处理优化** - 3.82x 加速
- [x] **学习率调度** - 动态学习率调整

### Seq2Seq 功能 ✨ 新增
- [x] **Decoder 层** - 完整的 Decoder 实现
- [x] **Masked Attention** - 因果掩码自注意力
- [x] **Cross-Attention** - Encoder-Decoder 注意力
- [x] **贪婪解码** - 快速生成策略
- [x] **束搜索** - 高质量生成
- [x] **生成示例** - 文本生成演示

## 下一步扩展

### 方向 A: 改进训练效果
- [x] **批处理训练** - 已实现 3.82x 加速
- [x] **学习率调度** - 已集成到训练循环
- [ ] **更多数据** - 扩大数据集规模
- [ ] **数据增强** - 文本增强技术
- [ ] **梯度裁剪** - 防止梯度爆炸

### 方向 B: 架构扩展
- [x] **Decoder 层** - 已实现完整 Seq2Seq
- [ ] **更多注意力类型** - 实现其他注意力变体
- [ ] **预训练** - 实现 BERT 风格的预训练
- [ ] **Transformer-XL** - 长距离依赖

### 方向 C: 性能优化
- [ ] **GPU 支持** - 迁移到 candle 或 burn
- [ ] **混合精度** - 使用 f16 加速训练
- [ ] **模型量化** - 压缩模型大小

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
