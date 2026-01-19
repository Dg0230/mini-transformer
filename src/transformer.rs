//! 完整的 Transformer Encoder
//!
//! 组合所有组件，构建可用的 Transformer 模型。

use ndarray::Array2;
use crate::embedding::InputEmbedding;
use crate::attention::MultiHeadAttention;
use crate::layers::{FeedForward, ResidualNorm};
use crate::tensor::TensorExt;

/// Transformer 配置
#[derive(Debug, Clone, Copy)]
pub struct TransformerConfig {
    /// 词表大小
    pub vocab_size: usize,
    /// 模型维度
    pub d_model: usize,
    /// 注意力头数
    pub n_heads: usize,
    /// Encoder 层数
    pub n_layers: usize,
    /// FFN 隐藏层维度
    pub d_ff: usize,
    /// 最大序列长度
    pub max_seq_len: usize,
    /// Dropout 比率
    pub dropout: f32,
}

impl Default for TransformerConfig {
    fn default() -> Self {
        Self {
            vocab_size: 10000,
            d_model: 512,
            n_heads: 8,
            n_layers: 6,
            d_ff: 2048,
            max_seq_len: 512,
            dropout: 0.1,
        }
    }
}

/// 单个 Transformer Encoder 层
///
/// 包含两个子层，每个子层都有残差连接和层归一化：
/// 1. Multi-Head Self-Attention
/// 2. Position-wise Feed-Forward Network
///
/// ```text
/// x → LayerNorm(x + Attention(x)) → LayerNorm(x + FFN(x)) → output
/// ```
#[derive(Debug, Clone)]
pub struct TransformerEncoderLayer {
    /// Multi-Head Self-Attention
    self_attention: MultiHeadAttention,
    /// Feed-Forward Network
    feed_forward: FeedForward,
    /// Attention 后的 Residual + Norm
    attn_norm: ResidualNorm,
    /// FFN 后的 Residual + Norm
    ffn_norm: ResidualNorm,
    /// Dropout
    dropout: f32,
}

impl TransformerEncoderLayer {
    /// 创建新的 Encoder Layer
    pub fn new(d_model: usize, n_heads: usize, d_ff: usize, dropout: f32) -> Self {
        Self {
            self_attention: MultiHeadAttention::new(d_model, n_heads),
            feed_forward: FeedForward::new(d_model, d_ff),
            attn_norm: ResidualNorm::new(d_model),
            ffn_norm: ResidualNorm::new(d_model),
            dropout,
        }
    }

    /// 前向传播
    ///
    /// # 参数
    /// - `x`: 输入 [seq_len, d_model]
    /// - `mask`: 可选的注意力掩码 [seq_len, seq_len]
    ///
    /// # 返回
    /// - [seq_len, d_model]
    pub fn forward(&self, x: &Array2<f32>, mask: Option<&Array2<f32>>) -> Array2<f32> {
        // 1. Multi-Head Self-Attention + Residual + Norm
        let (attn_output, _) = self.self_attention.forward(x, mask);
        let attn_with_residual = self.attn_norm.forward(x, &attn_output);

        // 2. Feed-Forward + Residual + Norm
        let ffn_output = self.feed_forward.forward(&attn_with_residual);
        let output = self.ffn_norm.forward(&attn_with_residual, &ffn_output);

        output
    }

    /// 获取层的参数数量
    pub fn param_count(&self) -> usize {
        let (w_q, w_k, w_v, w_o) = self.self_attention.weights();
        let (w1, w2, b1, b2) = self.feed_forward.params();

        w_q.len() + w_k.len() + w_v.len() + w_o.len()
            + w1.len() + w2.len() + b1.len() + b2.len()
    }
}

/// 完整的 Transformer Encoder
///
/// ```text
/// Input → Embedding → Positional Encoding →
///     [Encoder Layer × N] → Output
/// ```
///
/// 这个模型可以用于：
/// - 序列分类
/// - 语义理解
/// - 特征提取
#[derive(Debug, Clone)]
pub struct TransformerEncoder {
    /// 输入嵌入层
    embedding: InputEmbedding,
    /// Encoder 层堆叠
    layers: Vec<TransformerEncoderLayer>,
    /// 配置
    config: TransformerConfig,
}

impl TransformerEncoder {
    /// 创建新的 Transformer Encoder
    ///
    /// # 参数
    /// - `config`: 模型配置
    pub fn new(config: TransformerConfig) -> Self {
        // 验证配置
        assert_eq!(
            config.d_model % config.n_heads,
            0,
            "d_model must be divisible by n_heads"
        );

        // 创建嵌入层
        let embedding = InputEmbedding::new(
            config.vocab_size,
            config.d_model,
            config.max_seq_len,
        );

        // 创建 Encoder 层
        let layers = (0..config.n_layers)
            .map(|_| {
                TransformerEncoderLayer::new(
                    config.d_model,
                    config.n_heads,
                    config.d_ff,
                    config.dropout,
                )
            })
            .collect();

        Self {
            embedding,
            layers,
            config,
        }
    }

    /// 前向传播
    ///
    /// # 参数
    /// - `input`: 输入 token IDs [seq_len]
    /// - `mask`: 可选的注意力掩码 [seq_len, seq_len]
    ///
    /// # 返回
    /// - [seq_len, d_model] 输出表示
    pub fn forward(&self, input: &[usize], mask: Option<&Array2<f32>>) -> Array2<f32> {
        // 1. 嵌入 + 位置编码
        let mut x = self.embedding.forward(input);

        // 2. 通过所有 Encoder 层
        for layer in &self.layers {
            x = layer.forward(&x, mask);
        }

        x
    }

    /// 获取配置
    pub fn config(&self) -> &TransformerConfig {
        &self.config
    }

    /// 获取总参数数量
    pub fn param_count(&self) -> usize {
        let embed_params = self.config.vocab_size * self.config.d_model
            + self.config.max_seq_len * self.config.d_model;

        let layer_params = self.layers
            .iter()
            .map(|l| l.param_count())
            .sum::<usize>();

        embed_params + layer_params
    }

    /// 获取模型信息
    pub fn info(&self) -> String {
        format!(
            "TransformerEncoder(
  vocab_size: {},
  d_model: {},
  n_heads: {},
  n_layers: {},
  d_ff: {},
  max_seq_len: {},
  dropout: {:.2},
  total_params: {},
)",
            self.config.vocab_size,
            self.config.d_model,
            self.config.n_heads,
            self.config.n_layers,
            self.config.d_ff,
            self.config.max_seq_len,
            self.config.dropout,
            self.param_count(),
        )
    }

    /// 保存模型（简化版本）
    pub fn save(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        // 简化实现：只保存配置
        // 完整实现需要序列化所有权重
        use std::fs::File;
        use std::io::Write;

        let mut file = File::create(path)?;
        writeln!(file, "{}", self.info())?;
        writeln!(file, "Model serialized successfully")?;

        Ok(())
    }
}

/// 分类头（可选组件）
///
/// 将 Transformer 输出转换为类别预测
#[derive(Debug, Clone)]
pub struct ClassificationHead {
    /// 投影层: [d_model, n_classes]
    projection: Array2<f32>,
    n_classes: usize,
    d_model: usize,
}

impl ClassificationHead {
    pub fn new(d_model: usize, n_classes: usize) -> Self {
        Self {
            projection: Array2::random_xavier((d_model, n_classes)),
            n_classes,
            d_model,
        }
    }

    /// 前向传播：使用第一个 token 的表示进行分类
    ///
    /// # 参数
    /// - `x`: [seq_len, d_model] Transformer 输出
    ///
    /// # 返回
    /// - [1, n_classes] 类别 logits
    pub fn forward(&self, x: &Array2<f32>) -> Array2<f32> {
        // 使用第一个 token（通常是 [CLS] token）
        let cls_token = x.row(0).to_owned();
        let cls_2d = Array2::from_shape_vec((1, self.d_model), cls_token.to_vec()).unwrap();

        // 投影到类别空间
        cls_2d.matmul(&self.projection)
    }
}

/// 完整的分类模型：Transformer + 分类头
#[derive(Debug, Clone)]
pub struct TransformerClassifier {
    encoder: TransformerEncoder,
    classifier: ClassificationHead,
}

impl TransformerClassifier {
    pub fn new(config: TransformerConfig, n_classes: usize) -> Self {
        let encoder = TransformerEncoder::new(config);
        let classifier = ClassificationHead::new(config.d_model, n_classes);

        Self {
            encoder,
            classifier,
        }
    }

    pub fn forward(&self, input: &[usize]) -> Array2<f32> {
        let encoded = self.encoder.forward(input, None);
        self.classifier.forward(&encoded)
    }

    pub fn predict(&self, input: &[usize]) -> usize {
        let logits = self.forward(input);
        let probs = logits.softmax(1);

        // 找到最大概率的类别
        let mut max_class = 0;
        let mut max_prob = f32::NEG_INFINITY;

        for (i, &p) in probs.iter().enumerate() {
            if p > max_prob {
                max_prob = p;
                max_class = i;
            }
        }

        max_class
    }

    /// 获取参数总数
    pub fn param_count(&self) -> usize {
        self.encoder.param_count() + self.classifier.projection.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = TransformerConfig::default();
        assert_eq!(config.d_model, 512);
        assert_eq!(config.n_heads, 8);
        assert_eq!(config.n_layers, 6);
    }

    #[test]
    fn test_encoder_layer() {
        let layer = TransformerEncoderLayer::new(128, 4, 256, 0.1);
        let x = Array2::random_xavier((10, 128));

        let y = layer.forward(&x, None);

        assert_eq!(y.shape(), &[10, 128]);
    }

    #[test]
    fn test_transformer_encoder() {
        let config = TransformerConfig {
            vocab_size: 100,
            d_model: 64,
            n_heads: 4,
            n_layers: 2,
            d_ff: 128,
            max_seq_len: 32,
            dropout: 0.1,
        };

        let encoder = TransformerEncoder::new(config);
        let input = vec![0, 1, 2, 3, 4];

        let output = encoder.forward(&input, None);

        assert_eq!(output.shape(), &[5, 64]);
    }

    #[test]
    fn test_classifier() {
        let config = TransformerConfig {
            vocab_size: 100,
            d_model: 64,
            n_heads: 4,
            n_layers: 2,
            d_ff: 128,
            max_seq_len: 32,
            dropout: 0.1,
        };

        let model = TransformerClassifier::new(config, 10);
        let input = vec![0, 1, 2, 3, 4];

        let output = model.forward(&input);

        assert_eq!(output.shape(), &[1, 10]);

        let class = model.predict(&input);
        assert!(class < 10);
    }

    #[test]
    fn test_param_count() {
        let config = TransformerConfig {
            vocab_size: 1000,
            d_model: 128,
            n_heads: 4,
            n_layers: 2,
            d_ff: 256,
            max_seq_len: 64,
            dropout: 0.1,
        };

        let encoder = TransformerEncoder::new(config);

        // 确保参数数量合理
        let params = encoder.param_count();
        assert!(params > 0);
        assert!(params < 1_000_000);  // 小模型
    }
}
