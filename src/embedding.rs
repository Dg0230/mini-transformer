//! 嵌入层和位置编码
//!
//! 将离散的 token ID 转换为连续的向量表示，并添加位置信息。

use ndarray::Array2;
use crate::tensor::TensorExt;

/// 词嵌入层
///
/// 将 token ID 映射到 d_model 维向量
#[derive(Debug, Clone)]
pub struct Embedding {
    /// 权重矩阵: [vocab_size, d_model]
    weights: Array2<f32>,
    vocab_size: usize,
    d_model: usize,
}

impl Embedding {
    /// 创建新的嵌入层
    ///
    /// # 参数
    /// - `vocab_size`: 词表大小
    /// - `d_model`: 嵌入维度
    pub fn new(vocab_size: usize, d_model: usize) -> Self {
        // Xavier 初始化
        let weights = Array2::random_xavier((vocab_size, d_model));

        Self {
            weights,
            vocab_size,
            d_model,
        }
    }

    /// 前向传播
    ///
    /// # 输入
    /// - `input`: [seq_len] 或 [batch_size, seq_len]
    ///
    /// # 输出
    /// - [seq_len, d_model] 或 [batch_size, seq_len, d_model]
    pub fn forward(&self, input: &[usize]) -> Array2<f32> {
        let seq_len = input.len();

        // 查找每个 token 的嵌入向量
        let mut embedded = Array2::zeros((seq_len, self.d_model));

        for (i, &token_id) in input.iter().enumerate() {
            if token_id < self.vocab_size {
                embedded.row_mut(i).assign(&self.weights.row(token_id));
            } else {
                // 未知 token，使用零向量或随机向量
                embedded.row_mut(i).assign(&Array2::zeros((1, self.d_model)).row(0));
            }
        }

        embedded
    }

    /// 获取嵌入维度
    pub fn d_model(&self) -> usize {
        self.d_model
    }

    /// 获取权重（用于可视化或保存）
    pub fn weights(&self) -> &Array2<f32> {
        &self.weights
    }

    /// 设置权重（用于加载模型）
    pub fn set_weights(&mut self, weights: Array2<f32>) {
        self.weights = weights;
    }
}

/// 位置编码
///
/// 为输入序列添加位置信息，使用正弦和余弦函数。
///
/// 位置编码公式：
/// ```text
/// PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
/// PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
/// ```
#[derive(Debug, Clone)]
pub struct PositionalEncoding {
    /// 预计算的位置编码: [max_seq_len, d_model]
    pe: Array2<f32>,
    max_seq_len: usize,
    d_model: usize,
}

impl PositionalEncoding {
    /// 创建新的位置编码
    ///
    /// # 参数
    /// - `max_seq_len`: 最大序列长度
    /// - `d_model`: 嵌入维度
    pub fn new(max_seq_len: usize, d_model: usize) -> Self {
        let mut pe = Array2::zeros((max_seq_len, d_model));

        // 计算位置编码
        for pos in 0..max_seq_len {
            for i in (0..d_model).step_by(2) {
                // 计算分母: 10000^(2i/d_model)
                let div_term = (-2.0 * i as f32 / d_model as f32).exp();

                // 偶数维度: sin
                pe[[pos, i]] = ((pos as f32) * div_term).sin();

                // 奇数维度: cos (如果存在)
                if i + 1 < d_model {
                    pe[[pos, i + 1]] = ((pos as f32) * div_term).cos();
                }
            }
        }

        Self {
            pe,
            max_seq_len,
            d_model,
        }
    }

    /// 前向传播：将位置编码添加到输入嵌入
    ///
    /// # 输入
    /// - `x`: [seq_len, d_model] 嵌入向量
    ///
    /// # 输出
    /// - [seq_len, d_model] 添加位置编码后的向量
    pub fn forward(&self, x: &Array2<f32>) -> Array2<f32> {
        let seq_len = x.nrows();

        if seq_len > self.max_seq_len {
            panic!(
                "Sequence length {} exceeds maximum length {}",
                seq_len, self.max_seq_len
            );
        }

        // 取对应长度的位置编码并添加到输入
        let pe_slice = self.pe.slice(ndarray::s![..seq_len, ..]);
        x + &pe_slice
    }

    /// 获取位置编码（用于可视化）
    pub fn encoding(&self) -> &Array2<f32> {
        &self.pe
    }
}

/// 组合嵌入和位置编码
#[derive(Debug, Clone)]
pub struct InputEmbedding {
    embedding: Embedding,
    pos_encoding: PositionalEncoding,
}

impl InputEmbedding {
    /// 创建新的输入嵌入层
    ///
    /// # 参数
    /// - `vocab_size`: 词表大小
    /// - `d_model`: 嵌入维度
    /// - `max_seq_len`: 最大序列长度
    pub fn new(vocab_size: usize, d_model: usize, max_seq_len: usize) -> Self {
        let embedding = Embedding::new(vocab_size, d_model);
        let pos_encoding = PositionalEncoding::new(max_seq_len, d_model);

        Self {
            embedding,
            pos_encoding,
        }
    }

    /// 前向传播：嵌入 + 位置编码
    ///
    /// # 输入
    /// - `input`: [seq_len] token IDs
    ///
    /// # 输出
    /// - [seq_len, d_model] 包含位置信息的嵌入向量
    pub fn forward(&self, input: &[usize]) -> Array2<f32> {
        let embedded = self.embedding.forward(input);
        self.pos_encoding.forward(&embedded)
    }

    /// 获取嵌入维度
    pub fn d_model(&self) -> usize {
        self.embedding.d_model()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    #[test]
    fn test_embedding() {
        let embedding = Embedding::new(100, 64);
        let input = vec![0, 5, 10];
        let output = embedding.forward(&input);

        assert_eq!(output.shape(), &[3, 64]);
    }

    #[test]
    fn test_positional_encoding() {
        let pe = PositionalEncoding::new(10, 4);

        // 位置编码应该是有界的
        let encoding = pe.encoding();
        assert_eq!(encoding.shape(), &[10, 4]);

        // 验证第一个位置（应该接近 [0, 1, 0, 1]，因为 sin(0)=0, cos(0)=1）
        let first_pos = encoding.row(0);
        assert!(first_pos[0].abs() < 1e-6);  // sin(0) ≈ 0
        assert!((first_pos[1] - 1.0).abs() < 1e-6);  // cos(0) = 1
    }

    #[test]
    fn test_input_embedding() {
        let input_emb = InputEmbedding::new(100, 64, 128);
        let input = vec![0, 1, 2, 3];
        let output = input_emb.forward(&input);

        assert_eq!(output.shape(), &[4, 64]);

        // 验证位置信息已添加（不应该与纯嵌入相同）
        let pure_emb = input_emb.embedding.forward(&input);
        assert!(output != pure_emb);
    }
}
