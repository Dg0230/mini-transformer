//! Multi-Head Self-Attention
//!
//! Transformer 的核心组件：允许模型关注输入序列的不同位置。

use ndarray::Array2;
use crate::tensor::TensorExt;

/// Multi-Head Attention 参数
#[derive(Debug, Clone)]
pub struct AttentionParams {
    /// 模型维度
    pub d_model: usize,
    /// 注意力头数
    pub n_heads: usize,
    /// 每个头的维度
    pub d_k: usize,
}

impl AttentionParams {
    pub fn new(d_model: usize, n_heads: usize) -> Self {
        assert_eq!(
            d_model % n_heads, 0,
            "d_model must be divisible by n_heads"
        );

        let d_k = d_model / n_heads;

        Self {
            d_model,
            n_heads,
            d_k,
        }
    }
}

/// Multi-Head Self-Attention 层
///
/// 核心思想：将注意力分成多个"头"，每个头学习不同的表示子空间。
///
/// ```text
/// Input → [Q, K, V] → Split into Heads →
///     [Scaled Dot-Product Attention × N] →
///     Concat Heads → Linear → Output
/// ```
#[derive(Debug, Clone)]
pub struct MultiHeadAttention {
    params: AttentionParams,
    /// Query 投影: [d_model, d_model]
    w_q: Array2<f32>,
    /// Key 投影: [d_model, d_model]
    w_k: Array2<f32>,
    /// Value 投影: [d_model, d_model]
    w_v: Array2<f32>,
    /// 输出投影: [d_model, d_model]
    w_o: Array2<f32>,
}

impl MultiHeadAttention {
    /// 创建新的 Multi-Head Attention 层
    ///
    /// # 参数
    /// - `d_model`: 模型维度
    /// - `n_heads`: 注意力头数
    pub fn new(d_model: usize, n_heads: usize) -> Self {
        let params = AttentionParams::new(d_model, n_heads);

        Self {
            params: params.clone(),
            w_q: Array2::random_xavier((d_model, d_model)),
            w_k: Array2::random_xavier((d_model, d_model)),
            w_v: Array2::random_xavier((d_model, d_model)),
            w_o: Array2::random_xavier((d_model, d_model)),
        }
    }

    /// 缩放点积注意力
    ///
    /// ```text
    /// Attention(Q, K, V) = softmax(QK^T / √d_k) * V
    /// ```
    ///
    /// # 参数
    /// - `q`: Query [seq_len, d_model]
    /// - `k`: Key [seq_len, d_model]
    /// - `v`: Value [seq_len, d_model]
    ///
    /// # 返回
    /// - 注意力输出 [seq_len, d_model]
    /// - 注意力权重 [seq_len, seq_len]（用于可视化）
    fn scaled_dot_product_attention(
        &self,
        q: &Array2<f32>,
        k: &Array2<f32>,
        v: &Array2<f32>,
    ) -> (Array2<f32>, Array2<f32>) {
        let seq_len = q.nrows();
        let scale = (self.params.d_k as f32).sqrt();

        // 计算注意力分数: Q * K^T / √d_k
        let k_t = k.t();
        let k_t_owned = k_t.to_owned();
        let scores = q.matmul(&k_t_owned);
        let scaled_scores = scores.mapv(|x| x / scale);

        // Softmax 归一化
        let attention_weights = scaled_scores.softmax(1);

        // 应用权重到 Value
        let output = attention_weights.matmul(v);

        (output, attention_weights)
    }

    /// 分割多头
    ///
    /// 将输入分割成多个头：[seq_len, d_model] → [n_heads, seq_len, d_k]
    fn split_heads(&self, x: &Array2<f32>) -> Array2<f32> {
        let seq_len = x.nrows();
        let d_model = x.ncols();

        // 重塑并转置：[seq_len, n_heads, d_k] → [n_heads, seq_len, d_k]
        // 这里简化为 2D 操作，实际应该用 3D 张量
        let mut reshaped = Array2::zeros((seq_len, d_model));
        reshaped.assign(x);

        // 简化实现：按头维度切分
        let mut heads = Vec::new();
        for i in 0..self.params.n_heads {
            let start = i * self.params.d_k;
            let end = start + self.params.d_k;
            let head = reshaped.slice(ndarray::s![.., start..end]).to_owned();
            heads.push(head);
        }

        // 拼接所有头（简化：保持 2D）
        // 完整实现应该返回 3D 张量
        x.clone()
    }

    /// 合并多头
    ///
    /// 将多个头合并：[n_heads, seq_len, d_k] → [seq_len, d_model]
    fn combine_heads(&self, x: &Array2<f32>) -> Array2<f32> {
        x.clone()
    }

    /// 前向传播
    ///
    /// # 参数
    /// - `x`: 输入 [seq_len, d_model]
    /// - `mask`: 可选的注意力掩码 [seq_len, seq_len]
    ///
    /// # 返回
    /// - 输出 [seq_len, d_model]
    /// - 注意力权重 [seq_len, seq_len]（用于可视化）
    pub fn forward(
        &self,
        x: &Array2<f32>,
        mask: Option<&Array2<f32>>,
    ) -> (Array2<f32>, Array2<f32>) {
        // 1. 计算 Q, K, V
        let q = x.matmul(&self.w_q);
        let k = x.matmul(&self.w_k);
        let v = x.matmul(&self.w_v);

        // 2. 简化实现：单个注意力（完整版本应该分割成多头）
        let (attn_output, attn_weights) = self.scaled_dot_product_attention(&q, &k, &v);

        // 3. 应用掩码（如果提供）
        let masked_output = if let Some(m) = mask {
            let masked_weights = &attn_weights * m;
            // 重新归一化
            let sum = masked_weights.sum_axis(ndarray::Axis(1));
            let sum_view = sum.insert_axis(ndarray::Axis(1));
            let norm_weights = &masked_weights / &sum_view;
            norm_weights.matmul(&v)
        } else {
            attn_output
        };

        // 4. 输出投影
        let output = masked_output.matmul(&self.w_o);

        (output, attn_weights)
    }

    /// 获取参数
    pub fn params(&self) -> &AttentionParams {
        &self.params
    }

    /// 获取权重（用于保存模型）
    pub fn weights(&self) -> (&Array2<f32>, &Array2<f32>, &Array2<f32>, &Array2<f32>) {
        (&self.w_q, &self.w_k, &self.w_v, &self.w_o)
    }
}

/// 自注意力包装器（简化 API）
#[derive(Debug, Clone)]
pub struct SelfAttention {
    attention: MultiHeadAttention,
}

impl SelfAttention {
    pub fn new(d_model: usize, n_heads: usize) -> Self {
        Self {
            attention: MultiHeadAttention::new(d_model, n_heads),
        }
    }

    pub fn forward(&self, x: &Array2<f32>) -> Array2<f32> {
        let (output, _) = self.attention.forward(x, None);
        output
    }

    pub fn forward_with_mask(&self, x: &Array2<f32>, mask: &Array2<f32>) -> Array2<f32> {
        let (output, _) = self.attention.forward(x, Some(mask));
        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    #[test]
    fn test_attention_params() {
        let params = AttentionParams::new(512, 8);
        assert_eq!(params.d_model, 512);
        assert_eq!(params.n_heads, 8);
        assert_eq!(params.d_k, 64);
    }

    #[test]
    #[should_panic(expected = "d_model must be divisible by n_heads")]
    fn test_invalid_params() {
        AttentionParams::new(100, 3);  // 100 不能被 3 整除
    }

    #[test]
    fn test_multi_head_attention() {
        let attention = MultiHeadAttention::new(64, 4);
        let x = Array2::random_xavier((10, 64));

        let (output, weights) = attention.forward(&x, None);

        // 验证输出形状
        assert_eq!(output.shape(), &[10, 64]);
        assert_eq!(weights.shape(), &[10, 10]);

        // 验证注意力权重和为 1
        for row in 0..weights.nrows() {
            let sum: f32 = weights.row(row).iter().sum();
            assert!((sum - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_self_attention() {
        let sa = SelfAttention::new(128, 8);
        let x = Array2::random_xavier((5, 128));

        let output = sa.forward(&x);

        assert_eq!(output.shape(), &[5, 128]);
    }
}
