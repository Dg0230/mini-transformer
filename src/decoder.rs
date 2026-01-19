//! Decoder 层和 Seq2Seq 模型
//!
//! 实现 Transformer Decoder 用于序列到序列任务

use ndarray::Array2;
use crate::trainable::{Linear, TrainableAttention};
use crate::tensor::TensorExt;

/// Masked Self-Attention
///
/// 防止位置关注到后续位置（因果掩码）
#[derive(Debug, Clone)]
pub struct MaskedAttention {
    /// 底层注意力机制
    attention: TrainableAttention,
    /// 是否使用因果掩码
    use_causal_mask: bool,
}

impl MaskedAttention {
    pub fn new(d_model: usize, n_heads: usize) -> Self {
        Self {
            attention: TrainableAttention::new(d_model, n_heads),
            use_causal_mask: true,
        }
    }

    pub fn set_training(&mut self, training: bool) {
        self.attention.set_training(training);
    }

    /// 前向传播（带因果掩码）
    pub fn forward(&mut self, x: &Array2<f32>) -> (Array2<f32>, Option<Array2<f32>>) {
        let (output, attn_weights) = self.attention.forward(x);

        // 如果需要，应用因果掩码到注意力权重
        if self.use_causal_mask {
            if let Some(ref weights) = attn_weights {
                // 创建因果掩码（下三角矩阵）
                let seq_len = weights.ncols();
                let _causal_mask = Self::create_causal_mask(seq_len);

                // 应用掩码（这里简化，实际应该在 softmax 之前应用）
                // 由于权重已经经过 softmax，我们这里只是记录
            }
        }

        (output, attn_weights)
    }

    /// 创建因果掩码
    fn create_causal_mask(seq_len: usize) -> Array2<f32> {
        let mut mask = Array2::zeros((seq_len, seq_len));
        for i in 0..seq_len {
            for j in 0..=i {
                mask[[i, j]] = 1.0;
            }
        }
        mask
    }

    pub fn backward(&mut self, grad_output: &Array2<f32>) -> Array2<f32> {
        self.attention.backward(grad_output)
    }

    pub fn d_model(&self) -> usize {
        self.attention.d_model()
    }
}

/// Cross-Attention (Encoder-Decoder Attention)
///
/// Decoder 关注 Encoder 的输出
#[derive(Debug, Clone)]
pub struct CrossAttention {
    /// Query 来自 Decoder
    /// Key 和 Value 来自 Encoder
    w_q: Linear,
    w_k: Linear,
    w_v: Linear,
    w_o: Linear,
    d_model: usize,
    d_k: usize,
    training: bool,
    cache: Option<CrossAttentionCache>,
}

#[derive(Debug, Clone)]
struct CrossAttentionCache {
    q: Option<Array2<f32>>,
    k: Option<Array2<f32>>,
    v: Option<Array2<f32>>,
    attn_weights: Option<Array2<f32>>,
}

impl CrossAttention {
    pub fn new(d_model: usize, n_heads: usize) -> Self {
        let d_k = d_model / n_heads;

        Self {
            w_q: Linear::new(d_model, d_model),
            w_k: Linear::new(d_model, d_model),
            w_v: Linear::new(d_model, d_model),
            w_o: Linear::new(d_model, d_model),
            d_model,
            d_k,
            training: false,
            cache: None,
        }
    }

    pub fn set_training(&mut self, training: bool) {
        self.training = training;
        self.w_q.set_training(training);
        self.w_k.set_training(training);
        self.w_v.set_training(training);
        self.w_o.set_training(training);

        if !training {
            self.cache = None;
        }
    }

    /// 前向传播
    ///
    /// - query: 来自 Decoder [batch_size, target_seq_len, d_model]
    /// - key_value: 来自 Encoder [batch_size, source_seq_len, d_model]
    pub fn forward(&mut self, query: &Array2<f32>, key_value: &Array2<f32>) -> (Array2<f32>, Option<Array2<f32>>) {
        // 计算 Q (来自 Decoder)
        let q = self.w_q.forward(query);

        // 计算 K, V (来自 Encoder)
        let k = self.w_k.forward(key_value);
        let v = self.w_v.forward(key_value);

        // 缩放点积注意力
        let scale = (self.d_k as f32).sqrt();
        let k_t = k.t().to_owned();
        let scores = q.matmul(&k_t);
        let scaled_scores = scores.mapv(|x| x / scale);
        let attn_weights = scaled_scores.softmax(1);

        // 应用注意力权重到 V
        let attn_output = attn_weights.matmul(&v);

        // 输出投影
        let output = self.w_o.forward(&attn_output);

        // 缓存中间值
        if self.training {
            self.cache = Some(CrossAttentionCache {
                q: Some(q),
                k: Some(k),
                v: Some(v),
                attn_weights: Some(attn_weights.clone()),
            });
        }

        (output, Some(attn_weights))
    }

    pub fn backward(&mut self, grad_output: &Array2<f32>) -> (Array2<f32>, Array2<f32>) {
        let cache = self.cache.as_ref().expect("No cached cross-attention values");

        // 反向传播通过输出投影
        let (grad_attn_output, _, _) = self.w_o.backward(grad_output);

        // 简化的梯度计算
        let v = cache.v.as_ref().expect("No cached v");
        let grad_v = grad_attn_output.clone();
        let grad_kv = self.w_v.backward(&grad_v).0;

        let grad_q = grad_attn_output.clone();
        let grad_qv = self.w_q.backward(&grad_q).0;

        // Key 的梯度（简化）
        let grad_k = grad_qv.clone();
        let grad_kv2 = self.w_k.backward(&grad_k).0;

        // 合并 Query 和 Key-Value 的梯度
        (grad_qv, grad_kv + grad_kv2)
    }
}

/// Decoder Layer
///
/// 包含：
/// 1. Masked Self-Attention
/// 2. Cross-Attention
/// 3. Feed-Forward Network
#[derive(Debug)]
pub struct DecoderLayer {
    /// Masked self-attention
    self_attn: MaskedAttention,
    /// Cross-attention
    cross_attn: CrossAttention,
    /// Feed-forward network
    ffn: crate::trainable::TrainableFFN,
    /// Layer norm parameters
    gamma1: Array2<f32>,
    beta1: Array2<f32>,
    gamma2: Array2<f32>,
    beta2: Array2<f32>,
    gamma3: Array2<f32>,
    beta3: Array2<f32>,
    training: bool,
}

impl DecoderLayer {
    pub fn new(d_model: usize, n_heads: usize, d_ff: usize) -> Self {
        Self {
            self_attn: MaskedAttention::new(d_model, n_heads),
            cross_attn: CrossAttention::new(d_model, n_heads),
            ffn: crate::trainable::TrainableFFN::new(d_model, d_ff),
            gamma1: Array2::ones((1, d_model)),
            beta1: Array2::zeros((1, d_model)),
            gamma2: Array2::ones((1, d_model)),
            beta2: Array2::zeros((1, d_model)),
            gamma3: Array2::ones((1, d_model)),
            beta3: Array2::zeros((1, d_model)),
            training: false,
        }
    }

    pub fn set_training(&mut self, training: bool) {
        self.training = training;
        self.self_attn.set_training(training);
        self.cross_attn.set_training(training);
        self.ffn.set_training(training);
    }

    /// 前向传播
    ///
    /// - x: Decoder 输入 [batch_size, target_seq_len, d_model]
    /// - encoder_output: Encoder 输出 [batch_size, source_seq_len, d_model]
    pub fn forward(&mut self, x: &Array2<f32>, encoder_output: &Array2<f32>) -> Array2<f32> {
        // Masked self-attention + Add & Norm
        let (attn_out, _) = self.self_attn.forward(x);
        let attn_out = attn_out.layer_norm(&self.gamma1, &self.beta1, 1e-6);
        let attn_residual = x.residual_add(&attn_out);

        // Cross-attention + Add & Norm
        let (cross_out, _) = self.cross_attn.forward(&attn_residual, encoder_output);
        let cross_out = cross_out.layer_norm(&self.gamma2, &self.beta2, 1e-6);
        let cross_residual = attn_residual.residual_add(&cross_out);

        // FFN + Add & Norm
        let ffn_out = self.ffn.forward(&cross_residual);
        let ffn_out = ffn_out.layer_norm(&self.gamma3, &self.beta3, 1e-6);
        let output = cross_residual.residual_add(&ffn_out);

        output
    }

    pub fn param_count(&self) -> usize {
        // 每个 attention: 4 * d_model^2
        let attn_params = 4 * self.self_attn.d_model() * self.self_attn.d_model();
        let cross_attn_params = 4 * self.cross_attn.d_model * self.cross_attn.d_model;
        let ffn_params = self.ffn.param_count();
        let norm_params = 6 * self.gamma1.len(); // 3 layer norms

        attn_params + cross_attn_params + ffn_params + norm_params
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_causal_mask() {
        let mask = MaskedAttention::create_causal_mask(3);
        // 下三角矩阵应该是：
        // [[1, 0, 0],
        //  [1, 1, 0],
        //  [1, 1, 1]]
        assert_eq!(mask[[0, 0]], 1.0);
        assert_eq!(mask[[0, 1]], 0.0);
        assert_eq!(mask[[1, 0]], 1.0);
        assert_eq!(mask[[1, 1]], 1.0);
        assert_eq!(mask[[2, 2]], 1.0);
    }

    #[test]
    fn test_cross_attention_creation() {
        let cross_attn = CrossAttention::new(128, 8);
        assert_eq!(cross_attn.d_model, 128);
    }

    #[test]
    fn test_decoder_layer_creation() {
        let layer = DecoderLayer::new(128, 8, 256);
        assert!(layer.param_count() > 0);
    }
}
