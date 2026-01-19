//! Causal Language Model (GPT-2 风格)
//!
//! 实现类似 GPT-2 的 Decoder-only 架构
//! 特点：因果注意力、RoPE 位置编码、文本生成

use ndarray::{Array2, Array4};
use rand::Rng;

use crate::{
    attention::MultiHeadAttention,
    embedding::Embedding,
    layers::{FeedForward, LayerNorm},
    rope::{RoPEAttention, apply_rope_batch},
};

/// CausalLM 配置
#[derive(Debug, Clone)]
pub struct CausalLMConfig {
    /// 词汇表大小
    pub vocab_size: usize,
    /// 模型维度
    pub d_model: usize,
    /// 注意力头数
    pub n_heads: usize,
    /// 层数
    pub n_layers: usize,
    /// 前馈网络维度
    pub d_ff: usize,
    /// 最大序列长度
    pub max_seq_len: usize,
    /// Dropout 率
    pub dropout: f32,
    /// 是否使用 RoPE
    pub use_rope: bool,
}

impl Default for CausalLMConfig {
    fn default() -> Self {
        Self {
            vocab_size: 10000,
            d_model: 768,
            n_heads: 12,
            n_layers: 12,
            d_ff: 3072,
            max_seq_len: 1024,
            dropout: 0.1,
            use_rope: true,
        }
    }
}

impl CausalLMConfig {
    /// 创建小型配置（用于快速测试）
    pub fn small() -> Self {
        Self {
            vocab_size: 1000,
            d_model: 128,
            n_heads: 4,
            n_layers: 4,
            d_ff: 512,
            max_seq_len: 128,
            dropout: 0.1,
            use_rope: true,
        }
    }

    /// 创建基础配置
    pub fn base() -> Self {
        Self {
            vocab_size: 10000,
            d_model: 768,
            n_heads: 12,
            n_layers: 12,
            d_ff: 3072,
            max_seq_len: 1024,
            dropout: 0.1,
            use_rope: true,
        }
    }
}

/// CausalLM Decoder Block
///
/// 包含：因果自注意力 + 前馈网络
pub struct CausalLMBlock {
    /// 自注意力层
    pub self_attn: MultiHeadAttention,
    /// 前馈网络
    pub ffn: FeedForward,
    /// 注意力后的 Layer Norm
    pub norm1: LayerNorm,
    /// FFN 后的 Layer Norm
    pub norm2: LayerNorm,
    /// 是否使用 RoPE
    pub use_rope: bool,
}

impl CausalLMBlock {
    /// 创建新的 Decoder Block
    pub fn new(config: &CausalLMConfig) -> Self {
        Self {
            self_attn: MultiHeadAttention::new(config.d_model, config.n_heads),
            ffn: FeedForward::new(config.d_model, config.d_ff),
            norm1: LayerNorm::new(config.d_model, None),
            norm2: LayerNorm::new(config.d_model, None),
            use_rope: config.use_rope,
        }
    }

    /// 前向传播
    ///
    /// # Arguments
    /// * `x` - 输入 [seq_len, d_model]
    /// * `mask` - 因果掩码 [seq_len, seq_len]
    ///
    /// # Returns
    /// * 输出 [seq_len, d_model]
    pub fn forward(&self, x: &Array2<f32>, mask: &Array2<f32>) -> Array2<f32> {
        // 1. Layer Norm + Self-Attention
        let x_norm1 = self.norm1.forward(x);
        let (attn_output, _) = self.self_attn.forward(&x_norm1, Some(mask));

        // 残差连接
        let x = x + &attn_output;

        // 2. Layer Norm + FFN
        let x_norm2 = self.norm2.forward(&x);
        let ffn_output = self.ffn.forward(&x_norm2);

        // 残差连接
        x + ffn_output
    }
}

/// Causal Language Model
pub struct CausalLM {
    /// Token 嵌入
    pub embedding: Embedding,
    /// Decoder 层
    pub layers: Vec<CausalLMBlock>,
    /// 最终的 Layer Norm
    pub final_norm: LayerNorm,
    /// LM Head（词汇表投影）
    pub lm_head: Array2<f32>,
    /// 配置
    pub config: CausalLMConfig,
}

impl CausalLM {
    /// 创建新的 CausalLM
    pub fn new(config: CausalLMConfig) -> Self {
        // 创建嵌入层
        let embedding = Embedding::new(config.vocab_size, config.d_model);

        // 创建 Decoder 层
        let mut layers = Vec::new();
        for _ in 0..config.n_layers {
            layers.push(CausalLMBlock::new(&config));
        }

        // 最终 Layer Norm
        let final_norm = LayerNorm::new(config.d_model, None);

        // LM Head（线性投影到词汇表）
        let lm_head = Array2::zeros((config.d_model, config.vocab_size));

        Self {
            embedding,
            layers,
            final_norm,
            lm_head,
            config,
        }
    }

    /// 前向传播
    ///
    /// # Arguments
    /// * `input_ids` - 输入 token IDs [seq_len]
    ///
    /// # Returns
    /// * Logits [seq_len, vocab_size]
    pub fn forward(&self, input_ids: &[usize]) -> Array2<f32> {
        let seq_len = input_ids.len();

        // 1. Token 嵌入
        let x = self.embedding.forward(input_ids);

        // 2. 创建因果掩码
        let mask = self.create_causal_mask(seq_len);

        // 3. 通过所有 Decoder 层
        let mut x = x;
        for layer in &self.layers {
            x = layer.forward(&x, &mask);
        }

        // 4. 最终 Layer Norm
        let x = self.final_norm.forward(&x);

        // 5. LM Head 投影到词汇表
        let logits = self.project_to_vocab(&x);

        logits
    }

    /// 创建因果掩码
    ///
    /// 确保每个位置只能注意到它自己和之前的位置
    fn create_causal_mask(&self, seq_len: usize) -> Array2<f32> {
        let mut mask = Array2::zeros((seq_len, seq_len));

        for i in 0..seq_len {
            for j in 0..seq_len {
                // 只允许 j <= i（当前位置或之前的位置）
                mask[[i, j]] = if j <= i { 0.0 } else { f32::NEG_INFINITY };
            }
        }

        mask
    }

    /// 投影到词汇表
    fn project_to_vocab(&self, x: &Array2<f32>) -> Array2<f32> {
        // x @ lm_head
        let seq_len = x.nrows();
        let mut logits = Array2::zeros((seq_len, self.config.vocab_size));

        for i in 0..seq_len {
            for j in 0..self.config.vocab_size {
                let mut val = 0.0;
                for k in 0..self.config.d_model {
                    val += x[[i, k]] * self.lm_head[[k, j]];
                }
                logits[[i, j]] = val;
            }
        }

        logits
    }

    /// 生成文本（贪婪解码）
    ///
    /// # Arguments
    /// * `input_ids` - 输入 token IDs
    /// * `max_new_tokens` - 最大生成 token 数
    ///
    /// # Returns
    /// * 生成的 token IDs
    pub fn generate(&self, input_ids: &[usize], max_new_tokens: usize) -> Vec<usize> {
        let mut generated = input_ids.to_vec();

        for _ in 0..max_new_tokens {
            // 前向传播
            let logits = self.forward(&generated);

            // 获取最后一个位置的 logits
            let last_logits = logits.row(generated.len() - 1);

            // 贪婪选择
            let next_token = argmax(&last_logits.to_vec());

            // 添加到生成序列
            generated.push(next_token);

            // 可选：检查 EOS token
            // if next_token == eos_token_id { break; }
        }

        generated
    }

    /// 生成文本（带采样）
    ///
    /// # Arguments
    /// * `input_ids` - 输入 token IDs
    /// * `max_new_tokens` - 最大生成 token 数
    /// * `temperature` - 温度参数（1.0 = 不改变，< 1.0 = 更保守，> 1.0 = 更随机）
    /// * `top_k` - Top-K 采样（0 = 不使用）
    /// * `top_p` - Top-p (nucleus) 采样（0.0 = 不使用）
    ///
    /// # Returns
    /// * 生成的 token IDs
    pub fn generate_with_sampling(
        &self,
        input_ids: &[usize],
        max_new_tokens: usize,
        temperature: f32,
        top_k: usize,
        top_p: f32,
    ) -> Vec<usize> {
        let mut generated = input_ids.to_vec();
        let mut rng = rand::thread_rng();

        for _ in 0..max_new_tokens {
            // 前向传播
            let logits = self.forward(&generated);

            // 获取最后一个位置的 logits
            let mut last_logits = logits.row(generated.len() - 1).to_vec();

            // 应用温度缩放
            if temperature != 1.0 {
                for logit in &mut last_logits {
                    *logit /= temperature;
                }
            }

            // 应用 Top-K 采样
            if top_k > 0 && top_k < last_logits.len() {
                last_logits = apply_top_k(&last_logits, top_k);
            }

            // 应用 Top-p (nucleus) 采样
            if top_p > 0.0 && top_p < 1.0 {
                last_logits = apply_top_p(&last_logits, top_p);
            }

            // Softmax
            let probs = softmax(&last_logits);

            // 采样
            let next_token = sample_categorical(&probs, &mut rng);

            // 添加到生成序列
            generated.push(next_token);
        }

        generated
    }
}

/// Argmax：返回最大值的索引
fn argmax(values: &[f32]) -> usize {
    let mut max_idx = 0;
    let mut max_val = values[0];

    for (i, &val) in values.iter().enumerate() {
        if val > max_val {
            max_val = val;
            max_idx = i;
        }
    }

    max_idx
}

/// Softmax
fn softmax(logits: &[f32]) -> Vec<f32> {
    let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut exp_sum = 0.0;
    let mut exps = Vec::with_capacity(logits.len());

    for &logit in logits {
        let exp = (logit - max_logit).exp();
        exps.push(exp);
        exp_sum += exp;
    }

    exps.iter().map(|&e| e / exp_sum).collect()
}

/// 应用 Top-K 过滤
fn apply_top_k(logits: &[f32], k: usize) -> Vec<f32> {
    let mut filtered = logits.to_vec();

    // 找到第 k 大的值
    let mut sorted = logits.to_vec();
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let threshold = sorted[k.min(sorted.len() - 1)];

    // 将小于阈值的值设为负无穷
    for logit in &mut filtered {
        if *logit < threshold {
            *logit = f32::NEG_INFINITY;
        }
    }

    filtered
}

/// 应用 Top-p (nucleus) 过滤
fn apply_top_p(logits: &[f32], p: f32) -> Vec<f32> {
    let mut filtered = logits.to_vec();
    let mut sorted_indices: Vec<usize> = (0..logits.len()).collect();

    // 按值降序排序
    sorted_indices.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap_or(std::cmp::Ordering::Equal));

    let mut cumsum = 0.0;
    let mut cutoff_idx = 0;

    // 计算累积概率直到达到 p
    for &idx in &sorted_indices {
        cumsum += logits[idx].exp();
        cutoff_idx = idx;
        if cumsum >= p {
            break;
        }
    }

    // 将小于阈值的值设为负无穷
    let threshold = logits[cutoff_idx];
    for logit in &mut filtered {
        if *logit < threshold {
            *logit = f32::NEG_INFINITY;
        }
    }

    filtered
}

/// 分类采样
fn sample_categorical(probs: &[f32], rng: &mut impl Rng) -> usize {
    let mut cumsum = 0.0;
    let r: f32 = rng.gen();

    for (i, &p) in probs.iter().enumerate() {
        cumsum += p;
        if r <= cumsum {
            return i;
        }
    }

    probs.len() - 1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_causal_lm_config() {
        let config = CausalLMConfig::small();
        assert_eq!(config.d_model, 128);
        assert!(config.use_rope);
    }

    #[test]
    fn test_causal_lm_creation() {
        let config = CausalLMConfig::small();
        let model = CausalLM::new(config);
        assert_eq!(model.layers.len(), 4);
    }

    #[test]
    fn test_causal_mask() {
        let config = CausalLMConfig::small();
        let model = CausalLM::new(config);
        let mask = model.create_causal_mask(4);

        // 上三角应该是 0.0，下三角（含对角）应该是 -inf
        assert_eq!(mask[[0, 0]], 0.0);
        assert_eq!(mask[[1, 0]], 0.0);
        assert_eq!(mask[[1, 1]], 0.0);
        assert_eq!(mask[[2, 0]], 0.0);
        assert_eq!(mask[[2, 1]], 0.0);
        assert_eq!(mask[[2, 2]], 0.0);
        assert_eq!(mask[[0, 1]], f32::NEG_INFINITY);
        assert_eq!(mask[[1, 2]], f32::NEG_INFINITY);
    }

    #[test]
    fn test_forward() {
        let config = CausalLMConfig::small();
        let vocab_size = config.vocab_size;
        let model = CausalLM::new(config);

        let input_ids = vec![0, 1, 2, 3];
        let logits = model.forward(&input_ids);

        assert_eq!(logits.shape(), &[4, vocab_size]);
    }

    #[test]
    fn test_generate() {
        let config = CausalLMConfig::small();
        let model = CausalLM::new(config);

        let input_ids = vec![0, 1, 2];
        let generated = model.generate(&input_ids, 5);

        assert_eq!(generated.len(), 8); // 3 + 5
    }

    #[test]
    fn test_argmax() {
        let values = vec![0.1, 0.5, 0.3, 0.9, 0.2];
        let idx = argmax(&values);
        assert_eq!(idx, 3); // 0.9 是最大值
    }

    #[test]
    fn test_softmax() {
        let logits = vec![1.0, 2.0, 3.0];
        let probs = softmax(&logits);

        // 概率和应该为 1
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);

        // 应该是递增的（因为输入是递增的）
        assert!(probs[0] < probs[1]);
        assert!(probs[1] < probs[2]);
    }
}
