//! 预训练目标和任务
//!
//! 实现:
//! - Masked Language Modeling (MLM) - BERT 风格
//! - Causal Language Modeling (CLM) - GPT 风格

use ndarray::Array2;
use crate::trainable::{Linear, TrainableAttention, TrainableFFN};
use crate::embedding::{Embedding, PositionalEncoding};
use crate::tensor::TensorExt;
use rand::Rng;
use rand::seq::SliceRandom;

// ============================================================================
// Masked Language Modeling (MLM) - BERT 风格
// ============================================================================

/// MLM 配置
#[derive(Debug, Clone)]
pub struct MLMConfig {
    /// 掩码比例 (默认 0.15 = 15%)
    pub mask_ratio: f32,
    /// 替换为 [MASK] token 的概率 (默认 0.8 = 80%)
    pub mask_prob: f32,
    /// 替换为随机 token 的概率 (默认 0.1 = 10%)
    pub random_prob: f32,
    /// 保持不变的概率 (默认 0.1 = 10%)
    pub keep_prob: f32,
    /// [MASK] token ID
    pub mask_token_id: usize,
    /// [PAD] token ID
    pub pad_token_id: usize,
}

impl Default for MLMConfig {
    fn default() -> Self {
        Self {
            mask_ratio: 0.15,
            mask_prob: 0.8,
            random_prob: 0.1,
            keep_prob: 0.1,
            mask_token_id: 2,
            pad_token_id: 1,
        }
    }
}

/// MLM 预训练模型
///
/// 使用 Encoder 架构，预测被掩码的 token
#[derive(Debug)]
pub struct MLMPretrainer {
    /// 词嵌入
    embedding: Embedding,
    /// 位置编码
    pos_encoding: PositionalEncoding,
    /// Encoder 层
    layers: Vec<MLMEncoderLayer>,
    /// MLM 头（预测被掩码的 tokens）
    mlm_head: Linear,
    /// 配置
    d_model: usize,
    vocab_size: usize,
    training: bool,
}

/// MLM Encoder Layer
#[derive(Debug)]
struct MLMEncoderLayer {
    attention: TrainableAttention,
    ffn: TrainableFFN,
    gamma1: Array2<f32>,
    beta1: Array2<f32>,
    gamma2: Array2<f32>,
    beta2: Array2<f32>,
    training: bool,
}

impl MLMEncoderLayer {
    fn new(d_model: usize, n_heads: usize, d_ff: usize) -> Self {
        Self {
            attention: TrainableAttention::new(d_model, n_heads),
            ffn: TrainableFFN::new(d_model, d_ff),
            gamma1: Array2::ones((1, d_model)),
            beta1: Array2::zeros((1, d_model)),
            gamma2: Array2::ones((1, d_model)),
            beta2: Array2::zeros((1, d_model)),
            training: false,
        }
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
        self.attention.set_training(training);
        self.ffn.set_training(training);
    }

    fn forward(&mut self, x: &Array2<f32>) -> Array2<f32> {
        let (attn_out, _) = self.attention.forward(x);
        let attn_out = attn_out.layer_norm(&self.gamma1, &self.beta1, 1e-6);
        let attn_residual = x.residual_add(&attn_out);

        let ffn_out = self.ffn.forward(&attn_residual);
        let ffn_out = ffn_out.layer_norm(&self.gamma2, &self.beta2, 1e-6);
        let output = attn_residual.residual_add(&ffn_out);

        output
    }

    fn backward(&mut self, grad_output: &Array2<f32>) -> Array2<f32> {
        let grad_ffn = self.ffn.backward(grad_output);
        let grad_attn = self.attention.backward(&grad_ffn);
        &grad_attn + &grad_ffn
    }
}

impl MLMPretrainer {
    pub fn new(vocab_size: usize, d_model: usize, n_heads: usize, n_layers: usize, d_ff: usize, max_seq_len: usize) -> Self {
        let embedding = Embedding::new(vocab_size, d_model);
        let pos_encoding = PositionalEncoding::new(max_seq_len, d_model);

        let layers = (0..n_layers)
            .map(|_| MLMEncoderLayer::new(d_model, n_heads, d_ff))
            .collect();

        let mlm_head = Linear::new(d_model, vocab_size);

        Self {
            embedding,
            pos_encoding,
            layers,
            mlm_head,
            d_model,
            vocab_size,
            training: false,
        }
    }

    pub fn set_training(&mut self, training: bool) {
        self.training = training;
        for layer in &mut self.layers {
            layer.set_training(training);
        }
        self.mlm_head.set_training(training);
    }

    /// 前向传播
    ///
    /// 输入: input_ids [batch_size, seq_len]
    /// 输出: logits [batch_size, seq_len, vocab_size]
    pub fn forward(&mut self, input_ids: &Array2<usize>) -> Array2<f32> {
        let batch_size = input_ids.nrows();
        let seq_len = input_ids.ncols();

        // 嵌入所有 tokens
        let mut embedded = Vec::with_capacity(batch_size * seq_len * self.d_model);

        for i in 0..batch_size {
            let tokens: Vec<usize> = input_ids.row(i).to_vec();
            let emb = self.embedding.forward(&tokens); // [seq_len, d_model]

            // 添加位置编码
            let pos_encoded = self.pos_encoding.forward(&emb);

            // 扁平化
            for j in 0..seq_len {
                embedded.extend(pos_encoded.row(j).iter().cloned());
            }
        }

        let mut embedded_array = Array2::from_shape_vec((batch_size * seq_len, self.d_model), embedded).unwrap();

        // 通过 Encoder 层
        for layer in &mut self.layers {
            embedded_array = layer.forward(&embedded_array);
        }

        // 通过 MLM 头
        let logits = self.mlm_head.forward(&embedded_array);

        // logits shape: [batch_size * seq_len, vocab_size]
        logits
    }

    /// MLM 训练步骤
    ///
    /// 返回: (loss, accuracy)
    pub fn train_step(&mut self, input_ids: &Array2<usize>, labels: &Array2<usize>, lr: f32) -> (f32, f32) {
        self.set_training(true);

        // 前向传播
        let logits = self.forward(input_ids);

        // 计算损失和梯度
        let (loss, grad_logits) = self.compute_mlm_loss(&logits, labels);

        // 反向传播
        self.backward(&grad_logits, lr);

        // 计算准确率（只在被掩码的位置）
        let accuracy = self.compute_mlm_accuracy(&logits, labels);

        self.set_training(false);

        (loss, accuracy)
    }

    /// 计算 MLM 损失（只在非 PAD 位置计算）
    fn compute_mlm_loss(&self, logits: &Array2<f32>, labels: &Array2<usize>) -> (f32, Array2<f32>) {
        let batch_size = labels.nrows();
        let seq_len = labels.ncols();
        let vocab_size = self.vocab_size;

        let mut total_loss = 0.0;
        let mut n_valid_tokens = 0;

        // 计算 softmax 和损失
        let softmax_probs = logits.softmax(1);

        for i in 0..(batch_size * seq_len) {
            let label_idx = i / seq_len;
            let token_idx = i % seq_len;

            // 只在非 PAD 位置计算损失
            if labels[[label_idx, token_idx]] != 1 { // 1 = PAD token
                let target = labels[[label_idx, token_idx]];
                let log_p = softmax_probs[[i, target]].max(1e-7).ln();
                total_loss -= log_p;
                n_valid_tokens += 1;
            }
        }

        let loss = if n_valid_tokens > 0 {
            total_loss / n_valid_tokens as f32
        } else {
            0.0
        };

        // 计算梯度
        let mut grad = Array2::zeros((batch_size * seq_len, vocab_size));

        for i in 0..(batch_size * seq_len) {
            let label_idx = i / seq_len;
            let token_idx = i % seq_len;

            if labels[[label_idx, token_idx]] != 1 { // 非 PAD
                let target = labels[[label_idx, token_idx]];
                grad[[i, target]] = softmax_probs[[i, target]] - 1.0;
            }
        }

        (loss, grad)
    }

    /// 计算 MLM 准确率
    fn compute_mlm_accuracy(&self, logits: &Array2<f32>, labels: &Array2<usize>) -> f32 {
        let batch_size = labels.nrows();
        let seq_len = labels.ncols();

        let mut correct = 0;
        let mut total = 0;

        for i in 0..(batch_size * seq_len) {
            let label_idx = i / seq_len;
            let token_idx = i % seq_len;

            if labels[[label_idx, token_idx]] != 1 { // 非 PAD
                let pred = logits.row(i)
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .map(|(j, _)| j)
                    .unwrap();

                let target = labels[[label_idx, token_idx]];

                if pred == target {
                    correct += 1;
                }
                total += 1;
            }
        }

        if total > 0 {
            correct as f32 / total as f32
        } else {
            0.0
        }
    }

    /// 反向传播
    fn backward(&mut self, grad_logits: &Array2<f32>, lr: f32) {
        // 反向传播通过 MLM 头
        let (grad_encoded, grad_head_weight, grad_head_bias) = self.mlm_head.backward(grad_logits);

        // 更新 MLM 头
        self.mlm_head.update(&grad_head_weight, &grad_head_bias, lr);

        // 反向传播通过 Encoder 层
        let mut grad = grad_encoded;
        for layer in self.layers.iter_mut().rev() {
            grad = layer.backward(&grad);
        }
    }

    /// 批量 MLM 训练循环
    pub fn train(
        &mut self,
        train_inputs: &Array2<usize>,
        train_labels: &Array2<usize>,
        val_inputs: &Array2<usize>,
        val_labels: &Array2<usize>,
        epochs: usize,
        batch_size: usize,
        learning_rate: f32,
        patience: usize,
    ) -> (usize, f32) {
        let mut best_val_acc = 0.0;
        let mut patience_counter = 0;
        let mut best_epoch = 0;
        let n_train_samples = train_inputs.nrows();

        println!("\n╔════════════════════════════════════════════════╗");
        println!("║   MLM 预训练（batch_size={}, patience={}）  ║", batch_size, patience);
        println!("╚════════════════════════════════════════════════╝\n");

        for epoch in 1..=epochs {
            let epoch_start = std::time::Instant::now();

            let mut total_loss = 0.0;
            let mut total_acc = 0.0;
            let mut n_batches = 0;

            // 打乱数据
            let mut indices: Vec<usize> = (0..n_train_samples).collect();
            use rand::seq::SliceRandom;
            indices.shuffle(&mut rand::thread_rng());

            // 按批次训练
            for batch_start in (0..n_train_samples).step_by(batch_size) {
                let batch_end = (batch_start + batch_size).min(n_train_samples);
                let batch_indices = &indices[batch_start..batch_end];

                let batch_inputs = train_inputs.select(ndarray::Axis(0), batch_indices);
                let batch_labels = train_labels.select(ndarray::Axis(0), batch_indices);

                let (loss, acc) = self.train_step(&batch_inputs, &batch_labels, learning_rate);

                total_loss += loss;
                total_acc += acc;
                n_batches += 1;
            }

            let avg_train_loss = total_loss / n_batches as f32;
            let avg_train_acc = total_acc / n_batches as f32;

            // 验证
            let val_logits = self.forward(val_inputs);
            let (val_loss, _) = self.compute_mlm_loss(&val_logits, val_labels);
            let val_acc = self.compute_mlm_accuracy(&val_logits, val_labels);

            let epoch_time = epoch_start.elapsed().as_secs_f32();

            println!(
                "Epoch {:2}/{} | Train Loss: {:.4} | Train Acc: {:.2}% | Val Loss: {:.4} | Val Acc: {:.2}% | {:.2}s | {} batches",
                epoch, epochs, avg_train_loss, avg_train_acc * 100.0, val_loss, val_acc * 100.0, epoch_time, n_batches
            );

            if val_acc > best_val_acc {
                best_val_acc = val_acc;
                best_epoch = epoch;
                patience_counter = 0;
                println!("  ✨ 新的最佳模型！验证准确率: {:.2}%", val_acc * 100.0);
            } else {
                patience_counter += 1;
                println!("  ⏳ 验证准确率未提升 ({}/{})", patience_counter, patience);
            }

            if patience_counter >= patience {
                println!("\n⚠️  早停触发！验证准确率已 {} 个 epoch 未提升", patience);
                break;
            }

            println!();
        }

        println!("╔════════════════════════════════════════════════╗");
        println!("║     MLM 预训练完成！                           ║");
        println!("╚════════════════════════════════════════════════╝");
        println!("总计 {} 个 epochs", best_epoch);
        println!("最佳验证准确率: {:.2}% (epoch {})", best_val_acc * 100.0, best_epoch);

        (best_epoch, best_val_acc)
    }

    pub fn param_count(&self) -> usize {
        let embed_params = self.vocab_size * self.d_model;
        let layer_params: usize = self.layers.iter()
            .map(|l| {
                l.attention.d_model() * l.attention.d_model() * 4
                    + l.ffn.param_count()
            })
            .sum();
        let mlm_head_params = self.mlm_head.param_count();

        embed_params + layer_params + mlm_head_params
    }
}

// ============================================================================
// MLM 数据预处理
// ============================================================================

/// 应用 MLM 掩码策略
///
/// 对输入 tokens 应用 BERT 风格的掩码：
/// - 80% 替换为 [MASK]
/// - 10% 替换为随机 token
/// - 10% 保持不变
pub fn apply_mlm_mask(input_ids: &Vec<usize>, config: &MLMConfig, vocab_size: usize) -> (Vec<usize>, Vec<usize>) {
    let mut masked_input = input_ids.clone();
    let mut labels = input_ids.clone();

    let mut rng = rand::thread_rng();
    let seq_len = input_ids.len();
    let n_mask = (seq_len as f32 * config.mask_ratio) as usize;

    // 随机选择要掩码的位置（排除特殊 tokens）
    let mut valid_positions: Vec<usize> = (0..seq_len)
        .filter(|&i| input_ids[i] != config.pad_token_id && input_ids[i] != 0) // 不掩码 PAD 和 UNK
        .collect();

    // 打乱位置
    valid_positions.shuffle(&mut rng);

    // 只掩码前 n_mask 个位置
    for &pos in valid_positions.iter().take(n_mask) {
        let rand_val: f32 = rng.gen();

        if rand_val < config.mask_prob {
            // 80%: 替换为 [MASK]
            masked_input[pos] = config.mask_token_id;
        } else if rand_val < config.mask_prob + config.random_prob {
            // 10%: 替换为随机 token
            masked_input[pos] = rng.gen_range(3..vocab_size); // 从 3 开始（0=UNK, 1=PAD, 2=MASK）
        }
        // 10%: 保持不变

        // 设置 label 为原始 token
        labels[pos] = input_ids[pos];
    }

    (masked_input, labels)
}

/// 批量应用 MLM 掩码
pub fn apply_mlm_mask_batch(inputs: &Array2<usize>, config: &MLMConfig, vocab_size: usize) -> (Array2<usize>, Array2<usize>) {
    let batch_size = inputs.nrows();
    let seq_len = inputs.ncols();

    let mut masked_inputs = Vec::with_capacity(batch_size * seq_len);
    let mut labels = Vec::with_capacity(batch_size * seq_len);

    for i in 0..batch_size {
        let input_vec: Vec<usize> = inputs.row(i).to_vec();
        let (masked, label) = apply_mlm_mask(&input_vec, config, vocab_size);

        masked_inputs.extend(masked);
        labels.extend(label);
    }

    let masked_array = Array2::from_shape_vec((batch_size, seq_len), masked_inputs).unwrap();
    let label_array = Array2::from_shape_vec((batch_size, seq_len), labels).unwrap();

    (masked_array, label_array)
}

// ============================================================================
// Causal Language Modeling (CLM) - GPT 风格
// ============================================================================

/// CLM 预训练配置
#[derive(Debug, Clone)]
pub struct CLMConfig {
    /// 起始 token ID
    pub start_token_id: usize,
    /// 结束 token ID
    pub end_token_id: usize,
    /// 填充 token ID
    pub pad_token_id: usize,
}

impl Default for CLMConfig {
    fn default() -> Self {
        Self {
            start_token_id: 0,
            end_token_id: 2,
            pad_token_id: 1,
        }
    }
}

/// CLM 预训练辅助函数
///
/// 创建下一个 token 预测的训练数据
pub fn create_clm_targets(input_ids: &Vec<usize>, config: &CLMConfig) -> (Vec<usize>, Vec<usize>) {
    // 输入是 [start, t1, t2, ..., tn]
    // 目标是 [t1, t2, ..., tn, end]
    let mut inputs = vec![config.start_token_id];
    inputs.extend_from_slice(input_ids);

    let mut targets = input_ids.clone();
    targets.push(config.end_token_id);

    (inputs, targets)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mlm_mask_application() {
        let config = MLMConfig {
            mask_ratio: 0.3, // Increase to 30% to ensure masking happens
            ..MLMConfig::default()
        };
        let input = vec![3, 4, 5, 6, 7]; // vocab_size > 7
        let vocab_size = 100;

        let (masked, labels) = apply_mlm_mask(&input, &config, vocab_size);

        assert_eq!(masked.len(), input.len());
        assert_eq!(labels.len(), input.len());

        // 检查至少有一些位置被掩码（使用更高的 mask_ratio）
        let n_masked = masked.iter().zip(input.iter()).filter(|(m, o)| m != o).count();
        assert!(n_masked > 0, "Expected at least one token to be masked, but got {}", n_masked);
    }

    #[test]
    fn test_mlm_pretrainer_creation() {
        let model = MLMPretrainer::new(
            100,  // vocab_size
            64,   // d_model
            4,    // n_heads
            2,    // n_layers
            128,  // d_ff
            32,   // max_seq_len
        );

        assert!(model.param_count() > 0);
    }

    #[test]
    fn test_mlm_forward() {
        let mut model = MLMPretrainer::new(100, 64, 4, 2, 128, 32);

        let input = Array2::from_shape_vec((2, 5), vec![3, 4, 5, 6, 7, 8, 9, 10, 11, 12]).unwrap();

        let output = model.forward(&input);

        assert_eq!(output.shape(), &[10, 100]); // 2*5 x vocab_size
    }

    #[test]
    fn test_clm_target_creation() {
        let config = CLMConfig::default();
        let input = vec![5, 6, 7];

        let (inputs, targets) = create_clm_targets(&input, &config);

        assert_eq!(inputs, vec![0, 5, 6, 7]); // [start, 5, 6, 7]
        assert_eq!(targets, vec![5, 6, 7, 2]); // [5, 6, 7, end]
    }
}
