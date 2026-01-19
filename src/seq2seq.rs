//! Seq2Seq 模型
//!
//! 实现完整的序列到序列 Transformer 模型

use ndarray::Array2;
use crate::decoder::DecoderLayer;
use crate::embedding::Embedding;
use crate::embedding::PositionalEncoding;
use crate::trainable::Linear;
use crate::tensor::TensorExt;

/// Seq2Seq Transformer 模型
///
/// 用于序列到序列任务，如翻译、摘要等
#[derive(Debug)]
pub struct Seq2SeqTransformer {
    // Encoder 部分（简化：使用 embedding 作为编码器）
    pub embedding: Embedding,
    pub pos_encoding: PositionalEncoding,

    // Decoder 部分
    pub decoder_layers: Vec<DecoderLayer>,

    // 输出投影
    pub output_projection: Linear,

    // 配置
    pub vocab_size: usize,
    pub d_model: usize,
    pub n_layers: usize,
    pub max_seq_len: usize,
    training: bool,
}

impl Seq2SeqTransformer {
    pub fn new(vocab_size: usize, d_model: usize, n_heads: usize, n_layers: usize, d_ff: usize, max_seq_len: usize) -> Self {
        let embedding = Embedding::new(vocab_size, d_model);
        let pos_encoding = PositionalEncoding::new(max_seq_len, d_model);

        let decoder_layers = (0..n_layers)
            .map(|_| DecoderLayer::new(d_model, n_heads, d_ff))
            .collect();

        let output_projection = Linear::new(d_model, vocab_size);

        Self {
            embedding,
            pos_encoding,
            decoder_layers,
            output_projection,
            vocab_size,
            d_model,
            n_layers,
            max_seq_len,
            training: false,
        }
    }

    pub fn set_training(&mut self, training: bool) {
        self.training = training;
        for layer in &mut self.decoder_layers {
            layer.set_training(training);
        }
        self.output_projection.set_training(training);
    }

    /// 编码输入序列
    ///
    /// encoder_output: [batch_size, seq_len, d_model]
    pub fn encode(&mut self, input: &[usize]) -> Array2<f32> {
        // 嵌入
        let mut embedded = self.embedding.forward(input);

        // 位置编码
        let encoded = self.pos_encoding.forward(&embedded);

        encoded
    }

    /// 解码一步
    ///
    /// - decoder_input: [batch_size, 1, d_model] - 当前解码器输入
    /// - encoder_output: [batch_size, source_seq_len, d_model] - 编码器输出
    ///
    /// 返回: [batch_size, vocab_size] - 下一个 token 的 logits
    pub fn decode_step(&mut self, decoder_input: &Array2<f32>, encoder_output: &Array2<f32>) -> Array2<f32> {
        let mut output = decoder_input.clone();

        // 通过所有 decoder 层
        for layer in &mut self.decoder_layers {
            output = layer.forward(&output, encoder_output);
        }

        // 输出投影
        let logits = self.output_projection.forward(&output);

        logits
    }

    /// 训练步骤（教师强制）
    ///
    /// - source: 源序列 [batch_size, source_len]
    /// - target: 目标序列 [batch_size, target_len]
    ///
    /// 返回: (loss, accuracy)
    pub fn train_step(&mut self, source: &[usize], target: &[usize], learning_rate: f32) -> (f32, f32) {
        self.set_training(true);

        // 编码源序列
        let encoder_output = self.encode(source);

        // 解码目标序列（简化：只使用最后一个 token）
        let target_embedded = self.embedding.forward(target);
        let target_encoded = self.pos_encoding.forward(&target_embedded);

        // 取最后一个 token 作为 decoder 输入
        let decoder_input = target_encoded.slice(ndarray::s![..1, ..]).to_owned();

        // 解码
        let logits = self.decode_step(&decoder_input, &encoder_output);

        // 计算损失和准确率
        let loss = self.compute_loss(&logits, target[target.len() - 1]);
        let accuracy = if logits.row(0).iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap() == target[target.len() - 1]
        {
            1.0
        } else {
            0.0
        };

        self.set_training(false);

        (loss, accuracy)
    }

    /// 计算损失
    pub fn compute_loss(&self, logits: &Array2<f32>, target: usize) -> f32 {
        let batch_size = logits.nrows();
        let mut loss = 0.0;

        for i in 0..batch_size {
            let log_probs = logits.row(i).mapv(|x| x.max(1e-7).ln());
            loss -= log_probs[target];
        }

        loss / batch_size as f32
    }

    /// 贪婪解码生成
    ///
    /// - source: 源序列
    /// - max_len: 最大生成长度
    /// - start_token: 起始 token
    ///
    /// 返回: 生成的 token 序列
    pub fn generate_greedy(&mut self, source: &[usize], max_len: usize, start_token: usize) -> Vec<usize> {
        self.set_training(false);

        // 编码源序列
        let encoder_output = self.encode(source);

        // 初始化输出序列
        let mut output = vec![start_token];

        // 逐个生成 token
        for _ in 0..max_len {
            // 嵌入当前输出
            let embedded = self.embedding.forward(&output);
            let encoded = self.pos_encoding.forward(&embedded);

            // 取最后一个 token
            let decoder_input = encoded.slice(ndarray::s![..1, ..]).to_owned();

            // 解码
            let logits = self.decode_step(&decoder_input, &encoder_output);

            // 贪婪选择概率最大的 token
            let next_token = logits.row(0)
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i)
                .unwrap();

            output.push(next_token);

            // 如果生成了结束 token，停止
            if next_token == 2 { // 假设 2 是 EOS token
                break;
            }
        }

        output
    }

    /// 束搜索解码
    ///
    /// - source: 源序列
    /// - max_len: 最大生成长度
    /// - beam_width: 束宽度
    /// - start_token: 起始 token
    ///
    /// 返回: 生成的 token 序列
    pub fn generate_beam(&mut self, source: &[usize], max_len: usize, beam_width: usize, start_token: usize) -> Vec<usize> {
        self.set_training(false);

        // 编码源序列
        let encoder_output = self.encode(source);

        // 初始束：只包含 start_token
        let mut beams: Vec<(Vec<usize>, f32)> = vec![(vec![start_token], 0.0)];

        for _ in 0..max_len {
            let mut all_candidates: Vec<(Vec<usize>, f32)> = Vec::new();

            for (seq, score) in beams {
                // 嵌入当前序列
                let embedded = self.embedding.forward(&seq);
                let encoded = self.pos_encoding.forward(&embedded);

                // 取最后一个 token
                let decoder_input = encoded.slice(ndarray::s![..1, ..]).to_owned();

                // 解码
                let logits = self.decode_step(&decoder_input, &encoder_output);

                // 获取 top-k 个候选
                let mut token_probs: Vec<(usize, f32)> = logits.row(0)
                    .iter()
                    .enumerate()
                    .map(|(i, &logit)| (i, logit))
                    .collect();

                // 简化的 softmax
                let max_logit = token_probs.iter()
                    .map(|(_, l)| *l)
                    .reduce(f32::max)
                    .unwrap_or(0.0);

                let sum_exp: f32 = token_probs.iter()
                    .map(|(_, l)| (l - max_logit).exp())
                    .sum();

                token_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

                // 取 top beam_width 个
                for (token, logit) in token_probs.iter().take(beam_width) {
                    let new_seq = {
                        let mut s = seq.clone();
                        s.push(*token);
                        s
                    };

                    let new_score = score + (logit - max_logit - (sum_exp as f32).ln());
                    all_candidates.push((new_seq, new_score));
                }
            }

            // 按分数排序并保留 top beam_width 个
            all_candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            beams = all_candidates.into_iter().take(beam_width).collect();

            // 检查是否所有序列都已结束
            if beams.iter().all(|(seq, _)| seq.last() == Some(&2)) {
                break;
            }
        }

        // 返回分数最高的序列
        beams.into_iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .map(|(seq, _)| seq)
            .unwrap_or_else(|| vec![start_token])
    }

    pub fn param_count(&self) -> usize {
        let embed_params = self.vocab_size * self.d_model;
        let decoder_params: usize = self.decoder_layers
            .iter()
            .map(|l| l.param_count())
            .sum();
        let output_params = self.output_projection.param_count();

        embed_params + decoder_params + output_params
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_seq2seq_creation() {
        let model = Seq2SeqTransformer::new(
            100,  // vocab_size
            128,  // d_model
            4,    // n_heads
            2,    // n_layers
            256,  // d_ff
            32,   // max_seq_len
        );

        assert_eq!(model.vocab_size, 100);
        assert!(model.param_count() > 0);
    }

    #[test]
    fn test_greedy_generation() {
        let mut model = Seq2SeqTransformer::new(100, 64, 4, 2, 128, 32);

        let source = vec![1, 2, 3, 4, 5];
        let output = model.generate_greedy(&source, 10, 0);

        assert!(!output.is_empty());
        assert_eq!(output[0], 0); // start_token
    }

    #[test]
    fn test_beam_search() {
        let mut model = Seq2SeqTransformer::new(100, 64, 4, 2, 128, 32);

        let source = vec![1, 2, 3, 4, 5];
        let output = model.generate_beam(&source, 10, 3, 0);

        assert!(!output.is_empty());
        assert_eq!(output[0], 0); // start_token
    }
}
