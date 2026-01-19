//! 完整的可训练 Transformer
//!
//! 整合所有组件，实现端到端的训练

use ndarray::Array2;
use crate::trainable::{Linear, TrainableAttention, TrainableFFN};
use crate::embedding::{Embedding, PositionalEncoding};
use crate::tensor::TensorExt;

/// 可训练的 Transformer Encoder Layer
#[derive(Debug)]
pub struct TrainableEncoderLayer {
    /// Self-Attention
    attention: TrainableAttention,
    /// Feed-Forward Network
    ffn: TrainableFFN,
    /// Layer norm（简化：暂不训练）
    gamma1: Array2<f32>,
    beta1: Array2<f32>,
    gamma2: Array2<f32>,
    beta2: Array2<f32>,
    training: bool,
}

impl TrainableEncoderLayer {
    pub fn new(d_model: usize, n_heads: usize, d_ff: usize) -> Self {
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

    pub fn set_training(&mut self, training: bool) {
        self.training = training;
        self.attention.set_training(training);
        self.ffn.set_training(training);
    }

    pub fn forward(&mut self, x: &Array2<f32>) -> Array2<f32> {
        // Self-Attention + Add & Norm
        let (attn_out, _) = self.attention.forward(x);
        let attn_out = attn_out.layer_norm(&self.gamma1, &self.beta1, 1e-6);
        let attn_residual = x.residual_add(&attn_out);

        // FFN + Add & Norm
        let ffn_out = self.ffn.forward(&attn_residual);
        let ffn_out = ffn_out.layer_norm(&self.gamma2, &self.beta2, 1e-6);
        let output = attn_residual.residual_add(&ffn_out);

        output
    }

    pub fn backward(&mut self, grad_output: &Array2<f32>) -> Array2<f32> {
        // 简化：直接反向传播（忽略残差连接的梯度缩放）
        let grad_ffn = self.ffn.backward(grad_output);
        let grad_attn = self.attention.backward(&grad_ffn);

        &grad_attn + &grad_ffn
    }

    pub fn param_count(&self) -> usize {
        self.attention.d_model() * self.attention.d_model() * 4  // Q, K, V, O
            + self.ffn.param_count()
            + self.gamma1.len() + self.beta1.len()
            + self.gamma2.len() + self.beta2.len()
    }
}

/// 完整的可训练 Transformer
#[derive(Debug)]
pub struct TrainableTransformer {
    /// 词嵌入
    embedding: Embedding,
    /// 位置编码
    pos_encoding: PositionalEncoding,
    /// Encoder 层
    layers: Vec<TrainableEncoderLayer>,
    /// 分类头
    classifier: Linear,
    /// 配置
    d_model: usize,
    vocab_size: usize,
    n_classes: usize,
    training: bool,
}

impl TrainableTransformer {
    pub fn new(vocab_size: usize, d_model: usize, n_heads: usize, n_layers: usize, d_ff: usize, max_seq_len: usize, n_classes: usize) -> Self {
        let embedding = Embedding::new(vocab_size, d_model);
        let pos_encoding = PositionalEncoding::new(max_seq_len, d_model);

        let layers = (0..n_layers)
            .map(|_| TrainableEncoderLayer::new(d_model, n_heads, d_ff))
            .collect();

        let classifier = Linear::new(d_model, n_classes);

        Self {
            embedding,
            pos_encoding,
            layers,
            classifier,
            d_model,
            vocab_size,
            n_classes,
            training: false,
        }
    }

    pub fn set_training(&mut self, training: bool) {
        self.training = training;
        for layer in &mut self.layers {
            layer.set_training(training);
        }
        self.classifier.set_training(training);
    }

    /// 前向传播
    ///
    /// 输入: token IDs [batch_size, seq_len]
    /// 输出: logits [batch_size, n_classes]
    pub fn forward(&mut self, input: &Array2<usize>) -> Array2<f32> {
        let batch_size = input.nrows();
        let seq_len = input.ncols();

        // 嵌入（简化：使用第一个 token 的嵌入）
        let mut embedded = Array2::zeros((batch_size, self.d_model));
        for i in 0..batch_size {
            let tokens: Vec<usize> = input.row(i).to_vec();
            let emb = self.embedding.forward(&tokens);
            // emb is [seq_len, d_model], take only the first token
            embedded.row_mut(i).assign(&emb.row(0));
        }

        // 位置编码（简化：使用 seq_len 的前 seq_len 个位置）
        let pos_encoded = self.pos_encoding.forward(&embedded);

        // 通过 Encoder 层
        let mut encoded = pos_encoded;
        for layer in &mut self.layers {
            encoded = layer.forward(&encoded);
        }

        // 分类：使用第一个 token
        let cls_token = encoded.slice(ndarray::s![..1, ..]).to_owned();
        let logits = self.classifier.forward(&cls_token);

        logits
    }

    /// 训练步骤
    ///
    /// 返回: (loss, accuracy)
    pub fn train_step(&mut self, input: &Array2<usize>, targets: &Array2<f32>, lr: f32) -> (f32, f32) {
        self.set_training(true);

        // 前向传播
        let logits = self.forward(input);

        // 计算损失和梯度
        let (loss, grad_logits) = self.compute_loss_and_grad(&logits, targets);

        // 反向传播
        let _ = self.backward(&grad_logits, lr);

        // 计算准确率
        let accuracy = self.compute_accuracy(&logits, targets);

        self.set_training(false);

        (loss, accuracy)
    }

    /// 计算损失和梯度
    fn compute_loss_and_grad(&self, logits: &Array2<f32>, targets: &Array2<f32>) -> (f32, Array2<f32>) {
        // Cross-Entropy 损失
        let batch_size = logits.nrows();
        let mut loss = 0.0;

        let log_softmax = logits.softmax(1);
        for i in 0..batch_size {
            for j in 0..self.n_classes {
                if targets[[i, j]] > 0.0 {
                    let log_p = log_softmax[[i, j]].max(1e-7).ln();
                    loss -= log_p;
                }
            }
        }
        loss /= batch_size as f32;

        // 梯度: (pred - target) / batch_size
        let grad = (log_softmax - targets) / batch_size as f32;

        (loss, grad)
    }

    /// 反向传播
    fn backward(&mut self, grad_logits: &Array2<f32>, lr: f32) -> Array2<f32> {
        // 反向传播通过分类器
        let (grad_encoded, grad_clf_weight, grad_clf_bias) = self.classifier.backward(grad_logits);

        // 更新分类器
        self.classifier.update(&grad_clf_weight, &grad_clf_bias, lr);

        // 反向传播通过 Encoder 层（反向顺序）
        let mut grad = grad_encoded;
        for layer in self.layers.iter_mut().rev() {
            grad = layer.backward(&grad);
        }

        grad
    }

    /// 计算准确率
    fn compute_accuracy(&self, logits: &Array2<f32>, targets: &Array2<f32>) -> f32 {
        let batch_size = logits.nrows();
        let mut correct = 0;

        for i in 0..batch_size {
            let pred_class = logits.row(i)
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(j, _)| j)
                .unwrap();

            let target_class = targets.row(i)
                .iter()
                .position(|&x| x > 0.5)
                .unwrap();

            if pred_class == target_class {
                correct += 1;
            }
        }

        correct as f32 / batch_size as f32
    }

    /// 评估
    pub fn evaluate(&mut self, inputs: &[Vec<usize>], targets: &[usize]) -> (f32, f32) {
        self.set_training(false);

        let mut total_loss = 0.0;
        let mut total_acc = 0.0;
        let n = inputs.len();

        for (input, &target) in inputs.iter().zip(targets.iter()) {
            // 转换为 batch
            let input_batch = Array2::from_shape_vec((1, input.len()), input.clone()).unwrap();
            let target_onehot = Self::one_hot(target, self.n_classes);

            let logits = self.forward(&input_batch);
            let (loss, _) = self.compute_loss_and_grad(&logits, &target_onehot);
            let acc = self.compute_accuracy(&logits, &target_onehot);

            total_loss += loss;
            total_acc += acc;
        }

        self.set_training(false);

        (total_loss / n as f32, total_acc / n as f32)
    }

    /// One-hot 编码
    fn one_hot(class: usize, n_classes: usize) -> Array2<f32> {
        let mut arr = Array2::zeros((1, n_classes));
        arr[[0, class]] = 1.0;
        arr
    }

    pub fn param_count(&self) -> usize {
        let embed_params = self.vocab_size * self.d_model;
        let pos_params = self.embedding.d_model() * 1000; // 简化

        let layer_params: usize = self.layers.iter()
            .map(|l| l.param_count())
            .sum();

        let clf_params = self.classifier.param_count();

        embed_params + pos_params + layer_params + clf_params
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trainable_transformer_creation() {
        let model = TrainableTransformer::new(
            100,  // vocab_size
            64,   // d_model
            4,    // n_heads
            2,    // n_layers
            128,  // d_ff
            32,   // max_seq_len
            5,    // n_classes
        );

        assert_eq!(model.n_classes, 5);
        assert!(model.param_count() > 0);
    }

    #[test]
    fn test_forward_pass() {
        let mut model = TrainableTransformer::new(
            100, 64, 4, 2, 128, 32, 5
        );

        let input = Array2::from_shape_vec((2, 5), vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).unwrap();

        let output = model.forward(&input);

        assert_eq!(output.shape(), &[2, 5]);
    }

    #[test]
    fn test_train_step() {
        let mut model = TrainableTransformer::new(
            100, 64, 4, 2, 128, 32, 5
        );

        let input = Array2::from_shape_vec((2, 5), vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).unwrap();
        let targets = Array2::from_shape_vec((2, 5), vec![
            1.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0, 0.0,
        ]).unwrap();

        let (loss, acc) = model.train_step(&input, &targets, 0.001);

        assert!(loss >= 0.0);
        assert!(acc >= 0.0 && acc <= 1.0);
    }

    #[test]
    fn test_evaluation() {
        let mut model = TrainableTransformer::new(
            100, 64, 4, 2, 128, 32, 5
        );

        let inputs = vec![
            vec![1, 2, 3, 4, 5],
            vec![6, 7, 8, 9, 10],
        ];
        let targets = vec![0, 1];

        let (loss, acc) = model.evaluate(&inputs, &targets);

        assert!(loss >= 0.0);
        assert!(acc >= 0.0 && acc <= 1.0);
    }
}
