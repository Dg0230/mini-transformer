//! 完整的可训练 Transformer
//!
//! 整合所有组件，实现端到端的训练

use ndarray::Array2;
use crate::trainable::{Linear, TrainableAttention, TrainableFFN};
use crate::embedding::{Embedding, PositionalEncoding};
use crate::tensor::TensorExt;
use crate::lr_scheduler::LRScheduler;

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

        // 嵌入（取每个序列第一个 token 的嵌入）
        let mut embedded = Array2::zeros((batch_size, self.d_model));
        for i in 0..batch_size {
            let tokens: Vec<usize> = input.row(i).to_vec();
            let emb = self.embedding.forward(&tokens);
            // emb is [seq_len, d_model], take only the first token
            embedded.row_mut(i).assign(&emb.row(0));
        }

        // 位置编码
        let pos_encoded = self.pos_encoding.forward(&embedded);

        // 通过 Encoder 层
        let mut encoded = pos_encoded;
        for layer in &mut self.layers {
            encoded = layer.forward(&encoded);
        }

        // 分类：使用所有 batch 样本
        let logits = self.classifier.forward(&encoded);

        logits
    }

    /// 训练步骤（单样本）
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

    /// 批量训练步骤
    ///
    /// 输入:
    /// - input: [batch_size, seq_len]
    /// - targets: [batch_size, n_classes]
    ///
    /// 返回: (loss, accuracy)
    pub fn train_batch(&mut self, input: &Array2<usize>, targets: &Array2<f32>, lr: f32) -> (f32, f32) {
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
    pub fn one_hot(class: usize, n_classes: usize) -> Array2<f32> {
        let mut arr = Array2::zeros((1, n_classes));
        arr[[0, class]] = 1.0;
        arr
    }

    /// 批量 one-hot 编码
    pub fn one_hot_batch(classes: &[usize], n_classes: usize) -> Array2<f32> {
        let batch_size = classes.len();
        let mut arr = Array2::zeros((batch_size, n_classes));
        for (i, &class) in classes.iter().enumerate() {
            arr[[i, class]] = 1.0;
        }
        arr
    }

    /// 完整训练循环（带早停）
    ///
    /// # 参数
    /// - `train_inputs`: 训练输入
    /// - `train_targets`: 训练标签
    /// - `val_inputs`: 验证输入
    /// - `val_targets`: 验证标签
    /// - `epochs`: 最大训练轮数
    /// - `learning_rate`: 初始学习率
    /// - `patience`: 早停耐心值（验证准确率不提升的 epoch 数）
    ///
    /// # 返回
    /// - 训练的 epoch 数
    /// - 最佳验证准确率
    pub fn train(
        &mut self,
        train_inputs: &[Vec<usize>],
        train_targets: &[usize],
        val_inputs: &[Vec<usize>],
        val_targets: &[usize],
        epochs: usize,
        learning_rate: f32,
        patience: usize,
    ) -> (usize, f32) {
        let mut best_val_acc = 0.0;
        let mut patience_counter = 0;
        let mut best_epoch = 0;

        println!("\n╔════════════════════════════════════════════════╗");
        println!("║     开始训练（早停: patience={}）          ║", patience);
        println!("╚════════════════════════════════════════════════╝\n");

        for epoch in 1..=epochs {
            let epoch_start = std::time::Instant::now();

            // 训练一个 epoch
            let mut total_loss = 0.0;
            let mut total_acc = 0.0;

            for (input, &target) in train_inputs.iter().zip(train_targets.iter()) {
                let input_batch = Array2::from_shape_vec((1, input.len()), input.clone()).unwrap();
                let target_onehot = Self::one_hot(target, self.n_classes);

                let (loss, acc) = self.train_step(&input_batch, &target_onehot, learning_rate);

                total_loss += loss;
                total_acc += acc;
            }

            let avg_train_loss = total_loss / train_inputs.len() as f32;
            let avg_train_acc = total_acc / train_inputs.len() as f32;

            // 验证
            let (val_loss, val_acc) = self.evaluate(val_inputs, val_targets);

            let epoch_time = epoch_start.elapsed().as_secs_f32();

            // 打印进度
            println!(
                "Epoch {:2}/{} | Train Loss: {:.4} | Train Acc: {:.2}% | Val Loss: {:.4} | Val Acc: {:.2}% | {:.2}s",
                epoch,
                epochs,
                avg_train_loss,
                avg_train_acc * 100.0,
                val_loss,
                val_acc * 100.0,
                epoch_time
            );

            // 检查是否是最佳模型
            if val_acc > best_val_acc {
                best_val_acc = val_acc;
                best_epoch = epoch;
                patience_counter = 0;
                println!("  ✨ 新的最佳模型！验证准确率: {:.2}%", val_acc * 100.0);
            } else {
                patience_counter += 1;
                println!("  ⏳ 验证准确率未提升 ({}/{})", patience_counter, patience);
            }

            // 早停检查
            if patience_counter >= patience {
                println!("\n⚠️  早停触发！验证准确率已 {} 个 epoch 未提升", patience);
                break;
            }

            println!();
        }

        println!("╔════════════════════════════════════════════════╗");
        println!("║     训练完成！                                ║");
        println!("╚════════════════════════════════════════════════╝");
        println!("总计 {} 个 epochs", best_epoch);
        println!("最佳验证准确率: {:.2}% (epoch {})", best_val_acc * 100.0, best_epoch);

        (best_epoch, best_val_acc)
    }

    /// 批处理训练循环（带早停）
    ///
    /// # 参数
    /// - `train_inputs`: 训练输入 [batch_size, seq_len]
    /// - `train_targets`: 训练标签
    /// - `val_inputs`: 验证输入
    /// - `val_targets`: 验证标签
    /// - `epochs`: 最大训练轮数
    /// - `batch_size`: 批大小
    /// - `learning_rate`: 初始学习率
    /// - `patience`: 早停耐心值
    ///
    /// # 返回
    /// - 训练的 epoch 数
    /// - 最佳验证准确率
    pub fn train_with_batches(
        &mut self,
        train_inputs: &Array2<usize>,
        train_targets: &[usize],
        val_inputs: &Array2<usize>,
        val_targets: &[usize],
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
        println!("║   批处理训练（batch_size={}, patience={}）║", batch_size, patience);
        println!("╚════════════════════════════════════════════════╝\n");

        for epoch in 1..=epochs {
            let epoch_start = std::time::Instant::now();

            // 训练一个 epoch（使用批处理）
            let mut total_loss = 0.0;
            let mut total_acc = 0.0;
            let mut n_batches = 0;

            // 打乱数据索引
            let mut indices: Vec<usize> = (0..n_train_samples).collect();
            use rand::seq::SliceRandom;
            indices.shuffle(&mut rand::thread_rng());

            // 按批次训练
            for batch_start in (0..n_train_samples).step_by(batch_size) {
                let batch_end = (batch_start + batch_size).min(n_train_samples);
                let batch_indices = &indices[batch_start..batch_end];

                // 收集批次数据
                let batch_inputs: Vec<Vec<usize>> = batch_indices
                    .iter()
                    .map(|&i| train_inputs.row(i).to_vec())
                    .collect();

                let batch_targets: Vec<usize> = batch_indices
                    .iter()
                    .map(|&i| train_targets[i])
                    .collect();

                // 构建批次张量
                let seq_len = batch_inputs[0].len();
                let mut batch_data = Vec::with_capacity(batch_indices.len() * seq_len);
                for input in &batch_inputs {
                    batch_data.extend(input);
                }

                let input_batch = Array2::from_shape_vec((batch_indices.len(), seq_len), batch_data).unwrap();
                let target_batch = Self::one_hot_batch(&batch_targets, self.n_classes);

                // 训练一个批次
                let (loss, acc) = self.train_batch(&input_batch, &target_batch, learning_rate);

                total_loss += loss;
                total_acc += acc;
                n_batches += 1;
            }

            let avg_train_loss = total_loss / n_batches as f32;
            let avg_train_acc = total_acc / n_batches as f32;

            // 验证（使用完整验证集）
            let val_inputs_vec: Vec<Vec<usize>> = val_inputs
                .rows()
                .into_iter()
                .map(|row| row.to_vec())
                .collect();

            let (val_loss, val_acc) = self.evaluate(&val_inputs_vec, val_targets);

            let epoch_time = epoch_start.elapsed().as_secs_f32();

            // 打印进度
            println!(
                "Epoch {:2}/{} | Train Loss: {:.4} | Train Acc: {:.2}% | Val Loss: {:.4} | Val Acc: {:.2}% | {:.2}s | {} batches",
                epoch,
                epochs,
                avg_train_loss,
                avg_train_acc * 100.0,
                val_loss,
                val_acc * 100.0,
                epoch_time,
                n_batches
            );

            // 检查是否是最佳模型
            if val_acc > best_val_acc {
                best_val_acc = val_acc;
                best_epoch = epoch;
                patience_counter = 0;
                println!("  ✨ 新的最佳模型！验证准确率: {:.2}%", val_acc * 100.0);
            } else {
                patience_counter += 1;
                println!("  ⏳ 验证准确率未提升 ({}/{})", patience_counter, patience);
            }

            // 早停检查
            if patience_counter >= patience {
                println!("\n⚠️  早停触发！验证准确率已 {} 个 epoch 未提升", patience);
                break;
            }

            println!();
        }

        println!("╔════════════════════════════════════════════════╗");
        println!("║     训练完成！                                ║");
        println!("╚════════════════════════════════════════════════╝");
        println!("总计 {} 个 epochs", best_epoch);
        println!("最佳验证准确率: {:.2}% (epoch {})", best_val_acc * 100.0, best_epoch);

        (best_epoch, best_val_acc)
    }

    /// 批处理训练循环（带早停和学习率调度）
    ///
    /// # 参数
    /// - `train_inputs`: 训练输入 [batch_size, seq_len]
    /// - `train_targets`: 训练标签
    /// - `val_inputs`: 验证输入
    /// - `val_targets`: 验证标签
    /// - `epochs`: 最大训练轮数
    /// - `batch_size`: 批大小
    /// - `initial_lr`: 初始学习率
    /// - `patience`: 早停耐心值
    /// - `scheduler`: 学习率调度器
    ///
    /// # 返回
    /// - 训练的 epoch 数
    /// - 最佳验证准确率
    pub fn train_with_scheduler<S: LRScheduler>(
        &mut self,
        train_inputs: &Array2<usize>,
        train_targets: &[usize],
        val_inputs: &Array2<usize>,
        val_targets: &[usize],
        epochs: usize,
        batch_size: usize,
        initial_lr: f32,
        patience: usize,
        scheduler: &S,
    ) -> (usize, f32) {
        let mut best_val_acc = 0.0;
        let mut patience_counter = 0;
        let mut best_epoch = 0;
        let n_train_samples = train_inputs.nrows();

        println!("\n╔════════════════════════════════════════════════╗");
        println!("║  优化训练（batch={}, scheduler={}）       ║",
                batch_size, scheduler.name());
        println!("║  早停: patience={}                           ║", patience);
        println!("╚════════════════════════════════════════════════╝\n");

        for epoch in 1..=epochs {
            let epoch_start = std::time::Instant::now();

            // 获取当前学习率
            let current_lr = scheduler.get_lr(epoch - 1);

            // 训练一个 epoch（使用批处理）
            let mut total_loss = 0.0;
            let mut total_acc = 0.0;
            let mut n_batches = 0;

            // 打乱数据索引
            let mut indices: Vec<usize> = (0..n_train_samples).collect();
            use rand::seq::SliceRandom;
            indices.shuffle(&mut rand::thread_rng());

            // 按批次训练
            for batch_start in (0..n_train_samples).step_by(batch_size) {
                let batch_end = (batch_start + batch_size).min(n_train_samples);
                let batch_indices = &indices[batch_start..batch_end];

                // 收集批次数据
                let batch_inputs: Vec<Vec<usize>> = batch_indices
                    .iter()
                    .map(|&i| train_inputs.row(i).to_vec())
                    .collect();

                let batch_targets: Vec<usize> = batch_indices
                    .iter()
                    .map(|&i| train_targets[i])
                    .collect();

                // 构建批次张量
                let seq_len = batch_inputs[0].len();
                let mut batch_data = Vec::with_capacity(batch_indices.len() * seq_len);
                for input in &batch_inputs {
                    batch_data.extend(input);
                }

                let input_batch = Array2::from_shape_vec((batch_indices.len(), seq_len), batch_data).unwrap();
                let target_batch = Self::one_hot_batch(&batch_targets, self.n_classes);

                // 训练一个批次（使用当前学习率）
                let (loss, acc) = self.train_batch(&input_batch, &target_batch, current_lr);

                total_loss += loss;
                total_acc += acc;
                n_batches += 1;
            }

            let avg_train_loss = total_loss / n_batches as f32;
            let avg_train_acc = total_acc / n_batches as f32;

            // 验证（使用完整验证集）
            let val_inputs_vec: Vec<Vec<usize>> = val_inputs
                .rows()
                .into_iter()
                .map(|row| row.to_vec())
                .collect();

            let (val_loss, val_acc) = self.evaluate(&val_inputs_vec, val_targets);

            let epoch_time = epoch_start.elapsed().as_secs_f32();

            // 打印进度（包含学习率）
            println!(
                "Epoch {:2}/{} | LR: {:.6} | Train Loss: {:.4} | Train Acc: {:.2}% | Val Loss: {:.4} | Val Acc: {:.2}% | {:.2}s | {} batches",
                epoch,
                epochs,
                current_lr,
                avg_train_loss,
                avg_train_acc * 100.0,
                val_loss,
                val_acc * 100.0,
                epoch_time,
                n_batches
            );

            // 检查是否是最佳模型
            if val_acc > best_val_acc {
                best_val_acc = val_acc;
                best_epoch = epoch;
                patience_counter = 0;
                println!("  ✨ 新的最佳模型！验证准确率: {:.2}%", val_acc * 100.0);
            } else {
                patience_counter += 1;
                println!("  ⏳ 验证准确率未提升 ({}/{})", patience_counter, patience);
            }

            // 早停检查
            if patience_counter >= patience {
                println!("\n⚠️  早停触发！验证准确率已 {} 个 epoch 未提升", patience);
                break;
            }

            println!();
        }

        println!("╔════════════════════════════════════════════════╗");
        println!("║     训练完成！                                ║");
        println!("╚════════════════════════════════════════════════╝");
        println!("总计 {} 个 epochs", best_epoch);
        println!("最佳验证准确率: {:.2}% (epoch {})", best_val_acc * 100.0, best_epoch);

        (best_epoch, best_val_acc)
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
