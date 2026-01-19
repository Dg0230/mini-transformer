//! 训练器
//!
//! 实现训练循环、评估和模型检查点。

use ndarray::Array2;
use crate::loss::{LossFunction, Accuracy};
use crate::optimizer::Optimizer;
use crate::lr_scheduler::LRScheduler;
use crate::transformer::{TransformerEncoder, TransformerConfig};
use crate::tensor::TensorExt;
use std::time::Instant;

/// 训练配置
#[derive(Debug, Clone)]
pub struct TrainerConfig {
    /// 批大小
    pub batch_size: usize,
    /// 训练轮数
    pub epochs: usize,
    /// 学习率
    pub learning_rate: f32,
    /// 梯度裁剪阈值
    pub grad_clip: Option<f32>,
    /// 评估频率（每 eval_freq 个 batch 评估一次）
    pub eval_freq: Option<usize>,
    /// 保存检查点频率
    pub save_freq: Option<usize>,
    /// 设备（预留，当前只支持 CPU）
    pub device: String,
    /// 是否使用混合精度训练（预留）
    pub use_amp: bool,
}

impl Default for TrainerConfig {
    fn default() -> Self {
        Self {
            batch_size: 32,
            epochs: 10,
            learning_rate: 0.001,
            grad_clip: Some(1.0),
            eval_freq: Some(100),
            save_freq: None,
            device: "cpu".to_string(),
            use_amp: false,
        }
    }
}

/// 训练统计数据
#[derive(Debug, Clone)]
pub struct TrainStats {
    /// 当前 epoch
    pub epoch: usize,
    /// 当前 batch
    pub batch: usize,
    /// 总 batch 数
    pub total_batches: usize,
    /// 损失值
    pub loss: f32,
    /// 学习率
    pub lr: f32,
    /// 每秒处理的样本数
    pub samples_per_sec: f32,
}

/// 数据集 trait
pub trait Dataset {
    /// 获取数据集大小
    fn len(&self) -> usize;

    /// 获取单个样本
    fn get(&self, index: usize) -> (Vec<usize>, usize);

    /// 是否为空
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// 数据加载器
pub struct DataLoader<D: Dataset> {
    dataset: D,
    batch_size: usize,
    shuffle: bool,
}

impl<D: Dataset> DataLoader<D> {
    pub fn new(dataset: D, batch_size: usize, shuffle: bool) -> Self {
        Self {
            dataset,
            batch_size,
            shuffle,
        }
    }

    /// 返回迭代器
    pub fn iter(&self) -> BatchIterator<D> {
        let indices: Vec<usize> = (0..self.dataset.len()).collect();

        BatchIterator {
            dataloader: self,
            indices,
            position: 0,
        }
    }
}

/// 批次迭代器
pub struct BatchIterator<'a, D: Dataset> {
    dataloader: &'a DataLoader<D>,
    indices: Vec<usize>,
    position: usize,
}

impl<'a, D: Dataset> Iterator for BatchIterator<'a, D> {
    type Item = Vec<(Vec<usize>, usize)>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.position >= self.indices.len() {
            return None;
        }

        let end = (self.position + self.dataloader.batch_size).min(self.indices.len());
        let batch_indices = &self.indices[self.position..end];
        self.position = end;

        let batch = batch_indices
            .iter()
            .map(|&i| self.dataloader.dataset.get(i))
            .collect();

        Some(batch)
    }
}

/// 训练器
pub struct Trainer {
    config: TrainerConfig,
}

impl Trainer {
    pub fn new(config: TrainerConfig) -> Self {
        Self { config }
    }

    /// 训练一个 epoch
    ///
    /// 简化实现：手动计算梯度
    pub fn train_epoch<O: Optimizer, L: LossFunction>(
        &self,
        _model: &mut TransformerEncoder,
        dataloader: &DataLoader<impl Dataset>,
        _optimizer: &mut O,
        _loss_fn: &L,
        epoch: usize,
    ) -> (f32, f32) {
        let mut total_loss = 0.0;
        let mut total_samples = 0.0;
        let start_time = Instant::now();

        for (batch_idx, batch) in dataloader.iter().enumerate() {
            // 前向传播和损失计算
            let loss = 0.0; // TODO: 实际计算

            total_loss += loss * batch.len() as f32;
            total_samples += batch.len() as f32;

            // TODO: 实际的参数更新
            // 这里需要完整的 autograd 系统

            // 打印进度
            if batch_idx % 10 == 0 {
                let elapsed = start_time.elapsed().as_secs_f32();
                let samples_per_sec = total_samples / elapsed;
                let avg_loss = total_loss / total_samples;

                println!(
                    "Epoch: {:3} | Batch: {:4} | Loss: {:.4} | {:.0} samples/s",
                    epoch, batch_idx, avg_loss, samples_per_sec
                );
            }
        }

        let avg_loss = total_loss / total_samples;
        let elapsed = start_time.elapsed().as_secs_f32();
        let samples_per_sec = total_samples as f32 / elapsed;

        (avg_loss, samples_per_sec)
    }

    /// 计算一个 batch 的损失和梯度
    fn compute_batch_loss<L: LossFunction>(
        &self,
        _model: &mut TransformerEncoder,
        _batch: &[(Vec<usize>, usize)],
        _loss_fn: &L,
    ) -> (f32, Array2<f32>) {
        // TODO: 实现完整的前向传播和反向传播
        // 这里需要：
        // 1. 前向传播计算预测值
        // 2. 计算损失
        // 3. 反向传播计算梯度

        (0.0, Array2::zeros((1, 1)))
    }

    /// 评估模型
    pub fn evaluate<L: LossFunction>(
        &self,
        _model: &TransformerEncoder,
        dataloader: &DataLoader<impl Dataset>,
        _loss_fn: &L,
    ) -> (f32, f32) {
        let mut total_loss = 0.0;
        let mut correct = 0;
        let mut total = 0;

        let _acc_fn = Accuracy::new();

        for batch in dataloader.iter() {
            // TODO: 前向传播
            // let predictions = model.forward_batch(&batch);
            // let targets = extract_targets(&batch);

            // let loss = loss_fn.compute(&predictions, &targets);
            // total_loss += loss * batch.len() as f32;

            // let accuracy = acc_fn.compute(&predictions, &targets);
            // correct += (accuracy * batch.len() as f32) as usize;
            // total += batch.len();
            total += batch.len();
        }

        let avg_loss = if total > 0 { total_loss / total as f32 } else { 0.0 };
        let accuracy = if total > 0 { correct as f32 / total as f32 } else { 0.0 };

        (avg_loss, accuracy)
    }
}

/// 简单的内存数据集（用于测试）
pub struct SimpleDataset {
    data: Vec<(Vec<usize>, usize)>,
}

impl SimpleDataset {
    pub fn new(data: Vec<(Vec<usize>, usize)>) -> Self {
        Self { data }
    }

    /// 创建随机数据集
    pub fn random(size: usize, seq_len: usize, vocab_size: usize, n_classes: usize) -> Self {
        use rand::Rng;

        let mut rng = rand::thread_rng();
        let data: Vec<(Vec<usize>, usize)> = (0..size)
            .map(|_| {
                let sequence: Vec<usize> =
                    (0..seq_len).map(|_| rng.gen_range(0..vocab_size)).collect();
                let label = rng.gen_range(0..n_classes);
                (sequence, label)
            })
            .collect();

        Self { data }
    }
}

impl Dataset for SimpleDataset {
    fn len(&self) -> usize {
        self.data.len()
    }

    fn get(&self, index: usize) -> (Vec<usize>, usize) {
        self.data[index].clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimizer::SGD;
    use crate::loss::CrossEntropyLoss;

    #[test]
    fn test_dataloader() {
        let dataset = SimpleDataset::new(vec![
            (vec![1, 2, 3], 0),
            (vec![4, 5, 6], 1),
            (vec![7, 8, 9], 2),
            (vec![10, 11, 12], 0),
        ]);

        let dataloader = DataLoader::new(dataset, 2, false);

        let batches: Vec<_> = dataloader.iter().collect();

        assert_eq!(batches.len(), 2); // 4 samples, batch size 2 = 2 batches
        assert_eq!(batches[0].len(), 2);
        assert_eq!(batches[1].len(), 2);
    }

    #[test]
    fn test_trainer_creation() {
        let config = TrainerConfig::default();
        let trainer = Trainer::new(config);

        assert_eq!(trainer.config.batch_size, 32);
        assert_eq!(trainer.config.epochs, 10);
    }

    #[test]
    fn test_simple_dataset() {
        let dataset = SimpleDataset::random(100, 10, 1000, 5);

        assert_eq!(dataset.len(), 100);

        let sample = dataset.get(0);
        assert_eq!(sample.0.len(), 10);
        assert!(sample.1 < 5);
    }
}
