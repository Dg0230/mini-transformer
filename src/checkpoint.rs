//! 模型检查点保存和加载
//!
//! 支持保存和加载训练状态、模型权重等

use ndarray::Array2;
use serde::{Deserialize, Serialize};
use std::fs::{self, File};
use std::io::{BufReader, BufWriter};
use std::path::Path;

/// 训练检查点
///
/// 包含模型权重和训练状态
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Checkpoint {
    /// epoch 编号
    pub epoch: usize,
    /// 训练损失
    pub train_loss: f32,
    /// 验证损失
    pub val_loss: f32,
    /// 训练准确率
    pub train_acc: f32,
    /// 验证准确率
    pub val_acc: f32,
    /// 学习率
    pub learning_rate: f32,
    /// 模型权重（展平的向量）
    pub weights: Vec<f32>,
    /// 权重形状信息
    pub weight_shapes: Vec<(usize, usize)>,
}

impl Checkpoint {
    /// 创建新的检查点
    pub fn new(
        epoch: usize,
        train_loss: f32,
        val_loss: f32,
        train_acc: f32,
        val_acc: f32,
        learning_rate: f32,
    ) -> Self {
        Self {
            epoch,
            train_loss,
            val_loss,
            train_acc,
            val_acc,
            learning_rate,
            weights: Vec::new(),
            weight_shapes: Vec::new(),
        }
    }

    /// 保存为 JSON 格式
    pub fn save_json<P: AsRef<Path>>(&self, path: P) -> Result<(), Box<dyn std::error::Error>> {
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        serde_json::to_writer_pretty(writer, self)?;
        Ok(())
    }

    /// 从 JSON 格式加载
    pub fn load_json<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let checkpoint = serde_json::from_reader(reader)?;
        Ok(checkpoint)
    }

    /// 保存为二进制格式（更紧凑）
    pub fn save_binary<P: AsRef<Path>>(&self, path: P) -> Result<(), Box<dyn std::error::Error>> {
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        bincode::serialize_into(writer, self)?;
        Ok(())
    }

    /// 从二进制格式加载
    pub fn load_binary<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let checkpoint = bincode::deserialize_from(reader)?;
        Ok(checkpoint)
    }

    /// 获取检查点信息
    pub fn info(&self) -> String {
        format!(
            "Checkpoint at epoch {}:\n  Train Loss: {:.4}, Acc: {:.2}%\n  Val Loss: {:.4}, Acc: {:.2}%\n  LR: {:.6}",
            self.epoch,
            self.train_loss,
            self.train_acc * 100.0,
            self.val_loss,
            self.val_acc * 100.0,
            self.learning_rate
        )
    }
}

/// 训练历史记录
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingHistory {
    /// 训练损失历史
    pub train_losses: Vec<f32>,
    /// 验证损失历史
    pub val_losses: Vec<f32>,
    /// 训练准确率历史
    pub train_accs: Vec<f32>,
    /// 验证准确率历史
    pub val_accs: Vec<f32>,
    /// 学习率历史
    pub learning_rates: Vec<f32>,
}

impl Default for TrainingHistory {
    fn default() -> Self {
        Self::new()
    }
}

impl TrainingHistory {
    /// 创建新的训练历史
    pub fn new() -> Self {
        Self {
            train_losses: Vec::new(),
            val_losses: Vec::new(),
            train_accs: Vec::new(),
            val_accs: Vec::new(),
            learning_rates: Vec::new(),
        }
    }

    /// 添加一个 epoch 的记录
    pub fn push(&mut self, train_loss: f32, val_loss: f32, train_acc: f32, val_acc: f32, lr: f32) {
        self.train_losses.push(train_loss);
        self.val_losses.push(val_loss);
        self.train_accs.push(train_acc);
        self.val_accs.push(val_acc);
        self.learning_rates.push(lr);
    }

    /// 保存训练历史
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<(), Box<dyn std::error::Error>> {
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        serde_json::to_writer_pretty(writer, self)?;
        Ok(())
    }

    /// 加载训练历史
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let history = serde_json::from_reader(reader)?;
        Ok(history)
    }

    /// 获取最佳验证准确率
    pub fn best_val_acc(&self) -> Option<f32> {
        self.val_accs.iter()
            .cloned()
            .reduce(f32::max)
    }

    /// 获取最佳 epoch
    pub fn best_epoch(&self) -> Option<usize> {
        self.val_accs.iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
    }

    /// 打印训练历史摘要
    pub fn summary(&self) -> String {
        let n_epochs = self.train_losses.len();
        let best_acc = self.best_val_acc().unwrap_or(0.0);
        let best_epoch = self.best_epoch().unwrap_or(0);

        format!(
            "Training Summary:\n  Epochs: {}\n  Best Val Acc: {:.2}% (epoch {})\n  Final Train Loss: {:.4}\n  Final Val Loss: {:.4}",
            n_epochs,
            best_acc * 100.0,
            best_epoch + 1,
            self.train_losses.last().unwrap_or(&0.0),
            self.val_losses.last().unwrap_or(&0.0)
        )
    }
}

/// 检查点管理器
///
/// 管理训练过程中的检查点保存和加载
pub struct CheckpointManager {
    /// 保存目录
    save_dir: String,
    /// 保留的最佳检查点数量
    keep_best: usize,
    /// 当前最佳检查点
    best_checkpoint: Option<Checkpoint>,
}

impl CheckpointManager {
    /// 创建新的检查点管理器
    pub fn new(save_dir: &str, keep_best: usize) -> Self {
        // 创建保存目录
        fs::create_dir_all(save_dir).unwrap_or_else(|e| {
            eprintln!("Warning: Could not create checkpoint directory: {}", e);
        });

        Self {
            save_dir: save_dir.to_string(),
            keep_best,
            best_checkpoint: None,
        }
    }

    /// 保存检查点（如果是最佳模型）
    pub fn save_if_best(
        &mut self,
        epoch: usize,
        train_loss: f32,
        val_loss: f32,
        train_acc: f32,
        val_acc: f32,
        lr: f32,
    ) -> bool {
        let is_best = match &self.best_checkpoint {
            Some(best) => val_acc > best.val_acc,
            None => true,
        };

        if is_best {
            let checkpoint = Checkpoint::new(epoch, train_loss, val_loss, train_acc, val_acc, lr);

            // 保存最佳检查点
            let path = format!("{}/best_checkpoint.json", self.save_dir);
            if let Err(e) = checkpoint.save_json(&path) {
                eprintln!("Warning: Could not save checkpoint: {}", e);
            } else {
                println!("  💾 Saved new best checkpoint (val_acc: {:.2}%)", val_acc * 100.0);
            }

            self.best_checkpoint = Some(checkpoint);
            true
        } else {
            false
        }
    }

    /// 加载最佳检查点
    pub fn load_best(&self) -> Option<Checkpoint> {
        let path = format!("{}/best_checkpoint.json", self.save_dir);
        match Checkpoint::load_json(&path) {
            Ok(checkpoint) => {
                println!("  📥 Loaded checkpoint from epoch {}", checkpoint.epoch);
                Some(checkpoint)
            }
            Err(e) => {
                eprintln!("Warning: Could not load checkpoint: {}", e);
                None
            }
        }
    }

    /// 保存训练历史
    pub fn save_history(&self, history: &TrainingHistory) {
        let path = format!("{}/training_history.json", self.save_dir);
        if let Err(e) = history.save(&path) {
            eprintln!("Warning: Could not save training history: {}", e);
        }
    }

    /// 加载训练历史
    pub fn load_history(&self) -> Option<TrainingHistory> {
        let path = format!("{}/training_history.json", self.save_dir);
        match TrainingHistory::load(&path) {
            Ok(history) => {
                println!("  📥 Loaded training history");
                Some(history)
            }
            Err(e) => {
                eprintln!("Warning: Could not load training history: {}", e);
                None
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_checkpoint_creation() {
        let checkpoint = Checkpoint::new(1, 0.5, 0.6, 0.8, 0.75, 0.01);
        assert_eq!(checkpoint.epoch, 1);
        assert_eq!(checkpoint.train_loss, 0.5);
    }

    #[test]
    fn test_training_history() {
        let mut history = TrainingHistory::new();
        history.push(0.5, 0.6, 0.8, 0.75, 0.01);
        history.push(0.4, 0.55, 0.85, 0.8, 0.01);

        assert_eq!(history.train_losses.len(), 2);
        assert_eq!(history.best_val_acc(), Some(0.8));
        assert_eq!(history.best_epoch(), Some(1));
    }

    #[test]
    fn test_checkpoint_save_load() {
        let checkpoint = Checkpoint::new(5, 0.3, 0.4, 0.9, 0.85, 0.001);

        // 保存为 JSON
        let json_path = "/tmp/test_checkpoint.json";
        checkpoint.save_json(json_path).unwrap();

        // 加载 JSON
        let loaded = Checkpoint::load_json(json_path).unwrap();
        assert_eq!(loaded.epoch, 5);
        assert_eq!(loaded.train_loss, 0.3);

        // 清理
        std::fs::remove_file(json_path).ok();
    }

    #[test]
    fn test_checkpoint_manager() {
        let mut manager = CheckpointManager::new("/tmp/test_checkpoints", 3);

        // 第一次保存应该是最佳
        let saved = manager.save_if_best(1, 0.5, 0.6, 0.8, 0.75, 0.01);
        assert!(saved);

        // 更好的验证准确率
        let saved = manager.save_if_best(2, 0.4, 0.5, 0.85, 0.8, 0.01);
        assert!(saved);

        // 更差的验证准确率
        let saved = manager.save_if_best(3, 0.3, 0.6, 0.9, 0.7, 0.01);
        assert!(!saved);

        // 清理
        std::fs::remove_dir_all("/tmp/test_checkpoints").ok();
    }
}
