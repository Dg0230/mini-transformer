//! 模型检查点保存和加载
//!
//! 支持保存和加载训练状态、模型权重、优化器状态等

use ndarray::Array2;
use serde::{Deserialize, Serialize, Deserializer, Serializer};
use std::fs::{self, File};
use std::io::{BufReader, BufWriter};
use std::path::Path;

/// 可序列化的 Array2 包装器
///
/// ndarray::Array2 的序列化包装器，用于 serde
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableArray {
    /// 数据（行优先）
    pub data: Vec<f32>,
    /// 形状 (rows, cols)
    pub shape: (usize, usize),
}

impl From<Array2<f32>> for SerializableArray {
    fn from(arr: Array2<f32>) -> Self {
        let shape = (arr.nrows(), arr.ncols());
        let data = arr.into_raw_vec();
        Self { data, shape }
    }
}

impl From<SerializableArray> for Array2<f32> {
    fn from(sarr: SerializableArray) -> Self {
        Array2::from_shape_vec(sarr.shape, sarr.data).unwrap()
    }
}

impl SerializableArray {
    /// 转换为 Array2
    pub fn to_array(&self) -> Array2<f32> {
        Array2::from_shape_vec(self.shape, self.data.clone()).unwrap()
    }

    /// 从 Array2 创建
    pub fn from_array(arr: &Array2<f32>) -> Self {
        let shape = (arr.nrows(), arr.ncols());
        let data = arr.as_slice().unwrap().to_vec();
        Self { data, shape }
    }
}

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

/// 模型状态
///
/// 包含完整的模型权重
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelState {
    /// 嵌入层权重
    pub embedding_weights: Vec<SerializableArray>,
    /// 编码器层权重
    pub encoder_weights: Vec<LayerWeights>,
    /// 分类器权重
    pub classifier_weights: ClassifierWeights,
    /// 模型配置
    pub config: ModelConfig,
}

/// 单层的权重
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerWeights {
    /// Q, K, V, O 投影权重
    pub attn_weights: AttnWeights,
    /// FFN 权重
    pub ffn_weights: FFNWeights,
    /// LayerNorm 参数
    pub norm1_gamma: SerializableArray,
    pub norm1_beta: SerializableArray,
    pub norm2_gamma: SerializableArray,
    pub norm2_beta: SerializableArray,
}

/// 注意力权重
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttnWeights {
    pub w_q_weight: SerializableArray,
    pub w_q_bias: SerializableArray,
    pub w_k_weight: SerializableArray,
    pub w_k_bias: SerializableArray,
    pub w_v_weight: SerializableArray,
    pub w_v_bias: SerializableArray,
    pub w_o_weight: SerializableArray,
    pub w_o_bias: SerializableArray,
}

/// FFN 权重
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FFNWeights {
    pub linear1_weight: SerializableArray,
    pub linear1_bias: SerializableArray,
    pub linear2_weight: SerializableArray,
    pub linear2_bias: SerializableArray,
}

/// 分类器权重
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassifierWeights {
    pub weight: SerializableArray,
    pub bias: SerializableArray,
}

/// 模型配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub vocab_size: usize,
    pub d_model: usize,
    pub n_heads: usize,
    pub n_layers: usize,
    pub d_ff: usize,
    pub max_seq_len: usize,
    pub n_classes: usize,
}

/// 优化器状态（Adam）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizerState {
    /// 学习率
    pub learning_rate: f32,
    /// 一阶矩估计（m）
    pub moments: Vec<SerializableArray>,
    /// 二阶矩估计（v）
    pub variances: Vec<SerializableArray>,
    /// 时间步
    pub timestep: usize,
}

/// 完整检查点（包含模型和优化器状态）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FullCheckpoint {
    /// 模型状态
    pub model: ModelState,
    /// 优化器状态
    pub optimizer: Option<OptimizerState>,
    /// 训练状态
    pub training: TrainingState,
}

/// 训练状态
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingState {
    /// 当前 epoch
    pub epoch: usize,
    /// 训练损失
    pub train_loss: f32,
    /// 验证损失
    pub val_loss: f32,
    /// 训练准确率
    pub train_acc: f32,
    /// 验证准确率
    pub val_acc: f32,
    /// 全局步数
    pub global_step: usize,
}

impl FullCheckpoint {
    /// 保存完整检查点
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<(), Box<dyn std::error::Error>> {
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        bincode::serialize_into(writer, self)?;
        Ok(())
    }

    /// 加载完整检查点
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let checkpoint = bincode::deserialize_from(reader)?;
        Ok(checkpoint)
    }

    /// 保存为 JSON（可读性更好）
    pub fn save_json<P: AsRef<Path>>(&self, path: P) -> Result<(), Box<dyn std::error::Error>> {
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        serde_json::to_writer_pretty(writer, self)?;
        Ok(())
    }

    /// 从 JSON 加载
    pub fn load_json<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let checkpoint = serde_json::from_reader(reader)?;
        Ok(checkpoint)
    }
}

/// 模型保存/加载 trait
pub trait ModelSaveLoad {
    /// 保存模型状态
    fn save_model(&self) -> Result<ModelState, Box<dyn std::error::Error>>;

    /// 加载模型状态
    fn load_model(&mut self, state: ModelState) -> Result<(), Box<dyn std::error::Error>>;

    /// 保存模型到文件
    fn save<P: AsRef<Path>>(&self, path: P) -> Result<(), Box<dyn std::error::Error>> {
        let state = self.save_model()?;
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        bincode::serialize_into(writer, &state)?;
        Ok(())
    }

    /// 从文件加载模型
    fn load<P: AsRef<Path>>(&mut self, path: P) -> Result<(), Box<dyn std::error::Error>> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let state: ModelState = bincode::deserialize_from(reader)?;
        self.load_model(state)
    }

    /// 保存为 JSON
    fn save_json<P: AsRef<Path>>(&self, path: P) -> Result<(), Box<dyn std::error::Error>> {
        let state = self.save_model()?;
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        serde_json::to_writer_pretty(writer, &state)?;
        Ok(())
    }

    /// 从 JSON 加载
    fn load_json<P: AsRef<Path>>(&mut self, path: P) -> Result<(), Box<dyn std::error::Error>> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let state: ModelState = serde_json::from_reader(reader)?;
        self.load_model(state)
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
