//! 早停机制
//!
//! 基于验证集指标的早停，防止过拟合

/// 早停配置
#[derive(Debug, Clone)]
pub struct EarlyStoppingConfig {
    /// 耐心值：验证指标没有改善的 epoch 数
    /// 如果超过这个值，则停止训练
    pub patience: usize,

    /// 最小改善阈值
    /// 只有改善幅度超过此值才算改善
    pub min_delta: f32,

    /// 模式：min 表示指标越小越好（如 loss），max 表示越大越好（如 accuracy）
    pub mode: EarlyStoppingMode,

    /// 是否恢复最佳权重
    pub restore_best_weights: bool,

    /// 早停后是否调用回调（用于保存检查点等）
    pub callback_on_stop: bool,
}

/// 早停模式
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EarlyStoppingMode {
    /// 指标越小越好（如 loss）
    Min,
    /// 指标越大越好（如 accuracy）
    Max,
}

impl EarlyStoppingMode {
    /// 判断新指标是否优于旧指标
    pub fn is_better(&self, old_value: f32, new_value: f32, min_delta: f32) -> bool {
        match self {
            EarlyStoppingMode::Min => new_value < old_value - min_delta,
            EarlyStoppingMode::Max => new_value > old_value + min_delta,
        }
    }

    /// 获取初始值
    pub fn initial_value(&self) -> f32 {
        match self {
            EarlyStoppingMode::Min => f32::INFINITY,
            EarlyStoppingMode::Max => f32::NEG_INFINITY,
        }
    }
}

impl Default for EarlyStoppingConfig {
    fn default() -> Self {
        Self {
            patience: 5,
            min_delta: 0.0,
            mode: EarlyStoppingMode::Min,
            restore_best_weights: true,
            callback_on_stop: true,
        }
    }
}

impl EarlyStoppingConfig {
    /// 创建新的配置
    pub fn new(patience: usize, mode: EarlyStoppingMode) -> Self {
        Self {
            patience,
            min_delta: 0.0,
            mode,
            restore_best_weights: true,
            callback_on_stop: true,
        }
    }

    /// 设置最小改善阈值
    pub fn with_min_delta(mut self, min_delta: f32) -> Self {
        self.min_delta = min_delta;
        self
    }

    /// 设置是否恢复最佳权重
    pub fn with_restore_best_weights(mut self, restore: bool) -> Self {
        self.restore_best_weights = restore;
        self
    }

    /// 设置是否在停止时调用回调
    pub fn with_callback(mut self, callback: bool) -> Self {
        self.callback_on_stop = callback;
        self
    }

    /// 创建最小化模式（用于 loss）
    pub fn min(patience: usize) -> Self {
        Self::new(patience, EarlyStoppingMode::Min)
    }

    /// 创建最大化模式（用于 accuracy）
    pub fn max(patience: usize) -> Self {
        Self::new(patience, EarlyStoppingMode::Max)
    }
}

/// 早停器
///
/// # 使用示例
///
/// ```rust
/// use mini_transformer::early_stopping::{EarlyStopping, EarlyStoppingConfig};
///
/// let config = EarlyStoppingConfig::min(5);  // 5 epochs without improvement -> stop
/// let mut early_stopping = EarlyStopping::new(config);
///
/// for epoch in 0..100 {
///     // ... 训练 ...
///     let val_loss = 0.5;  // 假设验证损失
///
///     if early_stopping.update(val_loss) {
///         println!("Early stopping at epoch {}", epoch);
///         break;
///     }
/// }
/// ```
#[derive(Debug, Clone)]
pub struct EarlyStopping {
    config: EarlyStoppingConfig,
    best_score: f32,
    epochs_no_improve: usize,
    best_epoch: usize,
    stopped_epoch: Option<usize>,
    should_stop: bool,
}

impl EarlyStopping {
    /// 创建新的早停器
    pub fn new(config: EarlyStoppingConfig) -> Self {
        Self {
            best_score: config.mode.initial_value(),
            epochs_no_improve: 0,
            best_epoch: 0,
            stopped_epoch: None,
            should_stop: false,
            config,
        }
    }

    /// 更新早停器状态
    ///
    /// # Arguments
    /// * `current_score` - 当前验证指标
    /// * `epoch` - 当前 epoch 编号
    ///
    /// # Returns
    /// * `true` - 应该停止训练
    /// * `false` - 继续训练
    pub fn update(&mut self, current_score: f32, epoch: usize) -> bool {
        if self.should_stop {
            return true;
        }

        // 检查是否有改善
        if self.config.mode.is_better(self.best_score, current_score, self.config.min_delta) {
            // 有改善
            self.best_score = current_score;
            self.epochs_no_improve = 0;
            self.best_epoch = epoch;
        } else {
            // 无改善
            self.epochs_no_improve += 1;
        }

        // 检查是否应该停止
        if self.epochs_no_improve >= self.config.patience {
            self.stopped_epoch = Some(epoch);
            self.should_stop = true;
            return true;
        }

        false
    }

    /// 更新状态（简化版，自动计数 epoch）
    pub fn update_simple(&mut self, current_score: f32) -> bool {
        let epoch = self.best_epoch + self.epochs_no_improve + 1;
        self.update(current_score, epoch)
    }

    /// 重置早停器状态
    pub fn reset(&mut self) {
        self.best_score = self.config.mode.initial_value();
        self.epochs_no_improve = 0;
        self.best_epoch = 0;
        self.stopped_epoch = None;
        self.should_stop = false;
    }

    /// 获取最佳分数
    pub fn best_score(&self) -> f32 {
        self.best_score
    }

    /// 获取未改善的 epoch 数
    pub fn epochs_no_improve(&self) -> usize {
        self.epochs_no_improve
    }

    /// 获取最佳 epoch 编号
    pub fn best_epoch(&self) -> usize {
        self.best_epoch
    }

    /// 获取停止的 epoch（如果已停止）
    pub fn stopped_epoch(&self) -> Option<usize> {
        self.stopped_epoch
    }

    /// 是否应该停止训练
    pub fn should_stop(&self) -> bool {
        self.should_stop
    }

    /// 是否需要恢复最佳权重
    pub fn should_restore_best_weights(&self) -> bool {
        self.config.restore_best_weights && self.stopped_epoch.is_some()
    }

    /// 获取剩余耐心值
    pub fn remaining_patience(&self) -> usize {
        self.config.patience.saturating_sub(self.epochs_no_improve)
    }

    /// 获取改进进度（0.0 - 1.0）
    ///
    /// 0.0 表示刚开始，1.0 表示即将触发早停
    pub fn progress(&self) -> f32 {
        if self.config.patience == 0 {
            return 1.0;
        }
        self.epochs_no_improve as f32 / self.config.patience as f32
    }

    /// 获取早停原因描述
    pub fn stop_reason(&self) -> String {
        if !self.should_stop {
            return "Training is ongoing".to_string();
        }

        format!(
            "No improvement for {} epochs (patience: {}), best score: {:.6} at epoch {}",
            self.epochs_no_improve,
            self.config.patience,
            self.best_score,
            self.best_epoch
        )
    }

    /// 打印早停状态摘要
    pub fn print_summary(&self) {
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("早停状态摘要:");
        println!("  最佳分数: {:.6}", self.best_score);
        println!("  最佳 epoch: {}", self.best_epoch);
        println!("  未改善 epochs: {}", self.epochs_no_improve);
        println!("  耐心值: {}", self.config.patience);
        println!("  剩余耐心: {}", self.remaining_patience());
        if let Some(stopped) = self.stopped_epoch {
            println!("  停止 epoch: {}", stopped);
            println!("  早停原因: {}", self.stop_reason());
        }
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    }
}

/// 早停回调函数类型
pub type EarlyStoppingCallback = fn(epoch: usize, best_score: f32, reason: &str);

/// 带回调的早停器
///
/// 允许在触发早停时执行自定义操作（如保存检查点）
#[derive(Debug, Clone)]
pub struct EarlyStoppingWithCallback {
    early_stopping: EarlyStopping,
    callback: Option<EarlyStoppingCallback>,
}

impl EarlyStoppingWithCallback {
    /// 创建新的带回调的早停器
    pub fn new(config: EarlyStoppingConfig) -> Self {
        Self {
            early_stopping: EarlyStopping::new(config),
            callback: None,
        }
    }

    /// 设置回调函数
    pub fn with_callback(mut self, callback: EarlyStoppingCallback) -> Self {
        self.callback = Some(callback);
        self
    }

    /// 更新状态并检查是否应该停止
    pub fn update(&mut self, current_score: f32, epoch: usize) -> bool {
        let should_stop = self.early_stopping.update(current_score, epoch);

        if should_stop && self.early_stopping.config.callback_on_stop {
            if let Some(callback) = self.callback {
                callback(
                    epoch,
                    self.early_stopping.best_score,
                    &self.early_stopping.stop_reason(),
                );
            }
        }

        should_stop
    }

    /// 获取内部早停器的引用
    pub fn inner(&self) -> &EarlyStopping {
        &self.early_stopping
    }

    /// 获取内部早停器的可变引用
    pub fn inner_mut(&mut self) -> &mut EarlyStopping {
        &mut self.early_stopping
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_early_stopping_config_default() {
        let config = EarlyStoppingConfig::default();
        assert_eq!(config.patience, 5);
        assert_eq!(config.min_delta, 0.0);
        assert_eq!(config.mode, EarlyStoppingMode::Min);
        assert!(config.restore_best_weights);
    }

    #[test]
    fn test_early_stopping_mode_min() {
        let mode = EarlyStoppingMode::Min;

        // Loss 下降应该算改善
        assert!(mode.is_better(1.0, 0.5, 0.0));
        assert!(mode.is_better(1.0, 0.85, 0.1)); // 下降 0.15 > min_delta 0.1

        // Loss 上升不算改善
        assert!(!mode.is_better(1.0, 1.5, 0.0));
        assert!(!mode.is_better(1.0, 0.95, 0.1)); // 下降 0.05 < min_delta 0.1，不算改善

        assert_eq!(mode.initial_value(), f32::INFINITY);
    }

    #[test]
    fn test_early_stopping_mode_max() {
        let mode = EarlyStoppingMode::Max;

        // Accuracy 上升应该算改善
        assert!(mode.is_better(0.5, 0.8, 0.0));
        assert!(mode.is_better(0.5, 0.65, 0.1)); // 上升 0.15 > min_delta 0.1

        // Accuracy 下降不算改善
        assert!(!mode.is_better(0.8, 0.5, 0.0));
        assert!(!mode.is_better(0.8, 0.85, 0.1)); // 上升 0.05 < min_delta 0.1，不算改善

        assert_eq!(mode.initial_value(), f32::NEG_INFINITY);
    }

    #[test]
    fn test_early_stopping_min_mode() {
        let config = EarlyStoppingConfig::min(3);
        let mut early_stopping = EarlyStopping::new(config);

        // Epoch 0: loss = 1.0 (最佳)
        assert!(!early_stopping.update(1.0, 0));
        assert_eq!(early_stopping.best_score, 1.0);
        assert_eq!(early_stopping.epochs_no_improve, 0);

        // Epoch 1: loss = 0.8 (改善)
        assert!(!early_stopping.update(0.8, 1));
        assert_eq!(early_stopping.best_score, 0.8);
        assert_eq!(early_stopping.epochs_no_improve, 0);

        // Epoch 2-4: loss 保持 0.85 (无改善)
        assert!(!early_stopping.update(0.85, 2));
        assert_eq!(early_stopping.epochs_no_improve, 1);

        assert!(!early_stopping.update(0.85, 3));
        assert_eq!(early_stopping.epochs_no_improve, 2);

        // Epoch 4: loss = 0.85 (第3次无改善，触发早停)
        assert!(early_stopping.update(0.85, 4));
        assert_eq!(early_stopping.epochs_no_improve, 3);
        assert_eq!(early_stopping.stopped_epoch, Some(4));
        assert!(early_stopping.should_stop());
    }

    #[test]
    fn test_early_stopping_max_mode() {
        let config = EarlyStoppingConfig::max(2);
        let mut early_stopping = EarlyStopping::new(config);

        // Epoch 0: accuracy = 0.7 (最佳)
        assert!(!early_stopping.update(0.7, 0));
        assert_eq!(early_stopping.best_score, 0.7);

        // Epoch 1: accuracy = 0.75 (改善)
        assert!(!early_stopping.update(0.75, 1));
        assert_eq!(early_stopping.best_score, 0.75);

        // Epoch 2: accuracy = 0.73 (无改善)
        assert!(!early_stopping.update(0.73, 2));
        assert_eq!(early_stopping.epochs_no_improve, 1);

        // Epoch 3: accuracy = 0.72 (第2次无改善，触发早停)
        assert!(early_stopping.update(0.72, 3));
        assert_eq!(early_stopping.stopped_epoch, Some(3));
    }

    #[test]
    fn test_early_stopping_with_min_delta() {
        let config = EarlyStoppingConfig::min(3).with_min_delta(0.1);
        let mut early_stopping = EarlyStopping::new(config);

        // Epoch 0: loss = 1.0
        assert!(!early_stopping.update(1.0, 0));

        // Epoch 1: loss = 0.95 (改善小于 min_delta，不算改善)
        assert!(!early_stopping.update(0.95, 1));
        assert_eq!(early_stopping.best_score, 1.0); // 保持不变
        assert_eq!(early_stopping.epochs_no_improve, 1);

        // Epoch 2: loss = 0.89 (改善大于 min_delta，算改善)
        assert!(!early_stopping.update(0.89, 2));
        assert_eq!(early_stopping.best_score, 0.89);
        assert_eq!(early_stopping.epochs_no_improve, 0);
    }

    #[test]
    fn test_early_stopping_reset() {
        let config = EarlyStoppingConfig::min(2);
        let mut early_stopping = EarlyStopping::new(config);

        // 触发早停
        early_stopping.update(1.0, 0);
        early_stopping.update(1.0, 1);
        early_stopping.update(1.0, 2);
        assert!(early_stopping.should_stop());

        // 重置
        early_stopping.reset();
        assert!(!early_stopping.should_stop());
        assert_eq!(early_stopping.epochs_no_improve, 0);
        assert_eq!(early_stopping.best_score, f32::INFINITY);
    }

    #[test]
    fn test_early_stopping_remaining_patience() {
        let config = EarlyStoppingConfig::min(5);
        let mut early_stopping = EarlyStopping::new(config);

        assert_eq!(early_stopping.remaining_patience(), 5);

        // 第一次设置最佳值（不算无改善）
        early_stopping.update(1.0, 0);
        assert_eq!(early_stopping.remaining_patience(), 5);

        // 第二次无改善
        early_stopping.update(1.0, 1);
        assert_eq!(early_stopping.remaining_patience(), 4);

        // 第三次无改善
        early_stopping.update(1.0, 2);
        assert_eq!(early_stopping.remaining_patience(), 3);

        // 改善，重置
        early_stopping.update(0.9, 3);
        assert_eq!(early_stopping.remaining_patience(), 5);
    }

    #[test]
    fn test_early_stopping_progress() {
        let config = EarlyStoppingConfig::min(4);
        let mut early_stopping = EarlyStopping::new(config);

        assert_eq!(early_stopping.progress(), 0.0);

        // 第一次设置最佳值（不算无改善）
        early_stopping.update(1.0, 0);
        assert!((early_stopping.progress() - 0.0).abs() < 1e-6);

        // 第一次无改善
        early_stopping.update(1.0, 1);
        assert!((early_stopping.progress() - 0.25).abs() < 1e-6);

        // 第二次无改善
        early_stopping.update(1.0, 2);
        assert!((early_stopping.progress() - 0.5).abs() < 1e-6);

        // 第三次无改善
        early_stopping.update(1.0, 3);
        assert!((early_stopping.progress() - 0.75).abs() < 1e-6);
    }

    #[test]
    fn test_early_stopping_simple_update() {
        let config = EarlyStoppingConfig::min(3);
        let mut early_stopping = EarlyStopping::new(config);

        // 使用简化版更新（不需要手动传入 epoch）
        assert!(!early_stopping.update_simple(1.0));
        assert!(!early_stopping.update_simple(1.0));
        assert!(!early_stopping.update_simple(1.0));
        assert!(early_stopping.update_simple(1.0)); // 第4次，触发早停
    }

    #[test]
    fn test_stop_reason() {
        let config = EarlyStoppingConfig::min(2);
        let mut early_stopping = EarlyStopping::new(config);

        early_stopping.update(1.0, 0);
        early_stopping.update(0.9, 1);
        early_stopping.update(0.95, 2);
        early_stopping.update(0.95, 3);

        let reason = early_stopping.stop_reason();
        assert!(reason.contains("No improvement for 2 epochs"));
        assert!(reason.contains("best score: 0.900000"));
        assert!(reason.contains("epoch 1"));
    }
}
