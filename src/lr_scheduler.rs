//! 学习率调度器
//!
//! 实现常见的学习率调度策略。

/// 学习率调度器 trait
pub trait LRScheduler {
    /// 获取当前步骤的学习率
    fn get_lr(&self, step: usize) -> f32;

    /// 调度器名称
    fn name(&self) -> &str;
}

/// 固定学习率
#[derive(Debug, Clone, Copy)]
pub struct ConstantLR {
    lr: f32,
}

impl ConstantLR {
    pub fn new(lr: f32) -> Self {
        Self { lr }
    }
}

impl LRScheduler for ConstantLR {
    fn get_lr(&self, _step: usize) -> f32 {
        self.lr
    }

    fn name(&self) -> &str {
        "ConstantLR"
    }
}

/// 步进衰减
///
/// 每隔 `step_size` 个 epoch，学习率乘以 `gamma`
#[derive(Debug, Clone, Copy)]
pub struct StepLR {
    initial_lr: f32,
    step_size: usize,
    gamma: f32,
}

impl StepLR {
    pub fn new(initial_lr: f32, step_size: usize, gamma: f32) -> Self {
        Self {
            initial_lr,
            step_size,
            gamma,
        }
    }
}

impl LRScheduler for StepLR {
    fn get_lr(&self, step: usize) -> f32 {
        let decay_factor = self.gamma.powi((step / self.step_size) as i32);
        self.initial_lr * decay_factor
    }

    fn name(&self) -> &str {
        "StepLR"
    }
}

/// 指数衰减
///
/// ```text
/// lr = initial_lr * (gamma ^ step)
/// ```
#[derive(Debug, Clone, Copy)]
pub struct ExponentialLR {
    initial_lr: f32,
    gamma: f32,
}

impl ExponentialLR {
    pub fn new(initial_lr: f32, gamma: f32) -> Self {
        Self { initial_lr, gamma }
    }
}

impl LRScheduler for ExponentialLR {
    fn get_lr(&self, step: usize) -> f32 {
        self.initial_lr * self.gamma.powi(step as i32)
    }

    fn name(&self) -> &str {
        "ExponentialLR"
    }
}

/// 余弦退火
///
/// ```text
/// lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + cos(π * step / T_max))
/// ```
#[derive(Debug, Clone, Copy)]
pub struct CosineAnnealingLR {
    base_lr: f32,
    min_lr: f32,
    t_max: usize,
}

impl CosineAnnealingLR {
    pub fn new(base_lr: f32, min_lr: f32, t_max: usize) -> Self {
        Self {
            base_lr,
            min_lr,
            t_max,
        }
    }
}

impl LRScheduler for CosineAnnealingLR {
    fn get_lr(&self, step: usize) -> f32 {
        let progress = (step % self.t_max) as f32 / self.t_max as f32;
        self.min_lr
            + 0.5 * (self.base_lr - self.min_lr)
                * (1.0 + (std::f32::consts::PI * progress).cos())
    }

    fn name(&self) -> &str {
        "CosineAnnealingLR"
    }
}

/// 预热 + 余弦退火（推荐用于 Transformer）
///
/// 结合线性预热和余弦退火
#[derive(Debug, Clone, Copy)]
pub struct CosineAnnealingWarmRestarts {
    base_lr: f32,
    min_lr: f32,
    t_max: usize,
    warmup_steps: usize,
}

impl CosineAnnealingWarmRestarts {
    pub fn new(base_lr: f32, min_lr: f32, t_max: usize, warmup_steps: usize) -> Self {
        Self {
            base_lr,
            min_lr,
            t_max,
            warmup_steps,
        }
    }
}

impl LRScheduler for CosineAnnealingWarmRestarts {
    fn get_lr(&self, step: usize) -> f32 {
        if step < self.warmup_steps {
            // 线性预热
            self.base_lr * (step as f32 / self.warmup_steps as f32)
        } else {
            // 余弦退火
            let progress = ((step - self.warmup_steps) % self.t_max) as f32 / self.t_max as f32;
            self.min_lr
                + 0.5 * (self.base_lr - self.min_lr)
                    * (1.0 + (std::f32::consts::PI * progress).cos())
        }
    }

    fn name(&self) -> &str {
        "CosineAnnealingWarmRestarts"
    }
}

/// Warmup + 余弦退火（不重启）
///
/// 最常用的 Transformer 学习率调度策略，用于 GPT-2/GPT-3/BERT 等
///
/// # 公式
///
/// ```text
/// if step < warmup_steps:
///     lr = base_lr * step / warmup_steps
/// else:
///     progress = (step - warmup_steps) / (max_steps - warmup_steps)
///     lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + cos(π * progress))
/// ```
///
/// # 示例
///
/// ```rust
/// use mini_transformer::lr_scheduler::WarmupCosineAnnealing;
///
/// // 创建调度器：warmup 2000 步，总训练 100000 步
/// let scheduler = WarmupCosineAnnealing::new(0.0001, 0.0, 100000, 2000);
///
/// // warmup 阶段
/// assert_eq!(scheduler.get_lr(0), 0.0);
/// assert_eq!(scheduler.get_lr(1000), 0.00005);
/// assert_eq!(scheduler.get_lr(2000), 0.0001);
///
/// // 退火阶段
/// let lr_halfway = scheduler.get_lr(51000); // (100000-2000)/2 + 2000
/// assert!(lr_halfway < 0.0001 && lr_halfway > 0.0);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct WarmupCosineAnnealing {
    /// 基础学习率（warmup 后达到的学习率）
    base_lr: f32,
    /// 最小学习率（最终学习率）
    min_lr: f32,
    /// 总训练步数
    max_steps: usize,
    /// warmup 步数
    warmup_steps: usize,
}

impl WarmupCosineAnnealing {
    /// 创建新的 Warmup Cosine Annealing 调度器
    ///
    /// # Arguments
    /// * `base_lr` - 基础学习率（warmup 后达到）
    /// * `min_lr` - 最小学习率（训练结束时的学习率）
    /// * `max_steps` - 总训练步数
    /// * `warmup_steps` - warmup 步数
    ///
    /// # 推荐配置
    ///
    /// ```text
    /// 小模型 (< 1B 参数):
    ///     base_lr = 1e-4
    ///     warmup_steps = 2000
    ///
    /// 大模型 (> 1B 参数):
    ///     base_lr = 5e-5
    ///     warmup_steps = 2000-10000
    ///
    /// 超大模型 (> 10B 参数):
    ///     base_lr = 3e-5
    ///     warmup_steps = 10000-20000
    /// ```
    pub fn new(base_lr: f32, min_lr: f32, max_steps: usize, warmup_steps: usize) -> Self {
        assert!(warmup_steps < max_steps, "warmup_steps must be less than max_steps");
        assert!(warmup_steps > 0, "warmup_steps must be greater than 0");

        Self {
            base_lr,
            min_lr,
            max_steps,
            warmup_steps,
        }
    }

    /// 创建 GPT-2 风格的配置
    pub fn gpt2_style(max_steps: usize) -> Self {
        Self::new(1e-4, 0.0, max_steps, 2000)
    }

    /// 创建 GPT-3 风格的配置
    pub fn gpt3_style(max_steps: usize) -> Self {
        Self::new(6e-5, 0.0, max_steps, 2000)
    }

    /// 创建 BERT 风格的配置
    pub fn bert_style(max_steps: usize) -> Self {
        Self::new(1e-4, 0.0, max_steps, 10000)
    }

    /// 获取当前步数所处的阶段
    pub fn get_phase(&self, step: usize) -> WarmupPhase {
        if step < self.warmup_steps {
            WarmupPhase::Warmup
        } else if step < self.max_steps {
            WarmupPhase::Annealing
        } else {
            WarmupPhase::Finished
        }
    }
}

/// 训练阶段
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WarmupPhase {
    /// 预热阶段
    Warmup,
    /// 退火阶段
    Annealing,
    /// 训练完成
    Finished,
}

impl LRScheduler for WarmupCosineAnnealing {
    fn get_lr(&self, step: usize) -> f32 {
        if step < self.warmup_steps {
            // Warmup 阶段：线性增长
            self.base_lr * (step as f32 / self.warmup_steps as f32)
        } else if step < self.max_steps {
            // 余弦退火阶段
            let progress = (step - self.warmup_steps) as f32 / (self.max_steps - self.warmup_steps) as f32;
            self.min_lr
                + 0.5 * (self.base_lr - self.min_lr)
                    * (1.0 + (std::f32::consts::PI * progress).cos())
        } else {
            // 训练完成后保持最小学习率
            self.min_lr
        }
    }

    fn name(&self) -> &str {
        "WarmupCosineAnnealing"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constant_lr() {
        let scheduler = ConstantLR::new(0.01);

        assert_eq!(scheduler.get_lr(0), 0.01);
        assert_eq!(scheduler.get_lr(100), 0.01);
        assert_eq!(scheduler.get_lr(1000), 0.01);
    }

    #[test]
    fn test_step_lr() {
        let scheduler = StepLR::new(0.1, 30, 0.1);

        assert_eq!(scheduler.get_lr(0), 0.1);
        assert_eq!(scheduler.get_lr(29), 0.1);
        assert!((scheduler.get_lr(30) - 0.01).abs() < 1e-6);
        assert!((scheduler.get_lr(60) - 0.001).abs() < 1e-6);
    }

    #[test]
    fn test_exponential_lr() {
        let scheduler = ExponentialLR::new(0.1, 0.99);

        assert_eq!(scheduler.get_lr(0), 0.1);
        assert!((scheduler.get_lr(1) - 0.099).abs() < 0.001);
        assert!((scheduler.get_lr(10) - 0.09).abs() < 0.01);
    }

    #[test]
    fn test_cosine_annealing() {
        let scheduler = CosineAnnealingLR::new(0.1, 0.001, 100);

        // 开始时应该是最大学习率
        assert!((scheduler.get_lr(0) - 0.1).abs() < 0.01);

        // 中间时刻应该下降
        let mid_lr = scheduler.get_lr(50);
        assert!(mid_lr < 0.1 && mid_lr > 0.001);

        // 周期结束时应该是最小学习率
        assert!((scheduler.get_lr(100) - 0.001).abs() < 0.1);
    }

    #[test]
    fn test_warmup_cosine() {
        let scheduler = CosineAnnealingWarmRestarts::new(0.1, 0.001, 100, 10);

        // 预热阶段：学习率线性增长
        assert!((scheduler.get_lr(0) - 0.0).abs() < 0.001);
        assert!((scheduler.get_lr(5) - 0.05).abs() < 0.01);
        assert!((scheduler.get_lr(10) - 0.1).abs() < 0.01);

        // 预热后：余弦退火
        let after_warmup = scheduler.get_lr(20);
        assert!(after_warmup < 0.1 && after_warmup > 0.001);
    }

    #[test]
    fn test_warmup_cosine_annealing() {
        let scheduler = WarmupCosineAnnealing::new(0.0001, 0.0, 1000, 100);

        // Warmup 阶段
        assert_eq!(scheduler.get_phase(0), WarmupPhase::Warmup);
        assert!((scheduler.get_lr(0) - 0.0).abs() < 1e-9);
        assert!((scheduler.get_lr(50) - 0.00005).abs() < 1e-6);
        assert!((scheduler.get_lr(100) - 0.0001).abs() < 1e-6);

        // 退火阶段
        assert_eq!(scheduler.get_phase(101), WarmupPhase::Annealing);
        let lr_halfway = scheduler.get_lr(550); // (1000-100)/2 + 100
        assert!(lr_halfway < 0.0001 && lr_halfway > 0.0);

        // 训练结束
        assert_eq!(scheduler.get_phase(999), WarmupPhase::Annealing);
        assert!((scheduler.get_lr(999) - 0.0).abs() < 0.0001);

        // 训练完成
        assert_eq!(scheduler.get_phase(1000), WarmupPhase::Finished);
        assert_eq!(scheduler.get_phase(2000), WarmupPhase::Finished);
        assert_eq!(scheduler.get_lr(1000), 0.0);
    }

    #[test]
    fn test_warmup_cosine_gpt2_style() {
        let scheduler = WarmupCosineAnnealing::gpt2_style(10000);

        // GPT-2: base_lr = 1e-4, warmup = 2000
        assert!((scheduler.get_lr(0) - 0.0).abs() < 1e-9);
        assert!((scheduler.get_lr(2000) - 0.0001).abs() < 1e-6);

        // 在训练中期
        let mid_lr = scheduler.get_lr(6000);
        assert!(mid_lr < 0.0001 && mid_lr > 0.0);
    }

    #[test]
    fn test_warmup_cosine_bert_style() {
        let scheduler = WarmupCosineAnnealing::bert_style(100000);

        // BERT: base_lr = 1e-4, warmup = 10000
        assert_eq!(scheduler.get_phase(0), WarmupPhase::Warmup);
        assert_eq!(scheduler.get_phase(5000), WarmupPhase::Warmup);
        assert_eq!(scheduler.get_phase(10000), WarmupPhase::Annealing);

        assert!((scheduler.get_lr(0) - 0.0).abs() < 1e-9);
        assert!((scheduler.get_lr(10000) - 0.0001).abs() < 1e-6);
    }

    #[test]
    fn test_warmup_cosine_no_restart() {
        let scheduler = WarmupCosineAnnealing::new(0.1, 0.01, 100, 20);

        // Warmup
        assert!((scheduler.get_lr(10) - 0.05).abs() < 0.01);
        assert!((scheduler.get_lr(20) - 0.1).abs() < 0.01);

        // Annealing - 单调递减
        let lr_30 = scheduler.get_lr(30);
        let lr_50 = scheduler.get_lr(50);
        let lr_90 = scheduler.get_lr(90);
        let lr_100 = scheduler.get_lr(100);

        assert!(lr_30 < 0.1);
        assert!(lr_50 < lr_30);
        assert!(lr_90 < lr_50);
        assert!((lr_100 - 0.01).abs() < 0.001);
    }

    #[test]
    #[should_panic(expected = "warmup_steps must be less than max_steps")]
    fn test_warmup_cosine_invalid_warmup() {
        WarmupCosineAnnealing::new(0.1, 0.0, 100, 100);
    }
}
