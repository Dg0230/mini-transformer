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
}
