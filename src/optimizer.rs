//! 优化器
//!
//! 实现常见的优化算法：SGD、Adam、AdamW。

use ndarray::Array2;
use std::collections::HashMap;

/// 优化器 trait
pub trait Optimizer {
    /// 更新参数
    fn step(&mut self, param: &mut Array2<f32>, grad: &Array2<f32>, param_name: &str);

    /// 获取当前学习率
    fn lr(&self) -> f32;

    /// 设置学习率
    fn set_lr(&mut self, lr: f32);

    /// 优化器名称
    fn name(&self) -> &str;
}

/// SGD（随机梯度下降）
///
/// ```text
/// param = param - lr * grad
/// ```
#[derive(Debug, Clone)]
pub struct SGD {
    /// 学习率
    lr: f32,
    /// 动量系数
    momentum: f32,
    /// 权重衰减（L2 正则化）
    weight_decay: f32,
    /// 参数的动量（缓存）
    velocities: HashMap<String, Array2<f32>>,
    /// 是否使用 Nesterov 动量
    nesterov: bool,
}

impl SGD {
    pub fn new(lr: f32) -> Self {
        Self {
            lr,
            momentum: 0.0,
            weight_decay: 0.0,
            velocities: HashMap::new(),
            nesterov: false,
        }
    }

    pub fn with_momentum(mut self, momentum: f32) -> Self {
        self.momentum = momentum;
        self
    }

    pub fn with_weight_decay(mut self, weight_decay: f32) -> Self {
        self.weight_decay = weight_decay;
        self
    }

    pub fn with_nesterov(mut self, nesterov: bool) -> Self {
        self.nesterov = nesterov;
        self
    }
}

impl Optimizer for SGD {
    fn step(&mut self, param: &mut Array2<f32>, grad: &Array2<f32>, param_name: &str) {
        // 应用权重衰减
        let mut grad = grad.clone();
        if self.weight_decay > 0.0 {
            grad = grad + &(param.mapv(|x| x * self.weight_decay));
        }

        if self.momentum > 0.0 {
            // 获取或初始化速度
            let velocity = self
                .velocities
                .entry(param_name.to_string())
                .or_insert_with(|| Array2::zeros(param.dim()));

            if self.nesterov {
                // Nesterov 动量
                let velocity_clone = velocity.clone();
                *velocity = velocity_clone.mapv(|v| self.momentum * v) + &grad;
                let momentum_grad = &grad + &(velocity.mapv(|v| self.momentum * v));
                let param_clone = param.clone();
                *param = param_clone - &(momentum_grad.mapv(|g| self.lr * g));
            } else {
                // 标准动量
                let velocity_clone = velocity.clone();
                *velocity = velocity_clone.mapv(|v| self.momentum * v) + &grad.mapv(|g| self.lr * g);
                let param_clone = param.clone();
                let velocity_clone = velocity.clone();
                *param = param_clone - velocity_clone;
            }
        } else {
            // 标准 SGD
            let param_clone = param.clone();
            *param = param_clone - &(grad.mapv(|g| g * self.lr));
        }
    }

    fn lr(&self) -> f32 {
        self.lr
    }

    fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
    }

    fn name(&self) -> &str {
        "SGD"
    }
}

/// Adam 优化器
///
/// ```text
/// m = β1 * m + (1 - β1) * grad
/// v = β2 * v + (1 - β2) * grad²
/// m_hat = m / (1 - β1^t)
/// v_hat = v / (1 - β2^t)
/// param = param - lr * m_hat / (√v_hat + ε)
/// ```
#[derive(Debug, Clone)]
pub struct Adam {
    /// 学习率
    lr: f32,
    /// β1：一阶矩估计的指数衰减率
    beta1: f32,
    /// β2：二阶矩估计的指数衰减率
    beta2: f32,
    /// ε：数值稳定性常数
    eps: f32,
    /// 权重衰减
    weight_decay: f32,
    /// 一阶矩估计
    m: HashMap<String, Array2<f32>>,
    /// 二阶矩估计
    v: HashMap<String, Array2<f32>>,
    /// 时间步
    step: usize,
}

impl Adam {
    pub fn new(lr: f32) -> Self {
        Self {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.0,
            m: HashMap::new(),
            v: HashMap::new(),
            step: 0,
        }
    }

    pub fn with_betas(mut self, beta1: f32, beta2: f32) -> Self {
        self.beta1 = beta1;
        self.beta2 = beta2;
        self
    }

    pub fn with_eps(mut self, eps: f32) -> Self {
        self.eps = eps;
        self
    }

    pub fn with_weight_decay(mut self, weight_decay: f32) -> Self {
        self.weight_decay = weight_decay;
        self
    }
}

impl Optimizer for Adam {
    fn step(&mut self, param: &mut Array2<f32>, grad: &Array2<f32>, param_name: &str) {
        self.step += 1;

        // 应用权重衰减
        let mut grad = grad.clone();
        if self.weight_decay > 0.0 {
            grad = grad + &(param.mapv(|x| x * self.weight_decay));
        }

        // 获取或初始化一阶和二阶矩估计
        let m = self
            .m
            .entry(param_name.to_string())
            .or_insert_with(|| Array2::zeros(param.dim()));

        let v = self
            .v
            .entry(param_name.to_string())
            .or_insert_with(|| Array2::zeros(param.dim()));

        // 更新一阶矩估计
        let m_clone = m.clone();
        *m = m_clone.mapv(|x| self.beta1 * x) + &(grad.mapv(|g| g * (1.0 - self.beta1)));

        // 更新二阶矩估计
        let grad_sq = grad.mapv(|g| g * g);
        let v_clone = v.clone();
        *v = v_clone.mapv(|x| self.beta2 * x) + &(grad_sq.mapv(|g| g * (1.0 - self.beta2)));

        // 偏差修正
        let m_hat = m.mapv(|x| x / (1.0 - self.beta1.powi(self.step as i32)));
        let v_hat = v.mapv(|x| x / (1.0 - self.beta2.powi(self.step as i32)));

        // 参数更新
        let update = m_hat / &(v_hat.mapv(|x| x.sqrt() + self.eps));
        let param_clone = param.clone();
        *param = param_clone - &(update.mapv(|x| x * self.lr));
    }

    fn lr(&self) -> f32 {
        self.lr
    }

    fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
    }

    fn name(&self) -> &str {
        "Adam"
    }
}

/// AdamW 优化器（Adam with decoupled weight decay）
///
/// 与 Adam 的区别在于权重衰减的应用方式
#[derive(Debug, Clone)]
pub struct AdamW {
    adam: Adam,
}

impl AdamW {
    pub fn new(lr: f32) -> Self {
        Self {
            adam: Adam::new(lr).with_weight_decay(0.0),
        }
    }

    pub fn with_betas(mut self, beta1: f32, beta2: f32) -> Self {
        self.adam = self.adam.with_betas(beta1, beta2);
        self
    }

    pub fn with_eps(mut self, eps: f32) -> Self {
        self.adam = self.adam.with_eps(eps);
        self
    }

    pub fn with_weight_decay(mut self, weight_decay: f32) -> Self {
        // AdamW 将单独处理 weight decay
        self
    }

    fn step_with_wd(&mut self, param: &mut Array2<f32>, grad: &Array2<f32>, param_name: &str, weight_decay: f32) {
        // 先应用权重衰减
        if weight_decay > 0.0 {
            *param = param.mapv(|x| x * (1.0 - weight_decay));
        }

        // 然后使用 Adam 更新（不包含 weight_decay）
        self.adam.step(param, grad, param_name);
    }
}

impl Optimizer for AdamW {
    fn step(&mut self, param: &mut Array2<f32>, grad: &Array2<f32>, param_name: &str) {
        self.step_with_wd(param, grad, param_name, 0.01);
    }

    fn lr(&self) -> f32 {
        self.adam.lr()
    }

    fn set_lr(&mut self, lr: f32) {
        self.adam.set_lr(lr);
    }

    fn name(&self) -> &str {
        "AdamW"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    #[test]
    fn test_sgd() {
        let mut optimizer = SGD::new(0.01);
        let mut param = arr2(&[[1.0, 2.0]]);
        let grad = arr2(&[[0.1, 0.2]]);

        optimizer.step(&mut param, &grad, "test");

        // param = param - lr * grad
        assert!((param[[0, 0]] - 0.999).abs() < 1e-5);
        assert!((param[[0, 1]] - 1.998).abs() < 1e-5);
    }

    #[test]
    fn test_sgd_momentum() {
        let mut optimizer = SGD::new(0.01).with_momentum(0.9);
        let mut param = arr2(&[[1.0]]);
        let grad = arr2(&[[0.1]]);

        // 第一步
        optimizer.step(&mut param, &grad, "test");

        // 第二步（使用累积的动量）
        optimizer.step(&mut param, &grad, "test");

        assert!(param[[0, 0]] < 1.0); // 参数应该减小
    }

    #[test]
    fn test_adam() {
        let mut optimizer = Adam::new(0.01);
        let mut param = arr2(&[[1.0, 2.0]]);
        let grad = arr2(&[[0.1, 0.2]]);

        optimizer.step(&mut param, &grad, "test");

        // Adam 的更新更复杂，这里只检查参数确实改变了
        assert!((param[[0, 0]] - 1.0).abs() > 1e-6);
        assert!((param[[0, 1]] - 2.0).abs() > 1e-6);
    }

    #[test]
    fn test_lr_scheduling() {
        let mut optimizer = SGD::new(0.01);

        assert_eq!(optimizer.lr(), 0.01);

        optimizer.set_lr(0.001);

        assert_eq!(optimizer.lr(), 0.001);
    }
}
