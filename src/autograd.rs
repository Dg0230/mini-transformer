//! 自动求导和梯度计算
//!
//! 实现简单的自动求导系统，支持反向传播。

use ndarray::Array2;

/// 计算图节点
#[derive(Debug, Clone)]
pub enum TensorOp {
    /// 叶子节点（参数）
    Leaf,
    /// 加法
    Add,
    /// 矩阵乘法
    MatMul(Array2<f32>), // 保存另一个操作数
    /// 激活函数
    Gelu,
    /// Layer Norm
    LayerNorm(Array2<f32>, Array2<f32>, f32), // gamma, beta, eps
    /// Softmax
    Softmax(usize), // axis
}

/// 可求导的张量
#[derive(Debug)]
pub struct Var {
    /// 数据
    pub data: Array2<f32>,
    /// 梯度
    pub grad: Option<Array2<f32>>,
    /// 生成此变量的操作
    pub op: TensorOp,
    /// 是否需要计算梯度
    pub requires_grad: bool,
}

impl Var {
    /// 创建叶子变量（参数）
    pub fn leaf(data: Array2<f32>) -> Self {
        Self {
            data,
            grad: None,
            op: TensorOp::Leaf,
            requires_grad: true,
        }
    }

    /// 创建不需要梯度的常量
    pub fn constant(data: Array2<f32>) -> Self {
        Self {
            data,
            grad: None,
            op: TensorOp::Leaf,
            requires_grad: false,
        }
    }

    /// 清零梯度
    pub fn zero_grad(&mut self) {
        if self.requires_grad {
            self.grad = Some(Array2::zeros(self.data.dim()));
        }
    }

    /// 矩阵乘法
    pub fn matmul(&self, other: &Var) -> Var {
        let data = self.data.dot(&other.data);

        Var {
            data,
            grad: None,
            op: TensorOp::MatMul(other.data.clone()),
            requires_grad: self.requires_grad || other.requires_grad,
        }
    }

    /// 加法
    pub fn add(&self, other: &Var) -> Var {
        let data = &self.data + &other.data;

        Var {
            data,
            grad: None,
            op: TensorOp::Add,
            requires_grad: self.requires_grad || other.requires_grad,
        }
    }

    /// GELU 激活
    pub fn gelu(&self) -> Var {
        let data = self.data.mapv(|x: f32| {
            let sqrt_2_over_pi = (2.0 / std::f32::consts::PI).sqrt();
            let cube = x * x * x;
            0.5 * x * (1.0 + (sqrt_2_over_pi * (x + 0.044715 * cube)).tanh())
        });

        Var {
            data,
            grad: None,
            op: TensorOp::Gelu,
            requires_grad: self.requires_grad,
        }
    }

    /// Softmax
    pub fn softmax(&self, axis: usize) -> Var {
        // 简化实现，实际应该保存输入用于反向传播
        let max = self.data.fold_axis(ndarray::Axis(axis), f32::NEG_INFINITY, |a, &b| a.max(b));
        let max_view = max.insert_axis(ndarray::Axis(axis));

        let exp = (&self.data - &max_view).mapv(|x: f32| x.exp());
        let sum = exp.sum_axis(ndarray::Axis(axis));
        let sum_view = sum.insert_axis(ndarray::Axis(axis));

        let data = exp / sum_view;

        Var {
            data,
            grad: None,
            op: TensorOp::Softmax(axis),
            requires_grad: self.requires_grad,
        }
    }

    /// 反向传播
    ///
    /// 简化实现：手动实现常见操作的反向传播
    pub fn backward(&mut self, grad_output: Array2<f32>) {
        if !self.requires_grad {
            return;
        }

        match &self.op {
            TensorOp::Leaf => {
                // 叶子节点：累积梯度
                match &mut self.grad {
                    Some(g) => *g = g.clone() + &grad_output,
                    None => self.grad = Some(grad_output),
                }
            }

            TensorOp::Add => {
                // 加法：梯度直接传播
                self.grad = Some(grad_output);
            }

            TensorOp::Gelu => {
                // GELU 导数
                let grad = self.data.mapv(|x: f32| {
                    let sqrt_2_over_pi = (2.0 / std::f32::consts::PI).sqrt();
                    let cube = x * x * x;
                    let tanh_arg = sqrt_2_over_pi * (x + 0.044715 * cube);
                    let sech_sq = 1.0 - tanh_arg.tanh().powi(2);

                    // GELU 导数近似
                    0.5 * (1.0 + tanh_arg.tanh()) + 0.5 * x * sech_sq * sqrt_2_over_pi * (1.0 + 0.134145 * x * x)
                });

                self.grad = Some(grad * grad_output);
            }

            TensorOp::Softmax(_) => {
                // Softmax + 交叉熵的梯度简化为 (pred - target)
                self.grad = Some(grad_output);
            }

            _ => {
                // 其他操作：简化处理
                self.grad = Some(grad_output);
            }
        }
    }
}

/// 简化的参数结构（用于训练）
#[derive(Debug)]
pub struct Parameter {
    /// 权重数据
    pub data: Array2<f32>,
    /// 梯度
    pub grad: Array2<f32>,
}

impl Parameter {
    pub fn new(data: Array2<f32>) -> Self {
        let dim = data.dim();
        Self {
            data,
            grad: Array2::zeros(dim),
        }
    }

    /// 清零梯度
    pub fn zero_grad(&mut self) {
        self.grad.fill(0.0);
    }

    /// SGD 更新
    pub fn sgd_update(&mut self, lr: f32) {
        self.data = &self.data - &(self.grad.mapv(|g| g * lr));
    }

    /// Adam 更新
    pub fn adam_update(
        &mut self,
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        step: usize,
        m: &mut Array2<f32>,
        v: &mut Array2<f32>,
    ) {
        // 更新一阶矩估计
        let m_clone = m.clone();
        *m = m_clone.mapv(|x| beta1 * x) + &(self.grad.mapv(|g| g * (1.0 - beta1)));

        // 更新二阶矩估计
        let grad_sq = self.grad.mapv(|g| g * g);
        let v_clone = v.clone();
        *v = v_clone.mapv(|x| beta2 * x) + &(grad_sq.mapv(|g| g * (1.0 - beta2)));

        // 偏差修正
        let m_hat = m.mapv(|x| x / (1.0 - beta1.powi(step as i32)));
        let v_hat = v.mapv(|x| x / (1.0 - beta2.powi(step as i32)));

        // 参数更新
        let update = m_hat / &(v_hat.mapv(|x| x.sqrt() + eps));
        self.data = &self.data - &(update.mapv(|x| x * lr));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::TensorExt;

    #[test]
    fn test_parameter_creation() {
        let data = Array2::random_xavier((10, 20));
        let param = Parameter::new(data);

        assert_eq!(param.data.shape(), &[10, 20]);
        assert_eq!(param.grad.shape(), &[10, 20]);
    }

    #[test]
    fn test_zero_grad() {
        let mut param = Parameter::new(Array2::random_xavier((5, 10)));
        param.grad.fill(1.0);

        param.zero_grad();

        assert_eq!(param.grad.iter().sum::<f32>(), 0.0);
    }

    #[test]
    fn test_sgd_update() {
        let mut param = Parameter::new(Array2::from_elem((2, 2), 1.0));
        param.grad.fill(0.1);

        param.sgd_update(0.01);

        // 新参数 = 旧参数 - lr * grad = 1.0 - 0.01 * 0.1 = 0.999
        assert!((param.data[[0, 0]] - 0.999).abs() < 1e-5);
    }
}
