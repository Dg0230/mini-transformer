//! 损失函数
//!
//! 实现常见的损失函数：交叉熵、MSE 等。

use ndarray::Array2;

/// 损失函数 trait
pub trait LossFunction {
    /// 计算损失值
    fn compute(&self, predictions: &Array2<f32>, targets: &Array2<f32>) -> f32;

    /// 计算梯度（关于预测值）
    fn grad(&self, predictions: &Array2<f32>, targets: &Array2<f32>) -> Array2<f32>;

    /// 损失函数名称
    fn name(&self) -> &str;
}

/// 交叉熵损失（用于分类）
///
/// ```text
/// L = -Σ y_true * log(y_pred)
/// ```
///
/// 注意：通常结合 Softmax 使用
#[derive(Debug, Clone, Copy)]
pub struct CrossEntropyLoss;

impl CrossEntropyLoss {
    pub fn new() -> Self {
        Self
    }
}

impl Default for CrossEntropyLoss {
    fn default() -> Self {
        Self::new()
    }
}

impl LossFunction for CrossEntropyLoss {
    fn compute(&self, predictions: &Array2<f32>, targets: &Array2<f32>) -> f32 {
        assert_eq!(predictions.shape(), targets.shape(), "Shape mismatch");

        let mut loss = 0.0;
        let n = predictions.nrows();

        for i in 0..n {
            for (j, &target) in targets.row(i).iter().enumerate() {
                if target > 0.0 {
                    let pred = predictions[[i, j]].max(1e-7).min(1.0 - 1e-7); // 数值稳定性
                    loss -= target * pred.ln();
                }
            }
        }

        loss / n as f32
    }

    fn grad(&self, predictions: &Array2<f32>, targets: &Array2<f32>) -> Array2<f32> {
        // Softmax + 交叉熵的梯度简化为 (pred - target) / n
        let n = predictions.nrows() as f32;
        (predictions - targets) / n
    }

    fn name(&self) -> &str {
        "CrossEntropyLoss"
    }
}

/// 均方误差损失（用于回归）
///
/// ```text
/// L = (1/n) * Σ (y_pred - y_true)²
/// ```
#[derive(Debug, Clone, Copy)]
pub struct MSELoss;

impl MSELoss {
    pub fn new() -> Self {
        Self
    }
}

impl Default for MSELoss {
    fn default() -> Self {
        Self::new()
    }
}

impl LossFunction for MSELoss {
    fn compute(&self, predictions: &Array2<f32>, targets: &Array2<f32>) -> f32 {
        assert_eq!(predictions.shape(), targets.shape(), "Shape mismatch");

        let diff = predictions - targets;
        let sum_sq: f32 = diff.mapv(|x| x * x).sum();

        sum_sq / predictions.len() as f32
    }

    fn grad(&self, predictions: &Array2<f32>, targets: &Array2<f32>) -> Array2<f32> {
        // MSE 导数: 2 * (pred - target) / n
        let n = predictions.len() as f32;
        (predictions - targets).mapv(|x| 2.0 * x / n)
    }

    fn name(&self) -> &str {
        "MSELoss"
    }
}

/// 二元交叉熵损失
///
/// ```text
/// L = -[y * log(p) + (1-y) * log(1-p)]
/// ```
#[derive(Debug, Clone, Copy)]
pub struct BinaryCrossEntropyLoss {
    /// 数值稳定性的小常数
    pub eps: f32,
}

impl BinaryCrossEntropyLoss {
    pub fn new() -> Self {
        Self { eps: 1e-7 }
    }

    pub fn with_eps(eps: f32) -> Self {
        Self { eps }
    }
}

impl Default for BinaryCrossEntropyLoss {
    fn default() -> Self {
        Self::new()
    }
}

impl LossFunction for BinaryCrossEntropyLoss {
    fn compute(&self, predictions: &Array2<f32>, targets: &Array2<f32>) -> f32 {
        assert_eq!(predictions.shape(), targets.shape(), "Shape mismatch");

        let mut loss = 0.0;
        let n = predictions.len();

        for (&pred, &target) in predictions.iter().zip(targets.iter()) {
            let pred_clipped = pred.max(self.eps).min(1.0 - self.eps);
            let binary_loss = -(target * pred_clipped.ln()
                + (1.0 - target) * (1.0 - pred_clipped).ln());
            loss += binary_loss;
        }

        loss / n as f32
    }

    fn grad(&self, predictions: &Array2<f32>, targets: &Array2<f32>) -> Array2<f32> {
        let n = predictions.len() as f32;
        let mut grad = Array2::zeros(predictions.dim());

        for ((i, &pred), &target) in predictions
            .indexed_iter()
            .zip(targets.iter())
        {
            let pred_clipped = pred.max(self.eps).min(1.0 - self.eps);
            grad[i] = ((pred_clipped - target)
                / (pred_clipped * (1.0 - pred_clipped)))
                / n;
        }

        grad
    }

    fn name(&self) -> &str {
        "BinaryCrossEntropyLoss"
    }
}

/// 准确率（用于评估，不是真正的损失函数）
#[derive(Debug, Clone, Copy)]
pub struct Accuracy;

impl Accuracy {
    pub fn new() -> Self {
        Self
    }
}

impl Default for Accuracy {
    fn default() -> Self {
        Self::new()
    }
}

impl Accuracy {
    /// 计算分类准确率
    ///
    /// # 参数
    /// - `predictions`: [batch_size, n_classes] 预测 logits
    /// - `targets`: [batch_size] 真实类别索引
    pub fn compute(&self, predictions: &Array2<f32>, targets: &[usize]) -> f32 {
        let batch_size = predictions.nrows();

        let mut correct = 0;
        for i in 0..batch_size {
            let pred_class = predictions
                .row(i)
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(j, _)| j)
                .unwrap();

            if pred_class == targets[i] {
                correct += 1;
            }
        }

        correct as f32 / batch_size as f32
    }

    pub fn name(&self) -> &str {
        "Accuracy"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    #[test]
    fn test_cross_entropy_loss() {
        let loss_fn = CrossEntropyLoss::new();

        // 完美预测：损失应该接近 0
        let predictions = arr2(&[[0.1, 0.9, 0.0]]);
        let targets = arr2(&[[0.0, 1.0, 0.0]]);
        let loss = loss_fn.compute(&predictions, &targets);

        assert!(loss < 0.5);

        // 完全错误预测：损失应该很大
        let predictions = arr2(&[[0.9, 0.1, 0.0]]);
        let targets = arr2(&[[0.0, 1.0, 0.0]]);
        let loss = loss_fn.compute(&predictions, &targets);

        assert!(loss > 2.0);
    }

    #[test]
    fn test_mse_loss() {
        let loss_fn = MSELoss::new();

        let predictions = arr2(&[[1.0, 2.0, 3.0]]);
        let targets = arr2(&[[1.5, 2.5, 3.5]]);

        let loss = loss_fn.compute(&predictions, &targets);

        // 手动计算: ((0.5² + 0.5² + 0.5²) / 3) = 0.25
        assert!((loss - 0.25).abs() < 1e-5);
    }

    #[test]
    fn test_accuracy() {
        let acc_fn = Accuracy::new();

        let predictions = arr2(&[
            [0.1, 0.9, 0.0],  // 类别 1
            [0.8, 0.1, 0.1],  // 类别 0
            [0.1, 0.1, 0.8],  // 类别 2
        ]);

        let targets = vec![1, 0, 2];  // 全部正确

        let acc = acc_fn.compute(&predictions, &targets);
        assert!((acc - 1.0).abs() < 1e-5);

        let targets = vec![1, 1, 2];  // 中间错误
        let acc = acc_fn.compute(&predictions, &targets);
        assert!((acc - 0.6666667).abs() < 1e-5);
    }

    #[test]
    fn test_binary_cross_entropy() {
        let loss_fn = BinaryCrossEntropyLoss::new();

        // 完美预测
        let predictions = arr2(&[[0.9], [0.1]]);
        let targets = arr2(&[[1.0], [0.0]]);
        let loss = loss_fn.compute(&predictions, &targets);

        assert!(loss < 0.5);

        // 完全错误
        let predictions = arr2(&[[0.1], [0.9]]);
        let targets = arr2(&[[1.0], [0.0]]);
        let loss = loss_fn.compute(&predictions, &targets);

        assert!(loss > 2.0);
    }
}
