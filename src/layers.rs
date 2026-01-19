//! Transformer 的基础层
//!
//! 包含 Feed-Forward Network 和 Layer Normalization

use ndarray::Array2;
use crate::tensor::TensorExt;

/// Layer Normalization
///
/// 对每个样本的所有特征进行归一化，使均值为 0，方差为 1。
///
/// ```text
/// y = γ * ((x - μ) / √(σ² + ε)) + β
/// ```
///
/// 与 Batch Normalization 不同，Layer Norm 在序列数据（如 NLP）中更稳定。
#[derive(Debug, Clone)]
pub struct LayerNorm {
    /// 缩放参数 [1, d_model]
    gamma: Array2<f32>,
    /// 平移参数 [1, d_model]
    beta: Array2<f32>,
    /// 防止除零的小常数
    eps: f32,
    d_model: usize,
}

impl LayerNorm {
    /// 创建新的 Layer Norm 层
    ///
    /// # 参数
    /// - `d_model`: 特征维度
    /// - `eps`: 防止除零的小常数（默认 1e-6）
    pub fn new(d_model: usize, eps: Option<f32>) -> Self {
        // γ 初始化为 1，β 初始化为 0
        let gamma = Array2::ones((1, d_model));
        let beta = Array2::zeros((1, d_model));

        Self {
            gamma,
            beta,
            eps: eps.unwrap_or(1e-6),
            d_model,
        }
    }

    /// 前向传播
    ///
    /// # 输入
    /// - `x`: [seq_len, d_model]
    ///
    /// # 输出
    /// - [seq_len, d_model] 归一化后的值
    pub fn forward(&self, x: &Array2<f32>) -> Array2<f32> {
        x.layer_norm(&self.gamma, &self.beta, self.eps)
    }

    /// 获取参数（用于训练或保存）
    pub fn params(&self) -> (&Array2<f32>, &Array2<f32>) {
        (&self.gamma, &self.beta)
    }
}

/// Feed-Forward Network
///
/// 两层全连接网络，中间使用 GELU 激活函数。
///
/// ```text
/// FFN(x) = GELU(xW1 + b1)W2 + b2
/// ```
///
/// 这允许模型学习更复杂的非线性变换。
/// 通常 d_ff = 4 × d_model（例如 d_model=512 时，d_ff=2048）。
#[derive(Debug, Clone)]
pub struct FeedForward {
    /// 第一层权重: [d_model, d_ff]
    w1: Array2<f32>,
    /// 第二层权重: [d_ff, d_model]
    w2: Array2<f32>,
    /// 第一层偏置: [1, d_ff]
    b1: Array2<f32>,
    /// 第二层偏置: [1, d_model]
    b2: Array2<f32>,
    d_model: usize,
    d_ff: usize,
}

impl FeedForward {
    /// 创建新的 Feed-Forward Network
    ///
    /// # 参数
    /// - `d_model`: 输入/输出维度
    /// - `d_ff`: 隐藏层维度（通常为 d_model 的 4 倍）
    pub fn new(d_model: usize, d_ff: usize) -> Self {
        Self {
            w1: Array2::random_xavier((d_model, d_ff)),
            w2: Array2::random_xavier((d_ff, d_model)),
            b1: Array2::zeros((1, d_ff)),
            b2: Array2::zeros((1, d_model)),
            d_model,
            d_ff,
        }
    }

    /// 前向传播
    ///
    /// # 输入
    /// - `x`: [seq_len, d_model]
    ///
    /// # 输出
    /// - [seq_len, d_model]
    pub fn forward(&self, x: &Array2<f32>) -> Array2<f32> {
        // 第一层: xW1 + b1
        let hidden = x.matmul(&self.w1) + &self.b1;

        // GELU 激活
        let activated = hidden.gelu();

        // 第二层: GELU(xW1 + b1)W2 + b2
        let output = activated.matmul(&self.w2) + &self.b2;

        output
    }

    /// 获取维度
    pub fn dimensions(&self) -> (usize, usize) {
        (self.d_model, self.d_ff)
    }

    /// 获取参数（用于训练或保存）
    pub fn params(&self) -> (&Array2<f32>, &Array2<f32>, &Array2<f32>, &Array2<f32>) {
        (&self.w1, &self.w2, &self.b1, &self.b2)
    }
}

/// Dropout 层（简化实现，训练时使用）
///
/// 随机将一部分神经元的输出置零，防止过拟合。
/// 推理时通常关闭 dropout。
#[derive(Debug, Clone)]
pub struct Dropout {
    dropout_prob: f32,
    training: bool,
}

impl Dropout {
    pub fn new(dropout_prob: f32) -> Self {
        Self {
            dropout_prob,
            training: true,
        }
    }

    pub fn set_training(&mut self, training: bool) {
        self.training = training;
    }

    /// 前向传播
    ///
    /// 训练时：随机丢弃
    /// 推理时：不操作（但需要缩放）
    pub fn forward(&self, x: &Array2<f32>) -> Array2<f32> {
        if !self.training || self.dropout_prob == 0.0 {
            return x.clone();
        }

        // 简化实现：随机丢弃
        // 完整实现需要正确的随机数生成和缩放
        x.mapv(|v: f32| {
            if rand::random::<f32>() < self.dropout_prob {
                0.0
            } else {
                v / (1.0 - self.dropout_prob)  // 缩放以保持期望值
            }
        })
    }
}

/// 残差连接 + Layer Norm 包装器
///
/// Transformer 的标准模式：
/// ```text
/// output = LayerNorm(x + Sublayer(x))
/// ```
#[derive(Debug, Clone)]
pub struct ResidualNorm {
    layer_norm: LayerNorm,
}

impl ResidualNorm {
    pub fn new(d_model: usize) -> Self {
        Self {
            layer_norm: LayerNorm::new(d_model, None),
        }
    }

    /// 应用残差连接和层归一化
    pub fn forward(&self, x: &Array2<f32>, sublayer_output: &Array2<f32>) -> Array2<f32> {
        // x + Sublayer(x)
        let residual = x.residual_add(sublayer_output);

        // LayerNorm
        self.layer_norm.forward(&residual)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    #[test]
    fn test_layer_norm() {
        let ln = LayerNorm::new(4, Some(1e-6));

        let x = arr2(&[
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
        ]);

        let y = ln.forward(&x);

        // 验证形状不变
        assert_eq!(y.shape(), x.shape());

        // 验证每行的均值接近 0，标准差接近 1
        for row in 0..x.nrows() {
            let row_data: Vec<f32> = y.row(row).into_iter().copied().collect();
            let mean: f32 = row_data.iter().sum::<f32>() / row_data.len() as f32;
            let variance: f32 = row_data.iter().map(|&x| x * x).sum::<f32>() / row_data.len() as f32;

            assert!(mean.abs() < 1e-5, "Mean should be close to 0");
            assert!((variance - 1.0).abs() < 1e-4, "Variance should be close to 1");
        }
    }

    #[test]
    fn test_feed_forward() {
        let ffn = FeedForward::new(4, 16);

        let x = arr2(&[
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ]);

        let y = ffn.forward(&x);

        // 验证形状
        assert_eq!(y.shape(), &[2, 4]);

        // 验证非线性（输出不应该与输入成简单比例）
        assert!(y[[0, 0]] != y[[1, 1]]);
    }

    #[test]
    fn test_residual_norm() {
        let rn = ResidualNorm::new(4);

        let x = arr2(&[
            [1.0, 1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0, 2.0],
        ]);

        let sublayer = arr2(&[
            [0.5, 0.5, 0.5, 0.5],
            [-0.5, -0.5, -0.5, -0.5],
        ]);

        let y = rn.forward(&x, &sublayer);

        // 验证残差连接
        assert_eq!(y.shape(), &[2, 4]);

        // 验证归一化效果
        for row in 0..y.nrows() {
            let row_data: Vec<f32> = y.row(row).into_iter().copied().collect();
            let mean: f32 = row_data.iter().sum::<f32>() / row_data.len() as f32;
            assert!(mean.abs() < 1e-5);
        }
    }

    #[test]
    fn test_gelu() {
        let x = arr2(&[[0.0, 1.0, -1.0, 2.0]]);
        let y = x.gelu();

        // GELU(0) ≈ 0
        assert!((y[[0, 0]]).abs() < 1e-6);

        // GELU 应该是单调递增的
        assert!(y[[0, 1]] > y[[0, 0]]);
        assert!(y[[0, 3]] > y[[0, 1]]);
    }
}
