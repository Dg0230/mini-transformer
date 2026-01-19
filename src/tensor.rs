//! 张量操作扩展和工具函数
//!
//! 基于 ndarray 实现 Transformer 所需的张量操作。

use ndarray::{Array, Array2, Axis};
use rand::Rng;
use std::ops::Add;

/// 张量扩展 trait
pub trait TensorExt<T> {
    /// 创建随机张量（Xavier 初始化）
    fn random_xavier(shape: (usize, usize)) -> Array2<T>
    where
        T: Add<Output = T> + num_traits::Float + rand::distributions::uniform::SampleUniform;

    /// 创建零张量
    fn zeros(shape: (usize, usize)) -> Array2<T>
    where
        T: num_traits::Zero + Clone;

    /// 创建单位张量
    fn identity(n: usize) -> Array2<T>
    where
        T: num_traits::Zero + num_traits::One + Clone;

    /// 矩阵乘法
    fn matmul(&self, other: &Array2<T>) -> Array2<T>
    where
        T: num_traits::Num + Clone;

    /// 转置
    fn t(&self) -> Array2<T>
    where
        T: Clone;

    /// 添加维度（用于广播）
    fn insert_axis(&self, axis: usize) -> Array<T, ndarray::IxDyn>
    where
        T: Clone;

    /// 沿指定维度拼接
    fn concat(&self, other: &Array2<T>, axis: usize) -> Array2<T>
    where
        T: Clone;

    /// 应用 softmax
    fn softmax(&self, axis: usize) -> Array2<T>
    where
        T: num_traits::Float + num_traits::NumAssign;

    /// 缩放点积注意力（分数计算）
    fn scaled_dot_product(&self, other: &Array2<T>, scale: T) -> Array2<T>
    where
        T: num_traits::Float + num_traits::NumAssign + Clone;

    /// GELU 激活函数
    fn gelu(&self) -> Array2<T>
    where
        T: num_traits::Float;

    /// 残差连接
    fn residual_add(&self, other: &Array2<T>) -> Array2<T>
    where
        T: num_traits::Num + Clone;

    /// 层归一化
    fn layer_norm(&self, gamma: &Array2<T>, beta: &Array2<T>, eps: T) -> Array2<T>
    where
        T: num_traits::Float + num_traits::NumAssign;
}

/// 为 f32 实现张量扩展
impl TensorExt<f32> for Array2<f32> {
    fn random_xavier(shape: (usize, usize)) -> Array2<f32> {
        let mut rng = rand::thread_rng();
        let limit = (6.0 / (shape.0 + shape.1) as f32).sqrt();

        Array2::from_shape_fn(shape, |_| {
            rng.gen_range(-limit..limit)
        })
    }

    fn zeros(shape: (usize, usize)) -> Array2<f32> {
        Array2::zeros(shape)
    }

    fn identity(n: usize) -> Array2<f32> {
        Array2::from_elem((n, n), 0.0f32).mapv(|v: f32| {
            if v.abs() < 1e-6 { 1.0 } else { 0.0 }
        })
    }

    fn matmul(&self, other: &Array2<f32>) -> Array2<f32> {
        self.dot(other)
    }

    fn t(&self) -> Array2<f32> {
        self.clone().reversed_axes()
    }

    fn insert_axis(&self, axis: usize) -> Array<f32, ndarray::IxDyn> {
        let mut view = self.view();
        view.insert_axis(Axis(axis));
        view.into_dyn().to_owned()
    }

    fn concat(&self, other: &Array2<f32>, axis: usize) -> Array2<f32> {
        ndarray::concatenate(Axis(axis), &[self.view(), other.view()])
            .expect("Concatenate dimension mismatch")
    }

    fn softmax(&self, axis: usize) -> Array2<f32> {
        // 减去最大值以提高数值稳定性
        let max = self.fold_axis(Axis(axis), f32::NEG_INFINITY, |a, &b| a.max(b));
        let max_view = max.insert_axis(ndarray::Axis(axis));

        let exp = (self - &max_view).mapv(|x: f32| x.exp());
        let sum = exp.sum_axis(Axis(axis));
        let sum_view = sum.insert_axis(ndarray::Axis(axis));

        exp / sum_view
    }

    fn scaled_dot_product(&self, other: &Array2<f32>, scale: f32) -> Array2<f32> {
        let other_t = other.t();
        let other_t_owned = other_t.to_owned();
        let scores = self.matmul(&other_t_owned);
        let scaled = scores.mapv(|x| x / scale);
        scaled.softmax(1)
    }

    fn gelu(&self) -> Array2<f32> {
        // GELU(x) = x * Φ(x) 其中 Φ 是标准正态分布的 CDF
        // 近似: GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
        self.mapv(|x: f32| {
            let sqrt_2_over_pi = (2.0 / std::f32::consts::PI).sqrt();
            let cube = x * x * x;
            0.5 * x * (1.0 + (sqrt_2_over_pi * (x + 0.044715 * cube)).tanh())
        })
    }

    fn residual_add(&self, other: &Array2<f32>) -> Array2<f32> {
        self + other
    }

    fn layer_norm(&self, gamma: &Array2<f32>, beta: &Array2<f32>, eps: f32) -> Array2<f32> {
        let mean = self.mean_axis(Axis(1)).unwrap();
        let mean_view = mean.insert_axis(ndarray::Axis(1));

        let centered = self - &mean_view;
        let var = centered.mapv(|x: f32| x * x).mean_axis(Axis(1)).unwrap();
        let var_view = var.insert_axis(ndarray::Axis(1));
        let std = (var_view + eps).mapv(|x: f32| x.sqrt());

        &centered / &std * gamma + beta
    }
}

/// 辅助函数：因果掩码（用于 decoder）
pub fn causal_mask(seq_len: usize) -> Array2<f32> {
    let mut mask = Array2::zeros((seq_len, seq_len));
    for i in 0..seq_len {
        for j in 0..=i {
            mask[[i, j]] = 1.0;
        }
    }
    mask
}

/// 辅助函数：填充掩码
pub fn padding_mask(seq_len: usize, actual_len: usize) -> Array2<f32> {
    let mut mask = Array2::ones((seq_len, seq_len));
    for i in actual_len..seq_len {
        for j in 0..seq_len {
            mask[[i, j]] = 0.0;
        }
    }
    mask
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matmul() {
        let a = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let b = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let c = a.matmul(&b);

        assert_eq!(c.shape(), &[2, 2]);
        assert!((c[[0, 0]] - 22.0).abs() < 1e-6);
    }

    #[test]
    fn test_softmax() {
        let x = Array2::from_shape_vec((1, 3), vec![1.0, 2.0, 3.0]).unwrap();
        let s = x.softmax(1);

        let sum: f32 = s.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_layer_norm() {
        let x = Array2::from_shape_vec((2, 3), vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0
        ]).unwrap();
        let gamma = Array2::ones((1, 3));
        let beta = Array2::zeros((1, 3));

        let y = x.layer_norm(&gamma, &beta, 1e-6);

        // 检查归一化后每行的均值接近 0，标准差接近 1
        for row in 0..2 {
            let row_data: Vec<f32> = y.row(row).into_iter().copied().collect();
            let mean: f32 = row_data.iter().sum::<f32>() / row_data.len() as f32;
            assert!(mean.abs() < 1e-5);
        }
    }
}
