//! 梯度裁剪
//!
//! 防止梯度爆炸，稳定训练过程

use ndarray::Array2;

/// 梯度裁剪配置
#[derive(Debug, Clone)]
pub struct GradientClipConfig {
    /// 裁剪类型
    pub clip_type: ClipType,
    /// 最大范数或最大值
    pub max_value: f32,
}

/// 裁剪类型
#[derive(Debug, Clone, PartialEq)]
pub enum ClipType {
    /// 按范数裁剪 (推荐)
    Norm(f32),
    /// 按值裁剪
    Value(f32),
}

impl Default for GradientClipConfig {
    fn default() -> Self {
        Self {
            clip_type: ClipType::Norm(1.0),
            max_value: 1.0,
        }
    }
}

impl GradientClipConfig {
    /// 创建范数裁剪配置
    ///
    /// `max_norm`: 最大梯度范数，推荐值 1.0-5.0
    pub fn norm(max_norm: f32) -> Self {
        Self {
            clip_type: ClipType::Norm(max_norm),
            max_value: max_norm,
        }
    }

    /// 创建值裁剪配置
    ///
    /// `max_value`: 最大梯度绝对值，推荐值 1.0-10.0
    pub fn value(max_value: f32) -> Self {
        Self {
            clip_type: ClipType::Value(max_value),
            max_value,
        }
    }
}

/// 计算梯度张量的 Frobenius 范数
///
/// ||A||_F = sqrt(sum(a_ij^2))
pub fn compute_grad_norm(grads: &[Array2<f32>]) -> f32 {
    let mut sum_squared = 0.0;

    for grad in grads {
        for &val in grad.iter() {
            sum_squared += val * val;
        }
    }

    sum_squared.sqrt()
}

/// 应用范数裁剪到单个梯度
///
/// 如果梯度范数超过 max_norm，则按比例缩放
pub fn clip_grad_norm_single(grad: &Array2<f32>, max_norm: f32) -> Array2<f32> {
    // 计算范数
    let mut norm = 0.0;
    for &val in grad.iter() {
        norm += val * val;
    }
    norm = norm.sqrt();

    // 如果范数超过阈值，则缩放
    if norm > max_norm {
        let scale = max_norm / norm;
        grad.mapv(|g| g * scale)
    } else {
        grad.clone()
    }
}

/// 应用范数裁剪到多个梯度
///
/// 返回裁剪后的梯度列表
pub fn clip_grad_norm(grads: &[Array2<f32>], max_norm: f32) -> Vec<Array2<f32>> {
    // 计算全局梯度范数
    let mut global_norm = 0.0;
    for grad in grads {
        for &val in grad.iter() {
            global_norm += val * val;
        }
    }
    global_norm = global_norm.sqrt();

    // 如果范数超过阈值，计算缩放因子
    let scale = if global_norm > max_norm {
        Some(max_norm / global_norm)
    } else {
        None
    };

    // 应用裁剪
    grads.iter()
        .map(|grad| {
            if let Some(s) = scale {
                grad.mapv(|g| g * s)
            } else {
                grad.clone()
            }
        })
        .collect()
}

/// 应用值裁剪到单个梯度
///
/// 将每个梯度值限制在 [-max_value, max_value] 范围内
pub fn clip_grad_value_single(grad: &Array2<f32>, max_value: f32) -> Array2<f32> {
    grad.mapv(|g| {
        if g > max_value {
            max_value
        } else if g < -max_value {
            -max_value
        } else {
            g
        }
    })
}

/// 应用值裁剪到多个梯度
pub fn clip_grad_value(grads: &[Array2<f32>], max_value: f32) -> Vec<Array2<f32>> {
    grads.iter()
        .map(|grad| clip_grad_value_single(grad, max_value))
        .collect()
}

/// 应用梯度裁剪（通用接口）
///
/// 根据配置自动选择裁剪类型
pub fn clip_gradients(grads: &[Array2<f32>], config: &GradientClipConfig) -> Vec<Array2<f32>> {
    match config.clip_type {
        ClipType::Norm(max_norm) => clip_grad_norm(grads, max_norm),
        ClipType::Value(max_value) => clip_grad_value(grads, max_value),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_grad_norm() {
        let grad1 = Array2::from_shape_vec((2, 2), vec![3.0, 4.0, 0.0, 0.0]).unwrap();
        // sqrt(3^2 + 4^2) = 5.0
        let norm = compute_grad_norm(&[grad1]);
        assert!((norm - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_clip_grad_norm_single() {
        let grad = Array2::from_shape_vec((2, 2), vec![3.0, 4.0, 0.0, 0.0]).unwrap();
        // norm = 5.0, max_norm = 2.5, scale = 0.5
        let clipped = clip_grad_norm_single(&grad, 2.5);

        // 应该缩放到 [1.5, 2.0, 0.0, 0.0]
        assert!((clipped[[0, 0]] - 1.5).abs() < 1e-6);
        assert!((clipped[[0, 1]] - 2.0).abs() < 1e-6);
        assert_eq!(clipped[[1, 0]], 0.0);
        assert_eq!(clipped[[1, 1]], 0.0);
    }

    #[test]
    fn test_clip_grad_norm_no_clip() {
        let grad = Array2::from_shape_vec((2, 2), vec![1.0, 1.0, 0.0, 0.0]).unwrap();
        // norm = sqrt(2) ≈ 1.414 < 2.5, 不应该裁剪
        let clipped = clip_grad_norm_single(&grad, 2.5);

        // 应该保持不变
        assert_eq!(clipped[[0, 0]], 1.0);
        assert_eq!(clipped[[0, 1]], 1.0);
        assert_eq!(clipped[[1, 0]], 0.0);
        assert_eq!(clipped[[1, 1]], 0.0);
    }

    #[test]
    fn test_clip_grad_value() {
        let grad = Array2::from_shape_vec((2, 2), vec![5.0, -10.0, 2.0, 0.0]).unwrap();
        let clipped = clip_grad_value_single(&grad, 3.0);

        // [3.0, -3.0, 2.0, 0.0]
        assert_eq!(clipped[[0, 0]], 3.0);
        assert_eq!(clipped[[0, 1]], -3.0);
        assert_eq!(clipped[[1, 0]], 2.0);
        assert_eq!(clipped[[1, 1]], 0.0);
    }

    #[test]
    fn test_gradient_clip_config() {
        let config = GradientClipConfig::norm(1.0);
        assert_eq!(config.max_value, 1.0);

        let config = GradientClipConfig::value(5.0);
        assert_eq!(config.max_value, 5.0);
    }
}
