//! RoPE (Rotary Position Embedding)
//!
//! 实现旋转位置编码，用于替代或补充绝对位置编码
//! 参考: RoFormer: Enhanced Transformer with Rotary Position Embedding

use ndarray::Array2;
use std::f32::consts::PI;

/// RoPE 配置
#[derive(Debug, Clone)]
pub struct RoPEConfig {
    /// 维度
    pub d_model: usize,
    /// 计算维度（通常 = d_model，多头时 = d_model / n_heads）
    pub head_dim: usize,
}

impl RoPEConfig {
    pub fn new(d_model: usize) -> Self {
        Self {
            d_model,
            head_dim: d_model,
        }
    }

    /// 为多头注意力设置每个头的维度
    pub fn with_head_dim(mut self, head_dim: usize) -> Self {
        self.head_dim = head_dim;
        self
    }
}

/// 应用 RoPE 到查询和键
///
/// 这是 RoPE 的核心实现
/// - q: 查询向量 [seq_len, head_dim]
/// - k: 键向量 [seq_len, head_dim]
/// - 返回: 旋转后的 q 和 k
pub fn apply_rotary_pos_emb(q: &Array2<f32>, k: &Array2<f32>, position: usize) -> (Array2<f32>, Array2<f32>) {
    let _seq_len = q.nrows();
    let head_dim = q.ncols();

    // 确保维度是偶数（RoPE 要求）
    assert!(head_dim % 2 == 0, "head_dim must be even for RoPE");

    // 计算旋转角度
    let freqs = compute_rotary_frequencies(head_dim, position);

    // 应用旋转到 q
    let q_rotated = apply_rotation(q, &freqs);

    // 应用旋转到 k
    let k_rotated = apply_rotation(k, &freqs);

    (q_rotated, k_rotated)
}

/// 计算旋转频率
///
/// θ_i = 10000^(-2(i-1)/d) for i in [1, d/2]
fn compute_rotary_frequencies(head_dim: usize, position: usize) -> Vec<f32> {
    let half_dim = head_dim / 2;
    let mut freqs = Vec::with_capacity(half_dim);

    for i in 0..half_dim {
        // θ_i = position * 10000^(-2(i-1)/d)
        let exponent = -((2 * i) as f32) / head_dim as f32;
        let theta = (position as f32) * (10000.0_f32).powf(exponent);
        freqs.push(theta);
    }

    freqs
}

/// 应用 2D 旋转矩阵
///
/// 对于向量 [x0, x1, x2, x3, ...]，旋转：
/// [x0*cos(θ0) - x1*sin(θ0), x0*sin(θ0) + x1*cos(θ0), x2*cos(θ1) - x3*sin(θ1), ...]
fn apply_rotation(x: &Array2<f32>, thetas: &[f32]) -> Array2<f32> {
    let seq_len = x.nrows();
    let dim = x.ncols();
    let mut rotated = Array2::zeros((seq_len, dim));

    for pos in 0..seq_len {
        for i in (0..dim).step_by(2) {
            if i + 1 < dim {
                let x0 = x[[pos, i]];
                let x1 = x[[pos, i + 1]];
                let theta = thetas[i / 2];

                // 2D 旋转
                rotated[[pos, i]] = x0 * theta.cos() - x1 * theta.sin();
                rotated[[pos, i + 1]] = x0 * theta.sin() + x1 * theta.cos();
            } else {
                // 奇数维度，直接复制
                rotated[[pos, i]] = x[[pos, i]];
            }
        }
    }

    rotated
}

/// 批量应用 RoPE
///
/// 对所有位置的 q 和 k 应用 RoPE
pub fn apply_rope_batch(
    q: &Array2<f32>,
    k: &Array2<f32>,
) -> (Array2<f32>, Array2<f32>) {
    let seq_len = q.nrows();
    let head_dim = q.ncols();

    let mut q_rotated = Array2::zeros((seq_len, head_dim));
    let mut k_rotated = Array2::zeros((seq_len, head_dim));

    for pos in 0..seq_len {
        let q_row = q.row(pos);
        let k_row = k.row(pos);

        // 单独处理每个位置
        let q_vec: Vec<f32> = q_row.iter().cloned().collect();
        let k_vec: Vec<f32> = k_row.iter().cloned().collect();

        let q_single = Array2::from_shape_vec((1, head_dim), q_vec).unwrap();
        let k_single = Array2::from_shape_vec((1, head_dim), k_vec).unwrap();

        let (q_new, k_new) = apply_rotary_pos_emb(&q_single, &k_single, pos);

        // 复制结果
        for i in 0..head_dim {
            q_rotated[[pos, i]] = q_new[[0, i]];
            k_rotated[[pos, i]] = k_new[[0, i]];
        }
    }

    (q_rotated, k_rotated)
}

/// RoPE 注意力包装器
///
/// 在标准 attention 中应用 RoPE
pub struct RoPEAttention {
    pub d_model: usize,
    pub n_heads: usize,
    pub head_dim: usize,
}

impl RoPEAttention {
    pub fn new(d_model: usize, n_heads: usize) -> Self {
        let head_dim = d_model / n_heads;
        Self {
            d_model,
            n_heads,
            head_dim,
        }
    }

    /// 应用 RoPE 到多头注意力的 Q 和 K
    ///
    /// - q: [seq_len, d_model] 展平的查询
    /// - k: [seq_len, d_model] 展平的键
    /// - 返回: (q_rotated, k_rotated)
    pub fn apply_rope_multihead(
        &self,
        q: &Array2<f32>,
        k: &Array2<f32>,
    ) -> (Array2<f32>, Array2<f32>) {
        // 简化：直接对整个矩阵应用 RoPE
        // （多头时需要更复杂的实现）
        let (q_rotated, k_rotated) = apply_rope_batch(q, k);

        (q_rotated, k_rotated)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_rotary_frequencies() {
        let freqs = compute_rotary_frequencies(4, 0);
        assert_eq!(freqs.len(), 2);
        // position=0 时，所有频率都是 0
        assert!((freqs[0] - 0.0).abs() < 1e-6);
        assert!((freqs[1] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_apply_rotation() {
        let x = Array2::from_shape_vec((1, 4), vec![1.0, 0.0, 2.0, 0.0]).unwrap();
        let thetas = vec![PI / 4.0, PI / 2.0];

        let rotated = apply_rotation(&x, &thetas);

        // [1*cos(45°) - 0*sin(45°), 1*sin(45°) + 0*cos(45°),
        //  2*cos(90°) - 0*sin(90°), 2*sin(90°) + 0*cos(90°)]
        // [0.707, 0.707, 0.0, 2.0]
        assert!((rotated[[0, 0]] - 0.707).abs() < 0.01);
        assert!((rotated[[0, 1]] - 0.707).abs() < 0.01);
        assert!((rotated[[0, 2]] - 0.0).abs() < 1e-6);
        assert!((rotated[[0, 3]] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_apply_rope() {
        let q = Array2::from_shape_vec((2, 4), vec![
            1.0, 0.0, 1.0, 0.0,
            0.0, 1.0, 0.0, 1.0,
        ]).unwrap();
        let k = Array2::from_shape_vec((2, 4), vec![
            0.5, 0.5, 0.5, 0.5,
            1.0, 0.0, 1.0, 0.0,
        ]).unwrap();

        let (q_rot, k_rot) = apply_rope_batch(&q, &k);

        // 检查形状不变
        assert_eq!(q_rot.shape(), q.shape());
        assert_eq!(k_rot.shape(), k.shape());

        // 检查向量范数不变（RoPE 保持范数）
        let norm_before = q[[0, 0]] * q[[0, 0]] + q[[0, 1]] * q[[0, 1]];
        let norm_after = q_rot[[0, 0]] * q_rot[[0, 0]] + q_rot[[0, 1]] * q_rot[[0, 1]];
        assert!((norm_before - norm_after).abs() < 1e-6);
    }

    #[test]
    fn test_rope_config() {
        let config = RoPEConfig::new(128);
        assert_eq!(config.d_model, 128);
        assert_eq!(config.head_dim, 128);

        let config = RoPEConfig::new(128).with_head_dim(64);
        assert_eq!(config.d_model, 128);
        assert_eq!(config.head_dim, 64);
    }
}
