//! 文本生成采样策略
//!
//! 提供各种文本生成的采样方法和配置

use rand::Rng;

/// 采样配置
///
/// 控制文本生成的各种参数
#[derive(Debug, Clone)]
pub struct SamplingConfig {
    /// 采样方法
    pub method: SamplingMethod,
    /// 温度参数（控制随机性）
    /// - 1.0: 标准分布
    /// - < 1.0: 更保守（更集中）
    /// - > 1.0: 更随机（更均匀）
    pub temperature: f32,
    /// Top-k 采样（0 = 不使用）
    pub top_k: usize,
    /// Top-p (nucleus) 采样（0.0 = 不使用，1.0 = 使用所有）
    pub top_p: f32,
    /// 重复惩罚（1.0 = 不惩罚，> 1.0 = 惩罚重复）
    pub repeat_penalty: f32,
    /// 频率惩罚（用于有模型的情况）
    pub frequency_penalty: f32,
    /// 存在惩罚（惩罚已经生成的 token）
    pub presence_penalty: f32,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            method: SamplingMethod::Nucleus,
            temperature: 0.8,
            top_k: 0,
            top_p: 0.9,
            repeat_penalty: 1.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
        }
    }
}

impl SamplingConfig {
    /// 创建贪婪解码配置
    pub fn greedy() -> Self {
        Self {
            method: SamplingMethod::Greedy,
            temperature: 1.0,
            top_k: 0,
            top_p: 0.0,
            repeat_penalty: 1.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
        }
    }

    /// 创建 Top-k 采样配置
    pub fn top_k(k: usize) -> Self {
        Self {
            method: SamplingMethod::TopK,
            temperature: 1.0,
            top_k: k,
            top_p: 0.0,
            repeat_penalty: 1.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
        }
    }

    /// 创建 Nucleus (Top-p) 采样配置
    pub fn nucleus(p: f32) -> Self {
        Self {
            method: SamplingMethod::Nucleus,
            temperature: 0.8,
            top_k: 0,
            top_p: p,
            repeat_penalty: 1.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
        }
    }

    /// 创建典型的 GPT-2 风格配置
    pub fn gpt2_style() -> Self {
        Self {
            method: SamplingMethod::Nucleus,
            temperature: 0.7,
            top_k: 0,
            top_p: 0.9,
            repeat_penalty: 1.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
        }
    }

    /// 设置温度
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = temperature;
        self
    }

    /// 设置 Top-k
    pub fn with_top_k(mut self, top_k: usize) -> Self {
        self.top_k = top_k;
        self
    }

    /// 设置 Top-p
    pub fn with_top_p(mut self, top_p: f32) -> Self {
        self.top_p = top_p;
        self
    }

    /// 设置重复惩罚
    pub fn with_repeat_penalty(mut self, repeat_penalty: f32) -> Self {
        self.repeat_penalty = repeat_penalty;
        self
    }

    /// 设置频率惩罚
    pub fn with_frequency_penalty(mut self, frequency_penalty: f32) -> Self {
        self.frequency_penalty = frequency_penalty;
        self
    }

    /// 设置存在惩罚
    pub fn with_presence_penalty(mut self, presence_penalty: f32) -> Self {
        self.presence_penalty = presence_penalty;
        self
    }
}

/// 采样方法
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SamplingMethod {
    /// 贪婪解码（选择概率最大的）
    Greedy,
    /// Top-k 采样（从概率最大的 k 个词中采样）
    TopK,
    /// Nucleus (Top-p) 采样（从累积概率达到 p 的最小词集合中采样）
    Nucleus,
    /// 多项式采样（从所有词中采样）
    Multinomial,
}

/// 采样器
///
/// 根据配置对 logits 进行采样
pub struct Sampler {
    config: SamplingConfig,
    generated_tokens: Vec<usize>,
    generated_counts: Vec<usize>,
}

impl Sampler {
    /// 创建新的采样器
    pub fn new(config: SamplingConfig) -> Self {
        Self {
            config,
            generated_tokens: Vec::new(),
            generated_counts: Vec::new(),
        }
    }

    /// 重置采样器状态
    pub fn reset(&mut self) {
        self.generated_tokens.clear();
        self.generated_counts.clear();
    }

    /// 更新生成的 token
    pub fn update(&mut self, token: usize) {
        self.generated_tokens.push(token);

        // 更新计数（如果需要惩罚）
        if self.config.repeat_penalty != 1.0
            || self.config.frequency_penalty != 0.0
            || self.config.presence_penalty != 0.0
        {
            while self.generated_counts.len() <= token {
                self.generated_counts.push(0);
            }
            self.generated_counts[token] += 1;
        }
    }

    /// 对 logits 进行采样
    ///
    /// # 参数
    /// - `logits`: 模型输出的 logits [vocab_size]
    /// - `rng`: 随机数生成器
    ///
    /// # 返回
    /// - 采样的 token ID
    pub fn sample(&mut self, logits: &[f32], rng: &mut impl Rng) -> usize {
        let mut processed_logits = logits.to_vec();

        // 1. 应用惩罚
        processed_logits = self.apply_penalties(&processed_logits);

        // 2. 应用温度缩放
        if self.config.temperature != 1.0 {
            for logit in &mut processed_logits {
                *logit /= self.config.temperature;
            }
        }

        // 3. 应用采样方法
        match self.config.method {
            SamplingMethod::Greedy => {
                // 贪婪：选择概率最大的
                argmax(&processed_logits)
            }
            SamplingMethod::TopK => {
                // Top-k 采样
                if self.config.top_k > 0 && self.config.top_k < processed_logits.len() {
                    processed_logits = apply_top_k(&processed_logits, self.config.top_k);
                }
                sample_from_logits(&processed_logits, rng)
            }
            SamplingMethod::Nucleus => {
                // Top-p 采样
                if self.config.top_p > 0.0 && self.config.top_p < 1.0 {
                    processed_logits = apply_top_p(&processed_logits, self.config.top_p);
                }
                sample_from_logits(&processed_logits, rng)
            }
            SamplingMethod::Multinomial => {
                // 多项式采样
                sample_from_logits(&processed_logits, rng)
            }
        }
    }

    /// 应用各种惩罚
    fn apply_penalties(&self, logits: &[f32]) -> Vec<f32> {
        let mut penalized = logits.to_vec();

        for (token_id, &count) in self.generated_counts.iter().enumerate() {
            if count == 0 {
                continue;
            }

            // 重复惩罚
            if self.config.repeat_penalty != 1.0 {
                penalized[token_id] /= self.config.repeat_penalty.powi(count as i32 - 1);
            }

            // 频率惩罚
            if self.config.frequency_penalty != 0.0 {
                penalized[token_id] -= count as f32 * self.config.frequency_penalty;
            }

            // 存在惩罚
            if self.config.presence_penalty != 0.0 {
                penalized[token_id] -= self.config.presence_penalty;
            }
        }

        penalized
    }
}

/// Argmax：返回最大值的索引
pub fn argmax(values: &[f32]) -> usize {
    let mut max_idx = 0;
    let mut max_val = values[0];

    for (i, &val) in values.iter().enumerate() {
        if val > max_val {
            max_val = val;
            max_idx = i;
        }
    }

    max_idx
}

/// Softmax
pub fn softmax(logits: &[f32]) -> Vec<f32> {
    let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut exp_sum = 0.0;
    let mut exps = Vec::with_capacity(logits.len());

    for &logit in logits {
        let exp = (logit - max_logit).exp();
        exps.push(exp);
        exp_sum += exp;
    }

    exps.iter().map(|&e| e / exp_sum).collect()
}

/// 应用 Top-K 过滤
fn apply_top_k(logits: &[f32], k: usize) -> Vec<f32> {
    let mut filtered = logits.to_vec();

    // 找到第 k 大的值（使用 k-1 作为索引）
    let mut sorted = logits.to_vec();
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let threshold = sorted[(k - 1).min(sorted.len() - 1)];

    // 将小于阈值的值设为负无穷
    for logit in &mut filtered {
        if *logit < threshold {
            *logit = f32::NEG_INFINITY;
        }
    }

    filtered
}

/// 应用 Top-p (nucleus) 过滤
fn apply_top_p(logits: &[f32], p: f32) -> Vec<f32> {
    let mut filtered = logits.to_vec();

    // Softmax 得到概率
    let probs = softmax(&filtered);

    // 按概率排序
    let mut indexed_probs: Vec<_> = probs.iter().enumerate().collect();
    indexed_probs.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap_or(std::cmp::Ordering::Equal));

    // 计算累积概率
    let mut cumsum = 0.0;
    let mut cutoff = 0.0;

    for (_, &prob) in indexed_probs.iter() {
        cumsum += prob;
        if cumsum >= p {
            cutoff = prob;
            break;
        }
    }

    // 过滤掉概率小于阈值的
    for (i, &prob) in probs.iter().enumerate() {
        if prob < cutoff {
            filtered[i] = f32::NEG_INFINITY;
        }
    }

    filtered
}

/// 从 logits 采样
fn sample_from_logits(logits: &[f32], rng: &mut impl Rng) -> usize {
    let probs = softmax(logits);
    sample_categorical(&probs, rng)
}

/// 分类采样（根据概率分布采样）
fn sample_categorical(probs: &[f32], rng: &mut impl Rng) -> usize {
    let r = rng.gen::<f32>();

    let mut cumsum = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        cumsum += p;
        if r <= cumsum {
            return i;
        }
    }

    probs.len() - 1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_argmax() {
        let values = vec![0.1, 0.5, 0.3, 0.9, 0.2];
        assert_eq!(argmax(&values), 3); // 0.9 是最大的
    }

    #[test]
    fn test_softmax() {
        let values = vec![1.0, 2.0, 3.0];
        let probs = softmax(&values);

        assert!((probs.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert!(probs[2] > probs[1]); // 3.0 的概率最大
    }

    #[test]
    fn test_apply_top_k() {
        let logits = vec![0.1, 0.5, 0.3, 0.9, 0.2];
        let filtered = apply_top_k(&logits, 3);

        // 应该保留前 3 个最大的值
        let mut finite_count = 0;
        for &val in &filtered {
            if val.is_finite() {
                finite_count += 1;
            }
        }
        assert_eq!(finite_count, 3);
    }

    #[test]
    fn test_sampling_config_greedy() {
        let config = SamplingConfig::greedy();
        assert_eq!(config.method, SamplingMethod::Greedy);
        assert_eq!(config.temperature, 1.0);
    }

    #[test]
    fn test_sampling_config_top_k() {
        let config = SamplingConfig::top_k(50);
        assert_eq!(config.method, SamplingMethod::TopK);
        assert_eq!(config.top_k, 50);
    }

    #[test]
    fn test_sampling_config_nucleus() {
        let config = SamplingConfig::nucleus(0.9);
        assert_eq!(config.method, SamplingMethod::Nucleus);
        assert_eq!(config.top_p, 0.9);
    }

    #[test]
    fn test_sampler_with_penalties() {
        let config = SamplingConfig::default()
            .with_repeat_penalty(1.1)
            .with_presence_penalty(0.1);

        let mut sampler = Sampler::new(config);

        // 模拟生成了 token 0 两次
        sampler.update(0);
        sampler.update(0);

        // 检查计数
        assert_eq!(sampler.generated_counts[0], 2);
    }
}
