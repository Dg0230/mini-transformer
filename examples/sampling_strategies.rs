//! 文本生成采样策略示例
//!
//! 展示不同采样策略对生成文本的影响

use mini_transformer::{
    CausalLM, CausalLMConfig,
    SamplingConfig, SamplingMethod, Sampler,
};
use rand::SeedableRng;
use rand::rngs::StdRng;

/// 词汇表大小
const VOCAB_SIZE: usize = 50;

/// 创建一个简单的玩具词汇表
fn create_simple_vocab() -> Vec<&'static str> {
    vec![
        "The", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
        "A", "cat", "sat", "on", "mat", "The", "sun", "shines", "bright",
        "Hello", "world", "This", "is", "a", "test", "We", "love", "Rust",
        "Machine", "learning", "is", "fun", "AI", "models", "generate", "text",
        "Programming", "language", "code", "debug", "compile", "run",
        "Data", "structures", "algorithms", "efficiency", "performance",
    ]
}

/// 将 token IDs 转换为文本
fn tokens_to_text(tokens: &[usize], vocab: &[&str]) -> String {
    tokens.iter()
        .map(|&id| if id < vocab.len() { vocab[id] } else { "<UNK>" })
        .collect::<Vec<_>>()
        .join(" ")
}

fn main() {
    println!("╔════════════════════════════════════════════════╗");
    println!("║     文本生成采样策略示例                       ║");
    println!("╚════════════════════════════════════════════════╝\n");

    let vocab = create_simple_vocab();

    // ============================================================================
    // 1. 创建模型
    // ============================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("1. 创建 Causal Language Model");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let config = CausalLMConfig {
        vocab_size: VOCAB_SIZE,
        d_model: 64,
        n_heads: 2,
        n_layers: 2,
        d_ff: 128,
        max_seq_len: 32,
        dropout: 0.0,
        use_rope: true,
    };

    let mut model = CausalLM::new(config.clone());
    println!("✓ 模型创建成功！\n");
    println!("配置:");
    println!("  • Vocab Size: {}", config.vocab_size);
    println!("  • Model Dim: {}", config.d_model);
    println!("  • Num Layers: {}", config.n_layers);
    println!("  • Num Heads: {}", config.n_heads);
    println!();

    // ============================================================================
    // 2. 准备输入
    // ============================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("2. 准备输入文本");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let input_prompt = vec![0, 1, 2]; // "The quick brown"
    let prompt_text = tokens_to_text(&input_prompt, &vocab);
    println!("输入提示: \"{}\"\n", prompt_text);

    // ============================================================================
    // 3. 比较不同的采样策略
    // ============================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("3. 比较不同采样策略");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let max_new_tokens = 10;

    // 使用固定种子以确保可重复性
    let seed = 42;
    let mut rng = StdRng::seed_from_u64(seed);

    // ------------------------------------------------------------------------
    // 3.1 贪婪解码 (Greedy Decoding)
    // ------------------------------------------------------------------------
    println!("┌─────────────────────────────────────────────────────┐");
    println!("│  3.1 贪婪解码 (Greedy Decoding)                     │");
    println!("├─────────────────────────────────────────────────────┤");
    println!("│  特点: 总是选择概率最大的词                         │");
    println!("│  优点: 确定性强、输出一致                           │");
    println!("│  缺点: 可能重复、缺乏多样性                         │");
    println!("└─────────────────────────────────────────────────────┘\n");

    let greedy_config = SamplingConfig::greedy();
    let mut sampler = Sampler::new(greedy_config);
    sampler.reset();

    let mut greedy_tokens = input_prompt.clone();
    for token in &input_prompt {
        sampler.update(*token);
    }

    for _ in 0..max_new_tokens {
        let logits = model.forward(&greedy_tokens);
        let last_logits = logits.row(greedy_tokens.len() - 1).to_vec();

        let next_token = sampler.sample(&last_logits, &mut rng);
        greedy_tokens.push(next_token);
        sampler.update(next_token);
    }

    let greedy_text = tokens_to_text(&greedy_tokens, &vocab);
    println!("生成结果: \"{}\"\n", greedy_text);

    // ------------------------------------------------------------------------
    // 3.2 Top-k 采样
    // ------------------------------------------------------------------------
    println!("┌─────────────────────────────────────────────────────┐");
    println!("│  3.2 Top-k 采样 (k=5)                                │");
    println!("├─────────────────────────────────────────────────────┤");
    println!("│  特点: 从概率最大的 k 个词中随机采样                │");
    println!("│  优点: 平衡质量和多样性                             │");
    println!("│  缺点: k 太小会缺乏多样性，k 太大会质量下降         │");
    println!("└─────────────────────────────────────────────────────┘\n");

    let topk_config = SamplingConfig::top_k(5);
    let mut sampler = Sampler::new(topk_config);
    sampler.reset();

    let mut topk_tokens = input_prompt.clone();
    for token in &input_prompt {
        sampler.update(*token);
    }

    for _ in 0..max_new_tokens {
        let logits = model.forward(&topk_tokens);
        let last_logits = logits.row(topk_tokens.len() - 1).to_vec();

        let next_token = sampler.sample(&last_logits, &mut rng);
        topk_tokens.push(next_token);
        sampler.update(next_token);
    }

    let topk_text = tokens_to_text(&topk_tokens, &vocab);
    println!("生成结果: \"{}\"\n", topk_text);

    // ------------------------------------------------------------------------
    // 3.3 Nucleus (Top-p) 采样
    // ------------------------------------------------------------------------
    println!("┌─────────────────────────────────────────────────────┐");
    println!("│  3.3 Nucleus 采样 (p=0.9)                            │");
    println!("├─────────────────────────────────────────────────────┤");
    println!("│  特点: 从累积概率达到 p 的最小词集合中采样          │");
    println!("│  优点: 动态调整候选词数量                           │");
    println!("│  缺点: 需要调参                                     │");
    println!("└─────────────────────────────────────────────────────┘\n");

    let nucleus_config = SamplingConfig::nucleus(0.9);
    let mut sampler = Sampler::new(nucleus_config);
    sampler.reset();

    let mut nucleus_tokens = input_prompt.clone();
    for token in &input_prompt {
        sampler.update(*token);
    }

    for _ in 0..max_new_tokens {
        let logits = model.forward(&nucleus_tokens);
        let last_logits = logits.row(nucleus_tokens.len() - 1).to_vec();

        let next_token = sampler.sample(&last_logits, &mut rng);
        nucleus_tokens.push(next_token);
        sampler.update(next_token);
    }

    let nucleus_text = tokens_to_text(&nucleus_tokens, &vocab);
    println!("生成结果: \"{}\"\n", nucleus_text);

    // ------------------------------------------------------------------------
    // 3.4 Temperature 缩放
    // ------------------------------------------------------------------------
    println!("┌─────────────────────────────────────────────────────┐");
    println!("│  3.4 Temperature 变化对比                            │");
    println!("├─────────────────────────────────────────────────────┤");
    println!("│  Temperature 控制输出的随机性：                     │");
    println!("│  • T < 1.0: 更保守、更确定                          │");
    println!("│  • T = 1.0: 标准分布                                │");
    println!("│  • T > 1.0: 更随机、更多样                          │");
    println!("└─────────────────────────────────────────────────────┘\n");

    for temp in [0.3, 0.7, 1.0, 1.5] {
        let temp_config = SamplingConfig::nucleus(0.9).with_temperature(temp);
        let mut sampler = Sampler::new(temp_config);
        sampler.reset();

        let mut temp_tokens = input_prompt.clone();
        for token in &input_prompt {
            sampler.update(*token);
        }

        for _ in 0..max_new_tokens {
            let logits = model.forward(&temp_tokens);
            let last_logits = logits.row(temp_tokens.len() - 1).to_vec();

            let next_token = sampler.sample(&last_logits, &mut rng);
            temp_tokens.push(next_token);
            sampler.update(next_token);
        }

        let temp_text = tokens_to_text(&temp_tokens, &vocab);
        println!("  T={:.1}: \"{}\"", temp, temp_text);
    }
    println!();

    // ------------------------------------------------------------------------
    // 3.5 重复惩罚
    // ------------------------------------------------------------------------
    println!("┌─────────────────────────────────────────────────────┐");
    println!("│  3.5 重复惩罚对比                                    │");
    println!("├─────────────────────────────────────────────────────┤");
    println!("│  重复惩罚可以减少生成文本中的重复内容：             │");
    println!("│  • penalty = 1.0: 不惩罚                            │");
    println!("│  • penalty = 1.1: 轻度惩罚                          │");
    println!("│  • penalty = 1.3: 中度惩罚                          │");
    println!("│  • penalty = 1.5: 强度惩罚                          │");
    println!("└─────────────────────────────────────────────────────┘\n");

    for penalty in [1.0, 1.1, 1.3, 1.5] {
        let penalty_config = SamplingConfig::nucleus(0.9)
            .with_temperature(0.8)
            .with_repeat_penalty(penalty);
        let mut sampler = Sampler::new(penalty_config);
        sampler.reset();

        let mut penalty_tokens = input_prompt.clone();
        for token in &input_prompt {
            sampler.update(*token);
        }

        for _ in 0..max_new_tokens {
            let logits = model.forward(&penalty_tokens);
            let last_logits = logits.row(penalty_tokens.len() - 1).to_vec();

            let next_token = sampler.sample(&last_logits, &mut rng);
            penalty_tokens.push(next_token);
            sampler.update(next_token);
        }

        let penalty_text = tokens_to_text(&penalty_tokens, &vocab);

        // 计算唯一 token 比例
        let unique_tokens = penalty_tokens.iter().collect::<std::collections::HashSet<_>>().len();
        let unique_ratio = unique_tokens as f32 / penalty_tokens.len() as f32;

        println!("  penalty={:.1}: \"{}\"", penalty, penalty_text);
        println!("              (唯一词比例: {:.1}%)", unique_ratio * 100.0);
    }
    println!();

    // ============================================================================
    // 4. 组合策略推荐
    // ============================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("4. 推荐的采样策略配置");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    println!("┌─────────────────────────────────────────────────────┐");
    println!("│  场景 1: 创意写作                                     │");
    println!("├─────────────────────────────────────────────────────┤");
    println!("│  Temperature: 0.9-1.2                                │");
    println!("│  Top-p: 0.9-0.95                                    │");
    println!("│  Repeat Penalty: 1.1-1.2                            │");
    println!("│  说明: 需要多样性和创意                              │");
    println!("└─────────────────────────────────────────────────────┘\n");

    println!("┌─────────────────────────────────────────────────────┐");
    println!("│  场景 2: 代码生成                                     │");
    println!("├─────────────────────────────────────────────────────┤");
    println!("│  Temperature: 0.2-0.5                                │");
    println!("│  Top-k: 30-50                                       │");
    println!("│  Repeat Penalty: 1.0-1.1                            │");
    println!("│  说明: 需要准确性和一致性                            │");
    println!("└─────────────────────────────────────────────────────┘\n");

    println!("┌─────────────────────────────────────────────────────┐");
    println!("│  场景 3: 问答系统                                     │");
    println!("├─────────────────────────────────────────────────────┤");
    println!("│  Temperature: 0.6-0.8                                │");
    println!("│  Top-p: 0.9                                         │");
    println!("│  Repeat Penalty: 1.1                                │");
    println!("│  说明: 平衡准确性和流畅度                            │");
    println!("└─────────────────────────────────────────────────────┘\n");

    println!("┌─────────────────────────────────────────────────────┐");
    println!("│  场景 4: 翻译任务                                     │");
    println!("├─────────────────────────────────────────────────────┤");
    println!("│  Temperature: 0.3-0.5                                │");
    println!("│  Top-k: 20-30                                       │");
    println!("│  Repeat Penalty: 1.0                                │");
    println!("│  说明: 需要忠实于原文                                │");
    println!("└─────────────────────────────────────────────────────┘\n");

    // ============================================================================
    // 5. 总结
    // ============================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("5. 采样策略总结");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    println!("关键参数说明:");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    println!("1. Temperature (温度):");
    println!("   • 控制输出分布的平滑度");
    println!("   • 高温 → 更均匀分布 → 更多样");
    println!("   • 低温 → 更尖锐分布 → 更确定\n");

    println!("2. Top-k 采样:");
    println!("   • 只考虑概率最高的 k 个词");
    println!("   • k=1 等同于贪婪解码");
    println!("   • 典型值: 30-50\n");

    println!("3. Top-p (Nucleus) 采样:");
    println!("   • 动态选择候选词集合");
    println!("   • 累积概率达到 p 后停止");
    println!("   • 典型值: 0.9-0.95\n");

    println!("4. 重复惩罚:");
    println!("   • 惩罚已经生成过的词");
    println!("   • 减少\"循环生成\"问题");
    println!("   • 典型值: 1.1-1.3\n");

    println!("提示:");
    println!("  • 没有万能的最佳配置");
    println!("  • 根据具体任务调整参数");
    println!("  • 使用验证集评估不同配置");
    println!("  • 考虑模型大小和训练数据质量\n");

    println!("╔════════════════════════════════════════════════╗");
    println!("║     文本生成采样策略示例完成！                 ║");
    println!("╚════════════════════════════════════════════════╝\n");
}
