//! 改进的文本生成示例
//!
//! 使用CausalLM和增强数据集展示如何改进生成质量

use std::collections::HashMap;
use std::path::PathBuf;
use mini_transformer::{
    CausalLM, CausalLMConfig,
    SamplingConfig, Sampler,
};
use rand::SeedableRng;
use rand::rngs::StdRng;

fn main() {
    println!("╔════════════════════════════════════════════════╗");
    println!("║     改进的文本生成 - 质量提升演示              ║");
    println!("╚════════════════════════════════════════════════╝\n");

    // ============================================================================
    // 1. 加载增强数据集
    // ============================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("1. 加载增强数据集");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let dataset = mini_transformer::enhanced_datasets::EnhancedDataset::get("all")
        .expect("找不到数据集");

    let stats = dataset.stats();
    println!("数据集: {}", dataset.name);
    println!("文本数量: {}", stats.num_texts);
    println!("总字符数: {}", stats.total_chars);
    println!("唯一字符数: {}\n", stats.unique_chars);

    // 构建词汇表
    let mut vocab: Vec<char> = dataset.texts
        .iter()
        .flat_map(|text| text.chars())
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();
    
    // 添加特殊token
    vocab.insert(0, '〈'); // Padding
    vocab.insert(1, '〉'); // End-of-sequence
    vocab.insert(2, '〈'); // Start
    
    let vocab_size = vocab.len();
    println!("词汇表大小: {}\n", vocab_size);

    // 创建字符到ID的映射
    let char_to_id: HashMap<char, usize> = vocab
        .iter()
        .enumerate()
        .map(|(i, &c)| (c, i))
        .collect();

    let tokenize = |text: &str| -> Vec<usize> {
        text.chars()
            .map(|c| *char_to_id.get(&c).unwrap_or(&0))
            .collect()
    };

    let detokenize = |tokens: &[usize]| -> String {
        tokens.iter()
            .map(|&id| if id < vocab.len() { vocab[id] } else { '〈' })
            .collect()
    };

    // 准备训练文本
    let training_texts: Vec<Vec<usize>> = dataset.texts
        .iter()
        .map(|text| tokenize(text))
        .collect();

    // ============================================================================
    // 2. 创建CausalLM模型
    // ============================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("2. 创建Causal Language Model");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let config = CausalLMConfig {
        vocab_size,
        d_model: 256,       // 增大模型维度
        n_heads: 8,         // 增加注意力头
        n_layers: 4,        // 增加层数
        d_ff: 512,          // 增加FFN维度
        max_seq_len: 128,   // 增加序列长度
        dropout: 0.1,
        use_rope: true,
    };

    println!("模型配置:");
    println!("  • Vocab Size: {}", vocab_size);
    println!("  • Model Dim: {}", config.d_model);
    println!("  • Num Heads: {}", config.n_heads);
    println!("  • Num Layers: {}", config.n_layers);
    println!("  • Max Seq Len: {}", config.max_seq_len);
    println!();

    let mut model = CausalLM::new(config.clone());
    println!("✓ 模型创建成功！\n");

    // ============================================================================
    // 3. 简单的训练演示
    // ============================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("3. 模型训练（简化版）");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // 注意：CausalLM目前没有训练方法，这里只是演示
    // 在实际应用中，你需要实现训练逻辑
    println!("注意：当前CausalLM是推理模型，训练功能待实现\n");
    println!("使用随机初始化的权重进行生成演示...\n");

    // ============================================================================
    // 4. 改进的生成配置
    // ============================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("4. 改进的生成策略");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let prompts = vec![
        "春天",
        "小明",
        "学习",
        "一天",
    ];

    let sampling_configs = vec![
        ("贪婪解码", SamplingConfig::greedy()),
        ("Top-k (k=10)", SamplingConfig::top_k(10)),
        ("Nucleus (p=0.9)", SamplingConfig::nucleus(0.9)),
        ("GPT-2风格", SamplingConfig::gpt2_style()),
        ("创意模式", SamplingConfig::nucleus(0.95)
            .with_temperature(1.1)
            .with_repeat_penalty(1.2)),
        ("保守模式", SamplingConfig::nucleus(0.8)
            .with_temperature(0.7)
            .with_repeat_penalty(1.3)),
    ];

    let max_new_tokens = 30;  // 生成的字符数

    for prompt_text in prompts {
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("提示: \"{}\"\n", prompt_text);
        
        let prompt_tokens = tokenize(prompt_text);
        
        for (name, sampling_config) in &sampling_configs {
            let mut sampler = Sampler::new(sampling_config.clone());
            sampler.reset();

            let mut generated = prompt_tokens.clone();
            for &token in &prompt_tokens {
                sampler.update(token);
            }

            let mut rng = StdRng::seed_from_u64(42);
            let mut has_eos = false;

            for _ in 0..max_new_tokens {
                // 限制长度
                if generated.len() >= config.max_seq_len {
                    break;
                }

                let logits = model.forward(&generated);
                let last_logits = logits.row(generated.len() - 1).to_vec();
                
                let next_token = sampler.sample(&last_logits, &mut rng);
                
                // 检查EOS
                if next_token == 1 {  // EOS token
                    has_eos = true;
                    break;
                }
                
                generated.push(next_token);
                sampler.update(next_token);
            }

            let generated_text = detokenize(&generated);
            println!("  [{}]: \"{}\"{}", 
                     name, 
                     generated_text,
                     if has_eos { " [EOS]" } else { "" });
        }
        println!();
    }

    // ============================================================================
    // 5. 生成质量分析
    // ============================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("5. 生成策略对比分析");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    println!("不同采样策略的特点:\n");

    println!("1. 贪婪解码 (Greedy):");
    println!("   ✓ 总是选择概率最大的token");
    println!("   ✓ 确定性强，输出一致");
    println!("   ✗ 容易陷入重复循环");
    println!("   ✗ 缺乏多样性\n");

    println!("2. Top-k采样:");
    println!("   ✓ 从概率最高的k个token中随机选择");
    println!("   ✓ 平衡质量和多样性");
    println!("   ✗ k太小时缺乏多样性，k太大时质量下降\n");

    println!("3. Nucleus (Top-p)采样:");
    println!("   ✓ 动态调整候选token数量");
    println!("   ✓ 累积概率达到p后停止");
    println!("   ✓ 比Top-k更灵活");
    println!("   ✗ 需要调参\n");

    println!("4. Temperature缩放:");
    println!("   ✓ 控制输出的随机性");
    println!("   ✓ T<1: 更保守、更确定");
    println!("   ✓ T>1: 更随机、更多样");
    println!("   ✗ 需要根据任务调整\n");

    println!("5. Repeat Penalty:");
    println!("   ✓ 减少重复内容");
    println!("   ✓ 改善文本流畅度");
    println!("   ✗ 惩罚太大会影响正常重复\n");

    // ============================================================================
    // 6. 改进建议
    // ============================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("6. 进一步改进的建议");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    println!("要进一步提升生成质量，建议:\n");

    println!("A. 数据层面:");
    println!("  • 使用更大规模的训练数据（百万级文本）");
    println!("  • 提高数据质量和多样性");
    println!("  • 使用更好的tokenization（如BPE）");
    println!("  • 添加数据增强技术\n");

    println!("B. 模型层面:");
    println!("  • 增大模型规模（更多层、更大维度）");
    println!("  • 实现真正的训练流程");
    println!("  • 添加学习率调度和优化器");
    println!("  • 使用预训练+微调范式\n");

    println!("C. 训练技巧:");
    println!("  • 实现CausalLM的训练方法");
    println!("  • 添加梯度累积和混合精度");
    println!("  • 使用早停和checkpointing");
    println!("  • 实现warmup和cosine学习率调度\n");

    println!("D. 生成策略:");
    println!("  • 实现beam search");
    println!("  • 添加top-k和top-p组合");
    println!("  • 使用频率和存在惩罚");
    println!("  • 实现长度惩罚\n");

    println!("╔════════════════════════════════════════════════╗");
    println!("║     改进生成演示完成！                     ║");
    println!("╚════════════════════════════════════════════════╝\n");

    println!("总结:");
    println!("  ✓ 扩大了数据集（{}个文本，{}字符）", stats.num_texts, stats.total_chars);
    println!("  ✓ 增大了模型规模（维度{}，{}层，{}头）", 
             config.d_model, config.n_layers, config.n_heads);
    println!("  ✓ 实现了多种采样策略");
    println!("  ✓ 添加了repeat penalty和temperature控制");
    println!("  ✗ CausalLM训练功能待实现\n");
}
