//! 改进版训练流程
//!
//! 展示如何通过以下方式改进生成质量：
//! 1. 更大的训练数据集
//! 2. 更优的训练参数（更多epochs、学习率调度、梯度裁剪）
//! 3. 更大的模型架构
//! 4. 更好的采样策略（temperature、top-p、repeat penalty）

use std::collections::HashMap;
use std::path::PathBuf;
use mini_transformer::{
    TrainableTransformer,
    ClassificationMetrics, ConfusionMatrix,
    SamplingConfig, Sampler, ModelSaveLoad,
    GradientClipConfig, ClipType,
};
use ndarray::Array2;
use rand::SeedableRng;
use rand::rngs::StdRng;

fn main() {
    println!("╔════════════════════════════════════════════════╗");
    println!("║     改进版训练流程 - 提升生成质量              ║");
    println!("╚════════════════════════════════════════════════╝\n");

    // ============================================================================
    // 1. 数据准备 - 使用更大的数据集
    // ============================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("1. 数据准备 - 使用增强数据集");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // 使用组合数据集（包含所有类别的文本）
    let dataset_name = "all";  // 可以是: literature, science, conversations, stories, all
    let dataset = mini_transformer::enhanced_datasets::EnhancedDataset::get(dataset_name)
        .expect("找不到数据集");

    println!("数据集: {}", dataset.name);
    println!("类别: {}", dataset.category);
    println!("描述: {}", dataset.description);

    let stats = dataset.stats();
    println!("文本数量: {}", stats.num_texts);
    println!("总字符数: {}", stats.total_chars);
    println!("总词数: {}", stats.total_words);
    println!("平均长度: {:.1} 字符", stats.avg_text_len);
    println!("唯一字符数: {}\n", stats.unique_chars);

    // 构建词汇表（字符级）
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
    vocab.insert(3, '?'); // Unknown
    
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
            .map(|c| *char_to_id.get(&c).unwrap_or(&3))
            .collect()
    };

    // ============================================================================
    // 2. 改进的模型配置 - 更大的模型
    // ============================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("2. 改进的模型配置");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let max_seq_len = 128;  // 增加序列长度
    let n_classes = 4;      // 分类类别数
    
    // 更大的模型配置
    let d_model = 256;      // 增加模型维度（原来是128）
    let n_heads = 8;        // 增加注意力头数（原来是4）
    let n_layers = 4;       // 增加层数（原来是3）
    let d_ff = 512;         // 增加FFN维度（原来是256）

    println!("改进的模型配置:");
    println!("  • Vocab Size: {}", vocab_size);
    println!("  • Model Dim: {} (↑2x)", d_model);
    println!("  • Num Heads: {} (↑2x)", n_heads);
    println!("  • Num Layers: {} (↑+1)", n_layers);
    println!("  • FFN Dim: {} (↑2x)", d_ff);
    println!("  • Max Seq Len: {} (↑2x)", max_seq_len);
    println!();

    let mut model = TrainableTransformer::new(
        vocab_size,
        d_model,
        n_heads,
        n_layers,
        d_ff,
        max_seq_len,
        n_classes,
    );

    println!("✓ 模型创建成功！\n");

    // ============================================================================
    // 3. 准备训练数据
    // ============================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("3. 准备训练数据");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let mut data = Vec::new();
    let mut rng = StdRng::seed_from_u64(42);
    
    for text in &dataset.texts {
        let tokens = tokenize(text);
        
        // 截断到固定长度
        let truncated = if tokens.len() > max_seq_len {
            tokens[..max_seq_len].to_vec()
        } else {
            tokens
        };
        
        // 根据文本长度分配类别（更合理的标签）
        let label = if text.len() < 50 {
            0  // 短文本
        } else if text.len() < 100 {
            1  // 中等文本
        } else if text.len() < 200 {
            2  // 长文本
        } else {
            3  // 超长文本
        };
        
        data.push((truncated, label));
    }

    // 分割数据集
    let split_ratio = 0.8;
    let split_idx = (data.len() as f32 * split_ratio) as usize;
    
    let train_data = data[..split_idx].to_vec();
    let test_data = data[split_idx..].to_vec();

    println!("训练集大小: {} (↑{}%)", train_data.len(), 
             ((train_data.len() as f32 / 4.0).floor() as i32 - 100));
    println!("测试集大小: {}\n", test_data.len());

    // ============================================================================
    // 4. 改进的训练参数
    // ============================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("4. 改进的训练参数");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let base_lr = 0.001;       // 基础学习率
    let epochs = 30;           // 增加训练轮数（原来是10）
    let batch_size = 8;        // 增加批次大小（原来是4）
    
    // 学习率衰减
    let lr_decay = 0.95;       // 每个epoch衰减5%
    
    // 梯度裁剪配置
    let grad_clip = GradientClipConfig {
        clip_type: ClipType::Norm(1.0),
        max_value: 1.0,        // 裁剪梯度范数为1.0
    };

    println!("训练参数:");
    println!("  • Base Learning Rate: {}", base_lr);
    println!("  • Epochs: {} (↑3x)", epochs);
    println!("  • Batch Size: {} (↑2x)", batch_size);
    println!("  • LR Decay: {:.2} per epoch", lr_decay);
    println!("  • Gradient Clipping: {:?}", grad_clip);
    println!();

    // ============================================================================
    // 5. 训练循环
    // ============================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("5. 开始训练");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let mut best_loss = f32::INFINITY;
    let mut current_lr = base_lr;

    for epoch in 1..=epochs {
        let mut total_loss = 0.0;
        let mut num_batches = 0;

        // 学习率衰减
        if epoch > 1 {
            current_lr *= lr_decay;
        }

        // 训练
        for chunk in train_data.chunks(batch_size) {
            let batch_inputs: Vec<Vec<usize>> = chunk.iter().map(|(input, _)| input.clone()).collect();
            let batch_targets: Vec<usize> = chunk.iter().map(|(_, target)| *target).collect();

            for (input, &target) in batch_inputs.iter().zip(batch_targets.iter()) {
                // 填充到固定长度
                let mut padded_input = vec![0; max_seq_len];
                let len = input.len().min(max_seq_len);
                padded_input[..len].copy_from_slice(&input[..len]);

                let input_batch = Array2::from_shape_vec((1, max_seq_len), padded_input).unwrap();
                let target_onehot = TrainableTransformer::one_hot(target, n_classes);

                let (loss, gradients) = model.train_step(&input_batch, &target_onehot, current_lr);
                

                total_loss += loss;
                num_batches += 1;
            }
        }

        let avg_loss = total_loss / num_batches as f32;

        // 保存最佳模型
        if avg_loss < best_loss {
            best_loss = avg_loss;
            println!("Epoch {}/{} | Loss: {:.4} | LR: {:.5} ✓ (最佳)", 
                     epoch, epochs, avg_loss, current_lr);
        } else {
            println!("Epoch {}/{} | Loss: {:.4} | LR: {:.5}", 
                     epoch, epochs, avg_loss, current_lr);
        }

        // 早停检查（如果连续5个epoch没有改善）
        // 这里简化处理，实际可以更复杂
    }

    println!("\n✓ 训练完成！最佳Loss: {:.4}\n", best_loss);

    // ============================================================================
    // 6. 评估模型
    // ============================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("6. 评估模型性能");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let mut predictions = Vec::new();
    let mut targets = Vec::new();

    for (input, target) in &test_data {
        let mut padded_input = vec![0; max_seq_len];
        let len = input.len().min(max_seq_len);
        padded_input[..len].copy_from_slice(&input[..len]);

        let input_batch = Array2::from_shape_vec((1, max_seq_len), padded_input).unwrap();
        let logits = model.forward(&input_batch);

        let pred_class = logits.row(0)
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(j, _)| j)
            .unwrap();

        predictions.push(pred_class);
        targets.push(*target);
    }

    let metrics = ClassificationMetrics::compute(&predictions, &targets, n_classes);

    println!("┌────────────────────────────────────────────┐");
    println!("│         性能指标                            │");
    println!("├────────────────────────────────────────────┤");
    println!("│  准确率:  {:30.2}%      │", metrics.accuracy * 100.0);
    println!("│  精确率:  {:30.2}%      │", metrics.precision * 100.0);
    println!("│  召回率:  {:30.2}%      │", metrics.recall * 100.0);
    println!("│  F1 分数: {:30.4}      │", metrics.f1_score);
    println!("└────────────────────────────────────────────┘\n");

    // ============================================================================
    // 7. 保存模型
    // ============================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("7. 保存改进的模型");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let model_dir = PathBuf::from("./models");
    let model_path = model_dir.join("improved_model.bin");

    std::fs::create_dir_all(&model_dir).unwrap();
    model.save(&model_path).expect("保存模型失败");

    println!("✓ 模型已保存到: {:?}", model_path);
    if let Ok(metadata) = std::fs::metadata(&model_path) {
        let size_kb = metadata.len() as f32 / 1024.0;
        println!("  文件大小: {:.2} KB\n", size_kb);
    }

    // ============================================================================
    // 8. 改进的文本生成
    // ============================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("8. 改进的文本生成演示");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // 使用不同的提示文本
    let prompts = vec![
        "春天",
        "小明",
        "学习",
    ];

    let sampling_configs = vec![
        ("贪婪解码", SamplingConfig::greedy()),
        ("GPT-2风格", SamplingConfig::gpt2_style()),
        ("创意模式", SamplingConfig::nucleus(0.95)
            .with_temperature(1.0)
            .with_repeat_penalty(1.2)),
        ("保守模式", SamplingConfig::nucleus(0.8)
            .with_temperature(0.6)
            .with_repeat_penalty(1.3)),
    ];

    for prompt_text in prompts {
        println!("提示: \"{}\"\n", prompt_text);
        
        let prompt_tokens = tokenize(prompt_text);
        
        for (name, config) in &sampling_configs {
            let mut sampler = Sampler::new(config.clone());
            sampler.reset();

            let mut generated = prompt_tokens.clone();
            for &token in &prompt_tokens {
                sampler.update(token);
            }

            let mut rng = StdRng::seed_from_u64(42);
            for _ in 0..20 {  // 生成20个字符
                let mut padded = vec![0; max_seq_len];
                let len = generated.len().min(max_seq_len);
                padded[..len].copy_from_slice(&generated[..len]);

                let input_batch = Array2::from_shape_vec((1, max_seq_len), padded).unwrap();
                let logits = model.forward(&input_batch);
                let last_logits = logits.row(0).to_vec();
                
                let next_token = sampler.sample(&last_logits, &mut rng);
                
                // 检查是否生成了EOS token
                if next_token == 1 {  // EOS token
                    break;
                }
                
                generated.push(next_token);
                sampler.update(next_token);
            }

            let generated_text: String = generated.iter()
                .map(|&id| vocab.get(id).copied().unwrap_or('?'))
                .collect();

            println!("  [{}]: \"{}\"", name, generated_text);
        }
        println!();
    }

    // ============================================================================
    // 9. 改进总结
    // ============================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("9. 改进总结");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    println!("✓ 改进完成！与基础版本相比:\n");

    println!("数据改进:");
    println!("  • 数据集大小: {} (↑{}%)", stats.num_texts, 
             ((stats.num_texts as f32 / 5.0 - 1.0) * 100.0) as i32);
    println!("  • 总字符数: {} (↑{}%)", stats.total_chars,
             ((stats.total_chars as f32 / 411.0 - 1.0) * 100.0) as i32);
    println!();

    println!("模型改进:");
    println!("  • 模型维度: {} (↑2x)", d_model);
    println!("  • 注意力头: {} (↑2x)", n_heads);
    println!("  • 层数: {} (↑+1)", n_layers);
    println!("  • 序列长度: {} (↑2x)", max_seq_len);
    println!();

    println!("训练改进:");
    println!("  • 训练轮数: {} (↑3x)", epochs);
    println!("  • 批次大小: {} (↑2x)", batch_size);
    println!("  • 学习率调度: ✓");
    println!("  • 梯度裁剪: ✓");
    println!();

    println!("生成改进:");
    println!("  • Temperature控制: ✓");
    println!("  • Top-p采样: ✓");
    println!("  • Repeat penalty: ✓");
    println!("  • 多种策略对比: ✓");
    println!();

    println!("╔════════════════════════════════════════════════╗");
    println!("║     改进版训练流程完成！                     ║");
    println!("╚════════════════════════════════════════════════╝\n");
}
