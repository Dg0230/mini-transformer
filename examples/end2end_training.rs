//! 端到端训练流程示例
//!
//! 展示从数据准备到模型推理的完整流程：
//! 1. 加载内置数据集
//! 2. 数据预处理和tokenization
//! 3. 创建模型
//! 4. 训练
//! 5. 评估
//! 6. 保存模型
//! 7. 加载模型
//! 8. 文本生成

use std::collections::HashMap;
use rand::Rng;
use std::path::PathBuf;
use mini_transformer::{
    TrainableTransformer, SimpleDataset,
    ClassificationMetrics, ConfusionMatrix, Perplexity,
    SamplingConfig, Sampler, ModelSaveLoad,
};
use ndarray::Array2;
use rand::SeedableRng;
use rand::rngs::StdRng;

fn main() {
    println!("╔════════════════════════════════════════════════╗");
    println!("║     端到端训练流程示例                         ║");
    println!("╚════════════════════════════════════════════════╝\n");

    // ============================================================================
    // 1. 数据准备
    // ============================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("1. 数据准备");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // 加载内置数据集
    let dataset_name = "stories";  // 可以是: aesop, tech, stories, poems
    let builtin_dataset = mini_transformer::builtin_datasets::BuiltinDataset::get(dataset_name)
        .expect("找不到数据集");

    println!("数据集: {}", builtin_dataset.name);
    println!("描述: {}", builtin_dataset.description);
    println!("文本数量: {}", builtin_dataset.texts.len());

    let stats = builtin_dataset.stats();
    println!("总字符数: {}", stats.total_chars);
    println!("总词数: {}", stats.total_words);
    println!("平均长度: {:.1} 字符", stats.avg_text_len);
    println!("唯一字符数: {}\n", stats.unique_chars);

    // 显示样本
    println!("样本文本（前3个）:");
    for (i, text) in builtin_dataset.texts.iter().take(3).enumerate() {
        println!("  [{}] {}...", i + 1, text.chars().take(50).collect::<String>());
    }
    println!();

    // 构建词汇表（字符级）
    let mut vocab: Vec<char> = builtin_dataset.texts
        .iter()
        .flat_map(|text| text.chars())
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();
    
    // 添加特殊token
    vocab.insert(0, '<'); // Padding
    vocab.insert(1, '〉'); // End-of-sentence marker
    vocab.insert(2, '〈'); // Start marker
    vocab.insert(3, '?'); // Unknown
    
    let vocab_size = vocab.len();
    println!("词汇表大小: {}", vocab_size);

    // 创建字符到ID的映射
    let char_to_id: HashMap<char, usize> = vocab
        .iter()
        .enumerate()
        .map(|(i, &c)| (c, i))
        .collect();

    // Tokenize函数
    let tokenize = |text: &str| -> Vec<usize> {
        text.chars()
            .map(|c| *char_to_id.get(&c).unwrap_or(&3))  // 3是UNK
            .collect()
    };

    println!();

    // ============================================================================
    // 2. 创建训练数据集
    // ============================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("2. 创建训练数据集");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // 配置参数
    let max_seq_len = 64;  // 序列最大长度
    let n_classes = 3;     // 分类类别数（例如：故事类型）
    
    // 为每个文本分配一个随机类别标签
    let mut data = Vec::new();
    let mut rng = StdRng::seed_from_u64(42);
    
    for text in &builtin_dataset.texts {
        let tokens = tokenize(text);
        
        // 截断或填充到固定长度
        let truncated = if tokens.len() > max_seq_len {
            tokens[..max_seq_len].to_vec()
        } else {
            tokens
        };
        
        // 随机类别标签
        let label = rng.gen_range(0..n_classes);
        
        data.push((truncated, label));
    }

    // 分割训练集和测试集
    let split_ratio = 0.8;
    let split_idx = (data.len() as f32 * split_ratio) as usize;
    
    let train_data = data[..split_idx].to_vec();
    let test_data = data[split_idx..].to_vec();

    println!("训练集大小: {}", train_data.len());
    println!("测试集大小: {}", test_data.len());
    println!("序列长度: {}", max_seq_len);
    println!("分类数: {}\n", n_classes);

    // ============================================================================
    // 3. 创建模型
    // ============================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("3. 创建模型");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let d_model = 128;
    let n_heads = 4;
    let n_layers = 3;
    let d_ff = 256;

    println!("模型配置:");
    println!("  • Vocab Size: {}", vocab_size);
    println!("  • Model Dim: {}", d_model);
    println!("  • Num Heads: {}", n_heads);
    println!("  • Num Layers: {}", n_layers);
    println!("  • FFN Dim: {}", d_ff);
    println!("  • Max Seq Len: {}", max_seq_len);
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
    // 4. 训练模型
    // ============================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("4. 训练模型");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let learning_rate = 0.001;
    let epochs = 10;
    let batch_size = 4;

    println!("训练参数:");
    println!("  • Learning Rate: {}", learning_rate);
    println!("  • Epochs: {}", epochs);
    println!("  • Batch Size: {}", batch_size);
    println!();

    let mut best_loss = f32::INFINITY;

    for epoch in 1..=epochs {
        let mut total_loss = 0.0;
        let mut num_batches = 0;

        // 训练
        for chunk in train_data.chunks(batch_size) {
            let batch_inputs: Vec<Vec<usize>> = chunk.iter().map(|(input, _)| input.clone()).collect();
            let batch_targets: Vec<usize> = chunk.iter().map(|(_, target)| *target).collect();

            for (input, &target) in batch_inputs.iter().zip(batch_targets.iter()) {
                // 填充到固定长度
                let mut padded_input = vec![0; max_seq_len];  // 0是PAD
                let len = input.len().min(max_seq_len);
                padded_input[..len].copy_from_slice(&input[..len]);

                let input_batch = Array2::from_shape_vec((1, max_seq_len), padded_input).unwrap();
                let target_onehot = TrainableTransformer::one_hot(target, n_classes);

                let (loss, _) = model.train_step(&input_batch, &target_onehot, learning_rate);
                total_loss += loss;
                num_batches += 1;
            }
        }

        let avg_loss = total_loss / num_batches as f32;

        // 保存最佳模型
        if avg_loss < best_loss {
            best_loss = avg_loss;
            println!("Epoch {}/{} | Loss: {:.4} ✓ (最佳)", epoch, epochs, avg_loss);
        } else {
            println!("Epoch {}/{} | Loss: {:.4}", epoch, epochs, avg_loss);
        }
    }

    println!("\n✓ 训练完成！最佳Loss: {:.4}\n", best_loss);

    // ============================================================================
    // 5. 评估模型
    // ============================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("5. 评估模型");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // 获取预测结果
    let mut predictions = Vec::new();
    let mut targets = Vec::new();

    for (input, target) in &test_data {
        let mut padded_input = vec![0; max_seq_len];
        let len = input.len().min(max_seq_len);
        padded_input[..len].copy_from_slice(&input[..len]);

        let input_batch = Array2::from_shape_vec((1, max_seq_len), padded_input).unwrap();
        let logits = model.forward(&input_batch);

        // Argmax获取预测类别
        let pred_class = logits.row(0)
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(j, _)| j)
            .unwrap();

        predictions.push(pred_class);
        targets.push(*target);
    }

    // 计算分类指标
    let metrics = ClassificationMetrics::compute(&predictions, &targets, n_classes);

    println!("┌────────────────────────────────────────────┐");
    println!("│         分类性能指标                      │");
    println!("├────────────────────────────────────────────┤");
    println!("│  准确率 (Accuracy):  {:20.2}%      │", metrics.accuracy * 100.0);
    println!("│  精确率 (Precision): {:20.2}%      │", metrics.precision * 100.0);
    println!("│  召回率 (Recall):     {:20.2}%      │", metrics.recall * 100.0);
    println!("│  F1 分数 (F1 Score): {:20.4}      │", metrics.f1_score);
    println!("└────────────────────────────────────────────┘\n");

    // 混淆矩阵
    println!("混淆矩阵:\n");
    let cm = ConfusionMatrix::compute(&predictions, &targets, n_classes);
    println!("{}", cm.display());

    // ============================================================================
    // 6. 保存模型
    // ============================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("6. 保存模型");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let model_dir = PathBuf::from("./models");
    let model_path = model_dir.join("trained_model.bin");

    // 创建目录
    std::fs::create_dir_all(&model_dir).unwrap();

    // 保存模型
    model.save(&model_path).expect("保存模型失败");

    println!("✓ 模型已保存到: {:?}", model_path);

    // 显示文件大小
    if let Ok(metadata) = std::fs::metadata(&model_path) {
        let size_kb = metadata.len() as f32 / 1024.0;
        println!("  文件大小: {:.2} KB\n", size_kb);
    }

    // ============================================================================
    // 7. 加载模型
    // ============================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("7. 加载模型");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // 创建新模型
    let mut loaded_model = TrainableTransformer::new(
        vocab_size,
        d_model,
        n_heads,
        n_layers,
        d_ff,
        max_seq_len,
        n_classes,
    );

    // 加载权重
    loaded_model.load(&model_path).expect("加载模型失败");

    println!("✓ 模型加载成功！\n");

    // 验证加载是否正确
    println!("验证加载的模型...");
    let test_input = vec![0, 1, 2, 3];
    let test_input_batch = Array2::from_shape_vec((1, 4), test_input).unwrap();
    
    let output1 = model.forward(&test_input_batch);
    let output2 = loaded_model.forward(&test_input_batch);

    let max_diff = output1.iter()
        .zip(output2.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(f32::NEG_INFINITY, f32::max);

    println!("  输出差异: {:.10} {}", max_diff, if max_diff < 1e-5 { "✓" } else { "✗" });
    println!();

    // ============================================================================
    // 8. 文本生成演示
    // ============================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("8. 文本生成演示");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    println!("使用采样策略生成文本:\n");

    // 准备一个简单的提示
    let prompt_text = "小明";
    let prompt_tokens = tokenize(prompt_text);
    println!("提示: \"{}\" ({:?})\n", prompt_text, prompt_tokens);

    // 演示不同的采样策略
    println!("生成结果对比 (10个tokens):\n");

    let sampling_configs = vec![
        ("贪婪解码", SamplingConfig::greedy()),
        ("Top-k (k=5)", SamplingConfig::top_k(5)),
        ("Nucleus (p=0.9)", SamplingConfig::nucleus(0.9)),
        ("高温 (T=1.2)", SamplingConfig::nucleus(0.9).with_temperature(1.2)),
    ];

    for (name, config) in sampling_configs {
        let mut sampler = Sampler::new(config);
        sampler.reset();

        // 初始化
        let mut generated = prompt_tokens.clone();
        for &token in &prompt_tokens {
            sampler.update(token);
        }

        // 生成
        let mut rng = StdRng::seed_from_u64(42);
        for _ in 0..10 {
            let mut padded = vec![0; max_seq_len];
            let len = generated.len().min(max_seq_len);
            padded[..len].copy_from_slice(&generated[..len]);

            let input_batch = Array2::from_shape_vec((1, max_seq_len), padded).unwrap();
            let logits = loaded_model.forward(&input_batch);

            // 获取最后一个位置的logits
            let last_logits = logits.row(0).to_vec();
            
            let next_token = sampler.sample(&last_logits, &mut rng);
            generated.push(next_token);
            sampler.update(next_token);
        }

        // 解码为文本
        let generated_text: String = generated.iter()
            .map(|&id| vocab.get(id).copied().unwrap_or('?'))
            .collect();

        println!("  {}: \"{}\"", name, generated_text);
    }

    println!();

    // ============================================================================
    // 9. 总结
    // ============================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("9. 总结");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    println!("✓ 完整的训练流程已完成！\n");

    println!("流程总结:");
    println!("  1. ✓ 加载内置数据集 ({})", dataset_name);
    println!("  2. ✓ 构建词汇表并tokenize");
    println!("  3. ✓ 创建训练/测试数据集");
    println!("  4. ✓ 训练模型 ({} epochs)", epochs);
    println!("  5. ✓ 评估性能 (Accuracy: {:.1}%)", metrics.accuracy * 100.0);
    println!("  6. ✓ 保存模型到文件");
    println!("  7. ✓ 加载并验证模型");
    println!("  8. ✓ 文本生成演示\n");

    println!("关键成果:");
    println!("  • 训练了 {} 个样本", train_data.len());
    println!("  • 词汇表大小: {}", vocab_size);
    println!("  • 测试准确率: {:.1}%", metrics.accuracy * 100.0);
    println!("  • 模型已保存到: {:?}", model_path);
    println!();

    println!("提示:");
    println!("  • 使用真实的大规模数据集会获得更好的效果");
    println!("  • 调整模型大小和训练参数可以改善性能");
    println!("  • 可以尝试不同的采样策略来控制生成质量");
    println!();

    println!("╔════════════════════════════════════════════════╗");
    println!("║     端到端训练流程示例完成！                 ║");
    println!("╚════════════════════════════════════════════════╝\n");
}
