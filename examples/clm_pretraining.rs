//! Causal Language Modeling (CLM) 预训练示例
//!
//! 演示 GPT 风格的预训练：预测下一个 token

use mini_transformer::{Seq2SeqTransformer, CLMConfig, create_clm_targets, Vocabulary};

fn main() {
    println!("╔════════════════════════════════════════════════╗");
    println!("║   CLM 预训练示例 (GPT 风格)                  ║");
    println!("╚════════════════════════════════════════════════╝\n");

    // ============================================================================
    // 1. 准备预训练数据
    // ============================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("1. 准备预训练数据");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // 模拟预训练文本
    let corpus = vec![
        "the cat sat on the mat",
        "dogs are loyal animals",
        "birds can fly high",
        "fish swim in water",
        "trees grow tall",
        "the sun is bright",
        "moon shines at night",
        "stars twinkle brightly",
        "children love to play",
        "music brings joy",
        "reading books is educational",
        "writing helps express thoughts",
        "learning new skills",
        "practice makes perfect",
        "teamwork achieves goals",
    ];

    // 构建词汇表
    let mut vocab = Vocabulary::new();
    for text in &corpus {
        for token in text.split_whitespace() {
            if !vocab.token_to_id.contains_key(&token.to_string()) {
                let id = vocab.vocab_size;
                vocab.token_to_id.insert(token.to_string(), id);
                vocab.id_to_token.push(token.to_string());
                vocab.vocab_size += 1;
            }
        }
    }

    // 添加特殊 tokens
    let start_token_id = vocab.vocab_size;
    vocab.token_to_id.insert("<START>".to_string(), start_token_id);
    vocab.id_to_token.push("<START>".to_string());
    vocab.vocab_size += 1;

    let end_token_id = vocab.vocab_size;
    vocab.token_to_id.insert("<END>".to_string(), end_token_id);
    vocab.id_to_token.push("<END>".to_string());
    vocab.vocab_size += 1;

    println!("词汇表大小: {}", vocab.vocab_size);
    println!("特殊 tokens:");
    println!("  <UNK>=0, <PAD>=1");
    println!("  <START>={}", start_token_id);
    println!("  <END>={}\n", end_token_id);

    // 编码语料
    let mut encoded_corpus = Vec::new();
    for text in &corpus {
        let tokens: Vec<usize> = text.split_whitespace()
            .map(|token| *vocab.token_to_id.get(token).unwrap_or(&0))
            .collect();
        encoded_corpus.push(tokens);
    }

    println!("编码后的语料示例:");
    println!("  原始: {}", corpus[0]);
    println!("  编码: {:?}\n", encoded_corpus[0]);

    // ============================================================================
    // 2. 演示 CLM 目标创建
    // ============================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("2. CLM 目标创建演示");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let clm_config = CLMConfig {
        start_token_id,
        end_token_id,
        pad_token_id: 1,
    };

    println!("GPT 风格因果语言建模:\n");
    println!("  给定当前 token 序列，预测下一个 token");
    println!("  输入: [<START>, t1, t2, t3, ...]");
    println!("  目标: [t1, t2, t3, ..., <END>]\n");

    for (i, tokens) in encoded_corpus.iter().take(3).enumerate() {
        let (inputs, targets) = create_clm_targets(tokens, &clm_config);

        println!("样本 {}:", i + 1);
        println!("  原始:  {}", corpus[i]);
        println!("  输入:  {:?}", inputs);
        println!("  目标:  {:?}", targets);

        let input_text: Vec<String> = inputs.iter()
            .filter_map(|&id| vocab.id_to_token.get(id).cloned())
            .collect();
        let target_text: Vec<String> = targets.iter()
            .filter_map(|&id| vocab.id_to_token.get(id).cloned())
            .collect();

        println!("  输入文本: {}", input_text.join(" "));
        println!("  目标文本: {}\n", target_text.join(" "));
    }

    // ============================================================================
    // 3. 创建 CLM 预训练模型
    // ============================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("3. 创建 CLM 预训练模型");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let mut model = Seq2SeqTransformer::new(
        vocab.vocab_size,  // vocab_size
        128,               // d_model
        4,                 // n_heads
        2,                 // n_layers
        256,               // d_ff
        20,                // max_seq_len
    );

    println!("模型架构:");
    println!("  词汇表大小: {}", vocab.vocab_size);
    println!("  模型维度: {}", 128);
    println!("  注意力头数: {}", 4);
    println!("  Decoder 层数: {}", 2);
    println!("  FFN 维度: {}", 256);
    println!("  参数总数: {}\n", model.param_count());

    // ============================================================================
    // 4. 准备训练数据
    // ============================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("4. 准备 CLM 训练数据");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let mut train_inputs = Vec::new();
    let mut train_targets = Vec::new();
    let mut val_inputs = Vec::new();
    let mut val_targets = Vec::new();

    let n_train = (encoded_corpus.len() * 8 / 10) as usize;

    for (i, tokens) in encoded_corpus.iter().enumerate() {
        let (inputs, targets) = create_clm_targets(tokens, &clm_config);

        if i < n_train {
            train_inputs.push(inputs);
            train_targets.push(targets);
        } else {
            val_inputs.push(inputs);
            val_targets.push(targets);
        }
    }

    println!("训练样本: {}", train_inputs.len());
    println!("验证样本: {}\n", val_inputs.len());

    // ============================================================================
    // 5. 开始 CLM 预训练
    // ============================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("5. 开始 CLM 预训练");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let mut best_val_acc = 0.0;
    let mut patience_counter = 0;
    let epochs = 50;
    let learning_rate = 0.001;
    let patience = 10;

    println!("训练配置:");
    println!("  Epochs: {}", epochs);
    println!("  学习率: {}", learning_rate);
    println!("  早停 patience: {}\n", patience);

    for epoch in 1..=epochs {
        let epoch_start = std::time::Instant::now();

        // 训练
        let mut total_loss = 0.0;
        let mut total_acc = 0.0;

        for (input, target) in train_inputs.iter().zip(train_targets.iter()) {
            let (loss, acc) = model.train_step(input, target, learning_rate);
            total_loss += loss;
            total_acc += acc;
        }

        let avg_train_loss = total_loss / train_inputs.len() as f32;
        let avg_train_acc = total_acc / train_inputs.len() as f32;

        // 验证
        let mut val_loss = 0.0;
        let mut val_acc = 0.0;

        for (input, target) in val_inputs.iter().zip(val_targets.iter()) {
            model.set_training(false);
            let encoder_output = model.encode(input);

            let target_embedded = model.embedding.forward(target);
            let target_encoded = model.pos_encoding.forward(&target_embedded);
            let decoder_input = target_encoded.slice(ndarray::s![..1, ..]).to_owned();

            let logits = model.decode_step(&decoder_input, &encoder_output);

            let loss = model.compute_loss(&logits, target[target.len() - 1]);
            let accuracy = if logits.row(0).iter().enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i)
                .unwrap() == target[target.len() - 1]
            {
                1.0
            } else {
                0.0
            };

            val_loss += loss;
            val_acc += accuracy;
        }

        let avg_val_loss = val_loss / val_inputs.len() as f32;
        let avg_val_acc = val_acc / val_inputs.len() as f32;

        let epoch_time = epoch_start.elapsed().as_secs_f32();

        println!(
            "Epoch {:2}/{} | Train Loss: {:.4} | Train Acc: {:.2}% | Val Loss: {:.4} | Val Acc: {:.2}% | {:.2}s",
            epoch, epochs, avg_train_loss, avg_train_acc * 100.0, avg_val_loss, avg_val_acc * 100.0, epoch_time
        );

        if avg_val_acc > best_val_acc {
            best_val_acc = avg_val_acc;
            patience_counter = 0;
            println!("  ✨ 新的最佳模型！验证准确率: {:.2}%", avg_val_acc * 100.0);
        } else {
            patience_counter += 1;
            println!("  ⏳ 验证准确率未提升 ({}/{})", patience_counter, patience);
        }

        if patience_counter >= patience {
            println!("\n⚠️  早停触发！验证准确率已 {} 个 epoch 未提升", patience);
            break;
        }

        println!();
    }

    println!("╔════════════════════════════════════════════════╗");
    println!("║     CLM 预训练完成！                           ║");
    println!("╚════════════════════════━━━━━━━━━━━━━━━━━━━━━━━╝");
    println!("最佳验证准确率: {:.2}%\n", best_val_acc * 100.0);

    // ============================================================================
    // 6. 测试文本生成
    // ============================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("6. 测试文本生成");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    model.set_training(false);

    let test_prompts = vec![
        vec![start_token_id, *vocab.token_to_id.get("the").unwrap()],
        vec![start_token_id, *vocab.token_to_id.get("birds").unwrap()],
        vec![start_token_id, *vocab.token_to_id.get("dogs").unwrap()],
    ];

    for (i, prompt) in test_prompts.iter().enumerate() {
        println!("测试 {}:", i + 1);

        let prompt_text: Vec<String> = prompt.iter()
            .filter_map(|&id| vocab.id_to_token.get(id).cloned())
            .collect();
        println!("  提示: {}", prompt_text.join(" "));

        // 贪婪解码生成
        let output = model.generate_greedy(prompt, 10, start_token_id);

        let output_text: Vec<String> = output.iter()
            .filter_map(|&id| {
                if id == end_token_id {
                    Some("<END>".to_string())
                } else if id == start_token_id {
                    None
                } else {
                    vocab.id_to_token.get(id).cloned()
                }
            })
            .collect();

        println!("  生成: {}\n", output_text.join(" "));
    }

    // ============================================================================
    // 7. 技术细节
    // ============================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("CLM 预训练技术细节");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    println!("1. 因果语言建模:");
    println!("   - 自回归生成: P(t_n | t_1, ..., t_{{n-1}})");
    println!("   - 使用因果掩码防止位置关注后续位置");
    println!("   - 每个位置预测下一个 token");
    println!();

    println!("2. 损失函数:");
    println!("   - 在所有位置计算损失");
    println!("   - 使用 Cross-Entropy Loss");
    println!("   - 累加所有位置的预测损失");
    println!();

    println!("3. 模型架构:");
    println!("   - 基于 Transformer Decoder");
    println!("   - 单向（因果）注意力机制");
    println!("   - 输出投影到词汇表");
    println!();

    println!("4. 预训练优势:");
    println!("   - 学习生成能力");
    println!("   - 适合文本生成任务");
    println!("   - 为对话、摘要等任务提供基础");
    println!();

    println!("5. MLM vs CLM:");
    println!("   - MLM (BERT): 双向理解，适合分类");
    println!("   - CLM (GPT): 单向生成，适合创作");
    println!();

    println!("╔════════════════════════════════════════════════╗");
    println!("║     CLM 预训练示例完成！                       ║");
    println!("╚════════════════════════════════════════════════╝");
    println!();
    println!("💡 提示:");
    println!("   - 当前使用小规模数据演示");
    println!("   - 实际预训练需要数百B到数T tokens");
    println!("   - GPT-3 使用 300B tokens, GPT-4 使用更多");
}
