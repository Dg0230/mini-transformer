//! Masked Language Modeling (MLM) 预训练示例
//!
//! 演示 BERT 风格的预训练：随机掩码 tokens 并预测它们

use mini_transformer::{
    MLMPretrainer, MLMConfig, apply_mlm_mask_batch, Vocabulary,
};
use ndarray::Array2;

fn main() {
    println!("╔════════════════════════════════════════════════╗");
    println!("║   MLM 预训练示例 (BERT 风格)                  ║");
    println!("╚════════════════════════════════════════════════╝\n");

    // ============================================================================
    // 1. 准备预训练数据
    // ============================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("1. 准备预训练数据");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // 模拟预训练文本（实际应该是大规模文本）
    let corpus = vec![
        "the cat sat on the mat",
        "dogs are loyal animals",
        "birds can fly high in the sky",
        "fish swim in the water",
        "trees grow tall and strong",
        "the sun is very bright today",
        "moon shines at night time",
        "stars twinkle in darkness",
        "children love to play games",
        "music brings joy to people",
        "reading books is very educational",
        "writing helps express thoughts",
        "learning new skills is important",
        "practice makes perfect results",
        "teamwork achieves great goals",
        "honesty is the best policy",
        "kindness matters to everyone",
        "courage overcomes many fears",
        "wisdom comes from experience",
        "patience leads to success",
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
    let mask_token_id = vocab.vocab_size;
    vocab.token_to_id.insert("<MASK>".to_string(), mask_token_id);
    vocab.id_to_token.push("<MASK>".to_string());
    vocab.vocab_size += 1;

    println!("词汇表大小: {}", vocab.vocab_size);
    println!("特殊 tokens: <UNK>=0, <PAD>=1, <MASK>={}\n", mask_token_id);

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

    // 填充到固定长度
    let max_seq_len = 10;
    for tokens in &mut encoded_corpus {
        while tokens.len() < max_seq_len {
            tokens.push(1); // PAD
        }
        if tokens.len() > max_seq_len {
            tokens.truncate(max_seq_len);
        }
    }

    // 创建训练集和验证集
    let n_train = (encoded_corpus.len() * 8 / 10) as usize;

    let train_data: Vec<_> = encoded_corpus[..n_train].to_vec();
    let val_data: Vec<_> = encoded_corpus[n_train..].to_vec();

    // 转换为 Array2
    let mut train_flat = Vec::new();
    for tokens in &train_data {
        train_flat.extend(tokens);
    }
    let train_inputs = Array2::from_shape_vec((train_data.len(), max_seq_len), train_flat).unwrap();

    let mut val_flat = Vec::new();
    for tokens in &val_data {
        val_flat.extend(tokens);
    }
    let val_inputs = Array2::from_shape_vec((val_data.len(), max_seq_len), val_flat).unwrap();

    println!("训练样本: {}", train_data.len());
    println!("验证样本: {}\n", val_data.len());

    // ============================================================================
    // 2. 演示 MLM 掩码策略
    // ============================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("2. MLM 掩码策略演示");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let mlm_config = MLMConfig {
        mask_ratio: 0.3, // 提高到 30% 用于演示
        mask_prob: 0.8,
        random_prob: 0.1,
        keep_prob: 0.1,
        mask_token_id,
        pad_token_id: 1,
    };

    // 对前几个样本应用掩码
    let demo_batch = train_inputs.slice(ndarray::s![0..3, ..]).to_owned();
    let (masked_demo, labels_demo) = apply_mlm_mask_batch(&demo_batch, &mlm_config, vocab.vocab_size);

    println!("BERT 风格掩码策略 (15% tokens):\n");
    println!("  80% → 替换为  标记");
    println!("  10% → 替换为随机 token");
    println!("  10% → 保持不变\n");

    for i in 0..3 {
        println!("样本 {}:", i + 1);
        println!("  原始:  {}", decode_tokens(&train_inputs.row(i).to_vec(), &vocab));
        println!("  掩码:  {}", decode_tokens(&masked_demo.row(i).to_vec(), &vocab));
        println!("  标签:  {}", decode_tokens(&labels_demo.row(i).to_vec(), &vocab));
        println!();
    }

    // ============================================================================
    // 3. 创建 MLM 预训练模型
    // ============================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("3. 创建 MLM 预训练模型");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let mut model = MLMPretrainer::new(
        vocab.vocab_size,  // vocab_size
        128,               // d_model
        4,                 // n_heads
        2,                 // n_layers
        256,               // d_ff
        max_seq_len,       // max_seq_len
    );

    println!("模型架构:");
    println!("  词汇表大小: {}", vocab.vocab_size);
    println!("  模型维度: {}", 128);
    println!("  注意力头数: {}", 4);
    println!("  Encoder 层数: {}", 2);
    println!("  FFN 维度: {}", 256);
    println!("  参数总数: {}\n", model.param_count());

    // ============================================================================
    // 4. 准备预训练数据（应用掩码）
    // ============================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("4. 应用 MLM 掩码到训练数据");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let (train_masked, train_labels) = apply_mlm_mask_batch(&train_inputs, &MLMConfig::default(), vocab.vocab_size);
    let (val_masked, val_labels) = apply_mlm_mask_batch(&val_inputs, &MLMConfig::default(), vocab.vocab_size);

    println!("训练数据: {} 样本", train_masked.nrows());
    println!("验证数据: {} 样本\n", val_masked.nrows());

    // ============================================================================
    // 5. 开始预训练
    // ============================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("5. 开始 MLM 预训练");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let _result = model.train(
        &train_masked,
        &train_labels,
        &val_masked,
        &val_labels,
        50,      // epochs
        4,       // batch_size
        0.001,   // learning_rate
        10,      // patience
    );

    // ============================================================================
    // 6. 测试模型
    // ============================================================================
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("6. 测试预训练模型");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    model.set_training(false);

    // 创建测试样本
    let test_sentences = vec![
        "the cat sat on the <MASK>",
        "dogs are <MASK> animals",
        "birds can <MASK> high in the sky",
    ];

    println!("填空测试:\n");

    for sentence in test_sentences {
        println!("  输入: {}", sentence);

        // 编码并填充
        let tokens: Vec<usize> = sentence.split_whitespace()
            .map(|token| {
                if token == "<MASK>" {
                    mask_token_id
                } else {
                    *vocab.token_to_id.get(token).unwrap_or(&0)
                }
            })
            .collect();

        let mut padded_tokens = tokens.clone();
        while padded_tokens.len() < max_seq_len {
            padded_tokens.push(1); // PAD
        }

        // 前向传播
        let input_array = Array2::from_shape_vec((1, max_seq_len), padded_tokens).unwrap();
        let logits = model.forward(&input_array);

        // 找到  位置的预测
        let mask_pos = tokens.iter().position(|&t| t == mask_token_id).unwrap();
        let mask_logits = logits.row(mask_pos);

        let mut top_candidates: Vec<(usize, f32)> = mask_logits.iter()
            .enumerate()
            .map(|(i, &logit)| (i, logit))
            .filter(|(i, _)| *i > 2) // 排除特殊 tokens
            .collect();
        top_candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let top_5: Vec<(usize, f32)> = top_candidates.into_iter().take(5).collect();

        println!("  Top-5 预测:");
        for (rank, (token_id, score)) in top_5.iter().enumerate() {
            let token = &vocab.id_to_token[*token_id];
            println!("    {}. '{}' (logit={:.3})", rank + 1, token, score);
        }
        println!();
    }

    // ============================================================================
    // 7. 技术细节
    // ============================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("MLM 预训练技术细节");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    println!("1. 掩码策略:");
    println!("   - 随机选择 15% 的 token 位置");
    println!("   - 80% 概率替换为  ");
    println!("   - 10% 概率替换为随机 token");
    println!("   - 10% 概率保持原 token");
    println!();

    println!("2. 损失函数:");
    println!("   - 只在被掩码的位置计算损失");
    println!("   - 使用 Cross-Entropy Loss");
    println!("   - 排除  tokens");
    println!();

    println!("3. 模型架构:");
    println!("   - 基于 Transformer Encoder");
    println!("   - 双向注意力机制");
    println!("   - MLM 头映射到词汇表");
    println!();

    println!("4. 预训练优势:");
    println!("   - 学习双向上下文表示");
    println!("   - 适合理解任务（分类、NER 等）");
    println!("   - 为下游任务提供良好的初始化");
    println!();

    println!("╔════════════════════════════════════════════════╗");
    println!("║     MLM 预训练示例完成！                       ║");
    println!("╚════════════════════════════════════════════════╝");
    println!();
    println!("💡 提示:");
    println!("   - 当前使用小规模数据演示");
    println!("   - 实际预训练需要数B到数十B tokens");
    println!("   - 预训练后可进行下游任务微调");
}

/// 解码 tokens 为文本
fn decode_tokens(tokens: &[usize], vocab: &Vocabulary) -> String {
    tokens.iter()
        .filter_map(|&id| {
            if id < vocab.id_to_token.len() {
                Some(vocab.id_to_token[id].clone())
            } else {
                None
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}
