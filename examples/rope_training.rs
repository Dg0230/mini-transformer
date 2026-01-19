//! RoPE 训练示例
//!
//! 展示如何使用 RoPE (Rotary Position Embedding) 进行训练，
//! 并对比使用 RoPE 和不使用 RoPE 的模型性能。

use mini_transformer::{TrainableTransformer, SimpleDataset, Dataset};
use ndarray::Array2;

fn main() {
    println!("╔════════════════════════════════════════════════╗");
    println!("║     RoPE 训练示例                               ║");
    println!("╚════════════════════════════════════════════════╝\n");

    // ============================================================================
    // 1. 训练配置
    // ============================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("1. 训练配置");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let vocab_size = 1000;
    let d_model = 128;
    let n_heads = 4;
    let n_layers = 2;
    let d_ff = 256;
    let max_seq_len = 64;
    let n_classes = 10;

    let train_size = 1000;
    let val_size = 200;
    let seq_len = 32;
    let batch_size = 32;
    let epochs = 10;
    let learning_rate = 0.001;

    println!("模型配置:");
    println!("  词汇表大小: {}", vocab_size);
    println!("  模型维度: {}", d_model);
    println!("  注意力头数: {}", n_heads);
    println!("  层数: {}", n_layers);
    println!("  前馈维度: {}", d_ff);
    println!("  最大序列长度: {}\n", max_seq_len);

    println!("训练配置:");
    println!("  训练集大小: {}", train_size);
    println!("  验证集大小: {}", val_size);
    println!("  序列长度: {}", seq_len);
    println!("  批大小: {}", batch_size);
    println!("  训练轮数: {}", epochs);
    println!("  学习率: {:.6}\n", learning_rate);

    // ============================================================================
    // 2. 创建数据集
    // ============================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("2. 创建数据集");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let train_dataset = SimpleDataset::random(train_size, seq_len, vocab_size, n_classes);
    let val_dataset = SimpleDataset::random(val_size, seq_len, vocab_size, n_classes);

    // 准备训练数据
    let mut train_inputs = Vec::new();
    let mut train_targets = Vec::new();
    for i in 0..train_dataset.len() {
        let (input, target) = train_dataset.get(i);
        train_inputs.push(input);
        train_targets.push(target);
    }

    // 准备验证数据
    let mut val_inputs = Vec::new();
    let mut val_targets = Vec::new();
    for i in 0..val_dataset.len() {
        let (input, target) = val_dataset.get(i);
        val_inputs.push(input);
        val_targets.push(target);
    }

    // 转换为 Array2
    let train_inputs_array = vec_to_array2(&train_inputs);
    let val_inputs_array = vec_to_array2(&val_inputs);

    println!("✓ 数据集创建完成！\n");

    // ============================================================================
    // 3. 训练不使用 RoPE 的模型
    // ============================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("3. 训练不使用 RoPE 的模型");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let mut model_no_rope = TrainableTransformer::new(
        vocab_size,
        d_model,
        n_heads,
        n_layers,
        d_ff,
        max_seq_len,
        n_classes,
    );

    println!("模型参数量: {:.1}M\n", model_no_rope.param_count() as f32 / 1e6);

    let (epoch_no_rope, acc_no_rope) = model_no_rope.train_with_batches(
        &train_inputs_array,
        &train_targets,
        &val_inputs_array,
        &val_targets,
        epochs,
        batch_size,
        learning_rate,
        3,  // patience
    );

    println!("\n✓ 不使用 RoPE 的模型训练完成！");
    println!("  最佳验证准确率: {:.2}% (epoch {})\n", acc_no_rope * 100.0, epoch_no_rope);

    // ============================================================================
    // 4. 训练使用 RoPE 的模型
    // ============================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("4. 训练使用 RoPE 的模型");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let mut model_with_rope = TrainableTransformer::new_with_rope(
        vocab_size,
        d_model,
        n_heads,
        n_layers,
        d_ff,
        max_seq_len,
        n_classes,
        true,  // use_rope
        None,  // 使用默认 RoPE 配置
    );

    println!("模型参数量: {:.1}M", model_with_rope.param_count() as f32 / 1e6);
    println!("RoPE 配置: 启用");
    println!("  - d_model: {}", d_model);
    println!("  - max_seq_len: {}\n", max_seq_len);

    let (epoch_with_rope, acc_with_rope) = model_with_rope.train_with_batches(
        &train_inputs_array,
        &train_targets,
        &val_inputs_array,
        &val_targets,
        epochs,
        batch_size,
        learning_rate,
        3,  // patience
    );

    println!("\n✓ 使用 RoPE 的模型训练完成！");
    println!("  最佳验证准确率: {:.2}% (epoch {})\n", acc_with_rope * 100.0, epoch_with_rope);

    // ============================================================================
    // 5. 性能对比
    // ============================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("5. 性能对比");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    println!("模型对比:");
    println!("┌─────────────┬──────────────┬──────────────────┐");
    println!("│ 模型类型    │ 最佳 Epoch   │ 最佳验证准确率   │");
    println!("├─────────────┼──────────────┼──────────────────┤");
    println!(
        "│ 不使用 RoPE  │ {:12} │ {:14.2}% │",
        epoch_no_rope,
        acc_no_rope * 100.0
    );
    println!(
        "│ 使用 RoPE    │ {:12} │ {:14.2}% │",
        epoch_with_rope,
        acc_with_rope * 100.0
    );
    println!("└─────────────┴──────────────┴──────────────────┘\n");

    let improvement = (acc_with_rope - acc_no_rope) * 100.0;
    if improvement > 0.0 {
        println!("✨ RoPE 提升了模型性能：+{:.2}%", improvement);
    } else if improvement < 0.0 {
        println!("⚠️  RoPE 降低了模型性能：{:.2}%", improvement);
    } else {
        println!("➡️  RoPE 对模型性能无明显影响");
    }

    println!("\n╔════════════════════════════════════════════════╗");
    println!("║     RoPE 训练示例完成！                       ║");
    println!("╚════════════════════════════════════════════════╝\n");

    // ============================================================================
    // 6. RoPE 说明
    // ============================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("6. RoPE 技术说明");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    println!("什么是 RoPE?");
    println!("  RoPE (Rotary Position Embedding) 是一种旋转位置编码方法，");
    println!("  通过在注意力计算中引入旋转矩阵来编码位置信息。\n");

    println!("RoPE 的优势:");
    println!("  1. 相对位置编码：能够更好地捕捉 token 之间的相对位置关系");
    println!("  2. 外推性：能够处理比训练时更长的序列");
    println!("  3. 不增加参数：不像绝对位置编码那样需要额外的参数\n");

    println!("使用场景:");
    println!("  - 长序列建模（如长文本生成、长文档理解）");
    println!("  - 需要处理可变长度序列的任务");
    println!("  - 现代大语言模型（如 LLaMA、PaLM 等）\n");
}

/// 将 Vec<Vec<usize>> 转换为 Array2<usize>
fn vec_to_array2(vec: &[Vec<usize>]) -> Array2<usize> {
    if vec.is_empty() {
        return Array2::zeros((0, 0));
    }

    let nrows = vec.len();
    let ncols = vec[0].len();
    let mut data = Vec::with_capacity(nrows * ncols);

    for row in vec.iter() {
        data.extend(row);
    }

    Array2::from_shape_vec((nrows, ncols), data).unwrap()
}
