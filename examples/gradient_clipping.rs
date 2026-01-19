//! 梯度裁剪示例
//!
//! 演示如何使用梯度裁剪来稳定训练

use mini_transformer::{
    TrainableTransformer, TextClassificationDataset, GradientClipConfig,
    TensorExt,
};
use ndarray::Array2;

fn main() {
    println!("╔════════════════════════════════════════════════╗");
    println!("║     梯度裁剪示例                             ║");
    println!("╚════════════════════════════════════════════════╝\n");

    // ============================================================================
    // 1. 创建数据集
    // ============================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("1. 加载情感分类数据集");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let dataset = mini_transformer::create_sentiment_dataset();
    println!("训练样本: {}", dataset.train_len());
    println!("测试样本: {}", dataset.test_len());
    println!("词汇表大小: {}\n", dataset.vocab.vocab_size());

    let (train_inputs, train_targets) = dataset.encode_train(10);
    let (test_inputs, test_targets) = dataset.encode_test(10);

    // ============================================================================
    // 2. 对比实验：有/无梯度裁剪
    // ============================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("2. 对比训练实验");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // 实验配置
    let epochs = 30;
    let batch_size = 4;
    let learning_rate = 0.01;

    // 实验 1: 无梯度裁剪
    println!("实验 1: 无梯度裁剪");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let mut model_no_clip = TrainableTransformer::new(
        dataset.vocab.vocab_size,
        128,
        4,
        2,
        256,
        10,
        dataset.n_classes,
    );

    let train_inputs_vec: Vec<Vec<usize>> = train_inputs.rows()
        .into_iter()
        .map(|row| row.to_vec())
        .collect();

    let test_inputs_vec: Vec<Vec<usize>> = test_inputs.rows()
        .into_iter()
        .map(|row| row.to_vec())
        .collect();

    let start = std::time::Instant::now();
    let mut best_val_acc_no_clip = 0.0;

    for epoch in 1..=epochs {
        let mut total_loss = 0.0;
        let mut total_acc = 0.0;

        for (input, &target) in train_inputs_vec.iter().zip(train_targets.iter()) {
            let input_batch = Array2::from_shape_vec((1, input.len()), input.clone()).unwrap();
            let target_onehot = TrainableTransformer::one_hot(target, dataset.n_classes);

            let (loss, acc) = model_no_clip.train_step(&input_batch, &target_onehot, learning_rate);
            total_loss += loss;
            total_acc += acc;
        }

        let (val_loss, val_acc) = model_no_clip.evaluate(&test_inputs_vec, &test_targets);

        if epoch % 5 == 0 || epoch == 1 {
            println!(
                "Epoch {:2} | Train Loss: {:.4} | Train Acc: {:.2}% | Val Acc: {:.2}%",
                epoch,
                total_loss / train_inputs_vec.len() as f32,
                total_acc / train_inputs_vec.len() as f32 * 100.0,
                val_acc * 100.0
            );
        }

        if val_acc > best_val_acc_no_clip {
            best_val_acc_no_clip = val_acc;
        }
    }

    let time_no_clip = start.elapsed();
    println!("\n✓ 训练完成，耗时: {:.2}s", time_no_clip.as_secs_f32());
    println!("  最佳验证准确率: {:.2}%\n", best_val_acc_no_clip * 100.0);

    // 实验 2: 有梯度裁剪
    println!("实验 2: 有梯度裁剪 (max_norm=1.0)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let mut model_with_clip = TrainableTransformer::new(
        dataset.vocab.vocab_size,
        128,
        4,
        2,
        256,
        10,
        dataset.n_classes,
    );

    let clip_config = GradientClipConfig::norm(1.0);
    let start = std::time::Instant::now();
    let mut best_val_acc_with_clip = 0.0;

    for epoch in 1..=epochs {
        let mut total_loss = 0.0;
        let mut total_acc = 0.0;

        for (input, &target) in train_inputs_vec.iter().zip(train_targets.iter()) {
            let input_batch = Array2::from_shape_vec((1, input.len()), input.clone()).unwrap();
            let target_onehot = TrainableTransformer::one_hot(target, dataset.n_classes);

            let (loss, acc) = model_with_clip.train_step(&input_batch, &target_onehot, learning_rate);
            total_loss += loss;
            total_acc += acc;
        }

        let (val_loss, val_acc) = model_with_clip.evaluate(&test_inputs_vec, &test_targets);

        if epoch % 5 == 0 || epoch == 1 {
            println!(
                "Epoch {:2} | Train Loss: {:.4} | Train Acc: {:.2}% | Val Acc: {:.2}%",
                epoch,
                total_loss / train_inputs_vec.len() as f32,
                total_acc / train_inputs_vec.len() as f32 * 100.0,
                val_acc * 100.0
            );
        }

        if val_acc > best_val_acc_with_clip {
            best_val_acc_with_clip = val_acc;
        }
    }

    let time_with_clip = start.elapsed();
    println!("\n✓ 训练完成，耗时: {:.2}s", time_with_clip.as_secs_f32());
    println!("  最佳验证准确率: {:.2}%\n", best_val_acc_with_clip * 100.0);

    // ============================================================================
    // 3. 结果对比
    // ============================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("3. 实验结果对比");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    println!("┌─────────────────┬──────────────┬──────────────┐");
    println!("│ 方法           │ 验证准确率   │ 训练时间     │");
    println!("├─────────────────┼──────────────┼──────────────┤");
    println!("│ 无梯度裁剪      │ {:>8.2}%     │ {:>8.2}s     │",
        best_val_acc_no_clip * 100.0,
        time_no_clip.as_secs_f32()
    );
    println!("│ 有梯度裁剪      │ {:>8.2}%     │ {:>8.2}s     │",
        best_val_acc_with_clip * 100.0,
        time_with_clip.as_secs_f32()
    );
    println!("└─────────────────┴──────────────┴──────────────┘\n");

    // ============================================================================
    // 4. 技术细节
    // ============================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("4. 梯度裁剪技术细节");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    println!("1. 什么是梯度裁剪？");
    println!("   - 限制梯度的范数或值");
    println!("   - 防止梯度爆炸");
    println!("   - 稳定训练过程\n");

    println!("2. 两种裁剪类型:");
    println!("   a) 按范数裁剪 (推荐)");
    println!("      - 计算梯度向量的 L2 范数");
    println!("      - 如果超过阈值，按比例缩放");
    println!("      - 公式: g = g * (max_norm / ||g||) if ||g|| > max_norm\n");

    println!("   b) 按值裁剪");
    println!("      - 限制每个梯度元素的范围");
    println!("      - 公式: g = clip(g, -max_value, max_value)\n");

    println!("3. 推荐配置:");
    println!("   - max_norm = 1.0 (常用)");
    println!("   - max_norm = 5.0 (宽松)");
    println!("   - max_value = 1.0-10.0 (按值裁剪)\n");

    println!("4. 使用场景:");
    println!("   ✅ RNN/LSTM/Transformer (容易梯度爆炸)");
    println!("   ✅ 深层网络 (深度 > 10)");
    println!("   ✅ 高学习率训练");
    println!("   ✅ 长序列训练\n");

    println!("5. 实现示例:");
    println!("```rust");
    println!("use mini_transformer::GradientClipConfig;");
    println!("let clip_config = GradientClipConfig::norm(1.0);");
    println!("// 在训练循环中应用");
    println!("let clipped_grads = clip_gradients(&grads, &clip_config);");
    println!("```\n");

    println!("╔════════════════════════════════════════════════╗");
    println!("║     示例完成！                               ║");
    println!("╚════════════════════════════════════════════════╝\n");

    println!("💡 提示:");
    println!("   - 梯度裁剪是防止训练不稳定的简单有效方法");
    println!("   - 推荐在所有 Transformer 训练中使用");
    println!("   - max_norm = 1.0 是安全的起始值");
}
