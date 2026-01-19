//! 模型评估指标示例
//!
//! 展示如何使用各种评估指标来评估模型性能

use mini_transformer::{
    TrainableTransformer, SimpleDataset, Dataset,
    ClassificationMetrics, ConfusionMatrix, Perplexity,
};
use ndarray::Array2;

fn main() {
    println!("╔════════════════════════════════════════════════╗");
    println!("║     模型评估指标示例                         ║");
    println!("╚════════════════════════════════════════════════╝\n");

    // ============================================================================
    // 1. 创建模型和数据
    // ============================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("1. 创建模型和数据");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let vocab_size = 100;
    let d_model = 64;
    let n_heads = 4;
    let n_layers = 2;
    let d_ff = 128;
    let max_seq_len = 32;
    let n_classes = 5;

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

    // 准备数据
    let train_dataset = SimpleDataset::random(200, 16, vocab_size, n_classes);
    let test_dataset = SimpleDataset::random(50, 16, vocab_size, n_classes);

    let mut train_inputs = Vec::new();
    let mut train_targets = Vec::new();
    for i in 0..train_dataset.len() {
        let (input, target) = train_dataset.get(i);
        train_inputs.push(input);
        train_targets.push(target);
    }

    let mut test_inputs = Vec::new();
    let mut test_targets = Vec::new();
    for i in 0..test_dataset.len() {
        let (input, target) = test_dataset.get(i);
        test_inputs.push(input);
        test_targets.push(target);
    }

    println!("训练集大小: {}", train_inputs.len());
    println!("测试集大小: {}\n", test_inputs.len());

    // ============================================================================
    // 2. 训练模型
    // ============================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("2. 训练模型");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let lr = 0.01;
    let epochs = 5;

    for epoch in 1..=epochs {
        let mut total_loss = 0.0;

        for (input, &target) in train_inputs.iter().zip(train_targets.iter()) {
            let input_batch = Array2::from_shape_vec((1, input.len()), input.clone()).unwrap();
            let target_onehot = TrainableTransformer::one_hot(target, n_classes);

            let (loss, _) = model.train_step(&input_batch, &target_onehot, lr);
            total_loss += loss;
        }

        let avg_loss = total_loss / train_inputs.len() as f32;
        println!("Epoch {:}/{} | 平均 Loss: {:.4}", epoch, epochs, avg_loss);
    }

    println!("\n✓ 训练完成！\n");

    // ============================================================================
    // 3. 分类任务评估
    // ============================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("3. 分类任务评估");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // 获取预测结果
    let mut predictions = Vec::new();
    for input in &test_inputs {
        let input_batch = Array2::from_shape_vec((1, input.len()), input.clone()).unwrap();
        let logits = model.forward(&input_batch);

        // Argmax 获取预测类别
        let pred_class = logits.row(0)
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(j, _)| j)
            .unwrap();

        predictions.push(pred_class);
    }

    // 计算分类指标
    println!("计算分类指标...\n");

    let metrics = ClassificationMetrics::compute(&predictions, &test_targets, n_classes);

    println!("┌────────────────────────────────────────────┐");
    println!("│         分类性能指标                      │");
    println!("├────────────────────────────────────────────┤");
    println!("│  准确率 (Accuracy):  {:20.2}%      │", metrics.accuracy * 100.0);
    println!("│  精确率 (Precision): {:20.2}%      │", metrics.precision * 100.0);
    println!("│  召回率 (Recall):     {:20.2}%      │", metrics.recall * 100.0);
    println!("│  F1 分数 (F1 Score): {:20.4}      │", metrics.f1_score);
    println!("└────────────────────────────────────────────┘\n");

    // 混淆矩阵
    println!("混淆矩阵：\n");
    let cm = ConfusionMatrix::compute(&predictions, &test_targets, n_classes);
    println!("{}", cm.display());

    // 每个类别的准确率
    let per_class_acc = cm.per_class_accuracy();
    println!("各类别准确率:");
    for (i, &acc) in per_class_acc.iter().enumerate() {
        println!("  类别 {}: {:.2}%", i, acc * 100.0);
    }
    println!();

    // ============================================================================
    // 4. Perplexity 评估
    // ============================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("4. Perplexity 评估（语言模型指标）");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // 准备测试数据
    let test_data: Vec<(Vec<usize>, usize)> = test_inputs
        .iter()
        .zip(test_targets.iter())
        .map(|(input, target)| (input.clone(), *target))
        .collect();

    println!("计算测试集 Perplexity...\n");

    let perplexity = Perplexity::compute(&mut model, &test_data);
    println!("┌────────────────────────────────────────────┐");
    println!("│  Perplexity: {:28.2}      │", perplexity);
    println!("└────────────────────────────────────────────┘\n");

    println!("💡 Perplexity 解读:");
    println!("  • Perplexity 越低，表示模型预测越准确");
    println!("  • Perplexity = 1.0 表示完美预测");
    println!("  • Perplexity = 10.0 表示模型在 10 个等可能的词中犹豫");
    println!("  • 典型的语言模型 Perplexity 在 20-100 之间\n");

    // ============================================================================
    // 5. 指标对比和总结
    // ============================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("5. 评估总结");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    println!("模型性能评估完成！\n");

    println!("评估指标说明:");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    println!("1. 分类指标:");
    println!("   • Accuracy: 整体准确率，所有类别的预测正确比例");
    println!("   • Precision: 预测为正的样本中，真正为正的比例");
    println!("   • Recall: 真正为正的样本中，被正确预测的比例");
    println!("   • F1 Score: Precision 和 Recall 的调和平均\n");

    println!("2. 混淆矩阵:");
    println!("   • 展示每个类别的预测详情");
    println!("   • 对角线是正确预测的数量");
    println!("   • 可以发现模型的偏倚和混淆模式\n");

    println!("3. Perplexity:");
    println!("   • 语言模型的标准评估指标");
    println!("   • 基于交叉熵损失：exp(cross_entropy)");
    println!("   • 越低越好，表示模型不确定性越低\n");

    println!("使用建议:");
    println!("  • 分类任务: 主要关注 Accuracy 和 F1 Score");
    println!("  • 不平衡数据: 关注 Precision 和 Recall");
    println!("  • 语言模型: 关注 Perplexity");
    println!("  • 机器翻译: 使用 BLEU score（未在此示例中展示）\n");

    println!("╔════════════════════════════════════════════════╗");
    println!("║     评估指标示例完成！                     ║");
    println!("╚════════════════════════════════════════════════╝\n");
}
