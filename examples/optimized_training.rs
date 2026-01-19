//! 优化训练示例
//!
//! 展示学习率调度和其他优化技术的效果

use mini_transformer::{
    TrainableTransformer, TextClassificationDataset,
    lr_scheduler::{StepLR, ConstantLR, LRScheduler},
    TensorExt,
};
use ndarray::Array2;
use std::time::Instant;

fn main() {
    println!("╔════════════════════════════════════════════════╗");
    println!("║   训练优化对比                               ║");
    println!("╚════════════════════════════════════════════════╝\n");

    // 创建数据集
    println!("📊 加载情感分析数据集...");
    let dataset = mini_transformer::create_sentiment_dataset();

    println!("  训练样本: {}", dataset.train_len());
    println!("  测试样本: {}\n", dataset.test_len());

    // 准备数据
    let max_seq_len = 10;
    let (train_inputs, train_targets) = dataset.encode_train(max_seq_len);
    let (test_inputs, test_targets) = dataset.encode_test(max_seq_len);

    // ============ 实验 1: 固定学习率 ============
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("实验 1: 固定学习率 (0.01)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let mut model_constant = TrainableTransformer::new(
        dataset.vocab.vocab_size(),
        128,
        4,
        2,
        256,
        max_seq_len,
        dataset.n_classes,
    );

    let constant_scheduler = ConstantLR::new(0.01);
    let start_constant = Instant::now();
    let (epoch_constant, best_acc_constant) = model_constant.train_with_scheduler(
        &train_inputs,
        &train_targets,
        &test_inputs,
        &test_targets,
        30,
        8,
        0.01,
        5,
        &constant_scheduler,
    );
    let time_constant = start_constant.elapsed();

    let test_inputs_vec1: Vec<Vec<usize>> = test_inputs
        .rows()
        .into_iter()
        .map(|row| row.to_vec())
        .collect();

    let (final_test_loss_constant, final_test_acc_constant) =
        model_constant.evaluate(&test_inputs_vec1, &test_targets);

    println!("\n固定学习率结果:");
    println!("  训练时间: {:.2}s", time_constant.as_secs_f32());
    println!("  最佳 epoch: {}", epoch_constant);
    println!("  最佳验证准确率: {:.2}%", best_acc_constant * 100.0);
    println!("  最终测试准确率: {:.2}%", final_test_acc_constant * 100.0);

    // ============ 实验 2: 步进衰减 ============
    println!("\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("实验 2: 步进衰减 (每 10 epoch 衰减 0.5)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let mut model_step = TrainableTransformer::new(
        dataset.vocab.vocab_size(),
        128,
        4,
        2,
        256,
        max_seq_len,
        dataset.n_classes,
    );

    let step_scheduler = StepLR::new(0.01, 10, 0.5);
    let start_step = Instant::now();
    let (epoch_step, best_acc_step) = model_step.train_with_scheduler(
        &train_inputs,
        &train_targets,
        &test_inputs,
        &test_targets,
        30,
        8,
        0.01,
        5,
        &step_scheduler,
    );
    let time_step = start_step.elapsed();

    let test_inputs_vec2: Vec<Vec<usize>> = test_inputs
        .rows()
        .into_iter()
        .map(|row| row.to_vec())
        .collect();

    let (final_test_loss_step, final_test_acc_step) =
        model_step.evaluate(&test_inputs_vec2, &test_targets);

    println!("\n步进衰减结果:");
    println!("  训练时间: {:.2}s", time_step.as_secs_f32());
    println!("  最佳 epoch: {}", epoch_step);
    println!("  最佳验证准确率: {:.2}%", best_acc_step * 100.0);
    println!("  最终测试准确率: {:.2}%", final_test_acc_step * 100.0);

    // ============ 对比分析 ============
    println!("\n\n╔════════════════════════════════════════════════╗");
    println!("║   性能对比                                   ║");
    println!("╚════════════════════════════════════════════════╝\n");

    println!("训练时间:");
    println!("  固定学习率: {:.2}s", time_constant.as_secs_f32());
    println!("  步进衰减:   {:.2}s", time_step.as_secs_f32());

    let time_diff = if time_constant > time_step {
        (time_constant - time_step).as_secs_f32()
    } else {
        -(time_step - time_constant).as_secs_f32()
    };

    if time_diff > 0.0 {
        println!("  步进衰减快 {:.2}s\n", time_diff);
    } else if time_diff < 0.0 {
        println!("  步进衰减慢 {:.2}s\n", -time_diff);
    } else {
        println!("  时间相同\n");
    }

    println!("模型性能:");
    println!("  固定学习率测试准确率: {:.2}%", final_test_acc_constant * 100.0);
    println!("  步进衰减测试准确率:   {:.2}%", final_test_acc_step * 100.0);

    let acc_diff = (final_test_acc_step - final_test_acc_constant) * 100.0;
    if acc_diff > 0.0 {
        println!("  提升: +{:.2}%\n", acc_diff);
    } else if acc_diff < 0.0 {
        println!("  下降: {:.2}%\n", acc_diff);
    } else {
        println!("  持平\n");
    }

    // 学习率变化示例
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("学习率变化示例:");
    println!("Epoch | 固定学习率 | 步进衰减");
    println!("------|-----------|----------");
    for epoch in 0..30 {
        let const_lr = constant_scheduler.get_lr(epoch);
        let step_lr = step_scheduler.get_lr(epoch);
        if epoch % 5 == 0 || epoch < 5 {
            println!("{:5} | {:.6} | {:.6}", epoch + 1, const_lr, step_lr);
        }
    }

    // 推理示例
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("推理示例（步进衰减模型）:");

    let test_sentences = vec![
        "this movie is great",
        "this movie is terrible",
        "I love this film",
        "I hate this film",
    ];

    model_step.set_training(false);

    for sentence in test_sentences {
        let tokens = dataset.vocab.encode(sentence, max_seq_len);
        let input_batch = Array2::from_shape_vec((1, tokens.len()), tokens).unwrap();

        let logits = model_step.forward(&input_batch);
        let probs = logits.softmax(1);
        let prob_slice = probs.row(0);

        let pred_class = prob_slice
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap();

        let sentiment = if pred_class == 1 { "积极" } else { "消极" };
        let confidence = prob_slice[pred_class];

        println!("  文本: \"{}\"", sentence);
        println!("  预测: {} (置信度: {:.2}%)\n", sentiment, confidence * 100.0);
    }

    println!("╔════════════════════════════════════════════════╗");
    println!("║   实验完成！                                 ║");
    println!("╚════════════════════════════════════════════════╝");
}
