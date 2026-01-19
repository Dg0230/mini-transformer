//! 批处理训练示例
//!
//! 展示如何使用批处理训练提高性能

use mini_transformer::{
    TrainableTransformer, TextClassificationDataset, TensorExt,
};
use ndarray::Array2;
use std::time::Instant;

fn main() {
    println!("╔════════════════════════════════════════════════╗");
    println!("║   批处理训练 vs 单样本训练对比               ║");
    println!("╚════════════════════════════════════════════════╝\n");

    // 创建数据集
    println!("📊 加载情感分析数据集...");
    let dataset = mini_transformer::create_sentiment_dataset();

    println!("  词汇表大小: {}", dataset.vocab.vocab_size());
    println!("  训练样本: {}", dataset.train_len());
    println!("  测试样本: {}", dataset.test_len());
    println!("  类别数: {} (0=消极, 1=积极)\n", dataset.n_classes);

    // 准备数据
    let max_seq_len = 10;
    let (train_inputs, train_targets) = dataset.encode_train(max_seq_len);
    let (test_inputs, test_targets) = dataset.encode_test(max_seq_len);

    // ============ 实验 1: 单样本训练 ============
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("实验 1: 单样本训练");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let mut model_single = TrainableTransformer::new(
        dataset.vocab.vocab_size(),
        128,
        4,
        2,
        256,
        max_seq_len,
        dataset.n_classes,
    );

    let train_inputs_vec: Vec<Vec<usize>> = train_inputs
        .rows()
        .into_iter()
        .map(|row| row.to_vec())
        .collect();

    let test_inputs_vec: Vec<Vec<usize>> = test_inputs
        .rows()
        .into_iter()
        .map(|row| row.to_vec())
        .collect();

    let start_single = Instant::now();
    let (epoch_single, best_acc_single) = model_single.train(
        &train_inputs_vec,
        &train_targets,
        &test_inputs_vec,
        &test_targets,
        30,
        0.01,
        5,
    );
    let time_single = start_single.elapsed();

    let (final_test_loss_single, final_test_acc_single) =
        model_single.evaluate(&test_inputs_vec, &test_targets);

    println!("\n单样本训练结果:");
    println!("  训练时间: {:.2}s", time_single.as_secs_f32());
    println!("  最佳 epoch: {}", epoch_single);
    println!("  最佳验证准确率: {:.2}%", best_acc_single * 100.0);
    println!("  最终测试准确率: {:.2}%", final_test_acc_single * 100.0);

    // ============ 实验 2: 批处理训练 ============
    println!("\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("实验 2: 批处理训练 (batch_size=8)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let mut model_batch = TrainableTransformer::new(
        dataset.vocab.vocab_size(),
        128,
        4,
        2,
        256,
        max_seq_len,
        dataset.n_classes,
    );

    let test_inputs_vec2: Vec<Vec<usize>> = test_inputs
        .rows()
        .into_iter()
        .map(|row| row.to_vec())
        .collect();

    let start_batch = Instant::now();
    let (epoch_batch, best_acc_batch) = model_batch.train_with_batches(
        &train_inputs,
        &train_targets,
        &test_inputs,
        &test_targets,
        30,
        8, // batch_size
        0.01,
        5,
    );
    let time_batch = start_batch.elapsed();

    let (final_test_loss_batch, final_test_acc_batch) =
        model_batch.evaluate(&test_inputs_vec2, &test_targets);

    println!("\n批处理训练结果:");
    println!("  训练时间: {:.2}s", time_batch.as_secs_f32());
    println!("  最佳 epoch: {}", epoch_batch);
    println!("  最佳验证准确率: {:.2}%", best_acc_batch * 100.0);
    println!("  最终测试准确率: {:.2}%", final_test_acc_batch * 100.0);

    // ============ 对比分析 ============
    println!("\n\n╔════════════════════════════════════════════════╗");
    println!("║   性能对比                                   ║");
    println!("╚════════════════════════════════════════════════╝\n");

    println!("训练速度:");
    let speedup = time_single.as_secs_f32() / time_batch.as_secs_f32();
    println!("  单样本: {:.2}s", time_single.as_secs_f32());
    println!("  批处理: {:.2}s", time_batch.as_secs_f32());
    println!("  加速比: {:.2}x\n", speedup);

    println!("模型性能:");
    println!("  单样本测试准确率: {:.2}%", final_test_acc_single * 100.0);
    println!("  批处理测试准确率: {:.2}%", final_test_acc_batch * 100.0);

    let acc_diff = (final_test_acc_batch - final_test_acc_single) * 100.0;
    if acc_diff > 0.0 {
        println!("  提升: +{:.2}%\n", acc_diff);
    } else if acc_diff < 0.0 {
        println!("  下降: {:.2}%\n", acc_diff);
    } else {
        println!("  持平\n");
    }

    // 推理示例
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("推理示例（批处理模型）:");

    let test_sentences = vec![
        "this movie is great and wonderful",
        "this movie is terrible and boring",
        "I love this film it is amazing",
        "I hate this film it is awful",
    ];

    model_batch.set_training(false);

    for sentence in test_sentences {
        let tokens = dataset.vocab.encode(sentence, max_seq_len);
        let input_batch = Array2::from_shape_vec((1, tokens.len()), tokens).unwrap();

        let logits = model_batch.forward(&input_batch);
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
