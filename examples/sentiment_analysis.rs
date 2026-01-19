//! 情感分析训练示例
//!
//! 使用真实的文本数据进行情感分类训练

use mini_transformer::{
    TrainableTransformer, TextClassificationDataset, Vocabulary,
    CheckpointManager, TrainingHistory, TensorExt,
};
use ndarray::Array2;
use std::time::Instant;

fn main() {
    println!("╔════════════════════════════════════════════════╗");
    println!("║     Transformer 情感分析训练                 ║");
    println!("╚════════════════════════════════════════════════╝\n");

    // 创建数据集
    println!("📊 加载情感分析数据集...");
    let dataset = mini_transformer::create_sentiment_dataset();

    println!("  词汇表大小: {}", dataset.vocab.vocab_size());
    println!("  训练样本: {}", dataset.train_len());
    println!("  测试样本: {}", dataset.test_len());
    println!("  类别数: {} (0=消极, 1=积极)\n", dataset.n_classes);

    // 准备数据
    let max_seq_len = 10; // 最大序列长度
    let (train_inputs, train_targets) = dataset.encode_train(max_seq_len);
    let (test_inputs, test_targets) = dataset.encode_test(max_seq_len);

    // 将 Array2 转换为 Vec<Vec<usize>>
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

    // 创建模型
    println!("🏗️  构建模型...");
    let mut model = TrainableTransformer::new(
        dataset.vocab.vocab_size(),  // vocab_size
        128,                          // d_model
        4,                            // n_heads
        2,                            // n_layers
        256,                          // d_ff
        max_seq_len,                  // max_seq_len
        dataset.n_classes,            // n_classes
    );

    println!("  模型参数总数: {}\n", model.param_count());

    // 训练配置
    let epochs = 50;
    let learning_rate = 0.01;
    let patience = 10;

    // 训练前评估
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("训练前评估:");
    let (init_train_loss, init_train_acc) = model.evaluate(&train_inputs_vec, &train_targets);
    let (init_test_loss, init_test_acc) = model.evaluate(&test_inputs_vec, &test_targets);
    println!("  训练集: Loss={:.4}, Acc={:.2}%", init_train_loss, init_train_acc * 100.0);
    println!("  测试集: Loss={:.4}, Acc={:.2}%\n", init_test_loss, init_test_acc * 100.0);

    // 训练模型
    let start_time = Instant::now();
    let (best_epoch, best_val_acc) = model.train(
        &train_inputs_vec,
        &train_targets,
        &test_inputs_vec,
        &test_targets,
        epochs,
        learning_rate,
        patience,
    );
    let total_time = start_time.elapsed();

    println!("\n⏱️  总训练时间: {:.2} 秒", total_time.as_secs_f32());
    println!("   平均每个 epoch: {:.2} 秒", total_time.as_secs_f32() / best_epoch as f32);

    // 最终测试集评估
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("最终测试集评估:");
    let (final_test_loss, final_test_acc) = model.evaluate(&test_inputs_vec, &test_targets);
    println!("  测试集 Loss: {:.4}", final_test_loss);
    println!("  测试集准确率: {:.2}%", final_test_acc * 100.0);

    // 推理示例
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("推理示例:");

    let test_sentences = vec![
        "this movie is great and wonderful",
        "this movie is terrible and boring",
        "I love this film it is amazing",
        "I hate this film it is awful",
    ];

    model.set_training(false);

    for sentence in test_sentences {
        let tokens = dataset.vocab.encode(sentence, max_seq_len);
        let input_batch = Array2::from_shape_vec((1, tokens.len()), tokens).unwrap();

        let logits = model.forward(&input_batch);
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

    // 模型分析
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("模型分析:");
    println!("  训练样本数: {}", train_targets.len());
    println!("  测试样本数: {}", test_targets.len());
    println!("  模型参数量: {}", model.param_count());
    println!("  最佳验证准确率: {:.2}%", best_val_acc * 100.0);
    println!("  最佳 epoch: {}", best_epoch);

    // 与随机猜测对比
    println!("\n  与随机猜测对比:");
    println!("    随机猜测准确率: ~50.0% (二分类)");
    if final_test_acc > 0.5 {
        println!("    模型提升: +{:.2}%", (final_test_acc - 0.5) * 100.0);
        println!("    ✅ 模型学到了有用的模式！");
    } else {
        println!("    ⚠️  模型性能接近随机，需要更多训练或更大模型");
    }

    println!("\n╔════════════════════════════════════════════════╗");
    println!("║     训练完成！                                ║");
    println!("╚════════════════════════════════════════════════╝");
}
