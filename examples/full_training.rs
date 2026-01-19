//! 完整的训练示例
//!
//! 展示如何使用 TrainableTransformer 进行实际训练

use mini_transformer::{TrainableTransformer, TensorExt};
use ndarray::Array2;
use std::time::Instant;

fn main() {
    println!("╔════════════════════════════════════════════════╗");
    println!("║     Transformer 完整训练示例                   ║");
    println!("╚════════════════════════════════════════════════╝\n");

    // 创建模型
    println!("创建模型...");
    let mut model = TrainableTransformer::new(
        100,  // vocab_size
        64,   // d_model
        4,    // n_heads
        2,    // n_layers
        128,  // d_ff
        32,   // max_seq_len
        5,    // n_classes
    );

    println!("  模型参数总数: {}\n", model.param_count());

    // 创建训练数据
    println!("准备训练数据...");
    let train_inputs = generate_data(100, 5);  // 100 个样本，序列长度 5
    let train_targets: Vec<usize> = vec![0; 50].into_iter().chain(vec![1; 50]).collect();

    let val_inputs = generate_data(20, 5);
    let val_targets: Vec<usize> = (0..10).chain(std::iter::repeat(1).take(10)).map(|x| x % 5).collect();

    println!("  训练集: {} 样本", train_inputs.len());
    println!("  验证集: {} 样本\n", val_inputs.len());

    // 训练配置
    let epochs = 10;
    let learning_rate = 0.01;

    println!("╔════════════════════════════════════════════════╗");
    println!("║     开始训练                                  ║");
    println!("╚════════════════════════════════════════════════╝\n");

    let start_time = Instant::now();

    for epoch in 1..=epochs {
        let epoch_start = Instant::now();

        // 训练一个 epoch
        let mut total_loss = 0.0;
        let mut total_acc = 0.0;

        for (input, &target) in train_inputs.iter().zip(train_targets.iter()) {
            let input_batch = Array2::from_shape_vec((1, input.len()), input.clone()).unwrap();
            let target_onehot = Array2::from_shape_vec((1, 5), one_hot(target, 5)).unwrap();

            let (loss, acc) = model.train_step(&input_batch, &target_onehot, learning_rate);

            total_loss += loss;
            total_acc += acc;
        }

        let avg_loss = total_loss / train_inputs.len() as f32;
        let avg_acc = total_acc / train_inputs.len() as f32;

        // 评估
        let (val_loss, val_acc) = model.evaluate(&val_inputs, &val_targets);

        let epoch_time = epoch_start.elapsed().as_secs_f32();

        println!("Epoch {:2}/{} | Loss: {:.4} | Acc: {:.2}% | Val Loss: {:.4} | Val Acc: {:.2}% | {:.2}s",
            epoch,
            epochs,
            avg_loss,
            avg_acc * 100.0,
            val_loss,
            val_acc * 100.0,
            epoch_time
        );
    }

    let total_time = start_time.elapsed().as_secs_f32();

    println!("\n╔════════════════════════════════════════════════╗");
    println!("║     训练完成！                                ║");
    println!("╚════════════════════════════════════════════════╝");
    println!("总耗时: {:.2} 秒", total_time);

    // 测试推理
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("测试推理:");
    let test_input = vec![1, 2, 3, 4, 5];
    let input_batch = Array2::from_shape_vec((1, test_input.len()), test_input.clone()).unwrap();

    model.set_training(false);
    let logits = model.forward(&input_batch);
    let probs = logits.softmax(1);

    println!("输入: {:?}", test_input);
    println!("输出概率: {:?}", probs.row(0));

    let pred_class = probs.row(0)
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap();

    println!("预测类别: {}", pred_class);
}

/// 生成随机数据
fn generate_data(n_samples: usize, seq_len: usize) -> Vec<Vec<usize>> {
    use rand::Rng;
    let mut rng = rand::thread_rng();

    (0..n_samples)
        .map(|_| (0..seq_len).map(|_| rng.gen_range(0..100)).collect())
        .collect()
}

/// One-hot 编码
fn one_hot(class: usize, n_classes: usize) -> Vec<f32> {
    let mut v = vec![0.0; n_classes];
    v[class] = 1.0;
    v
}
