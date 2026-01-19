//! 完整训练示例
//!
//! 展示如何使用早停和 Warmup Cosine Annealing 学习率调度进行模型训练

use mini_transformer::{
    TrainableTransformer, TransformerConfig, Adam, CrossEntropyLoss,
    WarmupCosineAnnealing, EarlyStopping, EarlyStoppingConfig, EarlyStoppingMode,
    SimpleDataset, DataLoader,
    LossFunction, Optimizer, LRScheduler,
};
use ndarray::Array2;
use rand::Rng;
use std::time::Instant;

fn main() {
    println!("╔════════════════════════════════════════════════╗");
    println!("║     完整训练示例 (早停 + 学习率调度)         ║");
    println!("╚════════════════════════════════════════════════╝\n");

    // ============================================================================
    // 1. 训练配置
    // ============================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("1. 训练配置");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // 模型配置
    let model_config = TransformerConfig {
        vocab_size: 1000,
        d_model: 256,
        n_heads: 8,
        n_layers: 4,
        d_ff: 1024,
        max_seq_len: 128,
        dropout: 0.1,
    };

    // 训练超参数
    let batch_size = 32;
    let epochs = 1;
    let learning_rate = 0.0001;  // 1e-4
    let total_steps = 5000;      // 估计的总训练步数
    let warmup_steps = 500;      // warmup 步数
    let n_classes = 10;          // 分类数

    println!("模型配置:");
    println!("  词汇表大小: {}", model_config.vocab_size);
    println!("  模型维度: {}", model_config.d_model);
    println!("  注意力头数: {}", model_config.n_heads);
    println!("  层数: {}", model_config.n_layers);
    println!("  前馈维度: {}\n", model_config.d_ff);

    println!("训练超参数:");
    println!("  批大小: {}", batch_size);
    println!("  最大轮数: {}", epochs);
    println!("  学习率: {:.6}", learning_rate);
    println!("  总训练步数: {}", total_steps);
    println!("  Warmup 步数: {}\n", warmup_steps);

    // ============================================================================
    // 2. 创建模型
    // ============================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("2. 创建模型");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let mut model = TrainableTransformer::new(
        model_config.vocab_size,
        model_config.d_model,
        model_config.n_heads,
        model_config.n_layers,
        model_config.d_ff,
        model_config.max_seq_len,
        n_classes,
    );
    println!("✓ 模型创建成功！");
    println!("  参数量: {:.1}M\n", model.param_count() as f32 / 1e6);

    // ============================================================================
    // 3. 创建数据集
    // ============================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("3. 创建数据集");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let train_size = 5000;
    let val_size = 1000;
    let seq_len = 64;
    let vocab_size = model_config.vocab_size;

    println!("创建训练集: {} 样本", train_size);
    let train_dataset = SimpleDataset::random(train_size, seq_len, vocab_size, n_classes);

    println!("创建验证集: {} 样本\n", val_size);
    let val_dataset = SimpleDataset::random(val_size, seq_len, vocab_size, n_classes);

    let train_loader = DataLoader::new(train_dataset, batch_size, true);
    let val_loader = DataLoader::new(val_dataset, batch_size, false);

    // ============================================================================
    // 4. 创建优化器和学习率调度器
    // ============================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("4. 优化器和学习率调度");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let mut optimizer = Adam::new(learning_rate);

    // Warmup Cosine Annealing 学习率调度
    let lr_scheduler = WarmupCosineAnnealing::new(
        learning_rate,
        0.0,            // min_lr
        total_steps,
        warmup_steps,
    );

    println!("学习率调度: Warmup Cosine Annealing");
    println!("  初始学习率: {:.6}", lr_scheduler.get_lr(0));
    println!("  Warmup 后学习率: {:.6}", lr_scheduler.get_lr(warmup_steps));
    println!("  训练中期学习率: {:.6}", lr_scheduler.get_lr(total_steps / 2));
    println!("  最终学习率: {:.6}\n", lr_scheduler.get_lr(total_steps));

    // ============================================================================
    // 5. 创建早停器
    // ============================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("5. 早停配置");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let early_stopping_config = EarlyStoppingConfig::min(5)  // patience = 5 epochs
        .with_min_delta(0.001)
        .with_restore_best_weights(true);

    let mut early_stopping = EarlyStopping::new(early_stopping_config);

    println!("早停配置:");
    println!("  耐心值: 5 epochs");
    println!("  最小改善阈值: 0.001");
    println!("  恢复最佳权重: true\n");

    // ============================================================================
    // 6. 训练循环
    // ============================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("6. 开始训练");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let mut global_step = 0;
    let mut best_val_loss = f32::INFINITY;
    let loss_fn = CrossEntropyLoss::new();

    for epoch in 0..epochs {
        let epoch_start = Instant::now();

        // ────────────────────────────────────────────────────────────
        // 训练阶段
        // ────────────────────────────────────────────────────────────
        let mut train_loss = 0.0_f32;
        let mut train_samples = 0.0_f32;

        for batch in train_loader.iter() {
            // 更新学习率
            let current_lr = lr_scheduler.get_lr(global_step);
            update_optimizer_lr(&mut optimizer, current_lr);

            // 训练一个 batch
            let batch_loss = train_batch(&mut model, &batch, &mut optimizer, &loss_fn, current_lr, n_classes);

            train_loss += batch_loss * batch.len() as f32;
            train_samples += batch.len() as f32;

            global_step += 1;

            // 提前停止（达到总步数）
            if global_step >= total_steps {
                break;
            }
        }

        let avg_train_loss = train_loss / train_samples;

        // ────────────────────────────────────────────────────────────
        // 验证阶段
        // ────────────────────────────────────────────────────────────
        let mut val_loss = 0.0_f32;
        let mut val_samples = 0.0_f32;
        let mut correct = 0;

        for batch in val_loader.iter() {
            let (batch_loss, batch_correct) = validate_batch(&mut model, &batch, &loss_fn);

            val_loss += batch_loss * batch.len() as f32;
            val_samples += batch.len() as f32;
            correct += batch_correct;
        }

        let avg_val_loss = val_loss / val_samples;
        let val_accuracy = correct as f32 / val_samples as f32;

        let epoch_duration = epoch_start.elapsed();

        // ────────────────────────────────────────────────────────────
        // 打印训练信息
        // ────────────────────────────────────────────────────────────
        println!(
            "Epoch {:3}/{:3} | Train Loss: {:.4} | Val Loss: {:.4} | Val Acc: {:.2}% | LR: {:.2e} | Time: {:.1}s",
            epoch + 1,
            epochs,
            avg_train_loss,
            avg_val_loss,
            val_accuracy * 100.0,
            lr_scheduler.get_lr(global_step.min(total_steps)),
            epoch_duration.as_secs_f32()
        );

        // ────────────────────────────────────────────────────────────
        // 保存最佳模型
        // ────────────────────────────────────────────────────────────
        if avg_val_loss < best_val_loss {
            best_val_loss = avg_val_loss;
            // 在实际应用中，这里应该保存模型检查点
            // checkpoint_manager.save(&model, epoch);
        }

        // ────────────────────────────────────────────────────────────
        // 早停检查
        // ────────────────────────────────────────────────────────────
        if early_stopping.update(avg_val_loss, epoch) {
            println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            println!("⚠ 早停触发！");
            println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            early_stopping.print_summary();
            break;
        }

        // ────────────────────────────────────────────────────────────
        // 提前停止（达到总步数）
        // ────────────────────────────────────────────────────────────
        if global_step >= total_steps {
            println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            println!("✓ 达到总训练步数 {}，训练完成", total_steps);
            println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
            break;
        }
    }

    // ============================================================================
    // 7. 训练总结
    // ============================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("7. 训练总结");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    println!("训练统计:");
    println!("  总训练步数: {}", global_step);
    println!("  最佳验证损失: {:.6}", best_val_loss);
    println!("  最佳 epoch: {}", early_stopping.best_epoch());

    if early_stopping.should_stop() {
        println!("  早停触发: Yes");
        println!("  停止 epoch: {}", early_stopping.stopped_epoch().unwrap());
    } else {
        println!("  早停触发: No");
        println!("  完成所有 {} epochs", epochs);
    }

    println!("\n╔════════════════════════════════════════════════╗");
    println!("║     训练完成！                               ║");
    println!("╚════════════════════════════════════════════════╝\n");

    // ============================================================================
    // 8. 学习率调度可视化（文本）
    // ============================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("8. 学习率调度曲线");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    visualize_lr_schedule(&lr_scheduler, total_steps);
}

/// 训练一个 batch
fn train_batch(
    model: &mut TrainableTransformer,
    batch: &[(Vec<usize>, usize)],
    optimizer: &mut Adam,
    loss_fn: &CrossEntropyLoss,
    lr: f32,
    n_classes: usize,
) -> f32 {
    let mut total_loss = 0.0;

    for (input, target) in batch {
        // 将 Vec 转换为 Array2
        let input_array = vec_to_array2(input);

        // 将 target 转换为 one-hot 编码
        let target_array = vec_to_onehot(*target, n_classes);

        // 训练一步
        let (loss, _) = model.train_step(&input_array, &target_array, lr);
        total_loss += loss;
    }

    total_loss / batch.len() as f32
}

/// 验证一个 batch
fn validate_batch(
    model: &mut TrainableTransformer,
    batch: &[(Vec<usize>, usize)],
    loss_fn: &CrossEntropyLoss,
) -> (f32, usize) {
    let mut total_loss = 0.0;
    let mut correct = 0;

    for (input, target) in batch {
        // 将 Vec 转换为 Array2
        let input_array = vec_to_array2(input);

        // 前向传播
        let logits = model.forward(&input_array);

        // 将 target 转换为 one-hot 编码
        let n_classes = logits.ncols();
        let target_array = vec_to_onehot(*target, n_classes);

        // 计算损失
        let loss = loss_fn.compute(&logits, &target_array);
        total_loss += loss;

        // 计算准确率
        let predicted = argmax(&logits.row(0).to_vec());
        if predicted == *target {
            correct += 1;
        }
    }

    (
        total_loss / batch.len() as f32,
        correct,
    )
}

/// 将 target 转换为 one-hot 编码
fn vec_to_onehot(target: usize, n_classes: usize) -> Array2<f32> {
    let mut onehot = vec![0.0f32; n_classes];
    onehot[target] = 1.0;
    Array2::from_shape_vec((1, n_classes), onehot).unwrap()
}

/// 将 Vec<usize> 转换为 Array2<usize>
fn vec_to_array2(vec: &[usize]) -> Array2<usize> {
    Array2::from_shape_vec((1, vec.len()), vec.to_vec()).unwrap()
}

/// Argmax
fn argmax(values: &[f32]) -> usize {
    let mut max_idx = 0;
    let mut max_val = values[0];

    for (i, &val) in values.iter().enumerate() {
        if val > max_val {
            max_val = val;
            max_idx = i;
        }
    }

    max_idx
}

/// 更新优化器学习率（通过创建新的优化器）
fn update_optimizer_lr(optimizer: &mut Adam, new_lr: f32) {
    // 由于 Adam 的学习率是私有的，我们创建一个新的优化器
    // 在实际应用中，应该暴露 set_lr 方法
    let _ = new_lr; // 占位符，实际使用中需要优化器支持动态调整学习率
}

/// 可视化学习率调度（文本形式）
fn visualize_lr_schedule(scheduler: &WarmupCosineAnnealing, total_steps: usize) {
    let num_points = 20;
    let step_size = total_steps / num_points;

    println!("步数    | 学习率      | 阶段");
    println!("─────────┼─────────────┼──────────");

    for i in 0..=num_points {
        let step = (i * step_size).min(total_steps);
        let lr = scheduler.get_lr(step);
        let phase = match scheduler.get_phase(step) {
            mini_transformer::WarmupPhase::Warmup => "Warmup  ",
            mini_transformer::WarmupPhase::Annealing => "Annealing",
            mini_transformer::WarmupPhase::Finished => "Finished",
        };

        println!("{:7} | {:.4e} | {}", step, lr, phase);
    }

    println!();

    println!("💡 实践建议:");
    println!("  - Warmup 阶段有助于稳定训练初期");
    println!("  - 余弦退火有助于找到更好的局部最优");
    println!("  - 早停可以防止过拟合，节省训练时间");
    println!("  - 根据验证损失调整超参数\n");
}
