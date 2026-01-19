//! 模型保存与加载示例
//!
//! 展示如何保存训练好的模型并重新加载使用

use mini_transformer::{TrainableTransformer, SimpleDataset, Dataset, ModelSaveLoad};
use ndarray::Array2;
use std::path::Path;

fn main() {
    println!("╔════════════════════════════════════════════════╗");
    println!("║     模型保存与加载示例                         ║");
    println!("╚════════════════════════════════════════════════╝\n");

    // ============================================================================
    // 1. 创建并训练模型
    // ============================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("1. 创建并训练模型");
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

    println!("✓ 模型创建成功！");
    println!("  参数量: {:.1}M\n", model.param_count() as f32 / 1e6);

    // 准备训练数据
    let train_dataset = SimpleDataset::random(100, 16, vocab_size, n_classes);
    let val_dataset = SimpleDataset::random(20, 16, vocab_size, n_classes);

    let mut train_inputs = Vec::new();
    let mut train_targets = Vec::new();
    for i in 0..train_dataset.len() {
        let (input, target) = train_dataset.get(i);
        train_inputs.push(input);
        train_targets.push(target);
    }

    let mut val_inputs = Vec::new();
    let mut val_targets = Vec::new();
    for i in 0..val_dataset.len() {
        let (input, target) = val_dataset.get(i);
        val_inputs.push(input);
        val_targets.push(target);
    }

    let train_inputs_array = vec_to_array2(&train_inputs);
    let val_inputs_array = vec_to_array2(&val_inputs);

    println!("开始训练...\n");

    // 简单训练几个 epoch
    let lr = 0.01;
    for epoch in 1..=3 {
        let mut total_loss = 0.0;
        let mut total_acc = 0.0;

        for (i, (input, &target)) in train_inputs.iter().zip(train_targets.iter()).enumerate() {
            let input_batch = Array2::from_shape_vec((1, input.len()), input.clone()).unwrap();
            let target_onehot = TrainableTransformer::one_hot(target, n_classes);

            let (loss, acc) = model.train_step(&input_batch, &target_onehot, lr);
            total_loss += loss;
            total_acc += acc;

            if i == 0 || (i + 1) % 20 == 0 {
                println!(
                    "  Epoch {:} | Sample {:3}/100 | Loss: {:.4} | Acc: {:.2}%",
                    epoch,
                    i + 1,
                    loss,
                    acc * 100.0
                );
            }
        }

        let avg_loss = total_loss / train_inputs.len() as f32;
        let avg_acc = total_acc / train_inputs.len() as f32;

        println!(
            "  ➤ Epoch {:} 完成 | 平均 Loss: {:.4} | 平均 Acc: {:.2}%\n",
            epoch,
            avg_loss,
            avg_acc * 100.0
        );
    }

    // ============================================================================
    // 2. 保存模型
    // ============================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("2. 保存模型");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let model_path = "trained_model.bin";
    let model_json_path = "trained_model.json";

    // 保存为二进制格式（体积小，速度快）
    println!("保存为二进制格式: {} ...", model_path);
    match model.save(model_path) {
        Ok(()) => {
            let metadata = std::fs::metadata(model_path).unwrap();
            println!("  ✓ 二进制格式保存成功！");
            println!("  文件大小: {:.2} KB\n", metadata.len() as f32 / 1024.0);
        }
        Err(e) => {
            println!("  ✗ 保存失败: {}\n", e);
            return;
        }
    }

    // 保存为 JSON 格式（可读性好）
    println!("保存为 JSON 格式: {} ...", model_json_path);
    match model.save_json(model_json_path) {
        Ok(()) => {
            let metadata = std::fs::metadata(model_json_path).unwrap();
            println!("  ✓ JSON 格式保存成功！");
            println!("  文件大小: {:.2} KB\n", metadata.len() as f32 / 1024.0);
        }
        Err(e) => {
            println!("  ✗ 保存失败: {}\n", e);
            return;
        }
    }

    // ============================================================================
    // 3. 加载模型
    // ============================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("3. 加载模型");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // 创建一个新模型来加载权重
    let mut loaded_model = TrainableTransformer::new(
        vocab_size,
        d_model,
        n_heads,
        n_layers,
        d_ff,
        max_seq_len,
        n_classes,
    );

    println!("从二进制文件加载模型...");
    match loaded_model.load(model_path) {
        Ok(()) => {
            println!("  ✓ 二进制格式加载成功！\n");
        }
        Err(e) => {
            println!("  ✗ 加载失败: {}\n", e);
            return;
        }
    }

    // ============================================================================
    // 4. 验证模型
    // ============================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("4. 验证加载的模型");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // 使用原始模型和加载的模型进行预测，比较结果
    let test_input = Array2::from_shape_vec((1, 16), (0..16).collect()).unwrap();

    let original_output = model.forward(&test_input);
    let loaded_output = loaded_model.forward(&test_input);

    // 计算输出的差异
    let diff = &original_output - &loaded_output;
    let max_diff = diff.iter().map(|&x| x.abs()).reduce(f32::max).unwrap_or(0.0);

    println!("对比原始模型和加载模型的输出：");
    println!("  原始模型输出前5个值: {:?}", &original_output.row(0).to_vec()[..5]);
    println!("  加载模型输出前5个值: {:?}", &loaded_output.row(0).to_vec()[..5]);
    println!("  最大差异: {:.10}\n", max_diff);

    if max_diff < 1e-6 {
        println!("  ✨ 模型加载验证成功！输出完全一致。\n");
    } else {
        println!("  ⚠️  警告：输出存在差异，可能加载有问题。\n");
    }

    // ============================================================================
    // 5. 完整检查点保存/加载
    // ============================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("5. 完整检查点（包含训练状态）");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    use mini_transformer::{
        FullCheckpoint, TrainingState, OptimizerState, ModelState,
    };

    // 保存完整检查点
    println!("保存完整检查点...");
    let model_state = model.save_model().unwrap();
    let training_state = TrainingState {
        epoch: 3,
        train_loss: 2.5,
        val_loss: 2.6,
        train_acc: 0.25,
        val_acc: 0.24,
        global_step: 300,
    };

    let checkpoint = FullCheckpoint {
        model: model_state,
        optimizer: None,  // 暂不保存优化器状态
        training: training_state,
    };

    let checkpoint_path = "full_checkpoint.bin";
    match checkpoint.save(checkpoint_path) {
        Ok(()) => {
            println!("  ✓ 完整检查点保存成功！\n");
        }
        Err(e) => {
            println!("  ✗ 保存失败: {}\n", e);
            return;
        }
    }

    // 加载完整检查点
    println!("加载完整检查点...");
    match FullCheckpoint::load(checkpoint_path) {
        Ok(loaded_checkpoint) => {
            println!("  ✓ 完整检查点加载成功！");
            println!("  训练状态:");
            println!("    Epoch: {}", loaded_checkpoint.training.epoch);
            println!("    Train Loss: {:.4}", loaded_checkpoint.training.train_loss);
            println!("    Train Acc: {:.2}%", loaded_checkpoint.training.train_acc * 100.0);
            println!("    Global Step: {}\n", loaded_checkpoint.training.global_step);
        }
        Err(e) => {
            println!("  ✗ 加载失败: {}\n", e);
            return;
        }
    }

    // ============================================================================
    // 6. 清理和总结
    // ============================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("6. 总结");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    println!("模型保存与加载功能演示完成！");
    println!("\n支持的格式:");
    println!("  • 二进制格式 (.bin) - 体积小，速度快，推荐用于生产环境");
    println!("  • JSON 格式 (.json) - 可读性好，推荐用于调试和可视化");
    println!("\n使用场景:");
    println!("  • 保存训练好的模型用于部署");
    println!("  • 从检查点恢复训练");
    println!("  • 模型版本管理和回滚");
    println!("  • 迁移学习和微调");

    // 清理临时文件
    println!("\n清理临时文件...");
    let _ = std::fs::remove_file(model_path);
    let _ = std::fs::remove_file(model_json_path);
    let _ = std::fs::remove_file(checkpoint_path);
    println!("  ✓ 清理完成\n");

    println!("╔════════════════════════════════════════════════╗");
    println!("║     示例运行完成！                           ║");
    println!("╚════════════════════════════════════════════════╝\n");
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
