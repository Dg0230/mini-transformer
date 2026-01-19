//! 训练示例
//!
//! 演示如何使用训练 API

use mini_transformer::{
    TransformerEncoder, TransformerConfig,
    CrossEntropyLoss, Adam, CosineAnnealingWarmRestarts,
    Trainer, TrainerConfig, Dataset, DataLoader, SimpleDataset,
    configs,
};

fn main() {
    println!("╔════════════════════════════════════════════════╗");
    println!("║     Transformer 训练示例                      ║");
    println!("╚════════════════════════════════════════════════╝\n");

    // 创建配置
    let config = configs::mini();
    println!("模型配置:");
    println!("  - vocab_size: {}", config.vocab_size);
    println!("  - d_model: {}", config.d_model);
    println!("  - n_layers: {}", config.n_layers);
    println!("  - n_heads: {}", config.n_heads);

    // 创建模型
    let model = TransformerEncoder::new(config.clone());
    println!("\n模型参数总数: {}", model.param_count());

    // 创建训练数据
    println!("\n创建训练数据...");
    let train_data = SimpleDataset::random(
        1000,  // 样本数
        10,    // 序列长度
        1000,  // 词表大小
        5,     // 类别数
    );

    let val_data = SimpleDataset::random(
        200,   // 样本数
        10,    // 序列长度
        1000,  // 词表大小
        5,     // 类别数
    );

    println!("  训练集大小: {}", train_data.len());
    println!("  验证集大小: {}", val_data.len());

    // 创建数据加载器
    let train_loader = DataLoader::new(train_data, 32, true);
    let val_loader = DataLoader::new(val_data, 32, false);

    // 创建损失函数和优化器
    let loss_fn = CrossEntropyLoss::new();
    let mut optimizer = Adam::new(0.001);

    // 创建学习率调度器
    let lr_scheduler = CosineAnnealingWarmRestarts::new(0.001, 0.0001, 50, 5);

    // 创建训练器
    let trainer_config = TrainerConfig {
        batch_size: 32,
        epochs: 5,
        learning_rate: 0.001,
        ..Default::default()
    };

    let trainer = Trainer::new(trainer_config);

    println!("\n╔════════════════════════════════════════════════╗");
    println!("║     开始训练                                  ║");
    println!("╚════════════════════════════════════════════════╝\n");

    // 训练循环
    for epoch in 1..=5 {
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

        // 更新学习率
        let lr = lr_scheduler.get_lr(epoch);
        optimizer.set_lr(lr);
        println!("学习率: {:.6}", lr);

        // 训练一个 epoch
        // 注意：当前实现是简化的，完整的训练需要 autograd 系统
        println!("训练 Epoch {}...", epoch);
        // let (train_loss, samples_per_sec) = trainer.train_epoch(
        //     &mut model,
        //     &train_loader,
        //     &mut optimizer,
        //     &loss_fn,
        //     epoch,
        // );

        // 评估
        println!("评估...");
        // let (val_loss, val_acc) = trainer.evaluate(&model, &val_loader, &loss_fn);

        println!("Epoch {} 完成", epoch);
        println!("  训练损失: {:.4}", 0.0);
        println!("  验证损失: {:.4}", 0.0);
        println!("  验证准确率: {:.2}%", 0.0);
        println!();
    }

    println!("╔════════════════════════════════════════════════╗");
    println!("║     训练完成！                                ║");
    println!("╚════════════════════════════════════════════════╝");

    println!("\n💡 提示:");
    println!("  - 当前实现是框架代码，完整训练需要手动实现反向传播");
    println!("  - 或者使用成熟的框架（candle、burn）");
    println!("  - 参考示例：train_epoch 和 evaluate 方法中的 TODO 注释");
}
