//! 文本生成示例
//!
//! 展示 Seq2Seq Transformer 的生成能力

use mini_transformer::Seq2SeqTransformer;
use std::time::Instant;

fn main() {
    println!("╔════════════════════════════════════════════════╗");
    println!("║     Seq2Seq 文本生成示例                     ║");
    println!("╚════════════════════════════════════════════════╝\n");

    // 创建 Seq2Seq 模型
    println!("🏗️  构建 Seq2Seq 模型...");
    let mut model = Seq2SeqTransformer::new(
        1000,  // vocab_size
        128,   // d_model
        4,     // n_heads
        2,     // n_layers
        256,   // d_ff
        50,    // max_seq_len
    );

    println!("  模型参数总数: {}\n", model.param_count());

    // 示例：简单的序列转换任务
    // 假设我们要学习：输入序列 → 反转序列
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("任务：序列反转");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let test_cases = vec![
        vec![1, 2, 3, 4],
        vec![5, 6, 7],
        vec![10, 20, 30, 40, 50],
    ];

    println!("测试案例:\n");

    for (i, source) in test_cases.iter().enumerate() {
        println!("案例 {}:", i + 1);
        println!("  输入: {:?}", source);

        // 贪婪解码
        let start_greedy = Instant::now();
        let greedy_output = model.generate_greedy(source, 20, 0);
        let greedy_time = start_greedy.elapsed();

        println!("  贪婪解码输出: {:?}", greedy_output);
        println!("  耗时: {:?}", greedy_time);

        // 束搜索解码
        let start_beam = Instant::now();
        let beam_output = model.generate_beam(source, 20, 3, 0);
        let beam_time = start_beam.elapsed();

        println!("  束搜索输出: {:?}", beam_output);
        println!("  耗时: {:?}", beam_time);

        println!();
    }

    // 性能对比
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("生成策略对比");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let test_source = vec![1, 2, 3, 4, 5, 6, 7, 8];
    println!("测试序列: {:?}", test_source);
    println!();

    // 贪婪解码
    let start = Instant::now();
    let greedy = model.generate_greedy(&test_source, 50, 0);
    let greedy_time = start.elapsed();

    println!("贪婪解码:");
    println!("  输出: {:?}", greedy);
    println!("  生成长度: {}", greedy.len());
    println!("  耗时: {:?}", greedy_time);
    println!();

    // 束搜索（不同宽度）
    for beam_width in [1, 3, 5] {
        let start = Instant::now();
        let beam = model.generate_beam(&test_source, 50, beam_width, 0);
        let beam_time = start.elapsed();

        println!("束搜索 (width={}):", beam_width);
        println!("  输出: {:?}", beam);
        println!("  生成长度: {}", beam.len());
        println!("  耗时: {:?}", beam_time);
        println!();
    }

    // 技术细节
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("架构组件");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    println!("Seq2Seq Transformer 包含:");
    println!("  1. Encoder - 编码源序列");
    println!("     - Token Embedding");
    println!("     - Positional Encoding");
    println!();
    println!("  2. Decoder - 生成目标序列");
    println!("     - Masked Self-Attention (因果掩码)");
    println!("     - Cross-Attention (关注 Encoder)");
    println!("     - Feed-Forward Network");
    println!("     - Layer Normalization");
    println!("     - Residual Connections");
    println!();
    println!("  3. Output Projection");
    println!("     - 线性层映射到词汇表");
    println!();

    // 生成策略说明
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("生成策略");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    println!("1. 贪婪解码 (Greedy Decoding)");
    println!("   - 每步选择概率最高的 token");
    println!("   - 快速但可能陷入局部最优");
    println!("   - 时间复杂度: O(T × V)");
    println!();

    println!("2. 束搜索 (Beam Search)");
    println!("   - 保留 top-k 个候选序列");
    println!("   - 平衡质量和速度");
    println!("   - 时间复杂度: O(T × k × V)");
    println!("   - k = beam width");
    println!();

    println!("其中:");
    println!("  T = 目标序列长度");
    println!("  V = 词汇表大小");
    println!("  k = 束宽度");
    println!();

    println!("╔════════════════════════════════════════════════╗");
    println!("║     示例完成！                               ║");
    println!("╚════════════════════════════════════════════════╝");
    println!();
    println!("💡 提示: 当前模型未训练，输出是随机的");
    println!("   实际使用需要在真实数据上训练模型");
}
