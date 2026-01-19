//! 模型评估指标
//!
//! 提供各种机器学习评估指标的计算工具

use ndarray::Array2;
use std::collections::HashMap;
use crate::tensor::TensorExt;

/// 分类指标
///
/// 包含准确率、精确率、召回率和 F1 分数
#[derive(Debug, Clone, Copy)]
pub struct ClassificationMetrics {
    /// 准确率
    pub accuracy: f32,
    /// 精确率
    pub precision: f32,
    /// 召回率
    pub recall: f32,
    /// F1 分数
    pub f1_score: f32,
}

impl ClassificationMetrics {
    /// 计算分类指标
    ///
    /// # 参数
    /// - `predictions`: 预测的类别 [batch_size]
    /// - `targets`: 真实的类别 [batch_size]
    /// - `n_classes`: 类别总数
    pub fn compute(predictions: &[usize], targets: &[usize], n_classes: usize) -> Self {
        assert_eq!(predictions.len(), targets.len());

        let mut correct = 0;
        let mut tp_per_class = vec![0; n_classes];
        let mut fp_per_class = vec![0; n_classes];
        let mut fn_per_class = vec![0; n_classes];

        for (&pred, &target) in predictions.iter().zip(targets.iter()) {
            if pred == target {
                correct += 1;
                tp_per_class[pred] += 1;
            } else {
                fp_per_class[pred] += 1;
                fn_per_class[target] += 1;
            }
        }

        let accuracy = correct as f32 / predictions.len() as f32;

        // 计算宏平均精确率和召回率
        let precision: f32 = {
            let mut sum = 0.0;
            let mut count = 0;
            for i in 0..n_classes {
                let tp = tp_per_class[i] as f32;
                let fp = fp_per_class[i] as f32;
                if tp + fp > 0.0 {
                    sum += tp / (tp + fp);
                    count += 1;
                }
            }
            if count > 0 { sum / count as f32 } else { 0.0 }
        };

        let recall: f32 = {
            let mut sum = 0.0;
            let mut count = 0;
            for i in 0..n_classes {
                let tp = tp_per_class[i] as f32;
                let fn_ = fn_per_class[i] as f32;
                if tp + fn_ > 0.0 {
                    sum += tp / (tp + fn_);
                    count += 1;
                }
            }
            if count > 0 { sum / count as f32 } else { 0.0 }
        };

        let f1_score = if precision + recall > 0.0 {
            2.0 * precision * recall / (precision + recall)
        } else {
            0.0
        };

        Self {
            accuracy,
            precision,
            recall,
            f1_score,
        }
    }

    /// 打印指标摘要
    pub fn summary(&self) -> String {
        format!(
            "Accuracy: {:.2}% | Precision: {:.2}% | Recall: {:.2}% | F1 Score: {:.4}",
            self.accuracy * 100.0,
            self.precision * 100.0,
            self.recall * 100.0,
            self.f1_score
        )
    }
}

/// 混淆矩阵
#[derive(Debug, Clone)]
pub struct ConfusionMatrix {
    /// 矩阵数据 [n_classes, n_classes]
    matrix: Vec<Vec<usize>>,
    /// 类别数量
    n_classes: usize,
}

impl ConfusionMatrix {
    /// 创建混淆矩阵
    ///
    /// # 参数
    /// - `predictions`: 预测的类别
    /// - `targets`: 真实的类别
    /// - `n_classes`: 类别总数
    pub fn compute(predictions: &[usize], targets: &[usize], n_classes: usize) -> Self {
        assert_eq!(predictions.len(), targets.len());

        let mut matrix = vec![vec![0; n_classes]; n_classes];

        for (&pred, &target) in predictions.iter().zip(targets.iter()) {
            if pred < n_classes && target < n_classes {
                matrix[target][pred] += 1;
            }
        }

        Self { matrix, n_classes }
    }

    /// 获取矩阵
    pub fn matrix(&self) -> &Vec<Vec<usize>> {
        &self.matrix
    }

    /// 打印混淆矩阵
    pub fn display(&self) -> String {
        let mut output = String::from("Confusion Matrix:\n");
        output.push_str("     Pred →\n");

        // 表头
        output.push_str("True │");
        for i in 0..self.n_classes {
            output.push_str(&format!("{:5}", i));
        }
        output.push('\n');

        // 分隔线
        output.push_str("─────┼");
        for _ in 0..self.n_classes {
            output.push_str("─────");
        }
        output.push('\n');

        // 数据行
        for i in 0..self.n_classes {
            output.push_str(&format!("{:5}│", i));
            for j in 0..self.n_classes {
                output.push_str(&format!("{:5}", self.matrix[i][j]));
            }
            output.push('\n');
        }

        output
    }

    /// 获取每个类别的准确率
    pub fn per_class_accuracy(&self) -> Vec<f32> {
        let mut accuracies = Vec::new();

        for i in 0..self.n_classes {
            let total = (0..self.n_classes)
                .map(|j| self.matrix[i][j])
                .sum::<usize>();

            let correct = self.matrix[i][i];
            let acc = if total > 0 {
                correct as f32 / total as f32
            } else {
                0.0
            };

            accuracies.push(acc);
        }

        accuracies
    }
}

/// Perplexity（困惑度）
///
/// 语言模型的标准评估指标
///
/// # 公式
/// ```text
/// perplexity = exp(cross_entropy_loss)
/// ```
///
/// 越低越好，表示模型对数据的预测越准确
pub struct Perplexity;

impl Perplexity {
    /// 计算 perplexity
    ///
    /// # 参数
    /// - `model`: 模型（需实现前向传播）
    /// - `dataset`: 数据集
    ///
    /// # 返回
    /// - 平均困惑度
    pub fn compute<F>(model: &mut F, dataset: &[(Vec<usize>, usize)]) -> f32
    where
        F: ModelPredict,
    {
        let mut total_loss = 0.0;
        let mut total_samples = 0;

        for (input, target) in dataset.iter() {
            let input_array = vec_to_array2(&[input.clone()]);
            let logits = model.forward(&input_array);
            let n_classes = logits.ncols();

            // 计算 cross-entropy loss
            let log_softmax = logits.softmax(1);
            let loss = -log_softmax[[0, *target]].max(1e-10).ln();

            total_loss += loss;
            total_samples += 1;
        }

        let avg_loss = total_loss / total_samples as f32;
        avg_loss.exp()
    }

    /// 批量计算 perplexity
    ///
    /// # 参数
    /// - `model`: 模型
    /// - `inputs`: 输入数据 [batch_size, seq_len]
    /// - `targets`: 目标数据 [batch_size]
    pub fn compute_batch<F>(
        model: &mut F,
        inputs: &Array2<usize>,
        targets: &[usize],
    ) -> f32
    where
        F: ModelPredict,
    {
        let batch_size = inputs.nrows();
        let mut total_loss = 0.0;

        let logits = model.forward(inputs);
        let log_softmax = logits.softmax(1);

        for i in 0..batch_size {
            let target = targets[i];
            if target < log_softmax.ncols() {
                let loss = -log_softmax[[i, target]].max(1e-10).ln();
                total_loss += loss;
            }
        }

        let avg_loss = total_loss / batch_size as f32;
        avg_loss.exp()
    }
}

/// 模型预测 trait
///
/// 用于评估指标的统一接口
pub trait ModelPredict {
    /// 前向传播
    ///
    /// 输入: [batch_size, seq_len]
    /// 输出: [batch_size, n_classes]
    fn forward(&mut self, input: &Array2<usize>) -> Array2<f32>;
}

/// BLEU score
///
/// 机器翻译和文本生成的评估指标
///
/// # 参考文献
/// Papineni, K., Roukos, S., Ward, T., & Zhu, W. J. (2002).
/// BLEU: a method for automatic evaluation of machine translation.
pub struct BLEU {
    /// n-gram 大小（通常为 4）
    n: usize,
}

impl BLEU {
    /// 创建新的 BLEU 评估器
    pub fn new(n: usize) -> Self {
        Self { n }
    }

    /// 默认 BLEU-4
    pub fn bleu4() -> Self {
        Self { n: 4 }
    }

    /// 计算 BLEU score
    ///
    /// # 参数
    /// - `references`: 参考翻译（可能多个）[n_samples][n_references]
    /// - `hypotheses`: 模型生成结果 [n_samples]
    ///
    /// # 返回
    /// - BLEU score [0, 1]
    pub fn compute(&self, references: &[Vec<Vec<usize>>], hypotheses: &[Vec<usize>]) -> f32 {
        assert_eq!(references.len(), hypotheses.len());

        let mut precisions = Vec::new();
        let mut bp_len_ref = 0;
        let mut bp_len_hyp = 0;

        // 计算 1-gram 到 n-gram 的精确率
        for n in 1..=self.n {
            let mut correct = 0;
            let mut total = 0;

            for (refs, hyp) in references.iter().zip(hypotheses.iter()) {
                // 选择最短的参考作为基准
                let empty: &[usize] = &[];
                let best_ref = if let Some(r) = refs.iter().min_by_key(|r| r.len()) {
                    r
                } else {
                    empty
                };

                // 计算 n-gram 匹配
                let ref_ngrams = count_ngrams(best_ref, n);
                let hyp_ngrams = count_ngrams(hyp, n);

                for (ngram, &count) in hyp_ngrams.iter() {
                    let ref_count = ref_ngrams.get(ngram).unwrap_or(&0);
                    correct += *ref_count.min(&count);
                }

                total += hyp_ngrams.values().sum::<usize>();
            }

            let precision = if total > 0 {
                correct as f32 / total as f32
            } else {
                0.0
            };

            precisions.push(precision);
        }

        // 计算长度惩罚
        for refs in references.iter() {
            for ref_sent in refs.iter() {
                bp_len_ref += ref_sent.len();
            }
        }
        for hyp in hypotheses.iter() {
            bp_len_hyp += hyp.len();
        }

        let bp = if bp_len_hyp > 0 && bp_len_hyp < bp_len_ref {
            (1.0 - bp_len_ref as f32 / bp_len_hyp as f32).exp()
        } else {
            1.0
        };

        // 几何平均
        let geo_mean: f32 = precisions.iter().product::<f32>().powf(1.0 / precisions.len() as f32);

        bp * geo_mean
    }

    /// 计算单个样本的 BLEU
    ///
    /// # 参数
    /// - `reference`: 参考翻译
    /// - `hypothesis`: 模型生成结果
    pub fn compute_single(&self, reference: &[usize], hypothesis: &[usize]) -> f32 {
        let references = vec![vec![reference.to_vec()]];
        let hypotheses = vec![hypothesis.to_vec()];
        self.compute(&references, &hypotheses)
    }
}

/// 计算 n-gram 计数
fn count_ngrams(tokens: &[usize], n: usize) -> HashMap<Vec<usize>, usize> {
    let mut ngrams = HashMap::new();

    if tokens.len() < n {
        return ngrams;
    }

    for i in 0..=(tokens.len() - n) {
        let ngram = tokens[i..i + n].to_vec();
        *ngrams.entry(ngram).or_insert(0) += 1;
    }

    ngrams
}

/// 辅助函数：将 Vec 转换为 Array2
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classification_metrics() {
        let predictions = vec![0, 1, 2, 1, 0];
        let targets = vec![0, 1, 1, 1, 0];
        let n_classes = 3;

        let metrics = ClassificationMetrics::compute(&predictions, &targets, n_classes);

        assert_eq!(metrics.accuracy, 0.8); // 4/5 正确
    }

    #[test]
    fn test_confusion_matrix() {
        let predictions = vec![0, 1, 2, 1, 0];
        let targets = vec![0, 1, 1, 1, 0];

        let cm = ConfusionMatrix::compute(&predictions, &targets, 3);

        // 对角线元素是正确预测的数量
        assert_eq!(cm.matrix[0][0], 2); // 类别 0 正确 2 次
        assert_eq!(cm.matrix[1][1], 2); // 类别 1 正确 2 次
        assert_eq!(cm.matrix[2][2], 0); // 类别 2 正确 0 次
    }

    #[test]
    fn test_bleu_score() {
        let references = vec![
            vec![vec![1, 2, 3, 4, 5]],  // 完美匹配
        vec![vec![1, 2, 3, 4, 5, 6]], // 少一个词
        vec![vec![1, 2, 4, 5, 6]],    // 错误
        vec![vec![1, 2, 3, 4, 5]],    // 完美
        vec![vec![1, 2, 3, 4]],      // 少一个词
        ];
        let hypotheses = vec![
            vec![1, 2, 3, 4, 5],
            vec![1, 2, 3, 4, 5],
            vec![1, 2, 3, 4, 5],
            vec![1, 2, 3, 4, 5],
            vec![1, 2, 3, 4, 5],
        ];

        let bleu = BLEU::bleu4().compute(&references, &hypotheses);

        // 第一个完美匹配，其他不完美
        assert!(bleu > 0.0 && bleu <= 1.0);
    }

    #[test]
    fn test_ngram_counting() {
        let tokens = vec![1, 2, 3, 4, 5];

        let unigrams = count_ngrams(&tokens, 1);
        assert_eq!(unigrams.len(), 5);
        assert_eq!(*unigrams.get(&vec![1]).unwrap(), 1);
        assert_eq!(*unigrams.get(&vec![3]).unwrap(), 1);

        let bigrams = count_ngrams(&tokens, 2);
        assert_eq!(bigrams.len(), 4);
        assert_eq!(*bigrams.get(&vec![1, 2]).unwrap(), 1);
        assert_eq!(*bigrams.get(&vec![4, 5]).unwrap(), 1);
    }
}
