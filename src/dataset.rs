//! 数据集和数据处理
//!
//! 提供简单的文本分类数据集

use ndarray::Array2;
use std::collections::HashMap;

/// 简单的词汇表（Tokenizer）
#[derive(Debug, Clone)]
pub struct Vocabulary {
    /// token 到 ID 的映射
    token_to_id: HashMap<String, usize>,
    /// ID 到 token 的映射
    id_to_token: Vec<String>,
    /// 词汇表大小
    vocab_size: usize,
    /// 未知 token 标记
    unk_token: String,
    /// 填充 token 标记
    pad_token: String,
}

impl Vocabulary {
    /// 创建新的词汇表
    pub fn new() -> Self {
        let mut token_to_id = HashMap::new();
        let mut id_to_token = Vec::new();

        // 添加特殊标记
        let unk_token = "<UNK>".to_string();
        let pad_token = "<PAD>".to_string();

        token_to_id.insert(unk_token.clone(), 0);
        id_to_token.push(unk_token.clone());

        token_to_id.insert(pad_token.clone(), 1);
        id_to_token.push(pad_token.clone());

        Self {
            token_to_id,
            id_to_token,
            vocab_size: 2,
            unk_token,
            pad_token,
        }
    }

    /// 从文本列表构建词汇表
    pub fn from_texts(texts: &[String], min_freq: usize) -> Self {
        let mut vocab = Self::new();

        // 统计词频
        let mut freq = HashMap::new();
        for text in texts {
            for token in text.split_whitespace() {
                *freq.entry(token.to_string()).or_insert(0) += 1;
            }
        }

        // 添加高频词
        let mut words: Vec<_> = freq.into_iter()
            .filter(|(_, count)| *count >= min_freq)
            .collect();
        words.sort_by(|a, b| b.1.cmp(&a.1)); // 按频率降序

        for (word, _) in words {
            if !vocab.token_to_id.contains_key(&word) {
                vocab.token_to_id.insert(word.clone(), vocab.vocab_size);
                vocab.id_to_token.push(word);
                vocab.vocab_size += 1;
            }
        }

        vocab
    }

    /// 将文本转换为 token IDs
    pub fn encode(&self, text: &str, max_len: usize) -> Vec<usize> {
        let mut tokens = text.split_whitespace()
            .map(|token| {
                *self.token_to_id.get(token).unwrap_or(&0) // UNK = 0
            })
            .collect::<Vec<_>>();

        // 截断或填充
        if tokens.len() > max_len {
            tokens = tokens[..max_len].to_vec();
        } else {
            while tokens.len() < max_len {
                tokens.push(1); // PAD = 1
            }
        }

        tokens
    }

    /// 将 token IDs 转换回文本
    pub fn decode(&self, tokens: &[usize]) -> String {
        tokens.iter()
            .filter_map(|&id| {
                if id < self.id_to_token.len() && id != 1 { // 不包括 PAD
                    Some(self.id_to_token[id].clone())
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// 获取词汇表大小
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }
}

/// 文本分类样本
#[derive(Debug, Clone)]
pub struct TextSample {
    /// 文本内容
    pub text: String,
    /// 类别标签
    pub label: usize,
}

/// 文本分类数据集
#[derive(Debug, Clone)]
pub struct TextClassificationDataset {
    /// 训练样本
    pub train_samples: Vec<TextSample>,
    /// 测试样本
    pub test_samples: Vec<TextSample>,
    /// 类别数量
    pub n_classes: usize,
    /// 词汇表
    pub vocab: Vocabulary,
}

impl TextClassificationDataset {
    /// 创建新的数据集
    pub fn new(train_samples: Vec<TextSample>, test_samples: Vec<TextSample>, n_classes: usize, vocab: Vocabulary) -> Self {
        Self {
            train_samples,
            test_samples,
            n_classes,
            vocab,
        }
    }

    /// 获取训练样本数量
    pub fn train_len(&self) -> usize {
        self.train_samples.len()
    }

    /// 获取测试样本数量
    pub fn test_len(&self) -> usize {
        self.test_samples.len()
    }

    /// 批次编码训练数据
    pub fn encode_batch(&self, samples: &[TextSample], max_len: usize) -> (Array2<usize>, Vec<usize>) {
        let n_samples = samples.len();

        let mut input_ids = Vec::with_capacity(n_samples * max_len);
        let mut labels = Vec::with_capacity(n_samples);

        for sample in samples {
            let tokens = self.vocab.encode(&sample.text, max_len);
            input_ids.extend(tokens);
            labels.push(sample.label);
        }

        let inputs = Array2::from_shape_vec((n_samples, max_len), input_ids).unwrap();

        (inputs, labels)
    }

    /// 编码所有训练数据
    pub fn encode_train(&self, max_len: usize) -> (Array2<usize>, Vec<usize>) {
        self.encode_batch(&self.train_samples, max_len)
    }

    /// 编码所有测试数据
    pub fn encode_test(&self, max_len: usize) -> (Array2<usize>, Vec<usize>) {
        self.encode_batch(&self.test_samples, max_len)
    }
}

/// 创建简单的情感分析数据集
///
/// 返回一个预定义的小型情感分类数据集
pub fn create_sentiment_dataset() -> TextClassificationDataset {
    // 积极评价
    let positive = vec![
        "this movie is great and wonderful",
        "I love this film it is amazing",
        "excellent performance and great story",
        "best movie ever highly recommended",
        "fantastic acting and brilliant direction",
        "really enjoyed this masterpiece",
        "outstanding film with great characters",
        "superb storytelling and amazing visuals",
        "perfect movie loved every moment",
        "incredible film must watch",
        "great entertainment and fun",
        "wonderful experience and joy",
        "this film is fantastic",
        "amazing movie and excellent",
        "brilliant and outstanding work",
        "this is great love it",
        "best film ever seen",
        "excellent and wonderful",
        "fantastic amazing movie",
        "superb and perfect",
    ];

    // 消极评价
    let negative = vec![
        "this movie is terrible and boring",
        "I hate this film it is awful",
        "poor performance and bad story",
        "worst movie ever not recommended",
        "disappointing acting and terrible direction",
        "really disliked this disaster",
        "awful film with bad characters",
        "boring storytelling and poor visuals",
        "horrible movie hated every moment",
        "terrible film avoid it",
        "bad entertainment and dull",
        "awful experience and waste",
        "this film is horrible",
        "boring movie and terrible",
        "poor and awful work",
        "this is bad hate it",
        "worst film ever seen",
        "terrible and awful",
        "horrible bad movie",
        "boring and poor",
    ];

    // 创建训练样本
    let train_samples: Vec<TextSample> = positive.iter()
        .take(15)
        .chain(negative.iter().take(15))
        .enumerate()
        .map(|(i, text)| {
            TextSample {
                text: text.to_string(),
                label: if i < 15 { 1 } else { 0 }, // 1=积极, 0=消极
            }
        })
        .collect();

    // 创建测试样本
    let test_samples: Vec<TextSample> = positive.iter()
        .skip(15)
        .chain(negative.iter().skip(15))
        .enumerate()
        .map(|(i, text)| {
            TextSample {
                text: text.to_string(),
                label: if i < 5 { 1 } else { 0 },
            }
        })
        .collect();

    // 构建词汇表
    let all_texts: Vec<String> = positive.iter()
        .chain(negative.iter())
        .map(|s| s.to_string())
        .collect();

    let vocab = Vocabulary::from_texts(&all_texts, 1);

    TextClassificationDataset {
        train_samples,
        test_samples,
        n_classes: 2,
        vocab,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vocabulary_creation() {
        let vocab = Vocabulary::new();
        assert_eq!(vocab.vocab_size(), 2);
    }

    #[test]
    fn test_vocabulary_encode() {
        let vocab = Vocabulary::from_texts(&["hello world".to_string(), "hello rust".to_string()], 1);

        let tokens = vocab.encode("hello world", 10);
        assert_eq!(tokens.len(), 10);
        assert!(tokens.contains(&2)); // "hello" 的 ID 应该 >= 2
    }

    #[test]
    fn test_vocabulary_decode() {
        let vocab = Vocabulary::from_texts(&["hello world".to_string()], 1);

        // 先编码获取实际的 token IDs
        let tokens = vocab.encode("hello world", 5);

        // 然后解码回来
        let text = vocab.decode(&tokens);
        assert!(text.contains("hello") || text.contains("world"));
    }

    #[test]
    fn test_sentiment_dataset() {
        let dataset = create_sentiment_dataset();

        assert_eq!(dataset.train_len(), 30);
        assert_eq!(dataset.test_len(), 10);
        assert_eq!(dataset.n_classes, 2);
        assert!(dataset.vocab.vocab_size() > 10);
    }
}
