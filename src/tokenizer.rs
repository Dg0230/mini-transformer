//! BPE (Byte-Pair Encoding) Tokenizer
//!
//! 实现字节对编码分词器，用于子词级别的文本分词
//! 参考: Sennrich et al. (2016) - "Neural Machine Translation of Rare Words with Subword Units"

use std::collections::{HashMap, HashSet};
use std::io::{BufRead, BufReader};

/// BPE 分词器配置
#[derive(Debug, Clone)]
pub struct BPEConfig {
    /// 词汇表大小
    pub vocab_size: usize,
    /// 最小字符频率（低于此频率的字符将被忽略）
    pub min_frequency: usize,
    /// 特殊标记: UNK
    pub unk_token: String,
    /// 特殊标记: PAD
    pub pad_token: String,
    /// 特殊标记: BOS (Begin of Sequence)
    pub bos_token: String,
    /// 特殊标记: EOS (End of Sequence)
    pub eos_token: String,
    /// 是否在训练时包含特殊标记
    pub include_special_tokens: bool,
}

impl Default for BPEConfig {
    fn default() -> Self {
        Self {
            vocab_size: 10000,
            min_frequency: 2,
            unk_token: "[UNK]".to_string(),
            pad_token: "[PAD]".to_string(),
            bos_token: "[BOS]".to_string(),
            eos_token: "[EOS]".to_string(),
            include_special_tokens: true,
        }
    }
}

impl BPEConfig {
    /// 创建新的 BPE 配置
    pub fn new(vocab_size: usize) -> Self {
        Self {
            vocab_size,
            ..Default::default()
        }
    }

    /// 设置最小频率
    pub fn with_min_frequency(mut self, min_frequency: usize) -> Self {
        self.min_frequency = min_frequency;
        self
    }

    /// 设置 UNK token
    pub fn with_unk_token(mut self, unk: &str) -> Self {
        self.unk_token = unk.to_string();
        self
    }
}

/// BPE 分词器
#[derive(Debug, Clone)]
pub struct BPETokenizer {
    /// 词汇表：token -> id
    pub vocab: HashMap<String, usize>,
    /// 反向词汇表：id -> token
    pub inverse_vocab: Vec<String>,
    /// 合并规则：(pair) -> merged_token
    pub merges: Vec<(String, String)>,
    /// 配置
    pub config: BPEConfig,
    /// 特殊 token 的 ID
    pub unk_id: usize,
    pub pad_id: usize,
    pub bos_id: usize,
    pub eos_id: usize,
}

impl BPETokenizer {
    /// 创建新的 BPE 分词器
    pub fn new(config: BPEConfig) -> Self {
        let mut vocab = HashMap::new();
        let mut inverse_vocab = Vec::new();

        // 添加特殊 tokens
        let pad_id = if config.include_special_tokens {
            vocab.insert(config.pad_token.clone(), 0);
            inverse_vocab.push(config.pad_token.clone());
            vocab.insert(config.unk_token.clone(), 1);
            inverse_vocab.push(config.unk_token.clone());
            vocab.insert(config.bos_token.clone(), 2);
            inverse_vocab.push(config.bos_token.clone());
            vocab.insert(config.eos_token.clone(), 3);
            inverse_vocab.push(config.eos_token.clone());
            4
        } else {
            0
        };

        Self {
            vocab,
            inverse_vocab,
            merges: Vec::new(),
            config,
            unk_id: 1,
            pad_id: 0,
            bos_id: 2,
            eos_id: 3,
        }
    }

    /// 从文本语料训练 BPE 模型
    ///
    /// # Arguments
    /// * `corpus` - 文本语料列表
    pub fn train(&mut self, corpus: &[String]) {
        // 1. 初始化词汇表：所有字符
        let mut word_freqs = HashMap::new();

        for text in corpus {
            let words: Vec<&str> = text.split_whitespace().collect();
            for word in words {
                // 将单词拆分为字符
                let chars: Vec<String> = word.chars().map(|c| c.to_string()).collect();
                *word_freqs.entry(chars).or_insert(0) += 1;
            }
        }

        // 2. 初始化词汇表（所有字符）
        let mut char_vocab: HashSet<String> = HashSet::new();
        for (word_chars, freq) in word_freqs.iter() {
            if *freq >= self.config.min_frequency {
                for c in word_chars {
                    char_vocab.insert(c.clone());
                }
            }
        }

        // 添加字符到词汇表
        for c in char_vocab.iter() {
            if !self.vocab.contains_key(c) {
                self.vocab.insert(c.clone(), self.inverse_vocab.len());
                self.inverse_vocab.push(c.clone());
            }
        }

        println!("初始字符词汇表大小: {}", self.vocab.len());

        // 3. BPE 训练循环
        let mut current_word_freqs = word_freqs;
        let target_vocab_size = if self.config.include_special_tokens {
            self.config.vocab_size - 4 // 减去特殊 tokens
        } else {
            self.config.vocab_size
        };

        while self.vocab.len() < target_vocab_size {
            // 统计所有符号对的频率
            let pair_freqs = self.compute_pair_frequencies(&current_word_freqs);

            if pair_freqs.is_empty() {
                break;
            }

            // 找到频率最高的符号对
            let best_pair = pair_freqs
                .iter()
                .max_by_key(|(_, freq)| *freq)
                .map(|(pair, _)| pair.clone());

            let best_pair = match best_pair {
                Some(pair) => pair,
                None => break,
            };

            // 合并符号对
            let merged_token = format!("{}{}", best_pair.0, best_pair.1);

            // 添加到词汇表
            if !self.vocab.contains_key(&merged_token) {
                self.vocab.insert(merged_token.clone(), self.inverse_vocab.len());
                self.inverse_vocab.push(merged_token.clone());

                // 记录合并规则
                self.merges.push((best_pair.0.clone(), best_pair.1.clone()));

                println!(
                    "合并: {:?} + {:?} = {} (词汇表大小: {})",
                    best_pair.0,
                    best_pair.1,
                    merged_token,
                    self.vocab.len()
                );
            }

            // 更新单词频率
            current_word_freqs = self.apply_merge(&current_word_freqs, &best_pair);
        }

        println!("训练完成！最终词汇表大小: {}", self.vocab.len());
    }

    /// 计算所有符号对的频率
    fn compute_pair_frequencies(
        &self,
        word_freqs: &HashMap<Vec<String>, usize>,
    ) -> HashMap<(String, String), usize> {
        let mut pair_freqs = HashMap::new();

        for (word_chars, freq) in word_freqs.iter() {
            for i in 0..word_chars.len().saturating_sub(1) {
                let pair = (word_chars[i].clone(), word_chars[i + 1].clone());
                *pair_freqs.entry(pair).or_insert(0) += freq;
            }
        }

        pair_freqs
    }

    /// 应用合并规则到单词频率表
    fn apply_merge(
        &self,
        word_freqs: &HashMap<Vec<String>, usize>,
        pair: &(String, String),
    ) -> HashMap<Vec<String>, usize> {
        let mut new_word_freqs = HashMap::new();

        for (word_chars, freq) in word_freqs.iter() {
            let mut new_word = Vec::new();
            let mut i = 0;

            while i < word_chars.len() {
                if i + 1 < word_chars.len()
                    && &word_chars[i] == &pair.0
                    && &word_chars[i + 1] == &pair.1
                {
                    // 合并这对符号
                    new_word.push(format!("{}{}", pair.0, pair.1));
                    i += 2;
                } else {
                    new_word.push(word_chars[i].clone());
                    i += 1;
                }
            }

            *new_word_freqs.entry(new_word).or_insert(0) += freq;
        }

        new_word_freqs
    }

    /// 编码文本为 token IDs
    pub fn encode(&self, text: &str) -> Vec<usize> {
        let words: Vec<&str> = text.split_whitespace().collect();
        let mut tokens = Vec::new();

        for (i, word) in words.iter().enumerate() {
            let word_tokens = self.encode_word(word);
            tokens.extend(word_tokens);

            // 添加词尾标记（除了最后一个词）
            if i < words.len() - 1 {
                tokens.push(self.pad_id); // 使用 PAD token 作为分隔符
            }
        }

        tokens
    }

    /// 编码单个单词
    fn encode_word(&self, word: &str) -> Vec<usize> {
        let mut chars: Vec<String> = word.chars().map(|c| c.to_string()).collect();

        // 应用所有合并规则
        loop {
            let mut merged = false;

            for (token_a, token_b) in &self.merges {
                for i in 0..chars.len().saturating_sub(1) {
                    if &chars[i] == token_a && &chars[i + 1] == token_b {
                        // 合并
                        let merged_token = format!("{}{}", token_a, token_b);
                        chars[i] = merged_token;
                        chars.remove(i + 1);
                        merged = true;
                        break;
                    }
                }
                if merged {
                    break;
                }
            }

            if !merged {
                break;
            }
        }

        // 将 tokens 转换为 IDs
        chars
            .iter()
            .map(|token| self.vocab.get(token).copied().unwrap_or(self.unk_id))
            .collect()
    }

    /// 解码 token IDs 为文本
    pub fn decode(&self, token_ids: &[usize]) -> String {
        let mut text = String::new();

        for &id in token_ids {
            if id == self.pad_id {
                // PAD token 用作词分隔符
                text.push(' ');
            } else if let Some(token) = self.inverse_vocab.get(id) {
                // 跳过特殊 tokens（除了实际的文本内容）
                if !token.starts_with('[') || !token.ends_with(']') {
                    text.push_str(token);
                }
            }
        }

        text
    }

    /// 获取词汇表大小
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// 获取 token 的 ID（如果不存在返回 UNK ID）
    pub fn token_to_id(&self, token: &str) -> usize {
        self.vocab.get(token).copied().unwrap_or(self.unk_id)
    }

    /// 获取 ID 对应的 token（如果不存在返回 UNK token）
    pub fn id_to_token(&self, id: usize) -> String {
        self.inverse_vocab
            .get(id)
            .cloned()
            .unwrap_or_else(|| self.config.unk_token.clone())
    }

    /// 保存词汇表到文件
    pub fn save_vocab(&self, path: &str) -> std::io::Result<()> {
        use std::io::Write;
        let mut file = std::fs::File::create(path)?;

        for token in &self.inverse_vocab {
            writeln!(file, "{}", token)?;
        }

        Ok(())
    }

    /// 从文件加载词汇表
    pub fn load_vocab(&mut self, path: &str) -> std::io::Result<()> {
        let file = std::fs::File::open(path)?;
        let reader = BufReader::new(file);

        self.vocab.clear();
        self.inverse_vocab.clear();

        for (id, line) in reader.lines().enumerate() {
            let token = line?;
            self.vocab.insert(token.clone(), id);
            self.inverse_vocab.push(token);
        }

        Ok(())
    }

    /// 获取词汇表统计信息
    pub fn stats(&self) -> BPEStats {
        let mut token_lengths = Vec::new();

        for token in &self.inverse_vocab {
            token_lengths.push(token.chars().count());
        }

        let avg_length = if token_lengths.is_empty() {
            0.0
        } else {
            token_lengths.iter().sum::<usize>() as f32 / token_lengths.len() as f32
        };

        let max_length = token_lengths.iter().copied().max().unwrap_or(0);

        BPEStats {
            vocab_size: self.vocab.len(),
            num_merges: self.merges.len(),
            avg_token_length: avg_length,
            max_token_length: max_length,
        }
    }
}

/// BPE 统计信息
#[derive(Debug, Clone)]
pub struct BPEStats {
    pub vocab_size: usize,
    pub num_merges: usize,
    pub avg_token_length: f32,
    pub max_token_length: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bpe_config() {
        let config = BPEConfig::new(1000);
        assert_eq!(config.vocab_size, 1000);
        assert_eq!(config.min_frequency, 2);

        let config = BPEConfig::new(1000).with_min_frequency(5);
        assert_eq!(config.min_frequency, 5);
    }

    #[test]
    fn test_bpe_creation() {
        let config = BPEConfig::default();
        let tokenizer = BPETokenizer::new(config);
        assert_eq!(tokenizer.vocab.len(), 4); // 4 special tokens
    }

    #[test]
    fn test_bpe_train_simple() {
        let mut tokenizer = BPETokenizer::new(BPEConfig::new(50).with_min_frequency(1));

        let corpus = vec![
            "hello world".to_string(),
            "hello rust".to_string(),
            "hello hello".to_string(),
        ];

        tokenizer.train(&corpus);

        // 应该学习了 "he", "ll", "o", "hel", "llo", "hello" 等子词
        assert!(tokenizer.vocab.len() > 4);
    }

    #[test]
    fn test_bpe_encode_decode() {
        let mut tokenizer = BPETokenizer::new(BPEConfig::new(100).with_min_frequency(1));

        let corpus = vec![
            "hello world".to_string(),
            "hello rust".to_string(),
            "world rust".to_string(),
        ];

        tokenizer.train(&corpus);

        let text = "hello world";
        let token_ids = tokenizer.encode(text);
        let decoded = tokenizer.decode(&token_ids);

        assert_eq!(decoded, text);
    }

    #[test]
    fn test_token_to_id() {
        let mut tokenizer = BPETokenizer::new(BPEConfig::new(50).with_min_frequency(1));

        let corpus = vec!["test hello".to_string()];
        tokenizer.train(&corpus);

        // UNK token 应该存在
        assert_eq!(
            tokenizer.token_to_id("[UNK]"),
            tokenizer.unk_id
        );
    }

    #[test]
    fn test_bpe_stats() {
        let mut tokenizer = BPETokenizer::new(BPEConfig::new(100).with_min_frequency(1));

        let corpus = vec![
            "hello world".to_string(),
            "hello rust".to_string(),
        ];

        tokenizer.train(&corpus);

        let stats = tokenizer.stats();
        assert!(stats.vocab_size > 4);
        assert!(stats.num_merges > 0);
        assert!(stats.avg_token_length > 0.0);
    }
}
