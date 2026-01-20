//! 内置数据集
//!
//! 提供用于演示和测试的内置文本数据集

use std::collections::HashMap;

/// 内置文本数据集
pub struct BuiltinDataset {
    pub name: String,
    pub description: String,
    pub texts: Vec<String>,
}

impl BuiltinDataset {
    /// 获取所有内置数据集
    pub fn all() -> Vec<Self> {
        vec![
            Self::aesop_fables(),
            Self::tech_tutorials(),
            Self::simple_stories(),
            Self::poems(),
        ]
    }

    /// 获取指定名称的数据集
    pub fn get(name: &str) -> Option<Self> {
        match name {
            "aesop" => Some(Self::aesop_fables()),
            "tech" => Some(Self::tech_tutorials()),
            "stories" => Some(Self::simple_stories()),
            "poems" => Some(Self::poems()),
            _ => None,
        }
    }

    /// 伊索寓言（中文）
    pub fn aesop_fables() -> Self {
        Self {
            name: "aesop".to_string(),
            description: "伊索寓言（中文版）".to_string(),
            texts: vec![
                "乌鸦口渴了，找到一只水罐。水罐里的水不多，乌鸦喝不到。乌鸦看到地上有许多小石子，就把石子一颗一颗地叼起来，放进水罐里。水渐渐升高了，乌鸦终于喝到了水。这个故事告诉我们，遇到困难要动脑筋。".to_string(),

                "有一天，北风和太阳争论谁更强大。他们看到一个穿着大衣的行人。北风说：\"我能让他脱掉大衣。\"北风吹起了狂风，但是行人把大衣裹得更紧了。太阳出来照耀，行人觉得很热，就脱掉了大衣。这个故事告诉我们，温和比暴力更有效。".to_string(),

                "一只乌龟在和兔子赛跑。兔子跑得很快，但是他在半路睡着了。乌龟虽然爬得很慢，但是他不停地爬，最终到达了终点。兔子醒来后发现乌龟已经赢了。这个故事告诉我们，只要坚持不懈就能成功。".to_string(),

                "农夫在冬天发现一条冻僵的蛇。农夫很同情蛇，就把蛇放在怀里。蛇醒来后，咬了农夫一口。农夫中毒而死。这个故事告诉我们，对恶人不能仁慈。".to_string(),

                "一只狼掉进了井里。一只山羊也来到井边。狼对山羊说：\"井水很甜，你下来喝吧。\"山羊相信了狼的话，跳进了井里。狼踩着山羊的背爬出了井。山羊被困在井里。这个故事告诉我们，不要轻易相信坏人。".to_string(),
            ]
        }
    }

    /// 技术教程（编程相关）
    pub fn tech_tutorials() -> Self {
        Self {
            name: "tech".to_string(),
            description: "编程技术教程".to_string(),
            texts: vec![
                "Rust是一种系统编程语言，注重安全、并发和性能。Rust的所有权系统确保内存安全，无需垃圾回收。Rust的编译器非常严格，能在编译时捕获大多数错误。这使得Rust非常适合开发操作系统、数据库和Web服务。".to_string(),

                "Transformer是一种深度学习模型架构，通过自注意力机制处理序列数据。Transformer由编码器和解码器组成，可以并行处理所有位置的数据。这种架构被广泛应用于自然语言处理任务，如机器翻译、文本生成和问答系统。GPT和BERT都是基于Transformer的模型。".to_string(),

                "机器学习是人工智能的一个分支，通过数据训练模型来识别模式。监督学习需要标注数据，无监督学习不需要标注数据。深度学习使用多层神经网络，可以自动提取特征。常见的深度学习框架包括TensorFlow、PyTorch和Candle。".to_string(),

                "Git是分布式版本控制系统，用于跟踪代码变更。Git的基本操作包括add、commit和push。分支让开发者可以并行工作，合并将分支整合到主分支。GitHub和GitLab是托管Git仓库的流行平台。".to_string(),

                "数据结构是计算机存储和组织数据的方式。数组是连续存储的元素集合，链表通过指针连接节点。栈是后进先出的结构，队列是先进先出的结构。树和图用于表示层次关系和网络关系。选择合适的数据结构可以提高算法效率。".to_string(),
            ]
        }
    }

    /// 简单故事
    pub fn simple_stories() -> Self {
        Self {
            name: "stories".to_string(),
            description: "简单的儿童故事".to_string(),
            texts: vec![
                "小明喜欢画画。每天放学后，他都会拿起画笔，画出自己想象的世界。他画过蓝天白云，画过森林小动物，还画过未来的城市。老师说小明的画很有创意，同学们都很喜欢看他的画。小明希望长大后成为一名画家。".to_string(),

                "春天来了，花园里的花都开了。红的玫瑰，黄的向日葵，紫的薰衣草，五颜六色非常美丽。蝴蝶在花丛中飞舞，蜜蜂在采蜜。小兔子在草地上跳来跳去。这个花园就像一幅美丽的画。".to_string(),

                "小猫看到一只蝴蝶，就追了上去。蝴蝶飞到东，小猫追到东。蝴蝶飞到西，小猫追到西。蝴蝶飞到树上，小猫爬不上去，只好回家了。妈妈说：\"小猫，你真可爱。\"".to_string(),

                "今天是周末，爸爸带小明去公园玩。公园里有很多人，有的在跑步，有的在打球，有的在唱歌。小明和爸爸一起放风筝。风筝飞得很高很高，小明开心极了。他们玩了一整天，直到太阳下山才回家。".to_string(),

                "月亮姐姐出来了。她挂在天上，像一个银色的盘子。星星们围在月亮姐姐身边，一闪一闪的，好像在眨眼睛。小动物们都睡着了，只有萤火虫还在飞舞。这是一个宁静的夜晚。".to_string(),
            ]
        }
    }

    /// 诗歌
    pub fn poems() -> Self {
        Self {
            name: "poems".to_string(),
            description: "经典诗歌".to_string(),
            texts: vec![
                "床前明月光，疑是地上霜。举头望明月，低头思故乡。这是唐代诗人李白的《静夜思》，表达了诗人对故乡的思念。".to_string(),

                "春眠不觉晓，处处闻啼鸟。夜来风雨声，花落知多少。这是孟浩然的《春晓》，描写了春天早晨的景象。".to_string(),

                "白日依山尽，黄河入海流。欲穷千里目，更上一层楼。这是王之涣的《登鹳雀楼》，表达了不断向上的进取精神。".to_string(),

                "锄禾日当午，汗滴禾下土。谁知盘中餐，粒粒皆辛苦。这是李绅的《悯农》，告诉我们要珍惜粮食。".to_string(),

                "离离原上草，一岁一枯荣。野火烧不尽，春风吹又生。这是白居易的《赋得古原草送别》，赞美了小草顽强的生命力。".to_string(),
            ]
        }
    }

    /// 获取数据集统计信息
    pub fn stats(&self) -> DatasetStats {
        let total_chars: usize = self.texts.iter().map(|t| t.chars().count()).sum();
        let total_words: usize = self.texts.iter()
            .map(|t| t.split_whitespace().count())
            .sum();
        
        let avg_text_len = if self.texts.is_empty() {
            0.0
        } else {
            total_chars as f32 / self.texts.len() as f32
        };

        let mut char_freq = HashMap::new();
        for text in &self.texts {
            for ch in text.chars() {
                *char_freq.entry(ch).or_insert(0) += 1;
            }
        }

        let unique_chars = char_freq.len();

        DatasetStats {
            num_texts: self.texts.len(),
            total_chars,
            total_words,
            unique_chars,
            avg_text_len,
        }
    }
}

/// 数据集统计信息
#[derive(Debug, Clone)]
pub struct DatasetStats {
    pub num_texts: usize,
    pub total_chars: usize,
    pub total_words: usize,
    pub unique_chars: usize,
    pub avg_text_len: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builtin_datasets() {
        let datasets = BuiltinDataset::all();
        assert!(datasets.len() >= 4);
    }

    #[test]
    fn test_get_dataset() {
        let aesop = BuiltinDataset::get("aesop").unwrap();
        assert_eq!(aesop.name, "aesop");
        assert!(!aesop.texts.is_empty());
    }

    #[test]
    fn test_dataset_stats() {
        let dataset = BuiltinDataset::simple_stories();
        let stats = dataset.stats();
        assert!(stats.num_texts > 0);
        assert!(stats.total_chars > 0);
    }
}
