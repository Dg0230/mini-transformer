//! # Mini Transformer
//!
//! 一个从零实现的小型 Transformer 模型，用于学习 Transformer 原理。
//!
//! ## 架构概览
//!
//! ```text
//! Input → Embedding → Positional Encoding →
//!     [Transformer Encoder Layer × N] → Output
//!     ├── Multi-Head Self-Attention
//!     ├── Add & Norm
//!     ├── Feed Forward Network
//!     └── Add & Norm
//! ```

pub mod tensor;
pub mod embedding;
pub mod attention;
pub mod layers;
pub mod transformer;
pub mod autograd;
pub mod loss;
pub mod optimizer;
pub mod lr_scheduler;
pub mod trainer;
pub mod trainable;
pub mod trainable_transformer;
pub mod dataset;
pub mod checkpoint;
pub mod decoder;
pub mod seq2seq;
pub mod pretraining;
pub mod gradient_clip;
pub mod rope;

pub use tensor::TensorExt;
pub use embedding::{Embedding, PositionalEncoding};
pub use attention::{MultiHeadAttention, AttentionParams};
pub use layers::{FeedForward, LayerNorm};
pub use transformer::{TransformerEncoder, TransformerClassifier, TransformerConfig};
pub use loss::{LossFunction, CrossEntropyLoss, MSELoss, Accuracy};
pub use optimizer::{Optimizer, SGD, Adam, AdamW};
pub use lr_scheduler::{LRScheduler, ConstantLR, StepLR, CosineAnnealingWarmRestarts};
pub use trainer::{Trainer, TrainerConfig, Dataset, DataLoader, SimpleDataset};
pub use trainable_transformer::TrainableTransformer;
pub use dataset::{Vocabulary, TextSample, TextClassificationDataset, create_sentiment_dataset};
pub use checkpoint::{Checkpoint, TrainingHistory, CheckpointManager};
pub use seq2seq::Seq2SeqTransformer;
pub use decoder::{MaskedAttention, CrossAttention, DecoderLayer};
pub use pretraining::{MLMPretrainer, MLMConfig, CLMConfig, apply_mlm_mask, apply_mlm_mask_batch, create_clm_targets};
pub use gradient_clip::{GradientClipConfig, ClipType, clip_gradients, clip_grad_norm, clip_grad_value};
pub use rope::{RoPEConfig, RoPEAttention, apply_rope_batch, apply_rotary_pos_emb};

/// 预设配置
pub mod configs {
    use super::TransformerConfig;

    /// 小型模型（用于快速测试）
    pub fn mini() -> TransformerConfig {
        TransformerConfig {
            vocab_size: 1000,
            d_model: 128,
            n_heads: 4,
            n_layers: 2,
            d_ff: 512,
            max_seq_len: 64,
            dropout: 0.1,
        }
    }

    /// 基础模型（平衡性能和速度）
    pub fn base() -> TransformerConfig {
        TransformerConfig {
            vocab_size: 10000,
            d_model: 512,
            n_heads: 8,
            n_layers: 6,
            d_ff: 2048,
            max_seq_len: 512,
            dropout: 0.1,
        }
    }
}
