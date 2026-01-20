//! 增强版数据集
//!
//! 提供更大规模的内置文本数据集，用于改进生成质量

use std::collections::HashMap;

/// 增强版文本数据集
pub struct EnhancedDataset {
    pub name: String,
    pub description: String,
    pub texts: Vec<String>,
    pub category: String,
}

impl EnhancedDataset {
    /// 获取所有增强数据集
    pub fn all() -> Vec<Self> {
        vec![
            Self::chinese_literature(),
            Self::science_knowledge(),
            Self::daily_conversations(),
            Self::modern_stories(),
        ]
    }

    /// 获取指定名称的数据集
    pub fn get(name: &str) -> Option<Self> {
        match name {
            "literature" => Some(Self::chinese_literature()),
            "science" => Some(Self::science_knowledge()),
            "conversations" => Some(Self::daily_conversations()),
            "stories" => Some(Self::modern_stories()),
            "all" => Some(Self::combined()),
            _ => None,
        }
    }

    /// 中国文学经典（更大规模）
    pub fn chinese_literature() -> Self {
        Self {
            name: "literature".to_string(),
            description: "中国文学经典".to_string(),
            category: "文学".to_string(),
            texts: vec![
                // 古诗词
                "床前明月光，疑是地上霜。举头望明月，低头思故乡。".to_string(),
                "春眠不觉晓，处处闻啼鸟。夜来风雨声，花落知多少。".to_string(),
                "白日依山尽，黄河入海流。欲穷千里目，更上一层楼。".to_string(),
                "独在异乡为异客，每逢佳节倍思亲。遥知兄弟登高处，遍插茱萸少一人。".to_string(),
                "君不见黄河之水天上来，奔流到海不复回。君不见高堂明镜悲白发，朝如青丝暮成雪。".to_string(),
                "大江东去，浪淘尽，千古风流人物。故垒西边，人道是，三国周郎赤壁。".to_string(),
                "明月几时有？把酒问青天。不知天上宫阙，今夕是何年。".to_string(),
                
                // 成语故事
                "守株待兔：宋国有个农夫，看见一只兔子撞在树桩上死了，他就放下农具整天守在树桩旁，希望再捡到撞死的兔子。结果他再也没有捡到兔子，反而被宋国人嘲笑。这个故事告诉我们，不能心存侥幸，要靠努力才能成功。".to_string(),
                
                "画蛇添足：楚国有人祭祀祖先，赏给门客一壶酒。门客们觉得酒少人多，就约定在地上画蛇，先画好的人喝酒。一个人最先画好，拿起酒壶准备喝，见别人还没画好，就左手拿酒壶，右手给蛇画脚。脚还没画完，另一个人画好了，夺过酒壶说：蛇本来没有脚，你怎么能给它画脚呢？说完就把酒喝了。这个故事告诉我们，做多余的事有害无益。".to_string(),
                
                "刻舟求剑：楚国人坐船过江，不小心把剑掉进水里。他立刻在船边刻了个记号，说：我的剑是从这里掉下去的。船靠岸后，他顺着记号下水去找剑，但船已经移动了，而剑还在原地。这个故事告诉我们，情况变了，解决问题的方法也要跟着变。".to_string(),
                
                "掩耳盗铃：有人想偷一口钟，但是钟太重背不动。他想把钟敲碎了再搬，又怕别人听到钟声。于是他捂住自己的耳朵去敲钟，以为自己听不见，别人也就听不见了。结果还是被人抓住了。这个故事告诉我们，自己欺骗自己是没有用的。".to_string(),
                
                "亡羊补牢：有人养了几只羊，一天早上发现少了一只。原来羊圈破了个窟窿，狼从窟窿钻进去把羊叼走了。邻居劝他赶紧把羊圈修好，他却说：羊已经丢了，修羊圈有什么用？第二天他又丢了一只羊。他后悔了，赶紧把羊圈修好。从此他的羊再也没有丢过。这个故事告诉我们，出了问题要及时改正。".to_string(),
                
                // 散文
                "春天来了，大地复苏。小草从泥土里钻出来，嫩绿嫩绿的。桃花开了，粉红粉红的。燕子从南方飞回来了，在屋檐下筑巢。小溪解冻了，唱着欢快的歌谣流向远方。农民开始播种，孩子们在田野里奔跑。春天真是一年中最美好的季节。".to_string(),
                
                "夏天到了，天气炎热。太阳像个大火球，炙烤着大地。知了在树上不停地叫着：热啊，热啊！孩子们跳进河里游泳，老人们在树荫下乘凉。晚上，萤火虫在草丛中飞舞，像一颗颗闪亮的小星星。夏天虽然热，但也有夏天的乐趣。".to_string(),
                
                "秋天是个丰收的季节。稻田里金黄一片，麦浪翻滚。果园里，苹果红了，葡萄紫了，梨子黄了。农民们忙着收割庄稼，脸上洋溢着丰收的喜悦。秋天的天空特别蓝，云朵特别白。秋风一吹，树叶纷纷飘落，像一只只金色的蝴蝶。".to_string(),
                
                "冬天到了，北风呼啸，大雪纷飞。大地披上了银装，树枝上挂满了雪花。小河结冰了，孩子们在冰上滑冰、打雪仗。梅花在寒风中开放，散发着淡淡的幽香。冬天虽然寒冷，但也有着独特的美。".to_string(),
            ]
        }
    }

    /// 科学知识
    pub fn science_knowledge() -> Self {
        Self {
            name: "science".to_string(),
            description: "科学知识百科".to_string(),
            category: "科学".to_string(),
            texts: vec![
                "太阳系有八大行星：水星、金星、地球、火星、木星、土星、天王星和海王星。地球是唯一已知存在生命的行星。太阳是一颗恒星，为太阳系提供光和热。".to_string(),
                
                "水是生命之源。人体约70%由水组成。水在地球上以三种形态存在：液态（水）、固态（冰）和气态（水蒸气）。水循环包括蒸发、凝结和降水三个过程。".to_string(),
                
                "植物通过光合作用制造食物。它们需要阳光、水和二氧化碳。光合作用产生氧气和葡萄糖。氧气是动物和人类呼吸所必需的。".to_string(),
                
                "地球围绕太阳转，一年转一圈。地球自转一圈是一天。月亮围绕地球转，一个月转一圈。地球倾斜23.5度，这导致了四季的变化。".to_string(),
                
                "声音是由物体振动产生的。声音需要介质传播，可以在固体、液体和气体中传播，但不能在真空中传播。声音在固体中传播最快，在气体中最慢。".to_string(),
                
                "光是一种电磁波。光速是宇宙中最快的速度，约为每秒30万公里。光的颜色由波长决定。彩虹是阳光通过水滴折射形成的。".to_string(),
                
                "电是由电子运动产生的。电可以转化为热、光和机械能。发电机利用磁场产生电。电池通过化学反应产生电。".to_string(),
                
                "磁铁有两极：北极和南极。同极相斥，异极相吸。地球本身就是一个巨大的磁铁，指南针利用地球磁场指示方向。".to_string(),
                
                "动物分为脊椎动物和无脊椎动物。脊椎动物有哺乳类、鸟类、爬行类、两栖类和鱼类。哺乳动物温血、有毛发、用乳汁哺育后代。".to_string(),
                
                "人体有八大系统：消化系统、呼吸系统、循环系统、神经系统、内分泌系统、泌尿系统、生殖系统和运动系统。这些系统协调工作，维持人体的正常运转。".to_string(),
            ]
        }
    }

    /// 日常对话
    pub fn daily_conversations() -> Self {
        Self {
            name: "conversations".to_string(),
            description: "日常对话集锦".to_string(),
            category: "对话".to_string(),
            texts: vec![
                "你好，最近怎么样？我很好，谢谢你的关心。你呢？我也不错，就是工作有点忙。要注意休息，身体是革命的本钱。我会的，谢谢关心。".to_string(),
                
                "今天天气真好，要不要出去走走？好主意！我们去哪里？公园怎么样？可以散步，还可以欣赏风景。走吧，我准备一下。".to_string(),
                
                "你晚饭吃了吗？还没有，你有什么建议？我知道一家不错的餐厅，菜很好吃。太好了，我们去尝尝。那就这么定了，七点见。".to_string(),
                
                "这个周末你有什么安排？我想去书店买几本书。我可以陪你一起去吗？当然可以，我还想听听你的意见。没问题，周末见。".to_string(),
                
                "你能帮我个忙吗？当然可以，什么事？我电脑出了点问题，不会用了。我来帮你看看。应该是系统设置的问题，我帮你调一下。谢谢你，真是帮大忙了。".to_string(),
                
                "听说你要搬家了？是的，换个大点的房子。新家在哪里？离这儿不远，坐地铁半小时就到。那挺好的，祝贺你！谢谢，改天来家里玩。".to_string(),
                
                "你喜欢看电影吗？很喜欢，特别是科幻片。我也是，最近有部新片上映。真的吗？我们一起去看吧。好啊，我来买票。".to_string(),
                
                "你在学什么？我在学编程。编程很难吧？刚开始是有点难，但很有趣。我也想学，你能教我吗？当然可以，随时来问我。".to_string(),
                
                "你的爱好是什么？我喜欢读书和旅行。读什么书？历史和科学类的书。旅行呢？我去过很多地方，每个地方都有不同的风景和文化。真羡慕你，我也想多出去走走。".to_string(),
                
                "你觉得什么是幸福？我觉得幸福就是家人健康、工作顺利、朋友和睦。说得好，我也这么认为。珍惜当下，就是最大的幸福。".to_string(),
            ]
        }
    }

    /// 现代故事
    pub fn modern_stories() -> Self {
        Self {
            name: "stories".to_string(),
            description: "现代生活故事".to_string(),
            category: "故事".to_string(),
            texts: vec![
                "李华是一名程序员，每天都要写很多代码。他喜欢解决难题，每当找到bug的时候，他都会特别开心。他的梦想是开发出一个能帮助很多人的软件。经过不懈努力，他终于成功了。他的软件被很多人使用，他感到非常自豪。".to_string(),
                
                "王明是一名医生，每天都在医院工作。他对待病人非常认真，从不马虎。有一天，一位重症病人被送到医院。王明连夜为他做手术，终于把病人从死亡线上拉了回来。病人醒来后，紧紧握住王明的手，连声道谢。王明觉得，作为一名医生，能够救死扶伤，是他的荣幸。".to_string(),
                
                "张老师是一名小学教师，她热爱教育事业。她对待学生就像自己的孩子一样。有一次，一个学生因为家庭困难，想要退学。张老师知道后，主动帮助他，不仅给他补课，还资助他的生活费。这个学生后来考上重点中学，他永远不会忘记张老师的恩情。".to_string(),
                
                "刘洋是一名宇航员，从小就梦想飞向太空。经过刻苦训练，他终于有机会执行太空任务。在太空中，他看到了美丽的地球，感受到了宇宙的浩瀚。他在空间站进行了很多科学实验，为人类的太空探索做出了贡献。回到地球后，他成为了青少年的榜样，激励着更多人追求科学梦想。".to_string(),
                
                "赵敏是一名画家，她的画作充满灵气。她喜欢到大自然中写生，山川河流、花鸟虫鱼都是她的题材。有一次，她去山区写生，看到孩子们在艰苦的条件下依然努力学习，深受感动。她举办了一次义卖，把所有收入都捐给了山区的学校。孩子们用这些钱买了书和文具，他们给赵敏写了很多感谢信，赵敏觉得这是她收到的最好的礼物。".to_string(),
                
                "陈涛是一名志愿者，经常参加各种公益活动。周末，他会去敬老院看望老人，陪他们聊天，帮他们打扫卫生。节假日，他会去孤儿院，给孩子们讲故事，教他们画画。他的爱心感染了很多人，越来越多的人加入到志愿者的行列。陈涛说，帮助别人，快乐自己，这就是他做志愿者的动力。".to_string(),
                
                "周杰是一名厨师，他的菜做得特别好。他开了一家小餐馆，虽然不大，但是生意很好。他对食材很讲究，每天早上都去市场买最新鲜的蔬菜和肉类。他对待客人也很热情，客人们都喜欢来他这里吃饭。很多客人成了回头客，还带来了新客人。周杰觉得，用心做菜，客人能尝得到。".to_string(),
                
                "吴婷是一名舞蹈演员，从小练习舞蹈。她的舞姿优美，动作精准，每次演出都能赢得热烈的掌声。为了保持状态，她每天都要练习好几个小时。有时候练得浑身是伤，但她从不放弃。她说，舞蹈是她的生命，只要能跳舞，她就觉得幸福。她的努力没有白费，她成为了著名的舞蹈家，在世界各地演出。".to_string(),
                
                "郑强是一名消防员，工作很危险，但他从不畏惧。每次火灾报警，他都会第一时间冲向火场。有一次，一栋居民楼着火，还有很多人被困在里面。郑强冒着生命危险，一次又一次冲进火场，救出了十几个居民。最后所有人都安全了，他却累得瘫倒在地上。人们都说他是英雄，他说，这是他的职责。".to_string(),
                
                "孙丽是一名环保志愿者，她关注环境保护问题。她经常组织大家参加植树活动，还在社区宣传垃圾分类。在她的带动下，越来越多的人开始重视环保。小区的环境变好了，空气也更清新了。孙丽说，保护环境是每个人的责任，只要大家都出一份力，我们的家园会更美好。".to_string(),
            ]
        }
    }

    /// 组合数据集（所有数据）
    pub fn combined() -> Self {
        let lit = Self::chinese_literature();
        let sci = Self::science_knowledge();
        let conv = Self::daily_conversations();
        let stories = Self::modern_stories();

        let mut all_texts = Vec::new();
        all_texts.extend(lit.texts);
        all_texts.extend(sci.texts);
        all_texts.extend(conv.texts);
        all_texts.extend(stories.texts);

        Self {
            name: "all".to_string(),
            description: "组合数据集（所有文本）".to_string(),
            category: "综合".to_string(),
            texts: all_texts,
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
    fn test_enhanced_datasets() {
        let datasets = EnhancedDataset::all();
        assert!(datasets.len() >= 4);
    }

    #[test]
    fn test_combined_dataset() {
        let combined = EnhancedDataset::combined();
        assert!(combined.texts.len() >= 40); // 应该有很多文本
    }

    #[test]
    fn test_dataset_stats() {
        let dataset = EnhancedDataset::chinese_literature();
        let stats = dataset.stats();
        assert!(stats.num_texts > 0);
        assert!(stats.total_chars > 0);
    }
}
