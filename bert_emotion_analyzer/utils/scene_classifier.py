"""
场景分类器 - 识别文本所属的场景类别
"""
import json
import yaml
import os
from typing import List, Dict, Tuple, Any, Optional
import logging
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SceneClassifier:
    """场景分类器（重构版本，使用外部配置）"""
    
    def __init__(self, config_path: str = "configs/sentiment_config.yaml"):
        """
        初始化场景分类器
        
        Args:
            config_path: 配置文件路径
        """
        self.config = self._load_config(config_path)
        
        # 加载场景配置
        self.scene_categories = self._load_scene_categories()
        self.scene_weights = self._calculate_scene_weights()
        
        # 场景别名映射
        self.scene_aliases = {
            "工作": ["职场", "办公室", "公司", "职业"],
            "家庭": ["家人", "亲情", "婚姻", "亲子"],
            "学校": ["学习", "教育", "校园", "考试"],
            "购物消费": ["购物", "消费", "商品", "购买"],
            "餐饮美食": ["美食", "饮食", "餐厅", "食物"],
            "旅游出行": ["旅游", "旅行", "出行", "景点"],
            "健康医疗": ["健康", "医疗", "医院", "医生"],
            "娱乐休闲": ["娱乐", "休闲", "游戏", "电影"],
            "科技数码": ["科技", "数码", "手机", "电脑"],
            "社交人际关系": ["社交", "朋友", "关系", "人际"],
            "天气环境": ["天气", "气候", "环境", "自然"],
            "时事新闻": ["新闻", "时事", "政治", "社会"],
            "交通出行": ["交通", "出行", "道路", "驾驶"],
            "网络社交": ["网络", "社交", "微博", "微信"]
        }
        
        # 从配置加载参数
        self.min_score_threshold = self.config.get('scene_classifier', {}).get('min_score_threshold', 0.1)
        self.max_scenes = self.config.get('scene_classifier', {}).get('max_scenes', 3)
        
        logger.info(f"场景分类器初始化完成，共 {len(self.scene_categories)} 个场景类别")
    
    def _load_config(self, config_path: str) -> Dict:
        """加载配置文件"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                    return yaml.safe_load(f)
                elif config_path.endswith('.json'):
                    return json.load(f)
        except FileNotFoundError:
            logger.warning(f"配置文件 {config_path} 不存在，使用默认配置")
            return {'scene_classifier': {}}
    
    def _load_scene_categories(self) -> Dict[str, List[str]]:
        """
        加载场景分类配置
        
        Returns:
            场景分类字典 {场景名称: 关键词列表}
        """
        # 尝试从配置文件加载
        config_file = "configs/scene_categories.json"
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"加载场景配置文件失败 {config_file}: {e}")
        
        # 默认场景分类（与原始代码相同但使用外部化）
        return {
            "工作": ["工作", "加班", "上班", "同事", "老板", "会议", "项目", "任务", "汇报", "职场",
                     "办公室", "公司", "职业", "职位", "岗位", "业绩", "考核", "晋升", "离职", "跳槽",
                     "面试", "简历", "招聘", "薪资", "奖金", "福利", "培训", "出差", "商务", "合同"],
            
            "家庭": ["家庭", "家人", "父母", "妈妈", "爸爸", "孩子", "子女", "儿女", "老公", "老婆",
                     "妻子", "丈夫", "婚姻", "结婚", "离婚", "家务", "做饭", "洗衣", "打扫",
                     "装修", "买房", "租房", "房贷", "车贷", "亲子", "育儿", "教育", "孝顺", "团聚"],
            
            "学校": ["学校", "学习", "老师", "同学", "课程", "考试", "作业", "论文", "毕业", "学位",
                     "校园", "教室", "图书馆", "自习", "成绩", "分数", "奖学金", "社团", "活动", "比赛",
                     "专业", "学科", "教材", "辅导", "培训班", "高考", "中考", "大学", "中学", "小学"],
            
            "购物消费": ["购物", "商品", "产品", "质量", "价格", "贵", "便宜", "实惠", "划算", "优惠",
                         "折扣", "促销", "特价", "秒杀", "团购", "优惠券", "满减", "包邮", "物流", "快递",
                         "送货", "收货", "评价", "好评", "差评", "退货", "退款", "售后", "客服", "咨询"],
            
            "餐饮美食": ["美食", "食物", "餐厅", "饭店", "餐馆", "小吃", "早餐", "午餐", "晚餐", "夜宵",
                         "味道", "好吃", "难吃", "美味", "可口", "新鲜", "卫生", "服务", "环境", "氛围",
                         "饮料", "饮品", "奶茶", "咖啡", "酒", "啤酒", "白酒", "红酒", "烧烤", "火锅"],
            
            "旅游出行": ["旅游", "旅行", "景点", "景区", "酒店", "宾馆", "民宿", "交通", "飞机", "火车",
                         "高铁", "汽车", "公交", "地铁", "出租车", "导航", "地图", "行程", "计划", "攻略",
                         "风景", "拍照", "纪念", "体验", "导游", "旅行社", "门票", "预订", "假期", "度假"],
            
            "健康医疗": ["健康", "医疗", "医院", "医生", "护士", "看病", "挂号", "检查", "化验", "诊断",
                         "治疗", "手术", "药品", "药房", "药店", "疫苗", "体检", "保健", "养生", "运动",
                         "健身", "减肥", "饮食", "睡眠", "心理", "压力", "焦虑", "抑郁", "康复", "恢复"],
            
            "娱乐休闲": ["娱乐", "休闲", "电影", "电视", "电视剧", "综艺", "音乐", "歌曲", "演唱会", "KTV",
                         "游戏", "手游", "网游", "电竞", "体育", "运动", "比赛", "健身", "瑜伽", "舞蹈",
                         "阅读", "书籍", "小说", "杂志", "绘画", "摄影", "手工", "宠物", "养宠", "花草"],
            
            "科技数码": ["科技", "数码", "手机", "电脑", "平板", "笔记本", "硬件", "软件", "系统", "应用",
                         "APP", "游戏", "网络", "互联网", "5G", "AI", "人工智能", "数据", "云计算", "编程",
                         "开发", "设计", "产品", "体验", "性能", "功能", "电池", "屏幕", "拍照", "视频"],
            
            "社交人际关系": ["朋友", "友情", "友谊", "同学", "同事", "伙伴", "社交", "交往", "沟通",
                             "聊天", "对话", "联系", "聚会", "约会", "相亲", "恋爱", "爱情", "感情", "关系",
                             "矛盾", "冲突", "和解", "理解", "支持", "帮助", "信任", "尊重", "礼貌", "礼仪"],
            
            "天气环境": ["天气", "气候", "温度", "气温", "炎热", "寒冷", "凉爽", "温暖", "晴天", "阴天",
                         "雨天", "下雨", "下雪", "刮风", "台风", "暴雨", "雾霾", "空气", "环境", "污染",
                         "环保", "生态", "自然", "风景", "景色", "季节", "春天", "夏天", "秋天", "冬天"],
            
            "时事新闻": ["新闻", "时事", "热点", "事件", "政治", "经济", "社会", "文化", "教育", "科技",
                         "国际", "国内", "本地", "政策", "法律", "法规", "改革", "发展", "创新", "创业",
                         "市场", "股市", "房价", "就业", "收入", "消费", "投资", "理财", "保险", "税收"],
            
            "交通出行": ["交通", "出行", "道路", "公路", "高速公路", "交通", "堵车", "拥堵", "停车", "车位",
                         "驾驶", "开车", "司机", "乘客", "公共交通", "地铁", "公交", "出租车", "网约车",
                         "导航", "路线", "违章", "事故", "安全", "保险", "维修", "保养", "油价", "充电"],
            
            "网络社交": ["微博", "微信", "朋友圈", "QQ", "空间", "抖音", "快手", "B站", "小红书", "知乎",
                         "贴吧", "论坛", "社群", "群聊", "点赞", "评论", "转发", "分享", "关注", "粉丝",
                         "网红", "博主", "主播", "直播", "弹幕", "刷屏", "热搜", "话题", "超话", "打卡"],
        }
    
    def _calculate_scene_weights(self) -> Dict[str, float]:
        """
        计算场景权重
        
        Returns:
            场景权重字典 {场景名称: 权重}
        """
        weights = {}
        for scene, keywords in self.scene_categories.items():
            # 基础权重：关键词数量 * 系数
            base_weight = len(keywords) * 0.1
            
            # 根据场景重要性调整权重
            importance_factors = {
                "工作": 1.2,
                "家庭": 1.1,
                "学校": 1.1,
                "健康医疗": 1.3,
                "社交人际关系": 1.1,
                # 其他场景默认为1.0
            }
            
            importance = importance_factors.get(scene, 1.0)
            weights[scene] = base_weight * importance
        
        return weights
    
    def classify_scene(self, text: str) -> List[Dict[str, Any]]:
        """
        对文本进行场景分类
        
        Args:
            text: 输入文本
            
        Returns:
            场景分类结果列表 [{场景: 详情}]
        """
        scene_scores = {}
        text_lower = text
        
        # 第一轮：精确匹配
        for scene, keywords in self.scene_categories.items():
            score = 0
            matched_keywords = []
            
            for keyword in keywords:
                if keyword in text_lower:
                    # 精确匹配得分更高
                    score += 2.0
                    matched_keywords.append(keyword)
            
            if score > 0:
                # 应用场景权重
                weighted_score = score * self.scene_weights.get(scene, 1.0)
                scene_scores[scene] = {
                    'score': weighted_score,
                    'base_score': score,
                    'keywords': matched_keywords[:5],  # 只保留前5个关键词
                    'match_type': 'exact'
                }
        
        # 第二轮：别名匹配（如果精确匹配不够）
        if len(scene_scores) < 2:
            for scene, aliases in self.scene_aliases.items():
                if scene not in scene_scores:
                    score = 0
                    matched_aliases = []
                    
                    for alias in aliases:
                        if alias in text_lower:
                            score += 1.0
                            matched_aliases.append(alias)
                    
                    if score > 0:
                        weighted_score = score * self.scene_weights.get(scene, 1.0) * 0.7  # 别名匹配权重较低
                        scene_scores[scene] = {
                            'score': weighted_score,
                            'base_score': score,
                            'keywords': matched_aliases[:3],
                            'match_type': 'alias'
                        }
        
        # 如果没有检测到任何场景
        if not scene_scores:
            return []
        
        # 按得分排序
        sorted_scenes = sorted(scene_scores.items(), key=lambda x: x[1]['score'], reverse=True)
        
        # 过滤低分场景
        filtered_scenes = []
        for scene, info in sorted_scenes:
            if info['score'] >= self.min_score_threshold:
                filtered_scenes.append((scene, info))
        
        # 限制返回数量
        result_scenes = filtered_scenes[:self.max_scenes]
        
        # 构建返回结果
        results = []
        for scene, info in result_scenes:
            results.append({
                'scene': scene,
                'score': float(info['score']),
                'confidence': self._calculate_confidence(info['score']),
                'keywords': info['keywords'],
                'match_type': info['match_type']
            })
        
        return results
    
    def _calculate_confidence(self, score: float) -> float:
        """
        根据得分计算置信度
        
        Args:
            score: 场景得分
            
        Returns:
            置信度 (0.0-1.0)
        """
        # 简单的线性映射，可根据需要调整
        max_score = max(self.scene_weights.values()) * 10  # 估计最大可能得分
        
        confidence = min(1.0, score / max_score * 2.0)  # *2.0 使置信度更敏感
        return confidence
    
    def classify_with_details(self, text: str) -> Dict[str, Any]:
        """
        获取详细的场景分类结果
        
        Args:
            text: 输入文本
            
        Returns:
            详细分类结果
        """
        scenes = self.classify_scene(text)
        
        result = {
            'text': text,
            'scenes': scenes,
            'primary_scene': None,
            'scene_count': len(scenes),
            'has_scene': len(scenes) > 0
        }
        
        if scenes:
            result['primary_scene'] = scenes[0]
            
            # 生成场景描述
            scene_descriptions = []
            for scene_info in scenes:
                scene_name = scene_info['scene']
                confidence = scene_info['confidence']
                
                if confidence > 0.7:
                    level = "高度相关"
                elif confidence > 0.4:
                    level = "相关"
                else:
                    level = "可能相关"
                
                if scene_info['keywords']:
                    keywords_str = '、'.join(scene_info['keywords'][:3])
                    scene_descriptions.append(f"{scene_name}({level}, 关键词: {keywords_str})")
                else:
                    scene_descriptions.append(f"{scene_name}({level})")
            
            result['scene_description'] = "; ".join(scene_descriptions)
        
        return result
    
    def add_custom_scene(self, scene_name: str, keywords: List[str], aliases: Optional[List[str]] = None):
        """
        添加自定义场景
        
        Args:
            scene_name: 场景名称
            keywords: 场景关键词列表
            aliases: 场景别名列表（可选）
        """
        self.scene_categories[scene_name] = keywords
        
        if aliases:
            self.scene_aliases[scene_name] = aliases
        
        # 重新计算权重
        self.scene_weights = self._calculate_scene_weights()
        
        logger.info(f"添加自定义场景: {scene_name} ({len(keywords)} 个关键词)")
    
    def save_scene_config(self, file_path: str = "configs/scene_categories.json") -> bool:
        """
        保存场景配置到文件
        
        Args:
            file_path: 配置文件路径
            
        Returns:
            保存是否成功
        """
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        config_data = {
            'scene_categories': self.scene_categories,
            'scene_aliases': self.scene_aliases,
            'scene_weights': self.scene_weights
        }
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, ensure_ascii=False, indent=2)
            logger.info(f"场景配置已保存到: {file_path}")
            return True
        except Exception as e:
            logger.error(f"保存场景配置失败: {e}")
            return False
    
    def load_scene_config(self, file_path: str = "configs/scene_categories.json") -> bool:
        """
        从文件加载场景配置
        
        Args:
            file_path: 配置文件路径
            
        Returns:
            加载是否成功
        """
        if not os.path.exists(file_path):
            logger.warning(f"场景配置文件不存在: {file_path}")
            return False
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            self.scene_categories = config_data.get('scene_categories', self.scene_categories)
            self.scene_aliases = config_data.get('scene_aliases', self.scene_aliases)
            self.scene_weights = config_data.get('scene_weights', self._calculate_scene_weights())
            
            logger.info(f"场景配置已加载，共 {len(self.scene_categories)} 个场景")
            return True
        except Exception as e:
            logger.error(f"加载场景配置失败: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取分类器统计信息
        
        Returns:
            统计信息字典
        """
        stats = {
            'total_scenes': len(self.scene_categories),
            'keywords_per_scene': {},
            'total_keywords': 0,
            'average_keywords_per_scene': 0
        }
        
        total_keywords = 0
        for scene, keywords in self.scene_categories.items():
            keyword_count = len(keywords)
            stats['keywords_per_scene'][scene] = keyword_count
            total_keywords += keyword_count
        
        stats['total_keywords'] = total_keywords
        if stats['total_scenes'] > 0:
            stats['average_keywords_per_scene'] = total_keywords / stats['total_scenes']
        
        return stats


# 单例模式
_scene_classifier = None

def get_scene_classifier(config_path: str = "configs/sentiment_config.yaml") -> SceneClassifier:
    """获取场景分类器单例"""
    global _scene_classifier
    if _scene_classifier is None:
        _scene_classifier = SceneClassifier(config_path)
    return _scene_classifier


if __name__ == "__main__":
    # 测试代码
    classifier = get_scene_classifier()
    
    test_cases = [
        "今天工作很忙，开了三个会，加班到很晚",
        "考试没考好，被老师批评了，心情很差",
        "周末和朋友去新开的餐厅吃饭，味道很不错",
        "最近总是失眠，去医院看了医生，开了些药",
        "双十一买了很多东西，快递都快收不过来了",
        "计划去三亚旅游，正在看机票和酒店"
    ]
    
    print("场景分类器测试:")
    print("-" * 70)
    
    for text in test_cases:
        print(f"\n文本: {text[:40]}...")
        
        scenes = classifier.classify_scene(text)
        
        if scenes:
            print("  检测到场景:")
            for scene_info in scenes:
                print(f"    • {scene_info['scene']}: {scene_info['confidence']:.2f}")
                if scene_info['keywords']:
                    print(f"      关键词: {', '.join(scene_info['keywords'])}")
        else:
            print("  未检测到明确场景")
        
        # 详细分类
        details = classifier.classify_with_details(text)
        if details['scene_description']:
            print(f"  场景描述: {details['scene_description']}")
    
    # 显示统计信息
    stats = classifier.get_statistics()
    print(f"\n分类器统计:")
    print(f"  场景总数: {stats['total_scenes']}")
    print(f"  总关键词数: {stats['total_keywords']}")
    print(f"  平均每个场景的关键词数: {stats['average_keywords_per_scene']:.1f}")