"""
情感增强器 - 基于规则和多策略增强情感分析准确性
"""
import re
import json
from typing import List, Dict, Tuple, Any, Optional
import logging
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SentimentEnhancer:
    """情感增强器（重构版本，使用外部词典）"""
    
    def __init__(self, config_path: str = "configs/sentiment_config.yaml"):
        """
        初始化情感增强器
        
        Args:
            config_path: 配置文件路径
        """
        self.config = self._load_config(config_path)
        
        # 加载词典管理器
        from dictionary_manager import get_dictionary_manager
        self.dict_manager = get_dictionary_manager()
        
        # 加载词典
        self.positive_keywords = set(self.dict_manager.get_positive_keywords())
        self.negative_keywords = set(self.dict_manager.get_negative_keywords())
        self.negation_patterns = set(self.dict_manager.get_negation_patterns())
        self.slang_keywords = self.dict_manager.get_slang_keywords()
        self.mental_keywords = self.dict_manager.get_mental_health_keywords()
        
        # 情感修饰词
        self.intensifiers = {
            "positive": ["非常", "特别", "极其", "十分", "超级", "太", "真", "确实", "真的", "格外", "异常"],
            "negative": ["非常", "特别", "极其", "十分", "超级", "太", "真", "确实", "真的", "格外", "异常"]
        }
        
        # 否定词
        self.negation_words = ["不", "没", "无", "非", "未", "勿", "别", "莫", "休", "甭", "休想", "绝不"]
        
        # 双重否定词
        self.double_negation_words = ["不是不", "不能不", "不会不", "不可不", "不得不", "未必不", "未必没"]
        
        # 反讽模式
        self.irony_patterns = [
            ("真是", "好"),
            ("可真", "行"),
            ("太", "了"),
            ("真是", "不错"),
            ("够", "的"),
            ("真够", "的"),
            ("也太", "了吧")
        ]
        
        # 复合模式识别
        self.compound_patterns = {
            # 考试相关负面模式
            "exam_negative": [
                ("考试", "没考好"),
                ("考试", "考砸"),
                ("考试", "不及格"),
                ("考试", "挂科"),
                ("被老师", "批评"),
                ("被老师", "说"),
                ("被老师", "骂")
            ],
            # 工作压力模式
            "work_stress": [
                ("工作", "压力"),
                ("加班", "累"),
                ("工作", "疲惫"),
                ("项目", "困难")
            ],
            # 情绪低落模式
            "depression": [
                ("心情", "不好"),
                ("情绪", "低落"),
                ("没有", "兴趣"),
                ("提不起", "劲")
            ]
        }
        
        # 从配置加载权重
        self.weights = self.config.get('enhancer', {}).get('weights', {})
        self.thresholds = self.config.get('enhancer', {}).get('thresholds', {})
        
        logger.info(f"情感增强器初始化完成，加载了 {len(self.positive_keywords)} 个正面词和 {len(self.negative_keywords)} 个负面词")
    
    def _load_config(self, config_path: str) -> Dict:
        """加载配置文件"""
        import yaml
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"配置文件 {config_path} 不存在，使用默认配置")
            return {
                'enhancer': {
                    'weights': {
                        'keyword_positive': 1.5,
                        'keyword_negative': 2.0,
                        'pattern_positive': 1.8,
                        'pattern_negative': 2.5,
                        'depression_pattern': 4.0,
                        'compound_pattern': 3.0,
                        'slang_positive': 1.2,
                        'slang_negative': 1.5
                    },
                    'thresholds': {
                        'confidence_high': 0.85,
                        'confidence_medium': 0.7,
                        'confidence_low': 0.5
                    }
                }
            }
    
    def analyze_sentiment_structure(self, text: str) -> Dict[str, Any]:
        """
        分析文本的情感结构
        
        Args:
            text: 输入文本
            
        Returns:
            情感结构分析结果
        """
        result = {
            "has_negation": False,
            "has_double_negation": False,
            "specific_neg_count": 0,
            "depression_count": 0,
            "compound_patterns": [],
            "intensity_score": 0,
            "irony_score": 0
        }
        
        # 检测否定词
        for word in self.negation_words:
            if word in text:
                result["has_negation"] = True
                break
        
        # 检测双重否定
        for pattern in self.double_negation_words:
            if pattern in text:
                result["has_double_negation"] = True
                break
        
        # 检测特定否定模式
        for pattern in self.negation_patterns:
            if pattern in text:
                result["specific_neg_count"] += 1
        
        # 检测复合模式
        for pattern_type, patterns in self.compound_patterns.items():
            for pattern in patterns:
                if pattern[0] in text and pattern[1] in text:
                    result["compound_patterns"].append(pattern_type)
        
        # 检测情感强度修饰
        for intensifier in self.intensifiers["positive"]:
            if intensifier in text:
                result["intensity_score"] += 1
        
        # 检测可能的反讽
        for pattern in self.irony_patterns:
            if pattern[0] in text and pattern[1] in text:
                result["irony_score"] += 1
        
        return result
    
    def analyze_keywords(self, text: str) -> Tuple[int, int, List[Dict]]:
        """
        分析文本中的关键词
        
        Args:
            text: 输入文本
            
        Returns:
            (正面分数, 负面分数, 关键词详情列表)
        """
        pos_score = 0
        neg_score = 0
        keyword_details = []
        
        # 检查正面关键词
        for keyword in self.positive_keywords:
            if keyword in text:
                weight = self.weights.get('keyword_positive', 1.5)
                score = weight * (1.0 + text.count(keyword) * 0.1)
                pos_score += score
                keyword_details.append({
                    'keyword': keyword,
                    'score': score,
                    'sentiment': 'positive',
                    'type': 'keyword'
                })
        
        # 检查负面关键词
        for keyword in self.negative_keywords:
            if keyword in text:
                weight = self.weights.get('keyword_negative', 2.0)
                score = weight * (1.0 + text.count(keyword) * 0.1)
                neg_score += score
                keyword_details.append({
                    'keyword': keyword,
                    'score': score,
                    'sentiment': 'negative',
                    'type': 'keyword'
                })
        
        # 检查网络用语
        for sentiment, slang_dict in self.slang_keywords.items():
            for keyword in slang_dict:
                if keyword in text:
                    weight_key = f'slang_{sentiment}'
                    weight = self.weights.get(weight_key, 1.2 if sentiment == 'positive' else 1.5)
                    score = weight * 2.0  # 网络用语权重更高
                    
                    if sentiment == 'positive':
                        pos_score += score
                    else:
                        neg_score += score
                    
                    keyword_details.append({
                        'keyword': keyword,
                        'score': score,
                        'sentiment': sentiment,
                        'type': 'slang'
                    })
        
        # 检查复合模式
        structure = self.analyze_sentiment_structure(text)
        for pattern_type in structure['compound_patterns']:
            if 'negative' in pattern_type or 'stress' in pattern_type or 'depression' in pattern_type:
                weight = self.weights.get('compound_pattern', 3.0)
                neg_score += weight
                keyword_details.append({
                    'keyword': pattern_type,
                    'score': weight,
                    'sentiment': 'negative',
                    'type': 'compound_pattern'
                })
        
        # 检查心理健康关键词
        for category, keywords in self.mental_keywords.items():
            for keyword in keywords:
                if keyword in text:
                    if category == 'positive_mindset':
                        pos_score += 2.0
                        keyword_details.append({
                            'keyword': keyword,
                            'score': 2.0,
                            'sentiment': 'positive',
                            'type': 'mental_health'
                        })
                    else:  # stress 或 emotional_distress
                        weight = self.weights.get('depression_pattern', 4.0) if category == 'emotional_distress' else 2.5
                        neg_score += weight
                        keyword_details.append({
                            'keyword': keyword,
                            'score': weight,
                            'sentiment': 'negative',
                            'type': 'mental_health'
                        })
        
        return pos_score, neg_score, keyword_details
    
    def analyze_negation_context(self, text: str) -> List[Tuple[int, str]]:
        """
        分析否定词的上下文
        
        Args:
            text: 输入文本
            
        Returns:
            否定词位置和上下文的列表
        """
        words = list(text)
        negation_indices = []
        
        for i, word in enumerate(words):
            if word in self.negation_words:
                # 检查否定词附近的词语（前后各2个字）
                context_start = max(0, i - 2)
                context_end = min(len(words), i + 3)
                context = ''.join(words[context_start:context_end])
                negation_indices.append((i, context))
        
        return negation_indices
    
    def detect_negated_sentiment(self, text: str) -> Optional[str]:
        """
        检测否定修饰的情感
        
        Args:
            text: 输入文本
            
        Returns:
            被否定的情感类型，或None
        """
        negation_contexts = self.analyze_negation_context(text)
        
        for pos, context in negation_contexts:
            # 检查否定词附近是否有情感词
            for positive_word in self.positive_keywords:
                if positive_word in context:
                    # 检查否定词是否在情感词前面（否定修饰）
                    pos_in_context = context.find(positive_word)
                    neg_in_context = context.find(self._find_negation_word(context))
                    
                    if neg_in_context < pos_in_context and pos_in_context - neg_in_context <= 3:
                        return 'positive'
            
            for negative_word in self.negative_keywords:
                if negative_word in context:
                    # 检查否定词是否在情感词前面（否定修饰）
                    pos_in_context = context.find(negative_word)
                    neg_in_context = context.find(self._find_negation_word(context))
                    
                    if neg_in_context < pos_in_context and pos_in_context - neg_in_context <= 3:
                        return 'negative'
        
        return None
    
    def _find_negation_word(self, text: str) -> str:
        """在文本中查找否定词"""
        for word in self.negation_words:
            if word in text:
                return word
        return ""
    
    def enhance_prediction(self, text: str, model_emotion: str, model_confidence: float) -> Tuple[str, float]:
        """
        增强模型预测结果
        
        Args:
            text: 输入文本
            model_emotion: 模型预测的情感
            model_confidence: 模型预测的置信度
            
        Returns:
            (最终情感, 增强后置信度)
        """
        # 1. 关键词分析
        pos_score, neg_score, keyword_details = self.analyze_keywords(text)
        
        # 2. 结构分析
        structure = self.analyze_sentiment_structure(text)
        
        # 3. 否定上下文分析
        negated_sentiment = self.detect_negated_sentiment(text)
        
        # 4. 初始设置
        final_emotion = model_emotion
        final_confidence = model_confidence
        
        # 规则1：复合模式强烈指示（如"考试没考好"）
        if structure['compound_patterns']:
            # 如果是负面复合模式
            if any(p in structure['compound_patterns'] for p in ['exam_negative', 'work_stress', 'depression']):
                if model_emotion == "正面":
                    # 强烈修正为负面
                    final_confidence = max(model_confidence * 0.1, self.thresholds.get('confidence_high', 0.85))
                    return "负面", final_confidence
                else:
                    # 增强负面置信度
                    final_confidence = min(1.0, model_confidence * 1.3)
                    return "负面", final_confidence
        
        # 规则2：心理健康标签强烈指示
        mental_keywords_found = [k for k in keyword_details if k['type'] == 'mental_health']
        if mental_keywords_found:
            # 统计心理健康关键词的情感倾向
            mental_pos = sum(1 for k in mental_keywords_found if k['sentiment'] == 'positive')
            mental_neg = sum(1 for k in mental_keywords_found if k['sentiment'] == 'negative')
            
            if mental_neg > mental_pos * 2:  # 负面心理健康特征明显
                if model_emotion == "正面":
                    final_confidence = max(model_confidence * 0.15, self.thresholds.get('confidence_high', 0.85))
                    return "负面", final_confidence
            elif mental_pos > mental_neg * 2:  # 正面心理健康特征明显
                if model_emotion == "负面":
                    final_confidence = max(model_confidence * 0.2, self.thresholds.get('confidence_medium', 0.7))
                    return "正面", final_confidence
        
        # 规则3：强烈的负面关键词
        if neg_score > pos_score * 2 and neg_score > 3.0:
            if model_emotion == "正面":
                final_confidence = max(model_confidence * 0.2, self.thresholds.get('confidence_medium', 0.7))
                return "负面", final_confidence
            elif model_emotion == "负面":
                final_confidence = min(1.0, model_confidence * 1.2)
        
        # 规则4：强烈的正面关键词
        elif pos_score > neg_score * 2 and pos_score > 3.0:
            if model_emotion == "负面":
                final_confidence = max(model_confidence * 0.2, self.thresholds.get('confidence_medium', 0.7))
                return "正面", final_confidence
            elif model_emotion == "正面":
                final_confidence = min(1.0, model_confidence * 1.2)
        
        # 规则5：否定修饰情感
        if negated_sentiment:
            if negated_sentiment == 'positive' and model_emotion == "正面":
                # "不+正面词" -> 负面
                final_confidence = max(model_confidence * 0.3, self.thresholds.get('confidence_low', 0.5))
                return "负面", final_confidence
            elif negated_sentiment == 'negative' and model_emotion == "负面":
                # "不+负面词" -> 正面（需要谨慎，可能仍是负面）
                # 例如"不舒服"中的"不"不是否定修饰
                if "不舒服" not in text and "不开心" not in text:
                    final_confidence = max(model_confidence * 0.4, self.thresholds.get('confidence_low', 0.5))
                    return "正面", final_confidence
        
        # 规则6：双重否定 -> 正面
        if structure['has_double_negation']:
            if model_emotion == "负面":
                final_confidence = max(model_confidence * 0.4, self.thresholds.get('confidence_low', 0.5))
                return "正面", final_confidence
        
        # 规则7：强度修饰增强置信度
        if structure['intensity_score'] > 0:
            final_confidence = min(1.0, model_confidence * (1 + structure['intensity_score'] * 0.05))
        
        # 规则8：反讽模式调整
        if structure['irony_score'] > 0 and pos_score > 0:
            # 可能是反讽，降低正面置信度
            if model_emotion == "正面":
                final_confidence = model_confidence * 0.7
        
        # 规则9：特定否定模式（如"心情不好"）
        if structure['specific_neg_count'] > 0:
            if model_emotion == "正面":
                final_confidence = max(model_confidence * 0.25, self.thresholds.get('confidence_medium', 0.7))
                return "负面", final_confidence
            elif model_emotion == "负面":
                final_confidence = min(1.0, model_confidence * 1.15)
        
        return final_emotion, final_confidence
    
    def get_detailed_analysis(self, text: str, model_emotion: str, model_confidence: float) -> Dict[str, Any]:
        """
        获取详细的情感分析
        
        Args:
            text: 输入文本
            model_emotion: 模型预测的情感
            model_confidence: 模型预测的置信度
            
        Returns:
            详细分析结果
        """
        # 基础分析
        pos_score, neg_score, keyword_details = self.analyze_keywords(text)
        structure = self.analyze_sentiment_structure(text)
        negated_sentiment = self.detect_negated_sentiment(text)
        
        # 增强预测
        final_emotion, final_confidence = self.enhance_prediction(text, model_emotion, model_confidence)
        
        # 构建结果
        result = {
            'text': text,
            'model_prediction': {
                'emotion': model_emotion,
                'confidence': float(model_confidence)
            },
            'enhanced_prediction': {
                'emotion': final_emotion,
                'confidence': float(final_confidence)
            },
            'keyword_analysis': {
                'positive_score': float(pos_score),
                'negative_score': float(neg_score),
                'total_keywords': len(keyword_details),
                'keywords': keyword_details[:10]  # 只显示前10个
            },
            'structure_analysis': structure,
            'negation_analysis': {
                'has_negation': structure['has_negation'],
                'negated_sentiment': negated_sentiment,
                'double_negation': structure['has_double_negation']
            },
            'confidence_level': self._get_confidence_level(final_confidence),
            'is_corrected': model_emotion != final_emotion
        }
        
        return result
    
    def _get_confidence_level(self, confidence: float) -> str:
        """根据置信度确定等级"""
        if confidence >= self.thresholds.get('confidence_high', 0.85):
            return "high"
        elif confidence >= self.thresholds.get('confidence_medium', 0.7):
            return "medium"
        elif confidence >= self.thresholds.get('confidence_low', 0.5):
            return "low"
        else:
            return "very_low"
    
    def add_custom_keyword(self, keyword: str, sentiment: str, keyword_type: str = "custom"):
        """
        添加自定义关键词
        
        Args:
            keyword: 关键词
            sentiment: 情感倾向 ("positive" 或 "negative")
            keyword_type: 关键词类型
        """
        if sentiment == "positive":
            self.positive_keywords.add(keyword)
        elif sentiment == "negative":
            self.negative_keywords.add(keyword)
        
        logger.info(f"添加自定义关键词: {keyword} ({sentiment})")


# 单例模式
_enhancer = None

def get_sentiment_enhancer(config_path: str = "configs/sentiment_config.yaml") -> SentimentEnhancer:
    """获取情感增强器单例"""
    global _enhancer
    if _enhancer is None:
        _enhancer = SentimentEnhancer(config_path)
    return _enhancer


if __name__ == "__main__":
    # 测试代码
    enhancer = get_sentiment_enhancer()
    
    test_cases = [
        ("我今天很难过，考试没考好，被老师说了", "负面", 0.7),
        ("这个产品真的很好用，我非常满意", "正面", 0.8),
        ("我不是很开心，因为工作压力太大了", "正面", 0.6),  # 模型误判
        ("兴趣不高，什么都不想做", "正面", 0.65),  # 模型误判
        ("虽然很难，但我要加油坚持", "负面", 0.7),  # 模型误判
    ]
    
    print("情感增强器测试:")
    print("-" * 70)
    
    for text, model_emotion, model_confidence in test_cases:
        final_emotion, final_confidence = enhancer.enhance_prediction(text, model_emotion, model_confidence)
        
        corrected = "✅" if model_emotion != final_emotion else "➡️"
        
        print(f"文本: {text[:30]}...")
        print(f"  模型: {model_emotion} ({model_confidence:.3f})")
        print(f"  增强: {final_emotion} ({final_confidence:.3f}) {corrected}")
        
        # 详细分析
        if model_emotion != final_emotion:
            details = enhancer.get_detailed_analysis(text, model_emotion, model_confidence)
            print(f"  修正原因: {len(details['keyword_analysis']['keywords'])} 个关键词匹配")
            for kw in details['keyword_analysis']['keywords'][:3]:
                print(f"    {kw['keyword']}: {kw['score']:.2f} ({kw['sentiment']})")
        
        print()