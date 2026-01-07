"""
心理健康分析器 - 分析文本中的心理健康特征
"""
import json
from typing import List, Dict, Tuple, Any, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MentalHealthAnalyzer:
    """心理健康分析器（重构版本，使用外部词典）"""
    
    def __init__(self, config_path: str = "configs/sentiment_config.yaml"):
        """
        初始化心理健康分析器
        
        Args:
            config_path: 配置文件路径
        """
        self.config = self._load_config(config_path)
        
        # 加载词典管理器
        from dictionary_manager import get_dictionary_manager
        self.dict_manager = get_dictionary_manager()
        
        # 加载心理健康词典
        self.mental_keywords = self.dict_manager.get_mental_health_keywords()
        
        # 类别权重配置
        self.category_weights = self.config.get('mental_health', {}).get('category_weights', {
            "stress": 2.0,        # 压力倾诉
            "emotional_distress": 2.5,  # 情绪困扰
            "positive_mindset": 1.8     # 积极心态
        })
        
        # 阈值配置
        self.thresholds = self.config.get('mental_health', {}).get('thresholds', {
            "high_confidence": 0.7,
            "medium_confidence": 0.5,
            "low_confidence": 0.3
        })
        
        # 类别描述
        self.category_descriptions = {
            "stress": {
                "name": "压力倾诉",
                "description": "表达工作、学习或生活压力的内容",
                "examples": ["压力大", "累死了", "工作压力", "扛不住"]
            },
            "emotional_distress": {
                "name": "情绪困扰",
                "description": "表达负面情绪和心理困扰的内容",
                "examples": ["心情不好", "焦虑", "抑郁", "失眠"]
            },
            "positive_mindset": {
                "name": "积极心态",
                "description": "表达积极态度和心理调适的内容",
                "examples": ["加油", "努力", "乐观", "坚持"]
            }
        }
        
        logger.info(f"心理健康分析器初始化完成，加载了 {sum(len(v) for v in self.mental_keywords.values())} 个关键词")
    
    def _load_config(self, config_path: str) -> Dict:
        """加载配置文件"""
        import yaml
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"配置文件 {config_path} 不存在，使用默认配置")
            return {'mental_health': {}}
    
    def analyze(self, text: str) -> Dict[str, Dict[str, Any]]:
        """
        分析文本的心理健康标签
        
        Args:
            text: 输入文本
            
        Returns:
            心理健康标签字典 {类别: {详情}}
        """
        results = {}
        text_lower = text
        
        for category, keywords in self.mental_keywords.items():
            if not keywords:
                continue
                
            score = 0
            matched_keywords = []
            
            for keyword in keywords:
                if keyword in text_lower:
                    # 计算权重（长关键词权重更高）
                    weight = 1.0 + len(keyword) * 0.1
                    score += weight
                    matched_keywords.append({
                        'keyword': keyword,
                        'weight': weight
                    })
            
            if score > 0:
                # 应用类别权重
                category_weight = self.category_weights.get(category, 1.0)
                weighted_score = score * category_weight
                
                # 计算置信度
                max_possible_score = len(keywords) * 2.0  # 假设每个关键词最大权重为2.0
                confidence = min(0.99, weighted_score / max_possible_score * 2.0)
                
                # 确定置信等级
                confidence_level = self._get_confidence_level(confidence)
                
                results[category] = {
                    "score": float(weighted_score),
                    "confidence": float(confidence),
                    "confidence_level": confidence_level,
                    "keywords": [kw['keyword'] for kw in matched_keywords],
                    "keyword_details": matched_keywords[:5],  # 只保留前5个详情
                    "category_info": self.category_descriptions.get(category, {})
                }
        
        # 排序并选择最相关的标签（最多2个）
        sorted_results = sorted(results.items(), key=lambda x: x[1]["score"], reverse=True)
        
        # 构建返回结果
        final_results = {}
        for category, info in sorted_results[:2]:  # 最多返回2个标签
            final_results[category] = info
        
        return final_results
    
    def _get_confidence_level(self, confidence: float) -> str:
        """根据置信度确定等级"""
        if confidence >= self.thresholds.get("high_confidence", 0.7):
            return "高"
        elif confidence >= self.thresholds.get("medium_confidence", 0.5):
            return "中"
        elif confidence >= self.thresholds.get("low_confidence", 0.3):
            return "低"
        else:
            return "很低"
    
    def get_detailed_analysis(self, text: str) -> Dict[str, Any]:
        """
        获取详细的心理健康分析
        
        Args:
            text: 输入文本
            
        Returns:
            详细分析结果
        """
        analysis = self.analyze(text)
        
        result = {
            'text': text,
            'mental_health_labels': analysis,
            'has_concerns': len(analysis) > 0,
            'primary_concern': None,
            'recommendation': None
        }
        
        if not analysis:
            result['summary'] = "未检测到明显的心理健康特征"
            result['recommendation'] = "情绪状态较为平稳，继续保持"
            return result
        
        # 确定主要关注点
        if analysis:
            primary_category = max(analysis.items(), key=lambda x: x[1]['score'])
            result['primary_concern'] = {
                'category': primary_category[0],
                'name': self.category_descriptions.get(primary_category[0], {}).get('name', primary_category[0]),
                'confidence': primary_category[1]['confidence']
            }
        
        # 生成摘要
        labels_summary = []
        for category, info in analysis.items():
            category_name = self.category_descriptions.get(category, {}).get('name', category)
            confidence = info['confidence']
            keywords = info['keywords'][:2] if info['keywords'] else []
            
            if keywords:
                keywords_str = '、'.join(keywords)
                labels_summary.append(f"{category_name}({confidence:.2f}, 关键词: {keywords_str})")
            else:
                labels_summary.append(f"{category_name}({confidence:.2f})")
        
        result['summary'] = "检测到: " + "; ".join(labels_summary)
        
        # 生成建议
        result['recommendation'] = self._generate_recommendation(analysis)
        
        return result
    
    def _generate_recommendation(self, analysis: Dict) -> str:
        """根据分析结果生成建议"""
        if not analysis:
            return "情绪状态平稳，建议保持积极心态，适当放松。"
        
        recommendations = []
        
        # 检查各个类别
        if 'stress' in analysis:
            stress_conf = analysis['stress']['confidence']
            if stress_conf > 0.6:
                recommendations.append("检测到较高压力，建议适当休息、进行放松训练或寻求支持。")
            elif stress_conf > 0.3:
                recommendations.append("检测到一定压力，建议合理安排时间，注意工作生活平衡。")
        
        if 'emotional_distress' in analysis:
            distress_conf = analysis['emotional_distress']['confidence']
            if distress_conf > 0.6:
                recommendations.append("检测到明显情绪困扰，建议与亲友倾诉或考虑专业心理支持。")
            elif distress_conf > 0.3:
                recommendations.append("检测到一些情绪波动，建议进行情绪调节练习，如正念冥想。")
        
        if 'positive_mindset' in analysis:
            positive_conf = analysis['positive_mindset']['confidence']
            if positive_conf > 0.6:
                recommendations.append("检测到积极心态，继续保持，这种态度有助于应对挑战。")
            elif positive_conf > 0.3:
                recommendations.append("检测到积极倾向，可以进一步培养乐观思维。")
        
        # 如果没有具体建议，提供通用建议
        if not recommendations:
            if len(analysis) == 1:
                category = list(analysis.keys())[0]
                category_name = self.category_descriptions.get(category, {}).get('name', category)
                recommendations.append(f"检测到{category_name}特征，建议关注情绪健康，适当调节。")
            else:
                recommendations.append("检测到多种心理健康特征，建议综合关注情绪状态。")
        
        return " ".join(recommendations)
    
    def detect_crisis_signals(self, text: str) -> List[Dict[str, Any]]:
        """
        检测危机信号（需要紧急关注的内容）
        
        Args:
            text: 输入文本
            
        Returns:
            危机信号列表
        """
        crisis_keywords = {
            "自杀倾向": ["想死", "不想活了", "活够了", "自杀", "自残", "结束生命"],
            "严重抑郁": ["绝望", "无助", "毫无希望", "人生无意义", "重度抑郁"],
            "严重焦虑": ["惊恐发作", "崩溃", "失控", "要疯了", "极度恐惧"],
            "暴力倾向": ["想杀人", "报复", "毁灭", "同归于尽"]
        }
        
        crisis_signals = []
        
        for signal_type, keywords in crisis_keywords.items():
            matched = []
            for keyword in keywords:
                if keyword in text:
                    matched.append(keyword)
            
            if matched:
                crisis_signals.append({
                    'type': signal_type,
                    'keywords': matched,
                    'severity': 'high',
                    'recommendation': f"检测到{signal_type}表述，建议立即寻求专业帮助或联系危机干预热线。"
                })
        
        return crisis_signals
    
    def analyze_emotional_intensity(self, text: str) -> Dict[str, Any]:
        """
        分析情绪强度
        
        Args:
            text: 输入文本
            
        Returns:
            情绪强度分析
        """
        # 强度修饰词
        intensifiers = ["非常", "特别", "极其", "十分", "超级", "太", "真", "确实", "真的", "格外", "异常", "极度", "极度地"]
        
        # 检测强度修饰词
        intensity_score = 0
        found_intensifiers = []
        
        for intensifier in intensifiers:
            if intensifier in text:
                intensity_score += 1
                found_intensifiers.append(intensifier)
        
        # 确定强度等级
        if intensity_score >= 3:
            intensity_level = "极高"
        elif intensity_score >= 2:
            intensity_level = "高"
        elif intensity_score >= 1:
            intensity_level = "中等"
        else:
            intensity_level = "正常"
        
        return {
            'intensity_score': intensity_score,
            'intensity_level': intensity_level,
            'intensifiers': found_intensifiers
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取分析器统计信息
        
        Returns:
            统计信息字典
        """
        stats = {
            'total_categories': len(self.mental_keywords),
            'keywords_per_category': {},
            'category_weights': self.category_weights,
            'thresholds': self.thresholds
        }
        
        for category, keywords in self.mental_keywords.items():
            stats['keywords_per_category'][category] = len(keywords)
        
        stats['total_keywords'] = sum(stats['keywords_per_category'].values())
        
        return stats


# 单例模式
_mental_analyzer = None

def get_mental_health_analyzer(config_path: str = "configs/sentiment_config.yaml") -> MentalHealthAnalyzer:
    """获取心理健康分析器单例"""
    global _mental_analyzer
    if _mental_analyzer is None:
        _mental_analyzer = MentalHealthAnalyzer(config_path)
    return _mental_analyzer


if __name__ == "__main__":
    # 测试代码
    analyzer = get_mental_health_analyzer()
    
    test_cases = [
        "最近工作压力好大，天天加班到很晚，快撑不住了",
        "心情一直很低落，对什么都提不起兴趣，晚上也睡不着",
        "虽然遇到困难，但我要加油坚持，相信会越来越好的",
        "今天很开心，和朋友出去玩得很愉快",
        "最近总是焦虑不安，担心各种事情，感觉要崩溃了"
    ]
    
    print("心理健康分析器测试:")
    print("-" * 70)
    
    for text in test_cases:
        print(f"\n文本: {text[:40]}...")
        
        # 基础分析
        analysis = analyzer.analyze(text)
        
        if analysis:
            print("  检测到标签:")
            for category, info in analysis.items():
                category_name = analyzer.category_descriptions.get(category, {}).get('name', category)
                print(f"    • {category_name}: {info['confidence']:.2f} ({info['confidence_level']})")
                if info['keywords']:
                    print(f"      关键词: {', '.join(info['keywords'][:3])}")
        else:
            print("  未检测到明显的心理健康特征")
        
        # 详细分析
        details = analyzer.get_detailed_analysis(text)
        print(f"  建议: {details['recommendation']}")
        
        # 情绪强度分析
        intensity = analyzer.analyze_emotional_intensity(text)
        print(f"  情绪强度: {intensity['intensity_level']} (分数: {intensity['intensity_score']})")