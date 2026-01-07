"""
词典管理器 - 统一加载和管理所有外部词典
"""
import json
import yaml
import os
from typing import Dict, List, Any, Optional
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DictionaryManager:
    """统一词典管理器"""
    
    def __init__(self, config_path: str = "configs/paths_config.yaml"):
        """
        初始化词典管理器
        
        Args:
            config_path: 路径配置文件路径
        """
        self.config = self._load_config(config_path)
        self.dictionaries: Dict[str, Dict] = {}
        self.loaded = False
    
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
            return {
                'dictionary_paths': {
                    'sentiment': 'dictionaries/sentiment',
                    'slang': 'dictionaries/slang',
                    'mental_health': 'dictionaries/mental_health'
                }
            }
    
    def load_all_dictionaries(self) -> Dict[str, Dict]:
        """
        加载所有词典
        
        Returns:
            所有词典的字典
        """
        if self.loaded:
            return self.dictionaries
        
        logger.info("开始加载所有词典...")
        
        # 加载情感词典
        self._load_sentiment_dictionaries()
        
        # 加载网络用语词典
        self._load_slang_dictionaries()
        
        # 加载心理健康词典
        self._load_mental_health_dictionaries()
        
        self.loaded = True
        logger.info(f"词典加载完成，共加载 {len(self.dictionaries)} 类词典")
        return self.dictionaries
    
    def _load_sentiment_dictionaries(self):
        """加载情感词典"""
        sentiment_path = self.config.get('dictionary_paths', {}).get('sentiment', 'dictionaries/sentiment')
        
        sentiment_files = {
            'positive_keywords': 'positive_keywords.json',
            'negative_keywords': 'negative_keywords.json',
            'negation_patterns': 'negation_patterns.json'
        }
        
        for key, filename in sentiment_files.items():
            file_path = os.path.join(sentiment_path, filename)
            if os.path.exists(file_path):
                self.dictionaries[f'sentiment_{key}'] = self.load_json(file_path)
                logger.debug(f"加载情感词典: {key}")
            else:
                logger.warning(f"情感词典文件不存在: {file_path}")
                self.dictionaries[f'sentiment_{key}'] = {}
    
    def _load_slang_dictionaries(self):
        """加载网络用语词典"""
        slang_path = self.config.get('dictionary_paths', {}).get('slang', 'dictionaries/slang')
        
        slang_files = {
            'positive_slang': 'positive_slang.json',
            'negative_slang': 'negative_slang.json'
        }
        
        for key, filename in slang_files.items():
            file_path = os.path.join(slang_path, filename)
            if os.path.exists(file_path):
                self.dictionaries[f'slang_{key}'] = self.load_json(file_path)
                logger.debug(f"加载网络用语词典: {key}")
            else:
                logger.warning(f"网络用语词典文件不存在: {file_path}")
                self.dictionaries[f'slang_{key}'] = {}
    
    def _load_mental_health_dictionaries(self):
        """加载心理健康词典"""
        mental_health_path = self.config.get('dictionary_paths', {}).get('mental_health', 'dictionaries/mental_health')
        
        mental_health_files = {
            'stress': 'stress.json',
            'emotional_distress': 'emotional_distress.json',
            'positive_mindset': 'positive_mindset.json'
        }
        
        for key, filename in mental_health_files.items():
            file_path = os.path.join(mental_health_path, filename)
            if os.path.exists(file_path):
                self.dictionaries[f'mental_health_{key}'] = self.load_json(file_path)
                logger.debug(f"加载心理健康词典: {key}")
            else:
                logger.warning(f"心理健康词典文件不存在: {file_path}")
                self.dictionaries[f'mental_health_{key}'] = {}
    
    def load_json(self, file_path: str) -> Dict:
        """加载JSON文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"加载JSON文件失败 {file_path}: {e}")
            return {}
    
    def get_positive_keywords(self) -> List[str]:
        """获取正面关键词列表"""
        keywords = []
        if 'sentiment_positive_keywords' in self.dictionaries:
            data = self.dictionaries['sentiment_positive_keywords']
            if isinstance(data, dict):
                for category, words in data.items():
                    if isinstance(words, list):
                        keywords.extend(words)
            elif isinstance(data, list):
                keywords.extend(data)
        return list(set(keywords))
    
    def get_negative_keywords(self) -> List[str]:
        """获取负面关键词列表"""
        keywords = []
        if 'sentiment_negative_keywords' in self.dictionaries:
            data = self.dictionaries['sentiment_negative_keywords']
            if isinstance(data, dict):
                for category, words in data.items():
                    if isinstance(words, list):
                        keywords.extend(words)
            elif isinstance(data, list):
                keywords.extend(data)
        return list(set(keywords))
    
    def get_negation_patterns(self) -> List[str]:
        """获取否定模式列表"""
        patterns = []
        if 'sentiment_negation_patterns' in self.dictionaries:
            data = self.dictionaries['sentiment_negation_patterns']
            if isinstance(data, dict):
                for category, items in data.items():
                    if isinstance(items, list):
                        patterns.extend(items)
            elif isinstance(data, list):
                patterns.extend(data)
        return list(set(patterns))
    
    def get_slang_keywords(self) -> Dict[str, List[str]]:
        """获取网络用语关键词"""
        slang_dict = {}
        if 'slang_positive_slang' in self.dictionaries:
            slang_dict['positive'] = self._extract_keywords_from_dict(self.dictionaries['slang_positive_slang'])
        if 'slang_negative_slang' in self.dictionaries:
            slang_dict['negative'] = self._extract_keywords_from_dict(self.dictionaries['slang_negative_slang'])
        return slang_dict
    
    def get_mental_health_keywords(self) -> Dict[str, List[str]]:
        """获取心理健康关键词"""
        mental_dict = {}
        if 'mental_health_stress' in self.dictionaries:
            mental_dict['stress'] = self._extract_keywords_from_list(self.dictionaries['mental_health_stress'])
        if 'mental_health_emotional_distress' in self.dictionaries:
            mental_dict['emotional_distress'] = self._extract_keywords_from_list(self.dictionaries['mental_health_emotional_distress'])
        if 'mental_health_positive_mindset' in self.dictionaries:
            mental_dict['positive_mindset'] = self._extract_keywords_from_list(self.dictionaries['mental_health_positive_mindset'])
        return mental_dict
    
    def _extract_keywords_from_dict(self, data: Dict) -> List[str]:
        """从字典中提取关键词"""
        keywords = []
        if isinstance(data, dict):
            for item in data.values():
                if isinstance(item, dict) and 'keywords' in item:
                    keywords.extend(item['keywords'])
                elif isinstance(item, list):
                    keywords.extend(item)
        return list(set(keywords))
    
    def _extract_keywords_from_list(self, data: Any) -> List[str]:
        """从列表中提取关键词"""
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            return list(data.keys()) if data else []
        return []
    
    def update_dictionary(self, category: str, key: str, words: List[str]):
        """
        更新词典
        
        Args:
            category: 词典类别
            key: 词典键名
            words: 关键词列表
        """
        dict_key = f"{category}_{key}"
        if dict_key in self.dictionaries:
            if isinstance(self.dictionaries[dict_key], dict):
                self.dictionaries[dict_key]['keywords'] = words
            elif isinstance(self.dictionaries[dict_key], list):
                self.dictionaries[dict_key] = words
        else:
            self.dictionaries[dict_key] = words
    
    def save_dictionary(self, category: str, key: str):
        """
        保存词典到文件
        
        Args:
            category: 词典类别
            key: 词典键名
        """
        dict_key = f"{category}_{key}"
        if dict_key not in self.dictionaries:
            logger.error(f"词典不存在: {dict_key}")
            return False
        
        # 确定文件路径
        if category == 'sentiment':
            base_path = self.config.get('dictionary_paths', {}).get('sentiment', 'dictionaries/sentiment')
        elif category == 'slang':
            base_path = self.config.get('dictionary_paths', {}).get('slang', 'dictionaries/slang')
        elif category == 'mental_health':
            base_path = self.config.get('dictionary_paths', {}).get('mental_health', 'dictionaries/mental_health')
        else:
            base_path = 'dictionaries'
        
        file_path = os.path.join(base_path, f"{key}.json")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.dictionaries[dict_key], f, ensure_ascii=False, indent=2)
            logger.info(f"词典已保存到: {file_path}")
            return True
        except Exception as e:
            logger.error(f"保存词典失败 {file_path}: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取词典统计信息"""
        stats = {}
        for key, data in self.dictionaries.items():
            if isinstance(data, dict):
                # 如果是嵌套字典（如网络用语词典）
                if 'keywords' in data:
                    stats[key] = len(data['keywords'])
                else:
                    # 统计所有嵌套列表的总词数
                    total = 0
                    for sub_key, sub_data in data.items():
                        if isinstance(sub_data, list):
                            total += len(sub_data)
                    stats[key] = total
            elif isinstance(data, list):
                stats[key] = len(data)
            else:
                stats[key] = 0
        
        stats['total_dictionaries'] = len(self.dictionaries)
        stats['total_keywords'] = sum(stats.values())
        
        return stats


# 单例模式
_dict_manager = None

def get_dictionary_manager(config_path: str = "configs/paths_config.yaml") -> DictionaryManager:
    """获取词典管理器单例"""
    global _dict_manager
    if _dict_manager is None:
        _dict_manager = DictionaryManager(config_path)
        _dict_manager.load_all_dictionaries()
    return _dict_manager


if __name__ == "__main__":
    # 测试代码
    manager = get_dictionary_manager()
    stats = manager.get_statistics()
    
    print("词典统计信息:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print(f"\n正面关键词数量: {len(manager.get_positive_keywords())}")
    print(f"负面关键词数量: {len(manager.get_negative_keywords())}")
    print(f"否定模式数量: {len(manager.get_negation_patterns())}")