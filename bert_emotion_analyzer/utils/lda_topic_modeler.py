"""
LDA主题建模器 - 基于主题模型分析文本
"""
import pickle
import numpy as np
from gensim import corpora, models
from gensim.models.ldamodel import LdaModel
from gensim.models.coherencemodel import CoherenceModel
import jieba
import json
import os
from typing import List, Dict, Tuple, Any, Optional
import logging
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LDATopicModeler:
    """LDA主题建模器"""
    
    def __init__(self, model_path: str = "models/lda_model.pkl",
                 config_path: str = "configs/model_config.json"):
        """
        初始化LDA主题建模器
        
        Args:
            model_path: 模型保存路径
            config_path: 配置文件路径
        """
        self.model_path = model_path
        self.config = self._load_config(config_path)
        
        self.lda_model = None
        self.dictionary = None
        self.corpus = None
        self.topics = {}
        self.is_trained = False
        
        # 从配置获取参数
        lda_config = self.config.get('lda', {})
        self.num_topics = lda_config.get('num_topics', 10)
        self.passes = lda_config.get('passes', 10)
        self.iterations = lda_config.get('iterations', 50)
        self.random_state = lda_config.get('random_state', 42)
        self.no_below = lda_config.get('no_below', 2)
        self.no_above = lda_config.get('no_above', 0.5)
        self.alpha = lda_config.get('alpha', 'auto')
        self.eta = lda_config.get('eta', 'auto')
        
        # 加载停用词
        self.stopwords = self._load_stopwords()
        
        # 主题-情感映射
        self.topic_sentiment_map = self._load_topic_sentiment_map()
    
    def _load_config(self, config_path: str) -> Dict:
        """加载配置文件"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"配置文件 {config_path} 不存在，使用默认配置")
            return {'lda': {}}
    
    def _load_stopwords(self) -> set:
        """加载停用词表"""
        stopwords = set([
            '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个',
            '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好',
            '自己', '这', '那', '着', '给', '但是', '还', '个', '得', '也', '这', '那', '着',
            '中', '对', '下', '过', '次', '啊', '呢', '吧', '吗', '呀', '啦', '哇', '哦',
            '哟', '嗯', '呃', '嘛', '呗', '唉', '哎', '嗨', '哼', '呸', '呵', '哈'
        ])
        
        # 尝试从文件加载额外的停用词
        stopword_files = [
            "dictionaries/stopwords.txt",
            "configs/stopwords.txt"
        ]
        
        for file_path in stopword_files:
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            word = line.strip()
                            if word:
                                stopwords.add(word)
                    logger.info(f"从文件加载停用词: {file_path}")
                except Exception as e:
                    logger.warning(f"加载停用词文件失败 {file_path}: {e}")
        
        return stopwords
    
    def _load_topic_sentiment_map(self) -> Dict[int, str]:
        """加载主题-情感映射"""
        try:
            with open("configs/topic_sentiment_mapping.json", 'r', encoding='utf-8') as f:
                mapping = json.load(f)
                
            # 转换键为整数
            topic_sentiment_map = {}
            for key, value in mapping.items():
                topic_sentiment_map[int(key)] = value
            
            return topic_sentiment_map
        except FileNotFoundError:
            # 默认映射：前几个主题为负面，中间为中性，后几个为正面
            default_map = {}
            for i in range(self.num_topics):
                if i < self.num_topics // 3:
                    default_map[i] = "negative"
                elif i < 2 * self.num_topics // 3:
                    default_map[i] = "neutral"
                else:
                    default_map[i] = "positive"
            return default_map
    
    def preprocess_text(self, text: str) -> List[str]:
        """
        预处理文本
        
        Args:
            text: 输入文本
            
        Returns:
            分词后的词列表
        """
        if not text or not isinstance(text, str):
            return []
        
        # 中文分词
        words = jieba.lcut(text)
        
        # 过滤停用词和短词
        filtered_words = [
            word for word in words 
            if word not in self.stopwords 
            and len(word) > 1 
            and not word.isdigit()
            and not word.isspace()
        ]
        
        return filtered_words
    
    def train(self, texts: List[str]) -> bool:
        """
        训练LDA模型
        
        Args:
            texts: 训练文本列表
            
        Returns:
            训练是否成功
        """
        if not texts:
            logger.error("训练文本为空")
            return False
        
        logger.info(f"开始训练LDA模型，文本数量: {len(texts)}")
        
        try:
            # 预处理所有文本
            processed_texts = []
            for i, text in enumerate(texts):
                tokens = self.preprocess_text(text)
                if len(tokens) >= 2:  # 只保留有足够词汇的文本
                    processed_texts.append(tokens)
                
                # 进度显示
                if (i + 1) % 1000 == 0:
                    logger.info(f"  已预处理 {i + 1}/{len(texts)} 条文本")
            
            if len(processed_texts) < 10:
                logger.error(f"有效文本太少: {len(processed_texts)} 条")
                return False
            
            logger.info(f"预处理后文本数量: {len(processed_texts)}")
            
            # 创建词典
            self.dictionary = corpora.Dictionary(processed_texts)
            
            # 过滤极端词频
            self.dictionary.filter_extremes(
                no_below=self.no_below,
                no_above=self.no_above
            )
            
            # 创建语料库（词袋表示）
            self.corpus = [self.dictionary.doc2bow(text) for text in processed_texts]
            
            # 训练LDA模型
            self.lda_model = LdaModel(
                corpus=self.corpus,
                id2word=self.dictionary,
                num_topics=self.num_topics,
                passes=self.passes,
                iterations=self.iterations,
                random_state=self.random_state,
                alpha=self.alpha,
                eta=self.eta,
                chunksize=100,
                update_every=1
            )
            
            # 计算主题一致性
            coherence_model = CoherenceModel(
                model=self.lda_model,
                texts=processed_texts,
                dictionary=self.dictionary,
                coherence='c_v'
            )
            coherence_score = coherence_model.get_coherence()
            
            # 获取所有主题的关键词
            self.topics = self.get_all_topics(num_words=15)
            
            # 分析主题情感倾向
            self._analyze_topic_sentiments()
            
            self.is_trained = True
            
            logger.info(f"LDA模型训练完成，主题数: {self.num_topics}")
            logger.info(f"主题一致性分数: {coherence_score:.4f}")
            
            return True
            
        except Exception as e:
            logger.error(f"LDA模型训练失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def get_document_topics(self, text: str) -> List[Tuple[int, float]]:
        """
        获取文档的主题分布
        
        Args:
            text: 输入文本
            
        Returns:
            主题分布列表 [(主题ID, 概率), ...]
        """
        if not self.is_trained:
            return []
        
        # 预处理文本
        words = self.preprocess_text(text)
        if not words:
            return []
        
        # 转换为词袋表示
        bow = self.dictionary.doc2bow(words)
        
        # 获取主题分布
        topics = self.lda_model.get_document_topics(bow, minimum_probability=0.01)
        
        # 按概率排序
        topics = sorted(topics, key=lambda x: x[1], reverse=True)
        
        return topics
    
    def get_topic_keywords(self, topic_id: int, num_words: int = 10) -> List[Tuple[str, float]]:
        """
        获取主题的关键词
        
        Args:
            topic_id: 主题ID
            num_words: 返回的关键词数量
            
        Returns:
            关键词和权重的列表
        """
        if not self.is_trained or topic_id < 0 or topic_id >= self.num_topics:
            return []
        
        return self.lda_model.show_topic(topic_id, topn=num_words)
    
    def get_all_topics(self, num_words: int = 10) -> Dict[int, List[Tuple[str, float]]]:
        """
        获取所有主题的关键词
        
        Args:
            num_words: 每个主题返回的关键词数量
            
        Returns:
            所有主题的关键词字典
        """
        if not self.is_trained:
            return {}
        
        topics = {}
        for topic_id in range(self.num_topics):
            topics[topic_id] = self.lda_model.show_topic(topic_id, topn=num_words)
        
        return topics
    
    def analyze_text_topics(self, text: str) -> Dict[str, Any]:
        """
        分析文本的主题分布
        
        Args:
            text: 输入文本
            
        Returns:
            主题分析结果
        """
        topics = self.get_document_topics(text)
        
        result = {
            'text': text,
            'topic_distribution': [],
            'dominant_topic': None,
            'topic_sentiment': 'neutral',
            'topic_diversity': 0.0
        }
        
        if not topics:
            return result
        
        # 获取每个主题的详细信息
        topic_details = []
        for topic_id, probability in topics:
            keywords = self.get_topic_keywords(topic_id, 5)
            
            # 获取主题情感
            sentiment = self.topic_sentiment_map.get(topic_id, "neutral")
            
            topic_info = {
                'topic_id': topic_id,
                'probability': float(probability),
                'sentiment': sentiment,
                'keywords': [{'word': word, 'weight': float(weight)} for word, weight in keywords]
            }
            
            topic_details.append(topic_info)
        
        # 按概率排序
        topic_details.sort(key=lambda x: x['probability'], reverse=True)
        
        result['topic_distribution'] = topic_details
        
        # 确定主导主题
        if topic_details:
            result['dominant_topic'] = topic_details[0]
            
            # 确定文本的主题情感（基于主导主题）
            result['topic_sentiment'] = topic_details[0]['sentiment']
            
            # 计算主题多样性（熵）
            probabilities = [t['probability'] for t in topic_details]
            entropy = -sum(p * np.log(p + 1e-10) for p in probabilities)
            max_entropy = np.log(len(probabilities)) if probabilities else 1.0
            result['topic_diversity'] = float(entropy / max_entropy if max_entropy > 0 else 0.0)
        
        return result
    
    def _analyze_topic_sentiments(self):
        """分析主题的情感倾向"""
        if not self.is_trained:
            return
        
        # 从词典管理器获取情感关键词
        from dictionary_manager import get_dictionary_manager
        dict_manager = get_dictionary_manager()
        positive_keywords = set(dict_manager.get_positive_keywords())
        negative_keywords = set(dict_manager.get_negative_keywords())
        
        # 分析每个主题的情感倾向
        for topic_id in range(self.num_topics):
            keywords = self.get_topic_keywords(topic_id, num_words=20)
            
            # 计算正面和负面关键词的权重
            positive_score = 0.0
            negative_score = 0.0
            
            for word, weight in keywords:
                if word in positive_keywords:
                    positive_score += weight * 1.5
                elif word in negative_keywords:
                    negative_score += weight * 2.0
            
            # 确定主题情感
            if positive_score > negative_score and positive_score > 0.1:
                self.topic_sentiment_map[topic_id] = "positive"
            elif negative_score > positive_score and negative_score > 0.1:
                self.topic_sentiment_map[topic_id] = "negative"
            else:
                self.topic_sentiment_map[topic_id] = "neutral"
    
    def find_similar_topics(self, text: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        查找相似主题
        
        Args:
            text: 查询文本
            top_k: 返回最相似的K个主题
            
        Returns:
            相似主题列表
        """
        if not self.is_trained:
            return []
        
        # 获取文本的主题分布
        topics = self.get_document_topics(text)
        if not topics:
            return []
        
        # 构建文本的主题向量
        text_vector = np.zeros(self.num_topics)
        for topic_id, probability in topics:
            text_vector[topic_id] = probability
        
        # 计算所有主题的相似度（基于关键词重叠）
        similarities = []
        for topic_id in range(self.num_topics):
            # 获取主题的关键词
            topic_keywords = set([word for word, _ in self.get_topic_keywords(topic_id, 10)])
            
            # 预处理文本
            text_words = set(self.preprocess_text(text))
            
            # 计算Jaccard相似度
            if topic_keywords and text_words:
                intersection = len(topic_keywords.intersection(text_words))
                union = len(topic_keywords.union(text_words))
                similarity = intersection / union if union > 0 else 0
                
                similarities.append({
                    'topic_id': topic_id,
                    'similarity': similarity,
                    'keywords': list(topic_keywords)[:5]
                })
        
        # 按相似度排序
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        return similarities[:top_k]
    
    def save_model(self) -> bool:
        """
        保存LDA模型
        
        Returns:
            保存是否成功
        """
        if not self.is_trained:
            logger.error("没有可保存的模型，请先训练模型")
            return False
        
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        
        model_data = {
            'lda_model': self.lda_model,
            'dictionary': self.dictionary,
            'corpus': self.corpus,
            'topics': self.topics,
            'topic_sentiment_map': self.topic_sentiment_map,
            'config': {
                'num_topics': self.num_topics,
                'passes': self.passes,
                'iterations': self.iterations,
                'random_state': self.random_state,
                'no_below': self.no_below,
                'no_above': self.no_above,
                'alpha': self.alpha,
                'eta': self.eta
            }
        }
        
        try:
            with open(self.model_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            # 同时保存词典
            dict_path = self.model_path.replace('.pkl', '_dictionary.pkl')
            with open(dict_path, 'wb') as f:
                pickle.dump(self.dictionary, f)
            
            logger.info(f"LDA模型已保存到: {self.model_path}")
            return True
            
        except Exception as e:
            logger.error(f"保存LDA模型失败: {e}")
            return False
    
    def load_model(self) -> bool:
        """
        加载LDA模型
        
        Returns:
            加载是否成功
        """
        if not os.path.exists(self.model_path):
            logger.error(f"模型文件不存在: {self.model_path}")
            return False
        
        try:
            with open(self.model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.lda_model = model_data['lda_model']
            self.dictionary = model_data['dictionary']
            self.corpus = model_data['corpus']
            self.topics = model_data['topics']
            self.topic_sentiment_map = model_data.get('topic_sentiment_map', {})
            self.is_trained = True
            
            # 更新配置
            if 'config' in model_data:
                config = model_data['config']
                self.num_topics = config.get('num_topics', self.num_topics)
                self.passes = config.get('passes', self.passes)
                self.iterations = config.get('iterations', self.iterations)
            
            logger.info(f"LDA模型已加载，主题数: {self.num_topics}")
            return True
            
        except Exception as e:
            logger.error(f"加载LDA模型失败: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取模型统计信息
        
        Returns:
            统计信息字典
        """
        if not self.is_trained:
            return {'is_trained': False}
        
        stats = {
            'is_trained': True,
            'num_topics': self.num_topics,
            'vocabulary_size': len(self.dictionary) if self.dictionary else 0,
            'corpus_size': len(self.corpus) if self.corpus else 0,
            'topic_sentiment_distribution': {
                'positive': 0,
                'negative': 0,
                'neutral': 0
            }
        }
        
        # 统计主题情感分布
        for sentiment in self.topic_sentiment_map.values():
            if sentiment in stats['topic_sentiment_distribution']:
                stats['topic_sentiment_distribution'][sentiment] += 1
        
        return stats


if __name__ == "__main__":
    # 测试代码
    modeler = LDATopicModeler()
    
    # 示例文本
    test_texts = [
        "我今天很难过，考试没考好，被老师批评了",
        "工作压力大，天天加班，感觉很累很疲惫",
        "和朋友出去玩很开心，吃了很多美食",
        "学习新知识很有成就感，觉得自己在进步",
        "最近总是失眠，焦虑不安，不知道怎么办",
        "完成了重要项目，获得老板表扬，心情很好",
        "家庭关系紧张，经常吵架，感觉很烦恼",
        "旅行看到了美丽的风景，心情特别愉快",
        "投资失败损失了很多钱，感觉很沮丧",
        "坚持锻炼身体，感觉越来越健康有活力"
    ]
    
    # 训练模型
    success = modeler.train(test_texts)
    
    if success:
        # 测试单个文本
        text = "考试没考好，心情很糟糕，被老师说了"
        result = modeler.analyze_text_topics(text)
        
        print(f"文本: {text}")
        print(f"主题情感: {result['topic_sentiment']}")
        print(f"主题多样性: {result['topic_diversity']:.3f}")
        
        if result['dominant_topic']:
            dominant = result['dominant_topic']
            print(f"主导主题: {dominant['topic_id']} (概率: {dominant['probability']:.3f})")
            print("关键词:")
            for kw in dominant['keywords']:
                print(f"  {kw['word']}: {kw['weight']:.4f}")
        
        # 保存模型
        modeler.save_model()
        
        # 显示统计信息
        stats = modeler.get_statistics()
        print(f"\n模型统计:")
        for key, value in stats.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for k, v in value.items():
                    print(f"    {k}: {v}")
            else:
                print(f"  {key}: {value}")