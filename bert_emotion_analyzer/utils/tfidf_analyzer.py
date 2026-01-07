"""
TF-IDF分析器 - 基于统计方法提取文本特征
"""
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import json
import os
from typing import List, Dict, Tuple, Any, Optional
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TFIDFAnalyzer:
    """TF-IDF特征分析器"""
    
    def __init__(self, model_path: str = "models/tfidf_model.pkl", 
                 config_path: str = "configs/model_config.json"):
        """
        初始化TF-IDF分析器
        
        Args:
            model_path: 模型保存路径
            config_path: 配置文件路径
        """
        self.model_path = model_path
        self.config = self._load_config(config_path)
        
        self.vectorizer = None
        self.feature_names = []
        self.svd = None
        self.is_trained = False
        
        # 从配置获取参数
        self.max_features = self.config.get('tfidf', {}).get('max_features', 5000)
        self.min_df = self.config.get('tfidf', {}).get('min_df', 2)
        self.max_df = self.config.get('tfidf', {}).get('max_df', 0.95)
        self.ngram_range = tuple(self.config.get('tfidf', {}).get('ngram_range', (1, 2)))
        self.use_svd = self.config.get('tfidf', {}).get('use_svd', False)
        self.n_components = self.config.get('tfidf', {}).get('n_components', 100)
        
        # 词典管理器
        from dictionary_manager import get_dictionary_manager
        self.dict_manager = get_dictionary_manager()
    
    def _load_config(self, config_path: str) -> Dict:
        """加载配置文件"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"配置文件 {config_path} 不存在，使用默认配置")
            return {'tfidf': {}}
    
    def train(self, texts: List[str]) -> bool:
        """
        训练TF-IDF模型
        
        Args:
            texts: 训练文本列表
            
        Returns:
            训练是否成功
        """
        if not texts:
            logger.error("训练文本为空")
            return False
        
        logger.info(f"开始训练TF-IDF模型，文本数量: {len(texts)}")
        
        try:
            # 创建TF-IDF向量器
            self.vectorizer = TfidfVectorizer(
                max_features=self.max_features,
                min_df=self.min_df,
                max_df=self.max_df,
                ngram_range=self.ngram_range,
                stop_words=None  # 中文停用词需要特殊处理
            )
            
            # 拟合数据
            X = self.vectorizer.fit_transform(texts)
            self.feature_names = self.vectorizer.get_feature_names_out().tolist()
            
            logger.info(f"TF-IDF特征维度: {X.shape}")
            logger.info(f"词汇量: {len(self.feature_names)}")
            
            # 可选：使用SVD降维
            if self.use_svd:
                self.svd = TruncatedSVD(
                    n_components=self.n_components,
                    random_state=42
                )
                X_reduced = self.svd.fit_transform(X)
                logger.info(f"SVD降维后维度: {X_reduced.shape}")
            
            self.is_trained = True
            logger.info("TF-IDF模型训练完成")
            return True
            
        except Exception as e:
            logger.error(f"TF-IDF模型训练失败: {e}")
            return False
    
    def transform(self, texts: List[str]) -> np.ndarray:
        """
        将文本转换为TF-IDF特征
        
        Args:
            texts: 文本列表
            
        Returns:
            TF-IDF特征矩阵
        """
        if not self.is_trained:
            raise ValueError("请先调用 train() 方法训练模型")
        
        X = self.vectorizer.transform(texts)
        
        if self.svd is not None:
            X = self.svd.transform(X)
        
        return X
    
    def get_keyword_scores(self, text: str, top_n: int = 10) -> List[Tuple[str, float]]:
        """
        获取文本关键词的TF-IDF分数
        
        Args:
            text: 输入文本
            top_n: 返回前N个关键词
            
        Returns:
            关键词和分数的列表
        """
        if not self.is_trained:
            return []
        
        vector = self.vectorizer.transform([text])
        
        # 获取非零特征
        scores = []
        coo = vector.tocoo()
        
        for idx, value in zip(coo.col, coo.data):
            if idx < len(self.feature_names):
                scores.append((self.feature_names[idx], float(value)))
        
        # 按分数排序
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_n]
    
    def analyze_sentiment_keywords(self, text: str) -> Dict[str, Any]:
        """
        分析文本中的情感关键词（使用TF-IDF权重）
        
        Args:
            text: 输入文本
            
        Returns:
            情感分析结果
        """
        keyword_scores = self.get_keyword_scores(text, top_n=20)
        
        result = {
            'text': text,
            'sentiment_keywords': [],
            'positive_score': 0.0,
            'negative_score': 0.0,
            'sentiment_bias': 0.5,
            'keyword_count': 0
        }
        
        if not keyword_scores:
            return result
        
        # 获取情感词典
        positive_keywords = set(self.dict_manager.get_positive_keywords())
        negative_keywords = set(self.dict_manager.get_negative_keywords())
        
        # 分析关键词
        sentiment_keywords = []
        for keyword, score in keyword_scores:
            sentiment_info = {
                'keyword': keyword,
                'score': score,
                'sentiment': 'neutral'
            }
            
            if keyword in positive_keywords:
                sentiment_info['sentiment'] = 'positive'
                result['positive_score'] += score * 1.5  # 正面词权重更高
                sentiment_keywords.append(sentiment_info)
                
            elif keyword in negative_keywords:
                sentiment_info['sentiment'] = 'negative'
                result['negative_score'] += score * 2.0  # 负面词权重更高
                sentiment_keywords.append(sentiment_info)
        
        result['sentiment_keywords'] = sentiment_keywords
        result['keyword_count'] = len(sentiment_keywords)
        
        # 计算情感偏向
        total_score = result['positive_score'] + result['negative_score']
        if total_score > 0:
            result['sentiment_bias'] = result['positive_score'] / total_score
        
        # 确定情感倾向
        if result['positive_score'] > result['negative_score']:
            result['sentiment'] = 'positive'
        elif result['negative_score'] > result['positive_score']:
            result['sentiment'] = 'negative'
        else:
            result['sentiment'] = 'neutral'
        
        # 计算置信度
        if result['keyword_count'] > 0:
            result['confidence'] = min(0.95, total_score / result['keyword_count'])
        else:
            result['confidence'] = 0.0
        
        return result
    
    def find_similar_texts(self, query_text: str, texts: List[str], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        查找相似文本
        
        Args:
            query_text: 查询文本
            texts: 文本库
            top_k: 返回最相似的K个文本
            
        Returns:
            相似文本列表
        """
        if not self.is_trained:
            return []
        
        # 转换所有文本
        all_texts = [query_text] + texts
        vectors = self.transform(all_texts)
        
        # 计算余弦相似度
        query_vector = vectors[0:1]
        text_vectors = vectors[1:]
        
        similarities = cosine_similarity(query_vector, text_vectors)[0]
        
        # 构建结果
        results = []
        for i, similarity in enumerate(similarities):
            if i < len(texts):
                results.append({
                    'text': texts[i],
                    'similarity': float(similarity),
                    'rank': i + 1
                })
        
        # 按相似度排序
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:top_k]
    
    def extract_key_phrases(self, text: str, min_length: int = 2, max_length: int = 4) -> List[Dict[str, Any]]:
        """
        提取关键短语
        
        Args:
            text: 输入文本
            min_length: 最小短语长度
            max_length: 最大短语长度
            
        Returns:
            关键短语列表
        """
        if not text:
            return []
        
        # 简单的短语提取（基于n-gram）
        phrases = []
        words = list(text)  # 字符级n-gram
        
        for n in range(min_length, max_length + 1):
            for i in range(len(words) - n + 1):
                phrase = ''.join(words[i:i+n])
                
                # 检查短语是否在特征中
                if phrase in self.feature_names:
                    # 获取TF-IDF分数
                    vector = self.vectorizer.transform([phrase])
                    score = vector.max()
                    
                    phrases.append({
                        'phrase': phrase,
                        'length': n,
                        'score': float(score),
                        'position': i
                    })
        
        # 按分数排序
        phrases.sort(key=lambda x: x['score'], reverse=True)
        
        # 去重（去除重叠的短语）
        unique_phrases = []
        used_positions = set()
        
        for phrase in phrases:
            positions = set(range(phrase['position'], phrase['position'] + phrase['length']))
            
            if not positions.intersection(used_positions):
                unique_phrases.append(phrase)
                used_positions.update(positions)
        
        return unique_phrases[:10]  # 返回前10个
    
    def save_model(self) -> bool:
        """
        保存TF-IDF模型
        
        Returns:
            保存是否成功
        """
        if not self.is_trained:
            logger.error("没有可保存的模型，请先训练模型")
            return False
        
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        
        model_data = {
            'vectorizer': self.vectorizer,
            'feature_names': self.feature_names,
            'svd': self.svd,
            'config': {
                'max_features': self.max_features,
                'min_df': self.min_df,
                'max_df': self.max_df,
                'ngram_range': self.ngram_range,
                'use_svd': self.use_svd,
                'n_components': self.n_components
            }
        }
        
        try:
            with open(self.model_path, 'wb') as f:
                pickle.dump(model_data, f)
            logger.info(f"TF-IDF模型已保存到: {self.model_path}")
            return True
        except Exception as e:
            logger.error(f"保存TF-IDF模型失败: {e}")
            return False
    
    def load_model(self) -> bool:
        """
        加载TF-IDF模型
        
        Returns:
            加载是否成功
        """
        if not os.path.exists(self.model_path):
            logger.error(f"模型文件不存在: {self.model_path}")
            return False
        
        try:
            with open(self.model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.vectorizer = model_data['vectorizer']
            self.feature_names = model_data['feature_names']
            self.svd = model_data['svd']
            self.is_trained = True
            
            logger.info(f"TF-IDF模型已加载，特征数: {len(self.feature_names)}")
            return True
            
        except Exception as e:
            logger.error(f"加载TF-IDF模型失败: {e}")
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
            'vocabulary_size': len(self.feature_names),
            'max_features': self.max_features,
            'min_df': self.min_df,
            'max_df': self.max_df,
            'ngram_range': self.ngram_range,
            'use_svd': self.use_svd,
            'n_components': self.n_components if self.use_svd else None
        }
        
        return stats


if __name__ == "__main__":
    # 测试代码
    analyzer = TFIDFAnalyzer()
    
    # 示例文本
    test_texts = [
        "我今天很难过，考试没考好",
        "被老师批评了，心情不好",
        "我很开心，考试得了满分",
        "朋友送我礼物，非常高兴",
        "工作压力大，感觉很累"
    ]
    
    # 训练模型
    success = analyzer.train(test_texts)
    
    if success:
        # 测试单个文本
        text = "考试没考好，心情很糟糕"
        result = analyzer.analyze_sentiment_keywords(text)
        
        print(f"文本: {text}")
        print(f"情感: {result['sentiment']} (置信度: {result['confidence']:.3f})")
        print(f"情感偏向: {result['sentiment_bias']:.3f}")
        print(f"关键词: {len(result['sentiment_keywords'])} 个")
        
        for keyword in result['sentiment_keywords'][:5]:
            print(f"  {keyword['keyword']}: {keyword['score']:.4f} ({keyword['sentiment']})")
        
        # 保存模型
        analyzer.save_model()