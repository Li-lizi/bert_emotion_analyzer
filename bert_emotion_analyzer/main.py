"""
情感分析系统主程序 - 重构版本
整合所有模块，提供完整的微博评论情感分析功能
"""
import os
import sys
import json
import yaml
import pickle
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Any, Optional
import warnings

# 屏蔽警告
warnings.filterwarnings("ignore")

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("outputs/logs/main.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 导入自定义模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.dictionary_manager import get_dictionary_manager
from utils.sentiment_enhancer import get_sentiment_enhancer
from utils.mental_health_analyzer import get_mental_health_analyzer
from utils.scene_classifier import get_scene_classifier
from utils.tfidf_analyzer import TFIDFAnalyzer
from utils.lda_topic_modeler import LDATopicModeler


class EmotionAnalyzer:
    """情感分析主类"""
    
    def __init__(self, config_path: str = "configs/paths_config.yaml"):
        """
        初始化情感分析器
        
        Args:
            config_path: 配置文件路径
        """
        self.config = self._load_config(config_path)
        self.models = {}
        self.modules = {}
        self.is_initialized = False
        
        # 创建输出目录
        os.makedirs("outputs/logs", exist_ok=True)
        os.makedirs("models", exist_ok=True)
        
        logger.info("情感分析器初始化中...")
    
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
                'bert_model': {
                    'path': 'models/bert_model',
                    'max_len': 128,
                    'batch_size': 32
                },
                'data': {
                    'train_csv': 'data/train.csv',
                    'val_csv': 'data/val.csv',
                    'test_csv': 'data/test.csv',
                    'text_col': 'cleaned_text',
                    'label_col': 'label'
                }
            }
    
    def initialize_modules(self) -> bool:
        """
        初始化所有模块
        
        Returns:
            初始化是否成功
        """
        try:
            logger.info("开始初始化所有模块...")
            
            # 1. 初始化词典管理器
            logger.info("初始化词典管理器...")
            self.modules['dictionary_manager'] = get_dictionary_manager()
            dict_manager = self.modules['dictionary_manager']
            dict_manager.load_all_dictionaries()
            
            # 显示词典统计
            dict_stats = dict_manager.get_statistics()
            logger.info(f"词典加载完成，共 {dict_stats['total_dictionaries']} 类词典，{dict_stats['total_keywords']} 个关键词")
            
            # 2. 初始化情感增强器
            logger.info("初始化情感增强器...")
            self.modules['sentiment_enhancer'] = get_sentiment_enhancer()
            
            # 3. 初始化心理健康分析器
            logger.info("初始化心理健康分析器...")
            self.modules['mental_health_analyzer'] = get_mental_health_analyzer()
            
            # 4. 初始化场景分类器
            logger.info("初始化场景分类器...")
            self.modules['scene_classifier'] = get_scene_classifier()
            
            # 5. 初始化TF-IDF分析器（不立即训练）
            logger.info("初始化TF-IDF分析器...")
            self.modules['tfidf_analyzer'] = TFIDFAnalyzer()
            
            # 6. 初始化LDA主题建模器（不立即训练）
            logger.info("初始化LDA主题建模器...")
            self.modules['lda_modeler'] = LDATopicModeler()
            
            self.is_initialized = True
            logger.info("所有模块初始化完成")
            return True
            
        except Exception as e:
            logger.error(f"初始化模块失败: {e}")
            return False
    
    def load_models(self) -> bool:
        """
        加载所有预训练模型
        
        Returns:
            加载是否成功
        """
        try:
            logger.info("开始加载预训练模型...")
            
            # 1. 尝试加载TF-IDF模型
            tfidf_analyzer = self.modules['tfidf_analyzer']
            tfidf_model_path = self.config.get('models', {}).get('tfidf', 'models/tfidf_model.pkl')
            
            if os.path.exists(tfidf_model_path):
                if tfidf_analyzer.load_model():
                    logger.info(f"TF-IDF模型已加载，特征数: {len(tfidf_analyzer.feature_names)}")
                    self.models['tfidf'] = tfidf_analyzer
                else:
                    logger.warning("TF-IDF模型加载失败，将在需要时训练")
            else:
                logger.info("TF-IDF模型文件不存在，将在需要时训练")
            
            # 2. 尝试加载LDA模型
            lda_modeler = self.modules['lda_modeler']
            lda_model_path = self.config.get('models', {}).get('lda', 'models/lda_model.pkl')
            
            if os.path.exists(lda_model_path):
                if lda_modeler.load_model():
                    logger.info(f"LDA模型已加载，主题数: {lda_modeler.num_topics}")
                    self.models['lda'] = lda_modeler
                else:
                    logger.warning("LDA模型加载失败，将在需要时训练")
            else:
                logger.info("LDA模型文件不存在，将在需要时训练")
            
            # 3. 这里可以添加BERT模型的加载逻辑
            # 注意：BERT模型较大，可能需要单独处理
            
            logger.info("模型加载完成")
            return True
            
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            return False
    
    def train_tfidf_model(self, texts: List[str]) -> bool:
        """
        训练TF-IDF模型
        
        Args:
            texts: 训练文本列表
            
        Returns:
            训练是否成功
        """
        try:
            logger.info(f"开始训练TF-IDF模型，文本数量: {len(texts)}")
            
            tfidf_analyzer = self.modules['tfidf_analyzer']
            success = tfidf_analyzer.train(texts)
            
            if success:
                # 保存模型
                tfidf_analyzer.save_model()
                self.models['tfidf'] = tfidf_analyzer
                logger.info("TF-IDF模型训练完成并已保存")
                return True
            else:
                logger.error("TF-IDF模型训练失败")
                return False
                
        except Exception as e:
            logger.error(f"训练TF-IDF模型失败: {e}")
            return False
    
    def train_lda_model(self, texts: List[str]) -> bool:
        """
        训练LDA模型
        
        Args:
            texts: 训练文本列表
            
        Returns:
            训练是否成功
        """
        try:
            logger.info(f"开始训练LDA模型，文本数量: {len(texts)}")
            
            lda_modeler = self.modules['lda_modeler']
            success = lda_modeler.train(texts)
            
            if success:
                # 保存模型
                lda_modeler.save_model()
                self.models['lda'] = lda_modeler
                logger.info("LDA模型训练完成并已保存")
                return True
            else:
                logger.error("LDA模型训练失败")
                return False
                
        except Exception as e:
            logger.error(f"训练LDA模型失败: {e}")
            return False
    
    def analyze_single_text(self, text: str, use_bert: bool = True) -> Dict[str, Any]:
        """
        分析单个文本
        
        Args:
            text: 输入文本
            use_bert: 是否使用BERT模型（如果可用）
            
        Returns:
            完整的情感分析结果
        """
        if not self.is_initialized:
            logger.error("分析器未初始化，请先调用 initialize_modules()")
            return {'error': '分析器未初始化'}
        
        try:
            logger.info(f"分析文本: {text[:50]}...")
            
            result = {
                'text': text,
                'length': len(text),
                'analyses': {}
            }
            
            # 1. 基础BERT情感分析（模拟）
            # 注意：这里需要集成您原来的BERT模型
            bert_result = self._simulate_bert_analysis(text)
            result['analyses']['bert'] = bert_result
            
            # 2. 情感增强分析
            if 'sentiment_enhancer' in self.modules:
                enhancer = self.modules['sentiment_enhancer']
                enhanced_emotion, enhanced_confidence = enhancer.enhance_prediction(
                    text, bert_result['emotion'], bert_result['confidence']
                )
                
                result['analyses']['enhanced'] = {
                    'emotion': enhanced_emotion,
                    'confidence': enhanced_confidence,
                    'is_corrected': bert_result['emotion'] != enhanced_emotion
                }
            
            # 3. TF-IDF分析（如果模型已加载）
            if 'tfidf' in self.models:
                tfidf_result = self.models['tfidf'].analyze_sentiment_keywords(text)
                result['analyses']['tfidf'] = tfidf_result
            
            # 4. LDA主题分析（如果模型已加载）
            if 'lda' in self.models:
                lda_result = self.models['lda'].analyze_text_topics(text)
                result['analyses']['lda'] = lda_result
            
            # 5. 心理健康分析
            if 'mental_health_analyzer' in self.modules:
                mental_result = self.modules['mental_health_analyzer'].get_detailed_analysis(text)
                result['analyses']['mental_health'] = mental_result
            
            # 6. 场景分类
            if 'scene_classifier' in self.modules:
                scene_result = self.modules['scene_classifier'].classify_with_details(text)
                result['analyses']['scene'] = scene_result
            
            # 7. 综合决策
            final_decision = self._make_final_decision(result['analyses'])
            result['final_decision'] = final_decision
            
            logger.info(f"分析完成，最终情感: {final_decision['emotion']} ({final_decision['confidence']:.3f})")
            return result
            
        except Exception as e:
            logger.error(f"分析文本失败: {e}")
            return {'error': str(e), 'text': text}
    
    def _simulate_bert_analysis(self, text: str) -> Dict[str, Any]:
        """
        模拟BERT情感分析（占位函数）
        在实际使用中，这里应该调用您训练好的BERT模型
        
        Args:
            text: 输入文本
            
        Returns:
            模拟的分析结果
        """
        # 这里是一个简单的模拟实现
        # 实际应用中应该替换为真实的BERT模型调用
        
        # 简单规则：包含负面词则判为负面，否则判为正面
        negative_keywords = ['难过', '悲伤', '痛苦', '失望', '生气', '焦虑', '压力', '累']
        positive_keywords = ['开心', '高兴', '快乐', '满意', '喜欢', '好', '棒']
        
        # 统计关键词
        neg_count = sum(1 for word in negative_keywords if word in text)
        pos_count = sum(1 for word in positive_keywords if word in text)
        
        if neg_count > pos_count:
            emotion = "负面"
            confidence = min(0.95, 0.6 + neg_count * 0.05)
        elif pos_count > neg_count:
            emotion = "正面"
            confidence = min(0.95, 0.6 + pos_count * 0.05)
        else:
            emotion = "中性"
            confidence = 0.5
        
        return {
            'emotion': emotion,
            'confidence': confidence,
            'neg_count': neg_count,
            'pos_count': pos_count
        }
    
    def _make_final_decision(self, analyses: Dict[str, Any]) -> Dict[str, Any]:
        """
        综合所有分析结果做出最终决策
        
        Args:
            analyses: 所有分析结果
            
        Returns:
            最终决策结果
        """
        # 收集所有情感预测
        emotions = []
        confidences = []
        
        # 1. BERT结果
        if 'bert' in analyses:
            emotions.append(analyses['bert']['emotion'])
            confidences.append(analyses['bert']['confidence'])
        
        # 2. 增强后结果（如果有修正）
        if 'enhanced' in analyses and analyses['enhanced']['is_corrected']:
            emotions.append(analyses['enhanced']['emotion'])
            confidences.append(analyses['enhanced']['confidence'])
        
        # 3. TF-IDF结果
        if 'tfidf' in analyses:
            tfidf_sentiment = analyses['tfidf'].get('sentiment', 'neutral')
            emotions.append('正面' if tfidf_sentiment == 'positive' else '负面' if tfidf_sentiment == 'negative' else '中性')
            confidences.append(analyses['tfidf'].get('confidence', 0.5))
        
        # 4. LDA结果
        if 'lda' in analyses:
            lda_sentiment = analyses['lda'].get('topic_sentiment', 'neutral')
            emotions.append('正面' if lda_sentiment == 'positive' else '负面' if lda_sentiment == 'negative' else '中性')
            confidences.append(0.7)  # LDA置信度设为固定值
        
        # 投票决策
        if not emotions:
            return {'emotion': '未知', 'confidence': 0.0, 'decision_method': '无结果'}
        
        # 统计情感出现次数
        from collections import Counter
        emotion_counts = Counter(emotions)
        
        # 找到最频繁的情感
        most_common = emotion_counts.most_common(1)[0]
        final_emotion = most_common[0]
        
        # 计算平均置信度（只考虑匹配最终情感的结果）
        matching_confidences = [conf for emo, conf in zip(emotions, confidences) if emo == final_emotion]
        if matching_confidences:
            avg_confidence = sum(matching_confidences) / len(matching_confidences)
        else:
            avg_confidence = 0.5
        
        return {
            'emotion': final_emotion,
            'confidence': avg_confidence,
            'decision_method': '投票融合',
            'vote_count': len(emotions),
            'emotion_distribution': dict(emotion_counts)
        }
    
    def batch_analyze(self, texts: List[str], use_bert: bool = True) -> List[Dict[str, Any]]:
        """
        批量分析文本
        
        Args:
            texts: 文本列表
            use_bert: 是否使用BERT模型
            
        Returns:
            分析结果列表
        """
        results = []
        
        logger.info(f"开始批量分析，共 {len(texts)} 条文本")
        
        for i, text in enumerate(texts):
            try:
                result = self.analyze_single_text(text, use_bert)
                results.append(result)
                
                # 进度显示
                if (i + 1) % 10 == 0 or i == len(texts) - 1:
                    logger.info(f"  已分析 {i + 1}/{len(texts)} 条文本")
                    
            except Exception as e:
                logger.error(f"分析第 {i + 1} 条文本失败: {e}")
                results.append({'error': str(e), 'text': text})
        
        logger.info(f"批量分析完成，成功 {len([r for r in results if 'error' not in r])}/{len(texts)} 条")
        return results
    
    def save_results(self, results: List[Dict], output_path: str = "outputs/analysis_results.json"):
        """
        保存分析结果
        
        Args:
            results: 分析结果列表
            output_path: 输出文件路径
            
        Returns:
            保存是否成功
        """
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # 简化结果以便保存
            simplified_results = []
            for result in results:
                if 'error' in result:
                    simplified_results.append(result)
                    continue
                
                simplified = {
                    'text': result.get('text', ''),
                    'length': result.get('length', 0),
                    'final_decision': result.get('final_decision', {}),
                    'has_mental_health': 'mental_health' in result.get('analyses', {}),
                    'has_scene': 'scene' in result.get('analyses', {})
                }
                
                # 添加简要分析
                if 'analyses' in result:
                    analyses = result['analyses']
                    if 'enhanced' in analyses:
                        simplified['emotion'] = analyses['enhanced']['emotion']
                        simplified['confidence'] = analyses['enhanced']['confidence']
                    elif 'bert' in analyses:
                        simplified['emotion'] = analyses['bert']['emotion']
                        simplified['confidence'] = analyses['bert']['confidence']
                    
                    if 'scene' in analyses:
                        simplified['scenes'] = [s['scene'] for s in analyses['scene'].get('scenes', [])][:2]
                    
                    if 'mental_health' in analyses:
                        simplified['mental_labels'] = list(analyses['mental_health'].get('mental_health_labels', {}).keys())
                
                simplified_results.append(simplified)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(simplified_results, f, ensure_ascii=False, indent=2)
            
            logger.info(f"分析结果已保存到: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"保存结果失败: {e}")
            return False
    
    def generate_report(self, results: List[Dict]) -> Dict[str, Any]:
        """
        生成分析报告
        
        Args:
            results: 分析结果列表
            
        Returns:
            分析报告
        """
        if not results:
            return {'error': '无分析结果'}
        
        # 过滤错误结果
        valid_results = [r for r in results if 'error' not in r]
        
        if not valid_results:
            return {'error': '无有效分析结果'}
        
        report = {
            'total_texts': len(results),
            'valid_texts': len(valid_results),
            'error_rate': (len(results) - len(valid_results)) / len(results) if len(results) > 0 else 0,
            'emotion_distribution': {},
            'scene_distribution': {},
            'mental_health_distribution': {},
            'average_confidence': 0.0,
            'statistics': {}
        }
        
        # 情感分布
        emotions = []
        confidences = []
        scenes = []
        mental_labels = []
        
        for result in valid_results:
            # 情感统计
            if 'final_decision' in result:
                emotion = result['final_decision'].get('emotion', '未知')
                confidence = result['final_decision'].get('confidence', 0.0)
                emotions.append(emotion)
                confidences.append(confidence)
            
            # 场景统计
            if 'analyses' in result and 'scene' in result['analyses']:
                scene_result = result['analyses']['scene']
                for scene_info in scene_result.get('scenes', []):
                    scenes.append(scene_info.get('scene', '未知'))
            
            # 心理健康标签统计
            if 'analyses' in result and 'mental_health' in result['analyses']:
                mental_result = result['analyses']['mental_health']
                for label in mental_result.get('mental_health_labels', {}).keys():
                    mental_labels.append(label)
        
        # 计算情感分布
        from collections import Counter
        if emotions:
            emotion_counts = Counter(emotions)
            report['emotion_distribution'] = dict(emotion_counts)
            report['average_confidence'] = sum(confidences) / len(confidences) if confidences else 0.0
        
        # 计算场景分布
        if scenes:
            scene_counts = Counter(scenes)
            report['scene_distribution'] = dict(scene_counts)
        
        # 计算心理健康标签分布
        if mental_labels:
            mental_counts = Counter(mental_labels)
            report['mental_health_distribution'] = dict(mental_counts)
        
        # 统计数据
        report['statistics'] = {
            'emotion_count': len(emotions),
            'scene_count': len(scenes),
            'mental_label_count': len(mental_labels),
            'most_common_emotion': max(report['emotion_distribution'].items(), key=lambda x: x[1])[0] if report['emotion_distribution'] else '无',
            'most_common_scene': max(report['scene_distribution'].items(), key=lambda x: x[1])[0] if report['scene_distribution'] else '无',
            'most_common_mental_label': max(report['mental_health_distribution'].items(), key=lambda x: x[1])[0] if report['mental_health_distribution'] else '无'
        }
        
        return report
    
    def interactive_mode(self):
        """交互式分析模式"""
        print("\n" + "=" * 70)
        print("🎭 微博评论情感分析系统 - 交互模式")
        print("=" * 70)
        print("功能说明:")
        print("  1. 输入文本进行情感分析")
        print("  2. 输入 'batch' 进入批量分析模式")
        print("  3. 输入 'train' 进入模型训练模式")
        print("  4. 输入 'report' 查看系统状态")
        print("  5. 输入 'quit' 或 '退出' 结束程序")
        print("=" * 70)
        
        while True:
            try:
                user_input = input("\n📝 请输入文本或命令: ").strip()
                
                if user_input.lower() in ['quit', '退出', 'exit', 'q']:
                    print("感谢使用，再见！")
                    break
                
                elif user_input.lower() == 'batch':
                    self._batch_mode()
                
                elif user_input.lower() == 'train':
                    self._train_mode()
                
                elif user_input.lower() == 'report':
                    self._show_report()
                
                elif not user_input:
                    print("⚠️  请输入文本或命令")
                    continue
                
                else:
                    # 分析单条文本
                    result = self.analyze_single_text(user_input)
                    
                    if 'error' in result:
                        print(f"❌ 分析失败: {result['error']}")
                        continue
                    
                    # 显示结果
                    self._display_result(result)
                    
            except KeyboardInterrupt:
                print("\n\n检测到中断，退出程序")
                break
            except Exception as e:
                print(f"❌ 发生错误: {e}")
    
    def _display_result(self, result: Dict[str, Any]):
        """显示分析结果"""
        print("\n📊 分析结果:")
        print("-" * 50)
        
        text = result.get('text', '')
        if len(text) > 60:
            display_text = text[:57] + "..."
        else:
            display_text = text
        
        print(f"文本: {display_text}")
        print(f"长度: {result.get('length', 0)} 字符")
        
        # 显示最终决策
        if 'final_decision' in result:
            decision = result['final_decision']
            emotion = decision.get('emotion', '未知')
            confidence = decision.get('confidence', 0.0)
            
            # 情感颜色标记
            if emotion == "正面":
                emotion_display = f"✅ {emotion}"
            elif emotion == "负面":
                emotion_display = f"❌ {emotion}"
            else:
                emotion_display = f"⚪ {emotion}"
            
            print(f"最终情感: {emotion_display} (置信度: {confidence:.3f})")
            
            if 'decision_method' in decision:
                print(f"决策方法: {decision['decision_method']}")
        
        # 显示场景分类
        if 'analyses' in result and 'scene' in result['analyses']:
            scene_result = result['analyses']['scene']
            if scene_result.get('scenes'):
                scenes = [f"{s['scene']}({s['confidence']:.2f})" for s in scene_result['scenes'][:2]]
                print(f"场景识别: {', '.join(scenes)}")
        
        # 显示心理健康标签
        if 'analyses' in result and 'mental_health' in result['analyses']:
            mental_result = result['analyses']['mental_health']
            if mental_result.get('mental_health_labels'):
                labels = list(mental_result['mental_health_labels'].keys())
                print(f"心理健康: {', '.join(labels)}")
                
                if mental_result.get('recommendation'):
                    print(f"建议: {mental_result['recommendation'][:50]}...")
        
        # 显示是否被修正
        if 'analyses' in result and 'enhanced' in result['analyses']:
            enhanced = result['analyses']['enhanced']
            if enhanced.get('is_corrected'):
                print(f"📝 情感已修正 (BERT: {result['analyses'].get('bert', {}).get('emotion', '未知')} → {enhanced.get('emotion', '未知')})")
        
        print("-" * 50)
    
    def _batch_mode(self):
        """批量分析模式"""
        print("\n📚 批量分析模式")
        print("-" * 50)
        
        # 从文件加载或手动输入
        source = input("选择数据来源 (1=文件, 2=手动输入, 其他=返回): ").strip()
        
        texts = []
        
        if source == '1':
            # 从文件加载
            file_path = input("请输入文本文件路径 (每行一个文本): ").strip()
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        texts = [line.strip() for line in f if line.strip()]
                    print(f"从文件加载了 {len(texts)} 条文本")
                except Exception as e:
                    print(f"读取文件失败: {e}")
                    return
            else:
                print(f"文件不存在: {file_path}")
                return
        
        elif source == '2':
            # 手动输入
            print("请输入文本 (每行一个，空行结束):")
            while True:
                line = input().strip()
                if not line:
                    break
                texts.append(line)
            print(f"手动输入了 {len(texts)} 条文本")
        
        else:
            print("返回主菜单")
            return
        
        if not texts:
            print("没有文本可分析")
            return
        
        # 开始批量分析
        print(f"\n开始分析 {len(texts)} 条文本...")
        results = self.batch_analyze(texts)
        
        # 保存结果
        save_option = input("是否保存结果? (y/n): ").strip().lower()
        if save_option == 'y':
            output_path = input("请输入保存路径 (默认: outputs/batch_results.json): ").strip()
            if not output_path:
                output_path = "outputs/batch_results.json"
            
            if self.save_results(results, output_path):
                print(f"结果已保存到: {output_path}")
            
            # 生成报告
            report = self.generate_report(results)
            print("\n📈 批量分析报告:")
            print(f"  总文本数: {report.get('total_texts', 0)}")
            print(f"  有效分析: {report.get('valid_texts', 0)}")
            print(f"  错误率: {report.get('error_rate', 0):.2%}")
            
            if 'emotion_distribution' in report:
                print(f"  情感分布: {report['emotion_distribution']}")
            
            if 'average_confidence' in report:
                print(f"  平均置信度: {report['average_confidence']:.3f}")
        
        print("\n批量分析完成")
    
    def _train_mode(self):
        """模型训练模式"""
        print("\n🤖 模型训练模式")
        print("-" * 50)
        print("可训练的模型:")
        print("  1. TF-IDF 模型")
        print("  2. LDA 主题模型")
        print("  3. 返回主菜单")
        
        choice = input("请选择 (1-3): ").strip()
        
        if choice == '1':
            self._train_tfidf()
        elif choice == '2':
            self._train_lda()
        else:
            print("返回主菜单")
            return
    
    def _train_tfidf(self):
        """训练TF-IDF模型"""
        print("\n🔧 训练TF-IDF模型")
        
        # 获取训练数据
        data_source = input("选择数据来源 (1=使用内置训练数据, 2=自定义文件): ").strip()
        
        texts = []
        
        if data_source == '1':
            # 使用内置训练数据
            train_csv = self.config.get('data', {}).get('train_csv', 'data/train.csv')
            if os.path.exists(train_csv):
                try:
                    df = pd.read_csv(train_csv, encoding='utf-8-sig')
                    text_col = self.config.get('data', {}).get('text_col', 'cleaned_text')
                    
                    if text_col in df.columns:
                        texts = df[text_col].dropna().tolist()
                        print(f"从训练数据加载了 {len(texts)} 条文本")
                    else:
                        print(f"列 {text_col} 不存在于数据中")
                        return
                except Exception as e:
                    print(f"读取训练数据失败: {e}")
                    return
            else:
                print(f"训练数据文件不存在: {train_csv}")
                return
        
        elif data_source == '2':
            # 自定义文件
            file_path = input("请输入文本文件路径: ").strip()
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        texts = [line.strip() for line in f if line.strip()]
                    print(f"从文件加载了 {len(texts)} 条文本")
                except Exception as e:
                    print(f"读取文件失败: {e}")
                    return
            else:
                print(f"文件不存在: {file_path}")
                return
        
        else:
            print("返回")
            return
        
        if len(texts) < 10:
            print(f"文本数量不足 ({len(texts)} 条)，至少需要 10 条文本")
            return
        
        # 开始训练
        print(f"\n开始训练TF-IDF模型，使用 {len(texts)} 条文本...")
        success = self.train_tfidf_model(texts)
        
        if success:
            print("✅ TF-IDF模型训练完成")
            
            # 显示模型信息
            if 'tfidf' in self.models:
                stats = self.models['tfidf'].get_statistics()
                print(f"  特征数: {stats.get('vocabulary_size', '未知')}")
                print(f"  是否训练完成: {'是' if stats.get('is_trained', False) else '否'}")
        else:
            print("❌ TF-IDF模型训练失败")
    
    def _train_lda(self):
        """训练LDA模型"""
        print("\n🎯 训练LDA主题模型")
        
        # 获取训练数据（与TF-IDF类似）
        train_csv = self.config.get('data', {}).get('train_csv', 'data/train.csv')
        
        if os.path.exists(train_csv):
            try:
                df = pd.read_csv(train_csv, encoding='utf-8-sig')
                text_col = self.config.get('data', {}).get('text_col', 'cleaned_text')
                
                if text_col in df.columns:
                    texts = df[text_col].dropna().tolist()
                    print(f"从训练数据加载了 {len(texts)} 条文本")
                else:
                    print(f"列 {text_col} 不存在于数据中")
                    return
            except Exception as e:
                print(f"读取训练数据失败: {e}")
                return
        else:
            print(f"训练数据文件不存在: {train_csv}")
            return
        
        if len(texts) < 20:
            print(f"文本数量不足 ({len(texts)} 条)，至少需要 20 条文本")
            return
        
        # 开始训练
        print(f"\n开始训练LDA模型，使用 {len(texts)} 条文本...")
        success = self.train_lda_model(texts)
        
        if success:
            print("✅ LDA模型训练完成")
            
            # 显示模型信息
            if 'lda' in self.models:
                stats = self.models['lda'].get_statistics()
                print(f"  主题数: {stats.get('num_topics', '未知')}")
                print(f"  词汇量: {stats.get('vocabulary_size', '未知')}")
                
                # 显示主题情感分布
                if 'topic_sentiment_distribution' in stats:
                    dist = stats['topic_sentiment_distribution']
                    print(f"  主题情感分布: 正面={dist.get('positive', 0)}, 负面={dist.get('negative', 0)}, 中性={dist.get('neutral', 0)}")
        else:
            print("❌ LDA模型训练失败")
    
    def _show_report(self):
        """显示系统状态报告"""
        print("\n📈 系统状态报告")
        print("-" * 50)
        
        # 模块状态
        print("模块状态:")
        print(f"  词典管理器: {'✓ 已加载' if 'dictionary_manager' in self.modules else '✗ 未加载'}")
        print(f"  情感增强器: {'✓ 已加载' if 'sentiment_enhancer' in self.modules else '✗ 未加载'}")
        print(f"  心理健康分析器: {'✓ 已加载' if 'mental_health_analyzer' in self.modules else '✗ 未加载'}")
        print(f"  场景分类器: {'✓ 已加载' if 'scene_classifier' in self.modules else '✗ 未加载'}")
        print(f"  TF-IDF分析器: {'✓ 已加载' if 'tfidf_analyzer' in self.modules else '✗ 未加载'}")
        print(f"  LDA主题建模器: {'✓ 已加载' if 'lda_modeler' in self.modules else '✗ 未加载'}")
        
        # 模型状态
        print("\n模型状态:")
        print(f"  TF-IDF模型: {'✓ 已加载' if 'tfidf' in self.models else '✗ 未训练/未加载'}")
        print(f"  LDA模型: {'✓ 已加载' if 'lda' in self.models else '✗ 未训练/未加载'}")
        print(f"  BERT模型: {'⚠️  需要单独集成'}")
        
        # 词典统计
        if 'dictionary_manager' in self.modules:
            dict_manager = self.modules['dictionary_manager']
            stats = dict_manager.get_statistics()
            print(f"\n词典统计:")
            print(f"  词典总数: {stats.get('total_dictionaries', 0)}")
            print(f"  总关键词数: {stats.get('total_keywords', 0)}")
            
            # 显示各类词典数量
            for key, count in stats.items():
                if key.startswith('sentiment_') or key.startswith('slang_') or key.startswith('mental_health_'):
                    dict_name = key.replace('_', ' ').title()
                    print(f"  {dict_name}: {count} 词")
        
        print("\n提示: 使用 'train' 命令训练缺失的模型")


def main():
    """主函数"""
    print("=" * 70)
    print("🎭 微博评论情感分析系统")
    print("版本: 2.0 (重构版) | 集成TF-IDF + LDA + 多策略增强")
    print("=" * 70)
    
    try:
        # 创建分析器实例
        analyzer = EmotionAnalyzer()
        
        # 初始化模块
        print("初始化系统模块...")
        if not analyzer.initialize_modules():
            print("❌ 模块初始化失败")
            return
        
        # 加载模型
        print("加载预训练模型...")
        analyzer.load_models()
        
        print("✅ 系统准备就绪")
        
        # 进入交互模式
        analyzer.interactive_mode()
        
    except Exception as e:
        print(f"❌ 程序执行失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()