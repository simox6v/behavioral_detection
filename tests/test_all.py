"""
اختبارات الوحدات | Tests Unitaires | Unit Tests
اختبارات شاملة لجميع وحدات النظام
Tests complets pour tous les modules du système
"""

import os
import sys
import time
import json
import tempfile
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# إضافة المسار | Ajouter le chemin
sys.path.insert(0, str(Path(__file__).parent.parent))


# ==================== اختبارات مراقب العمليات ====================
# ==================== Tests Moniteur Processus ====================

class TestProcessMonitor:
    """اختبارات مراقب العمليات | Tests du Moniteur Processus"""
    
    def test_import(self):
        """اختبار الاستيراد | Test d'importation"""
        from src.collector.process_monitor import ProcessMonitor, ProcessEvent
        assert ProcessMonitor is not None
        assert ProcessEvent is not None
    
    def test_init(self):
        """اختبار التهيئة | Test d'initialisation"""
        from src.collector.process_monitor import ProcessMonitor
        monitor = ProcessMonitor(interval=1.0)
        assert monitor.interval == 1.0
    
    def test_collect_once(self):
        """اختبار الجمع الواحد | Test de collecte unique"""
        from src.collector.process_monitor import ProcessMonitor
        monitor = ProcessMonitor(interval=1.0)
        events = monitor.collect_once()
        assert isinstance(events, list)
        assert len(events) >= 0  # قد يكون هناك 0 أو أكثر
    
    def test_system_stats(self):
        """اختبار إحصائيات النظام | Test des stats système"""
        from src.collector.process_monitor import ProcessMonitor
        monitor = ProcessMonitor()
        stats = monitor.get_system_stats()
        assert 'cpu_percent' in stats
        assert 'memory_percent' in stats
        assert 'process_count' in stats


# ==================== اختبارات مراقب الشبكة ====================
# ==================== Tests Moniteur Réseau ====================

class TestNetworkMonitor:
    """اختبارات مراقب الشبكة | Tests du Moniteur Réseau"""
    
    def test_import(self):
        """اختبار الاستيراد | Test d'importation"""
        from src.collector.network_monitor import NetworkMonitor, NetworkEvent
        assert NetworkMonitor is not None
        assert NetworkEvent is not None
    
    def test_init(self):
        """اختبار التهيئة | Test d'initialisation"""
        from src.collector.network_monitor import NetworkMonitor
        monitor = NetworkMonitor(interval=1.0)
        assert monitor.interval == 1.0
    
    def test_collect_once(self):
        """اختبار الجمع الواحد | Test de collecte unique"""
        from src.collector.network_monitor import NetworkMonitor
        monitor = NetworkMonitor(interval=1.0)
        event = monitor.collect_once()
        assert event is not None
        assert hasattr(event, 'total_connections')


# ==================== اختبارات مراقب الملفات ====================
# ==================== Tests Moniteur Fichiers ====================

class TestFileMonitor:
    """اختبارات مراقب الملفات | Tests du Moniteur Fichiers"""
    
    def test_import(self):
        """اختبار الاستيراد | Test d'importation"""
        from src.collector.file_monitor import SimpleFileMonitor, FileEvent
        assert SimpleFileMonitor is not None
        assert FileEvent is not None
    
    def test_init(self):
        """اختبار التهيئة | Test d'initialisation"""
        from src.collector.file_monitor import SimpleFileMonitor
        with tempfile.TemporaryDirectory() as tmpdir:
            monitor = SimpleFileMonitor(watch_directories=[tmpdir])
            assert len(monitor.watch_directories) == 1


# ==================== اختبارات السيناريوهات ====================
# ==================== Tests Scénarios ====================

class TestBenignScenarios:
    """اختبارات السيناريوهات الحميدة | Tests Scénarios Bénins"""
    
    def test_import(self):
        """اختبار الاستيراد | Test d'importation"""
        from src.generator.benign_scenarios import BenignScenarios
        assert BenignScenarios is not None
    
    def test_init(self):
        """اختبار التهيئة | Test d'initialisation"""
        from src.generator.benign_scenarios import BenignScenarios
        with tempfile.TemporaryDirectory() as tmpdir:
            scenarios = BenignScenarios(sandbox_dir=tmpdir)
            assert scenarios.sandbox_dir.exists()
    
    def test_web_browsing(self):
        """اختبار محاكاة التصفح | Test simulation navigation"""
        from src.generator.benign_scenarios import BenignScenarios
        with tempfile.TemporaryDirectory() as tmpdir:
            scenarios = BenignScenarios(sandbox_dir=tmpdir)
            events = scenarios.simulate_web_browsing(duration=2, intensity='high')
            assert events > 0


class TestMaliciousScenarios:
    """اختبارات السيناريوهات المشبوهة | Tests Scénarios Malveillants"""
    
    def test_import(self):
        """اختبار الاستيراد | Test d'importation"""
        from src.generator.malicious_scenarios import MaliciousScenarios
        assert MaliciousScenarios is not None
    
    def test_init(self):
        """اختبار التهيئة | Test d'initialisation"""
        from src.generator.malicious_scenarios import MaliciousScenarios
        with tempfile.TemporaryDirectory() as tmpdir:
            scenarios = MaliciousScenarios(sandbox_dir=tmpdir)
            assert scenarios.sandbox_dir.exists()
    
    def test_file_burst(self):
        """اختبار انفجار الملفات | Test burst fichiers"""
        from src.generator.malicious_scenarios import MaliciousScenarios
        with tempfile.TemporaryDirectory() as tmpdir:
            scenarios = MaliciousScenarios(sandbox_dir=tmpdir)
            events = scenarios.simulate_file_burst(duration=2, files_count=50)
            assert events > 0


# ==================== اختبارات هندسة الميزات ====================
# ==================== Tests Feature Engineering ====================

class TestFeatureExtractor:
    """اختبارات مستخرج الميزات | Tests Extracteur Features"""
    
    def test_import(self):
        """اختبار الاستيراد | Test d'importation"""
        from src.features.feature_engineering import FeatureExtractor
        assert FeatureExtractor is not None
    
    def test_init(self):
        """اختبار التهيئة | Test d'initialisation"""
        from src.features.feature_engineering import FeatureExtractor
        extractor = FeatureExtractor(window_size=10)
        assert extractor.window_size == 10
    
    def test_empty_features(self):
        """اختبار الميزات الفارغة | Test features vides"""
        from src.features.feature_engineering import FeatureExtractor
        extractor = FeatureExtractor()
        features = extractor.extract_features_from_events([])
        assert isinstance(features, dict)
        assert len(features) > 0
    
    def test_feature_names(self):
        """اختبار أسماء الميزات | Test noms des features"""
        from src.features.feature_engineering import FeatureExtractor
        extractor = FeatureExtractor()
        names = extractor.get_feature_names()
        assert isinstance(names, list)
        assert len(names) >= 15  # على الأقل 15 ميزة
    
    def test_extract_with_events(self):
        """اختبار الاستخراج مع أحداث | Test extraction avec événements"""
        from src.features.feature_engineering import FeatureExtractor
        extractor = FeatureExtractor()
        
        events = [
            {'timestamp': 1000, 'source': 'file', 'data': {'operation': 'created', 'path': '/test/a.txt', 'extension': '.txt'}},
            {'timestamp': 2000, 'source': 'file', 'data': {'operation': 'deleted', 'path': '/test/b.txt', 'extension': '.txt'}},
            {'timestamp': 3000, 'source': 'process', 'data': {'cpu_percent': 10, 'memory_percent': 30}},
        ]
        
        features = extractor.extract_features_from_events(events)
        assert 'file_ops_per_sec' in features
        assert 'cpu_mean' in features


# ==================== اختبارات النماذج ====================
# ==================== Tests Modèles ====================

class TestModelTrainer:
    """اختبارات مدرب النماذج | Tests Entraîneur Modèles"""
    
    def test_import(self):
        """اختبار الاستيراد | Test d'importation"""
        from src.models.train_models import ModelTrainer
        assert ModelTrainer is not None
    
    def test_init(self):
        """اختبار التهيئة | Test d'initialisation"""
        from src.models.train_models import ModelTrainer
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = ModelTrainer(models_dir=tmpdir)
            assert trainer.models_dir.exists()
    
    def test_default_config(self):
        """اختبار الإعدادات الافتراضية | Test config par défaut"""
        from src.models.train_models import ModelTrainer
        trainer = ModelTrainer()
        config = trainer._get_default_config()
        assert 'isolation_forest' in config
        assert 'random_forest' in config


# ==================== اختبارات الكاشف الفوري ====================
# ==================== Tests Détecteur Temps Réel ====================

class TestRealtimeDetector:
    """اختبارات الكاشف الفوري | Tests Détecteur Temps Réel"""
    
    def test_import(self):
        """اختبار الاستيراد | Test d'importation"""
        from src.detector.realtime_detector import RealtimeDetector, DetectionResult
        assert RealtimeDetector is not None
        assert DetectionResult is not None
    
    def test_init(self):
        """اختبار التهيئة | Test d'initialisation"""
        from src.detector.realtime_detector import RealtimeDetector
        detector = RealtimeDetector(model_name='isolation_forest')
        assert detector.model_name == 'isolation_forest'
    
    def test_predict_without_model(self):
        """اختبار التنبؤ بدون نموذج | Test prédiction sans modèle"""
        from src.detector.realtime_detector import RealtimeDetector
        detector = RealtimeDetector()
        
        features = {'file_ops_per_sec': 1.0, 'cpu_mean': 10.0}
        result = detector.predict(features)
        
        assert result is not None
        assert hasattr(result, 'prediction')
        assert hasattr(result, 'confidence')
    
    def test_get_status(self):
        """اختبار حالة الكاشف | Test état du détecteur"""
        from src.detector.realtime_detector import RealtimeDetector
        detector = RealtimeDetector()
        status = detector.get_status()
        
        assert 'running' in status
        assert 'model_name' in status
        assert 'total_detections' in status


# ==================== اختبارات التكامل ====================
# ==================== Tests d'Intégration ====================

class TestIntegration:
    """اختبارات التكامل | Tests d'Intégration"""
    
    def test_full_pipeline(self):
        """اختبار خط الأنابيب الكامل | Test pipeline complet"""
        from src.features.feature_engineering import FeatureExtractor
        from src.detector.realtime_detector import RealtimeDetector
        
        # إنشاء أحداث | Créer des événements
        events = [
            {'timestamp': i * 100, 'source': 'file', 'data': {'operation': 'created', 'path': f'/test/{i}.txt', 'extension': '.txt'}}
            for i in range(10)
        ]
        
        # استخراج الميزات | Extraire les features
        extractor = FeatureExtractor(window_size=10)
        features = extractor.extract_features_from_events(events)
        
        # التنبؤ | Prédire
        detector = RealtimeDetector()
        result = detector.predict(features)
        
        assert result.prediction in ['benign', 'malicious', 'error']


# ==================== اختبارات الأداء ====================
# ==================== Tests de Performance ====================

class TestPerformance:
    """اختبارات الأداء | Tests de Performance"""
    
    def test_feature_extraction_speed(self):
        """اختبار سرعة استخراج الميزات | Test vitesse extraction"""
        from src.features.feature_engineering import FeatureExtractor
        
        extractor = FeatureExtractor()
        events = [
            {'timestamp': i, 'source': 'file', 'data': {'operation': 'created', 'path': f'/test/{i}.txt', 'extension': '.txt'}}
            for i in range(1000)
        ]
        
        start = time.time()
        features = extractor.extract_features_from_events(events)
        duration = time.time() - start
        
        # يجب أن يكون أقل من ثانية | Doit être < 1 seconde
        assert duration < 1.0, f"استخراج بطيء جداً | Extraction trop lente: {duration:.2f}s"
    
    def test_prediction_latency(self):
        """اختبار زمن التنبؤ | Test latence prédiction"""
        from src.detector.realtime_detector import RealtimeDetector
        
        detector = RealtimeDetector()
        features = {name: 0.5 for name in ['file_ops_per_sec', 'cpu_mean', 'memory_mean']}
        
        start = time.time()
        for _ in range(100):
            result = detector.predict(features)
        duration = time.time() - start
        
        avg_latency = duration / 100 * 1000  # ms
        
        # يجب أن يكون أقل من 20ms | Doit être < 20ms
        assert avg_latency < 20, f"تنبؤ بطيء جداً | Prédiction trop lente: {avg_latency:.2f}ms"


# ==================== تشغيل الاختبارات ====================
# ==================== Exécution des tests ====================

if __name__ == "__main__":
    # تشغيل الاختبارات | Exécuter les tests
    pytest.main([__file__, "-v", "--tb=short"])
