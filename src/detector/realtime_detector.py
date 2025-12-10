"""
الكشف الفوري | Détection en Temps Réel | Real-time Detection
كشف السلوك المشبوه في الوقت الحقيقي
Détection du comportement suspect en temps réel
"""

import os
import sys
import time
import json
import threading
import tracemalloc
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path
from datetime import datetime
from collections import deque
from dataclasses import dataclass
import logging
import joblib
import numpy as np

# استيراد الوحدات | Importer les modules
try:
    from ..collector.behavior_collector import BehaviorCollector
    from ..features.feature_engineering import FeatureExtractor
except ImportError:
    BehaviorCollector = None
    FeatureExtractor = None

# إعداد التسجيل | Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    """
    نتيجة الكشف | Résultat de Détection
    """
    timestamp: float
    timestamp_iso: str
    prediction: str  # benign, malicious
    confidence: float  # 0.0 - 1.0
    model_name: str
    features: Dict[str, float]
    alert_level: str  # normal, warning, danger
    latency_ms: float
    memory_mb: float
    
    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp,
            'timestamp_iso': self.timestamp_iso,
            'prediction': self.prediction,
            'confidence': self.confidence,
            'model_name': self.model_name,
            'features': self.features,
            'alert_level': self.alert_level,
            'latency_ms': self.latency_ms,
            'memory_mb': self.memory_mb
        }
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


class RealtimeDetector:
    """
    الكاشف الفوري | Détecteur en Temps Réel
    يكشف السلوك المشبوه في الوقت الحقيقي
    Détecte le comportement suspect en temps réel
    
    المتطلبات | Exigences:
    - زمن الاستجابة < 2 ثانية | Latence < 2 secondes
    - استخدام RAM < 80 Mo | RAM < 80 Mo
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        scaler_path: Optional[str] = None,
        model_name: str = 'isolation_forest',
        config_path: Optional[str] = None,
        alert_threshold: float = 0.7,
        window_size: float = 10.0,
        max_latency: float = 2.0,
        max_memory_mb: float = 80.0
    ):
        """
        تهيئة الكاشف | Initialisation du détecteur
        
        Args:
            model_path: مسار النموذج | Chemin du modèle
            scaler_path: مسار المعياري | Chemin du scaler
            model_name: اسم النموذج | Nom du modèle
            config_path: مسار التكوين | Chemin de configuration
            alert_threshold: عتبة التنبيه | Seuil d'alerte
            window_size: حجم النافذة | Taille de la fenêtre
            max_latency: الحد الأقصى للتأخير | Latence maximale
            max_memory_mb: الحد الأقصى للذاكرة | RAM maximale
        """
        self.model_name = model_name
        self.alert_threshold = alert_threshold
        self.window_size = window_size
        self.max_latency = max_latency
        self.max_memory_mb = max_memory_mb
        
        # تحميل النموذج | Charger le modèle
        self.model = None
        self.scaler = None
        self.is_anomaly_detector = model_name in ['isolation_forest', 'one_class_svm', 'lof']
        
        if model_path and os.path.exists(model_path):
            self.model = joblib.load(model_path)
            logger.info(f"تم تحميل النموذج | Modèle chargé: {model_path}")
        
        if scaler_path and os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
            logger.info(f"تم تحميل المعياري | Scaler chargé: {scaler_path}")
        
        # مستخرج الميزات | Extracteur de features
        self.feature_extractor = FeatureExtractor(window_size=window_size) if FeatureExtractor else None
        
        # المخزن المؤقت للأحداث | Buffer d'événements
        self._event_buffer: deque = deque(maxlen=10000)
        
        # سجل النتائج | Historique des résultats
        self._detection_history: deque = deque(maxlen=1000)
        
        # حالة التشغيل | État d'exécution
        self._running = False
        self._detection_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        
        # دالة التنبيه | Callback d'alerte
        self._alert_callback: Optional[Callable[[DetectionResult], None]] = None
        
        # الإحصائيات | Statistiques
        self._stats = {
            'total_detections': 0,
            'malicious_count': 0,
            'benign_count': 0,
            'avg_latency_ms': 0.0,
            'max_latency_ms': 0.0,
            'current_memory_mb': 0.0
        }
        
        logger.info(f"تم تهيئة الكاشف الفوري | Détecteur initialisé")
        logger.info(f"   النموذج | Modèle: {model_name}")
        logger.info(f"   النافذة | Fenêtre: {window_size}s")
        logger.info(f"   التأخير الأقصى | Latence max: {max_latency}s")
        logger.info(f"   الذاكرة القصوى | RAM max: {max_memory_mb} MB")
    
    def set_alert_callback(self, callback: Callable[[DetectionResult], None]):
        """
        تعيين دالة التنبيه | Définir le callback d'alerte
        """
        self._alert_callback = callback
    
    def add_event(self, event: Dict):
        """
        إضافة حدث للمعالجة | Ajouter un événement à traiter
        """
        with self._lock:
            self._event_buffer.append(event)
        
        if self.feature_extractor:
            self.feature_extractor.add_event(event)
    
    def predict(self, features: Dict[str, float]) -> DetectionResult:
        """
        التنبؤ من الميزات | Prédire à partir des features
        
        Args:
            features: الميزات | Features
            
        Returns:
            نتيجة الكشف | Résultat de détection
        """
        start_time = time.time()
        tracemalloc.start()
        
        try:
            # تحويل الميزات إلى مصفوفة | Convertir en array
            feature_names = list(features.keys())
            feature_values = np.array([[features.get(name, 0) for name in feature_names]])
            
            # التطبيع | Normaliser
            if self.scaler is not None:
                try:
                    feature_values = self.scaler.transform(feature_values)
                except:
                    pass
            
            # التنبؤ | Prédire
            prediction = 'benign'
            confidence = 0.5
            
            if self.model is not None:
                if self.is_anomaly_detector:
                    raw_pred = self.model.predict(feature_values)[0]
                    # -1 = anomaly, 1 = normal
                    prediction = 'malicious' if raw_pred == -1 else 'benign'
                    
                    # الثقة من درجة الشذوذ | Confiance depuis le score
                    if hasattr(self.model, 'score_samples'):
                        score = -self.model.score_samples(feature_values)[0]
                        # تطبيع الدرجة | Normaliser le score
                        confidence = min(max(score, 0), 1)
                    elif hasattr(self.model, 'decision_function'):
                        score = -self.model.decision_function(feature_values)[0]
                        confidence = 1 / (1 + np.exp(-score))  # sigmoid
                    else:
                        confidence = 0.8 if prediction == 'malicious' else 0.2
                else:
                    prediction = 'malicious' if self.model.predict(feature_values)[0] == 1 else 'benign'
                    if hasattr(self.model, 'predict_proba'):
                        confidence = float(self.model.predict_proba(feature_values)[0][1])
                    else:
                        confidence = 0.9 if prediction == 'malicious' else 0.1
            
            # تحديد مستوى التنبيه | Déterminer le niveau d'alerte
            if prediction == 'malicious' and confidence >= self.alert_threshold:
                alert_level = 'danger'
            elif prediction == 'malicious' or confidence >= 0.5:
                alert_level = 'warning'
            else:
                alert_level = 'normal'
            
            # قياس الأداء | Mesurer les performances
            latency = (time.time() - start_time) * 1000  # ms
            current, peak = tracemalloc.get_traced_memory()
            memory_mb = peak / 1024 / 1024
            tracemalloc.stop()
            
            # إنشاء النتيجة | Créer le résultat
            result = DetectionResult(
                timestamp=time.time() * 1000,
                timestamp_iso=datetime.now().isoformat(),
                prediction=prediction,
                confidence=float(confidence),
                model_name=self.model_name,
                features=features,
                alert_level=alert_level,
                latency_ms=latency,
                memory_mb=memory_mb
            )
            
            # تحديث الإحصائيات | Mettre à jour les stats
            self._update_stats(result)
            
            # حفظ في السجل | Sauvegarder dans l'historique
            with self._lock:
                self._detection_history.append(result)
            
            # استدعاء التنبيه | Appeler le callback
            if alert_level != 'normal' and self._alert_callback:
                self._alert_callback(result)
            
            return result
            
        except Exception as e:
            logger.error(f"خطأ في التنبؤ | Erreur de prédiction: {e}")
            tracemalloc.stop()
            
            return DetectionResult(
                timestamp=time.time() * 1000,
                timestamp_iso=datetime.now().isoformat(),
                prediction='error',
                confidence=0.0,
                model_name=self.model_name,
                features=features,
                alert_level='warning',
                latency_ms=(time.time() - start_time) * 1000,
                memory_mb=0.0
            )
    
    def detect_current(self) -> DetectionResult:
        """
        الكشف من الأحداث الحالية | Détecter depuis les événements actuels
        """
        if self.feature_extractor:
            features = self.feature_extractor.get_current_features()
        else:
            features = {}
        
        return self.predict(features)
    
    def _update_stats(self, result: DetectionResult):
        """
        تحديث الإحصائيات | Mettre à jour les statistiques
        """
        self._stats['total_detections'] += 1
        
        if result.prediction == 'malicious':
            self._stats['malicious_count'] += 1
        else:
            self._stats['benign_count'] += 1
        
        # المتوسط المتحرك للتأخير | Moyenne mobile de la latence
        n = self._stats['total_detections']
        old_avg = self._stats['avg_latency_ms']
        self._stats['avg_latency_ms'] = old_avg + (result.latency_ms - old_avg) / n
        
        self._stats['max_latency_ms'] = max(self._stats['max_latency_ms'], result.latency_ms)
        self._stats['current_memory_mb'] = result.memory_mb
    
    def _detection_loop(self, interval: float = 1.0):
        """
        حلقة الكشف الرئيسية | Boucle de détection principale
        """
        logger.info("بدء حلقة الكشف | Démarrage de la boucle de détection")
        
        while self._running:
            try:
                result = self.detect_current()
                
                # التحقق من الأداء | Vérifier les performances
                if result.latency_ms > self.max_latency * 1000:
                    logger.warning(f"⚠️ تأخير عالي | Latence élevée: {result.latency_ms:.1f}ms")
                
                if result.memory_mb > self.max_memory_mb:
                    logger.warning(f"⚠️ ذاكرة عالية | RAM élevée: {result.memory_mb:.1f}MB")
                
                time.sleep(interval)
                
            except Exception as e:
                logger.error(f"خطأ في حلقة الكشف | Erreur dans la boucle: {e}")
                time.sleep(1)
        
        logger.info("توقف حلقة الكشف | Boucle de détection arrêtée")
    
    def start(self, interval: float = 1.0):
        """
        بدء الكشف الفوري | Démarrer la détection en temps réel
        """
        if self._running:
            logger.warning("الكاشف يعمل بالفعل | Détecteur déjà en cours")
            return
        
        self._running = True
        self._detection_thread = threading.Thread(
            target=self._detection_loop,
            args=(interval,),
            daemon=True
        )
        self._detection_thread.start()
        
        logger.info("✅ تم بدء الكاشف الفوري | Détecteur démarré")
    
    def stop(self):
        """
        إيقاف الكشف | Arrêter la détection
        """
        self._running = False
        if self._detection_thread:
            self._detection_thread.join(timeout=2)
        
        logger.info("✅ تم إيقاف الكاشف | Détecteur arrêté")
    
    def get_history(self, count: int = 100) -> List[DetectionResult]:
        """
        الحصول على سجل الكشف | Obtenir l'historique de détection
        """
        with self._lock:
            return list(self._detection_history)[-count:]
    
    def get_stats(self) -> Dict:
        """
        الحصول على الإحصائيات | Obtenir les statistiques
        """
        return self._stats.copy()
    
    def get_status(self) -> Dict:
        """
        الحصول على حالة الكاشف | Obtenir l'état du détecteur
        """
        return {
            'running': self._running,
            'model_name': self.model_name,
            'model_loaded': self.model is not None,
            'scaler_loaded': self.scaler is not None,
            'window_size': self.window_size,
            'alert_threshold': self.alert_threshold,
            'events_in_buffer': len(self._event_buffer),
            'detections_count': len(self._detection_history),
            **self._stats
        }
    
    def print_status(self):
        """
        طباعة الحالة | Afficher l'état
        """
        status = self.get_status()
        
        print("\n" + "=" * 60)
        print("🛡️ حالة الكاشف الفوري | État du Détecteur")
        print("=" * 60)
        print(f"   يعمل | Running: {'✅' if status['running'] else '❌'}")
        print(f"   النموذج | Modèle: {status['model_name']}")
        print(f"   النموذج محمّل | Modèle chargé: {'✅' if status['model_loaded'] else '❌'}")
        print(f"   المعياري محمّل | Scaler chargé: {'✅' if status['scaler_loaded'] else '❌'}")
        print(f"   الأحداث في المخزن | Événements buffer: {status['events_in_buffer']}")
        print(f"   إجمالي الكشوفات | Total détections: {status['total_detections']}")
        print(f"   حميدة | Bénins: {status['benign_count']}")
        print(f"   مشبوهة | Malveillants: {status['malicious_count']}")
        print(f"   متوسط التأخير | Latence moyenne: {status['avg_latency_ms']:.2f}ms")
        print(f"   الذاكرة الحالية | RAM actuelle: {status['current_memory_mb']:.2f}MB")
        print("=" * 60)


class IntegratedDetector:
    """
    الكاشف المتكامل | Détecteur Intégré
    يجمع بين الجامع والكاشف في وحدة واحدة
    Combine le collecteur et le détecteur
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        scaler_path: Optional[str] = None,
        model_name: str = 'isolation_forest',
        config_path: Optional[str] = None
    ):
        """
        تهيئة الكاشف المتكامل | Initialisation du détecteur intégré
        """
        # الكاشف | Détecteur
        self.detector = RealtimeDetector(
            model_path=model_path,
            scaler_path=scaler_path,
            model_name=model_name,
            config_path=config_path
        )
        
        # الجامع | Collecteur
        if BehaviorCollector:
            self.collector = BehaviorCollector(config_path=config_path)
        else:
            self.collector = None
            logger.warning("الجامع غير متوفر | Collecteur non disponible")
        
        self._running = False
        
        logger.info("تم تهيئة الكاشف المتكامل | Détecteur intégré initialisé")
    
    def start(self, detection_interval: float = 1.0):
        """
        بدء الكشف المتكامل | Démarrer la détection intégrée
        """
        self._running = True
        
        # بدء الجامع | Démarrer le collecteur
        if self.collector:
            self.collector.start()
        
        # بدء الكاشف | Démarrer le détecteur
        self.detector.start(interval=detection_interval)
        
        logger.info("✅ تم بدء الكاشف المتكامل | Détecteur intégré démarré")
    
    def stop(self):
        """
        إيقاف الكشف | Arrêter la détection
        """
        self._running = False
        
        if self.collector:
            self.collector.stop()
        
        self.detector.stop()
        
        logger.info("✅ تم إيقاف الكاشف المتكامل | Détecteur intégré arrêté")
    
    def get_status(self) -> Dict:
        """
        الحصول على الحالة الكاملة | Obtenir l'état complet
        """
        status = {
            'running': self._running,
            'detector': self.detector.get_status(),
            'collector': self.collector.get_stats() if self.collector else {}
        }
        return status


def main():
    """
    الدالة الرئيسية | Fonction principale
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="الكاشف الفوري | Détecteur en Temps Réel"
    )
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='مسار النموذج | Chemin du modèle'
    )
    parser.add_argument(
        '--scaler',
        type=str,
        default=None,
        help='مسار المعياري | Chemin du scaler'
    )
    parser.add_argument(
        '--model-name',
        type=str,
        default='isolation_forest',
        choices=['isolation_forest', 'one_class_svm', 'lof', 'random_forest', 'xgboost'],
        help='اسم النموذج | Nom du modèle'
    )
    parser.add_argument(
        '--duration',
        type=int,
        default=60,
        help='مدة التشغيل بالثواني | Durée en secondes'
    )
    parser.add_argument(
        '--interval',
        type=float,
        default=1.0,
        help='فترة الكشف بالثواني | Intervalle de détection'
    )
    
    args = parser.parse_args()
    
    # دالة التنبيه | Callback d'alerte
    def on_alert(result: DetectionResult):
        level_icons = {'normal': '🟢', 'warning': '🟡', 'danger': '🔴'}
        icon = level_icons.get(result.alert_level, '❓')
        print(f"\n{icon} تنبيه | Alerte: {result.prediction.upper()}")
        print(f"   الثقة | Confiance: {result.confidence:.2%}")
        print(f"   التأخير | Latence: {result.latency_ms:.1f}ms")
    
    # إنشاء الكاشف | Créer le détecteur
    detector = RealtimeDetector(
        model_path=args.model,
        scaler_path=args.scaler,
        model_name=args.model_name
    )
    detector.set_alert_callback(on_alert)
    
    try:
        print("\n" + "=" * 60)
        print("🛡️ الكاشف الفوري | Détecteur en Temps Réel")
        print("=" * 60)
        
        detector.start(interval=args.interval)
        
        print(f"\n⏳ تشغيل لمدة {args.duration} ثانية | Running for {args.duration}s...")
        print("اضغط Ctrl+C للإيقاف | Press Ctrl+C to stop\n")
        
        for i in range(args.duration):
            time.sleep(1)
            if (i + 1) % 10 == 0:
                detector.print_status()
        
    except KeyboardInterrupt:
        print("\n\n⚠️ توقف بواسطة المستخدم | Arrêt par l'utilisateur")
    
    finally:
        detector.stop()
        detector.print_status()


if __name__ == "__main__":
    main()
