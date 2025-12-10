"""
هندسة الميزات | Feature Engineering | Ingénierie des Features
استخراج وحساب الميزات من الأحداث الخام
Extraction et calcul des features à partir des événements bruts
"""

import os
import json
import math
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path
import logging

# إعداد التسجيل | Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureExtractor:
    """
    مستخرج الميزات | Extracteur de Features
    يحسب 15+ ميزة من الأحداث على نافذة متحركة
    Calcule 15+ features à partir des événements sur une fenêtre glissante
    """
    
    def __init__(self, window_size: float = 10.0):
        """
        تهيئة المستخرج | Initialisation de l'extracteur
        
        Args:
            window_size: حجم النافذة المتحركة بالثواني | Taille de la fenêtre en secondes
        """
        self.window_size = window_size
        self._event_buffer: deque = deque()
        self._last_features: Optional[Dict] = None
        
        logger.info(f"تم تهيئة مستخرج الميزات | Extracteur initialisé: window={window_size}s")
    
    # ==================== حساب الإنتروبيا ====================
    # ==================== Calcul de l'Entropie ====================
    
    def _calculate_entropy(self, values: List[str]) -> float:
        """
        حساب إنتروبيا شانون | Calculer l'entropie de Shannon
        
        Args:
            values: قائمة القيم | Liste des valeurs
            
        Returns:
            قيمة الإنتروبيا | Valeur de l'entropie
        """
        if not values:
            return 0.0
        
        # حساب التكرارات | Calculer les fréquences
        freq = defaultdict(int)
        for v in values:
            freq[v] += 1
        
        total = len(values)
        entropy = 0.0
        
        for count in freq.values():
            if count > 0:
                p = count / total
                entropy -= p * math.log2(p)
        
        return entropy
    
    # ==================== حساب معامل الانفجار ====================
    # ==================== Calcul du Coefficient de Burstiness ====================
    
    def _calculate_burstiness(self, timestamps: List[float]) -> float:
        """
        حساب معامل الانفجار | Calculer le coefficient de burstiness
        يقيس مدى تجمع الأحداث في فترات قصيرة
        Mesure le degré de regroupement des événements
        
        Args:
            timestamps: قائمة الأوقات | Liste des timestamps
            
        Returns:
            معامل الانفجار (0-1) | Coefficient de burstiness
        """
        if len(timestamps) < 2:
            return 0.0
        
        # حساب الفترات بين الأحداث | Calculer les intervalles
        sorted_ts = sorted(timestamps)
        intervals = [sorted_ts[i+1] - sorted_ts[i] for i in range(len(sorted_ts)-1)]
        
        if not intervals:
            return 0.0
        
        # حساب المتوسط والانحراف المعياري | Calculer moyenne et écart-type
        mean = np.mean(intervals)
        std = np.std(intervals)
        
        if mean == 0:
            return 0.0
        
        # معامل الانفجار = (std - mean) / (std + mean)
        # قيم موجبة تعني انفجار، سالبة تعني انتظام
        burstiness = (std - mean) / (std + mean) if (std + mean) > 0 else 0
        
        # تطبيع إلى 0-1 | Normaliser à 0-1
        return (burstiness + 1) / 2
    
    # ==================== استخراج الميزات ====================
    # ==================== Extraction des Features ====================
    
    def extract_features_from_events(self, events: List[Dict]) -> Dict[str, float]:
        """
        استخراج جميع الميزات من قائمة الأحداث
        Extraire toutes les features d'une liste d'événements
        
        Args:
            events: قائمة الأحداث | Liste des événements
            
        Returns:
            قاموس الميزات | Dictionnaire des features
        """
        if not events:
            return self._get_empty_features()
        
        # تصنيف الأحداث حسب المصدر | Classifier par source
        process_events = [e for e in events if e.get('source') == 'process']
        network_events = [e for e in events if e.get('source') == 'network']
        file_events = [e for e in events if e.get('source') == 'file']
        
        # الأوقات | Timestamps
        all_timestamps = [e.get('timestamp', 0) for e in events]
        
        # حساب المدة | Calculer la durée
        if all_timestamps:
            duration = (max(all_timestamps) - min(all_timestamps)) / 1000  # بالثواني
            duration = max(duration, 0.001)  # تجنب القسمة على صفر
        else:
            duration = self.window_size
        
        features = {}
        
        # ==================== ميزات الملفات ====================
        # ==================== Features Fichiers ====================
        
        # 1. عمليات الملفات في الثانية | Opérations fichiers par seconde
        features['file_ops_per_sec'] = len(file_events) / duration
        
        # 2. نسبة الملفات الفريدة | Ratio fichiers uniques
        file_paths = [e.get('data', {}).get('path', '') for e in file_events]
        unique_files = len(set(file_paths))
        features['unique_files_ratio'] = unique_files / max(len(file_paths), 1)
        
        # 3. نسبة الحذف/الإنشاء | Ratio suppression/création
        create_ops = sum(1 for e in file_events if e.get('data', {}).get('operation') == 'created')
        delete_ops = sum(1 for e in file_events if e.get('data', {}).get('operation') == 'deleted')
        features['delete_create_ratio'] = delete_ops / max(create_ops, 1)
        
        # 4. إنتروبيا المسارات | Entropie des chemins
        features['path_entropy'] = self._calculate_entropy(file_paths)
        
        # 5. إنتروبيا الامتدادات | Entropie des extensions
        extensions = [e.get('data', {}).get('extension', '') for e in file_events]
        features['file_extension_entropy'] = self._calculate_entropy(extensions)
        
        # ==================== ميزات العمليات ====================
        # ==================== Features Processus ====================
        
        # استخراج بيانات العمليات | Extraire les données processus
        cpu_values = []
        memory_values = []
        io_read_values = []
        io_write_values = []
        
        for e in process_events:
            data = e.get('data', {})
            if 'cpu_percent' in data:
                cpu_values.append(data['cpu_percent'])
            if 'memory_percent' in data:
                memory_values.append(data['memory_percent'])
            if 'io_read_bytes' in data:
                io_read_values.append(data['io_read_bytes'])
            if 'io_write_bytes' in data:
                io_write_values.append(data['io_write_bytes'])
        
        # 6. متوسط CPU | Moyenne CPU
        features['cpu_mean'] = np.mean(cpu_values) if cpu_values else 0.0
        
        # 7. انحراف CPU | Écart-type CPU
        features['cpu_std'] = np.std(cpu_values) if cpu_values else 0.0
        
        # 8. متوسط الذاكرة | Moyenne mémoire
        features['memory_mean'] = np.mean(memory_values) if memory_values else 0.0
        
        # 9. معدل القراءة | Taux de lecture
        if io_read_values and len(io_read_values) > 1:
            io_read_rate = (max(io_read_values) - min(io_read_values)) / duration
        else:
            io_read_rate = 0.0
        features['io_read_rate'] = io_read_rate / 1_000_000  # MB/s
        
        # 10. معدل الكتابة | Taux d'écriture
        if io_write_values and len(io_write_values) > 1:
            io_write_rate = (max(io_write_values) - min(io_write_values)) / duration
        else:
            io_write_rate = 0.0
        features['io_write_rate'] = io_write_rate / 1_000_000  # MB/s
        
        # 11. عدم تماثل I/O | Asymétrie I/O
        total_io = features['io_read_rate'] + features['io_write_rate']
        if total_io > 0:
            features['io_asymmetry'] = abs(features['io_read_rate'] - features['io_write_rate']) / total_io
        else:
            features['io_asymmetry'] = 0.0
        
        # ==================== ميزات الشبكة ====================
        # ==================== Features Réseau ====================
        
        # 12. معدل الاتصالات | Taux de connexions
        features['net_connections_rate'] = len(network_events) / duration
        
        # 13. نسبة المنافذ الفريدة | Ratio ports uniques
        remote_ports = []
        for e in network_events:
            data = e.get('data', {})
            if 'unique_remote_ports' in data:
                remote_ports.append(data['unique_remote_ports'])
        features['unique_ports_ratio'] = np.mean(remote_ports) if remote_ports else 0.0
        
        # 14. انفجار الاتصالات | Burst de connexions
        net_timestamps = [e.get('timestamp', 0) for e in network_events]
        features['connection_burst'] = self._calculate_burstiness(net_timestamps)
        
        # ==================== ميزات زمنية ====================
        # ==================== Features Temporelles ====================
        
        # 15. معامل الانفجار العام | Coefficient de burstiness global
        features['burstiness'] = self._calculate_burstiness(all_timestamps)
        
        # 16. الانتظام الزمني | Régularité temporelle
        if len(all_timestamps) > 2:
            sorted_ts = sorted(all_timestamps)
            intervals = [sorted_ts[i+1] - sorted_ts[i] for i in range(len(sorted_ts)-1)]
            if intervals:
                cv = np.std(intervals) / np.mean(intervals) if np.mean(intervals) > 0 else 0
                features['temporal_regularity'] = 1 / (1 + cv)  # أعلى = أكثر انتظاماً
            else:
                features['temporal_regularity'] = 0.5
        else:
            features['temporal_regularity'] = 0.5
        
        # ==================== ميزات إضافية ====================
        # ==================== Features Supplémentaires ====================
        
        # 17. كثافة الأحداث | Densité d'événements
        features['event_density'] = len(events) / duration
        
        # 18. تنوع المصادر | Diversité des sources
        sources = [e.get('source', '') for e in events]
        features['source_diversity'] = len(set(sources)) / 3  # max 3 sources
        
        self._last_features = features
        return features
    
    def _get_empty_features(self) -> Dict[str, float]:
        """
        الحصول على ميزات فارغة | Obtenir des features vides
        """
        return {
            'file_ops_per_sec': 0.0,
            'unique_files_ratio': 0.0,
            'delete_create_ratio': 0.0,
            'path_entropy': 0.0,
            'file_extension_entropy': 0.0,
            'cpu_mean': 0.0,
            'cpu_std': 0.0,
            'memory_mean': 0.0,
            'io_read_rate': 0.0,
            'io_write_rate': 0.0,
            'io_asymmetry': 0.0,
            'net_connections_rate': 0.0,
            'unique_ports_ratio': 0.0,
            'connection_burst': 0.0,
            'burstiness': 0.0,
            'temporal_regularity': 0.5,
            'event_density': 0.0,
            'source_diversity': 0.0
        }
    
    def get_feature_names(self) -> List[str]:
        """
        الحصول على أسماء الميزات | Obtenir les noms des features
        """
        return list(self._get_empty_features().keys())
    
    def add_event(self, event: Dict):
        """
        إضافة حدث إلى المخزن المؤقت | Ajouter un événement au buffer
        """
        self._event_buffer.append(event)
        
        # إزالة الأحداث القديمة | Supprimer les anciens événements
        current_time = event.get('timestamp', 0)
        cutoff = current_time - (self.window_size * 1000)  # تحويل إلى ميلي ثانية
        
        while self._event_buffer and self._event_buffer[0].get('timestamp', 0) < cutoff:
            self._event_buffer.popleft()
    
    def get_current_features(self) -> Dict[str, float]:
        """
        الحصول على الميزات الحالية من المخزن المؤقت
        Obtenir les features actuelles du buffer
        """
        return self.extract_features_from_events(list(self._event_buffer))
    
    def clear_buffer(self):
        """مسح المخزن المؤقت | Vider le buffer"""
        self._event_buffer.clear()


class DatasetFeatureProcessor:
    """
    معالج ميزات مجموعة البيانات | Processeur de Features du Dataset
    يحول مجموعة البيانات الخام إلى ميزات جاهزة للتدريب
    Transforme le dataset brut en features prêtes pour l'entraînement
    """
    
    def __init__(self, window_size: float = 10.0, step_size: float = 1.0):
        """
        تهيئة المعالج | Initialisation du processeur
        
        Args:
            window_size: حجم النافذة بالثواني | Taille de la fenêtre en secondes
            step_size: حجم الخطوة بالثواني | Taille du pas en secondes
        """
        self.window_size = window_size
        self.step_size = step_size
        self.extractor = FeatureExtractor(window_size=window_size)
        
        logger.info(f"تم تهيئة معالج البيانات | Processeur initialisé: window={window_size}s, step={step_size}s")
    
    def process_jsonl_file(
        self,
        input_file: str,
        output_file: Optional[str] = None
    ) -> pd.DataFrame:
        """
        معالجة ملف JSONL وتحويله إلى DataFrame مع الميزات
        Traiter un fichier JSONL et le convertir en DataFrame avec features
        
        Args:
            input_file: ملف الإدخال | Fichier d'entrée
            output_file: ملف الإخراج (اختياري) | Fichier de sortie (optionnel)
            
        Returns:
            DataFrame مع الميزات | DataFrame avec features
        """
        logger.info(f"معالجة | Traitement: {input_file}")
        
        # قراءة الأحداث | Lire les événements
        events = []
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    event = json.loads(line.strip())
                    events.append(event)
                except json.JSONDecodeError:
                    continue
        
        logger.info(f"تم قراءة {len(events)} حدث | {len(events)} événements lus")
        
        if not events:
            return pd.DataFrame()
        
        # ترتيب حسب الوقت | Trier par temps
        events.sort(key=lambda x: x.get('timestamp', 0))
        
        # تقسيم إلى نوافذ | Diviser en fenêtres
        features_list = []
        labels = []
        
        min_ts = events[0].get('timestamp', 0)
        max_ts = events[-1].get('timestamp', 0)
        
        window_ms = self.window_size * 1000
        step_ms = self.step_size * 1000
        
        current_start = min_ts
        
        while current_start + window_ms <= max_ts:
            # جمع أحداث النافذة | Collecter les événements de la fenêtre
            window_events = [
                e for e in events
                if current_start <= e.get('timestamp', 0) < current_start + window_ms
            ]
            
            if window_events:
                # استخراج الميزات | Extraire les features
                features = self.extractor.extract_features_from_events(window_events)
                features['window_start'] = current_start
                features['window_end'] = current_start + window_ms
                features['event_count'] = len(window_events)
                features_list.append(features)
                
                # تحديد التسمية (الأغلبية) | Déterminer le label (majorité)
                window_labels = [e.get('label', 'benign') for e in window_events]
                malicious_count = sum(1 for l in window_labels if l == 'malicious')
                label = 'malicious' if malicious_count > len(window_labels) / 2 else 'benign'
                labels.append(label)
            
            current_start += step_ms
        
        # إنشاء DataFrame | Créer le DataFrame
        df = pd.DataFrame(features_list)
        df['label'] = labels
        df['label_numeric'] = df['label'].map({'benign': 0, 'malicious': 1})
        
        logger.info(f"تم إنشاء {len(df)} نافذة | {len(df)} fenêtres créées")
        
        # حفظ إذا طُلب | Sauvegarder si demandé
        if output_file:
            df.to_csv(output_file, index=False)
            logger.info(f"تم الحفظ في | Sauvegardé: {output_file}")
        
        return df
    
    def get_feature_statistics(self, df: pd.DataFrame) -> Dict:
        """
        الحصول على إحصائيات الميزات | Obtenir les statistiques des features
        """
        feature_cols = self.extractor.get_feature_names()
        
        stats = {
            'total_samples': len(df),
            'benign_samples': len(df[df['label'] == 'benign']),
            'malicious_samples': len(df[df['label'] == 'malicious']),
            'features': {}
        }
        
        for col in feature_cols:
            if col in df.columns:
                stats['features'][col] = {
                    'mean': float(df[col].mean()),
                    'std': float(df[col].std()),
                    'min': float(df[col].min()),
                    'max': float(df[col].max())
                }
        
        return stats
    
    def print_statistics(self, df: pd.DataFrame):
        """
        طباعة إحصائيات الميزات | Afficher les statistiques
        """
        stats = self.get_feature_statistics(df)
        
        print("\n" + "=" * 70)
        print("📊 إحصائيات الميزات | Statistiques des Features")
        print("=" * 70)
        print(f"📦 إجمالي العينات | Total échantillons: {stats['total_samples']}")
        print(f"🌿 حميدة | Bénins: {stats['benign_samples']}")
        print(f"🔴 مشبوهة | Malveillants: {stats['malicious_samples']}")
        print("\n📈 الميزات | Features:")
        print("-" * 70)
        print(f"{'Feature':<30} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
        print("-" * 70)
        
        for name, values in stats['features'].items():
            print(f"{name:<30} {values['mean']:>10.3f} {values['std']:>10.3f} "
                  f"{values['min']:>10.3f} {values['max']:>10.3f}")
        
        print("=" * 70)


# اختبار الوحدة | Test du module
if __name__ == "__main__":
    print("=" * 60)
    print("اختبار هندسة الميزات | Test Feature Engineering")
    print("=" * 60)
    
    # إنشاء أحداث وهمية للاختبار | Créer des événements factices
    import time
    
    extractor = FeatureExtractor(window_size=10)
    
    # محاكاة أحداث | Simuler des événements
    test_events = []
    base_time = time.time() * 1000
    
    # أحداث ملفات | Événements fichiers
    for i in range(50):
        test_events.append({
            'timestamp': base_time + i * 100,
            'source': 'file',
            'data': {
                'operation': 'created' if i % 3 != 0 else 'deleted',
                'path': f'/test/file_{i % 10}.txt',
                'extension': '.txt'
            },
            'label': 'benign'
        })
    
    # أحداث عمليات | Événements processus
    for i in range(30):
        test_events.append({
            'timestamp': base_time + i * 150,
            'source': 'process',
            'data': {
                'cpu_percent': 10 + i % 20,
                'memory_percent': 30 + i % 10,
                'io_read_bytes': 1000000 + i * 10000,
                'io_write_bytes': 500000 + i * 5000
            },
            'label': 'benign'
        })
    
    # أحداث شبكة | Événements réseau
    for i in range(20):
        test_events.append({
            'timestamp': base_time + i * 200,
            'source': 'network',
            'data': {
                'unique_remote_ports': i % 10
            },
            'label': 'benign'
        })
    
    # استخراج الميزات | Extraire les features
    features = extractor.extract_features_from_events(test_events)
    
    print("\n📊 الميزات المستخرجة | Features Extraites:")
    print("-" * 40)
    for name, value in features.items():
        print(f"  {name}: {value:.4f}")
    
    print(f"\n✅ تم استخراج {len(features)} ميزة | {len(features)} features extraites")
