"""
الوكيل الرئيسي لجمع السلوك | Agent Principal de Collecte | Main Behavior Collector
يجمع جميع الأحداث من مراقبي العمليات والشبكة والملفات
Collecte tous les événements des moniteurs de processus, réseau et fichiers
"""

import os
import sys
import json
import time
import yaml
import threading
import argparse
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import logging
from collections import deque

# استيراد المراقبين | Importer les moniteurs
from .process_monitor import ProcessMonitor, ProcessEvent
from .network_monitor import NetworkMonitor, NetworkEvent
from .file_monitor import FileMonitor, SimpleFileMonitor, FileEvent

# إعداد التسجيل | Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class UnifiedEvent:
    """
    حدث موحد يجمع جميع أنواع الأحداث
    Événement unifié regroupant tous les types
    """
    timestamp: float
    timestamp_iso: str
    source: str  # process, network, file
    event_type: str
    data: Dict[str, Any]
    label: str = "benign"  # benign, malicious
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


class BehaviorCollector:
    """
    الوكيل الرئيسي لجمع السلوك | Agent Principal de Collecte
    يجمع ويوحد جميع الأحداث من المراقبين المختلفين
    Collecte et unifie tous les événements des différents moniteurs
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        تهيئة الوكيل | Initialisation de l'agent
        
        Args:
            config_path: مسار ملف التكوين | Chemin du fichier de configuration
        """
        # تحميل التكوين | Charger la configuration
        self.config = self._load_config(config_path)
        
        # قائمة الأحداث الموحدة | Liste des événements unifiés
        self._events: deque = deque(maxlen=self.config.get('max_events_buffer', 100000))
        self._lock = threading.Lock()
        
        # إعداد مسارات البيانات | Configurer les chemins de données
        self.data_dir = Path(self.config.get('data_dir', './data'))
        self.raw_dir = self.data_dir / 'raw'
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        
        # مجلد المراقبة | Répertoire de surveillance
        self.watch_dir = self.data_dir / 'test_sandbox'
        self.watch_dir.mkdir(parents=True, exist_ok=True)
        
        # إنشاء المراقبين | Créer les moniteurs
        self._init_monitors()
        
        # ملف الإخراج | Fichier de sortie
        self._output_file = None
        self._output_format = self.config.get('output_format', 'jsonl')
        
        # التسمية الحالية | Label actuel
        self._current_label = "benign"
        
        # الإحصائيات | Statistiques
        self._stats = {
            'process_events': 0,
            'network_events': 0,
            'file_events': 0,
            'total_events': 0,
            'start_time': None
        }
        
        logger.info("تم تهيئة وكيل جمع السلوك | Agent de collecte initialisé")
    
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """
        تحميل التكوين من ملف | Charger la configuration
        """
        default_config = {
            'collection_interval': 0.5,
            'feature_window': 10,
            'max_events_buffer': 100000,
            'data_dir': './data',
            'output_format': 'jsonl',
            'process_monitor': {
                'enabled': True,
                'excluded_processes': ['System', 'System Idle Process', 'Registry']
            },
            'network_monitor': {
                'enabled': True,
                'excluded_ports': []
            },
            'file_monitor': {
                'enabled': True,
                'watch_directories': ['./data/test_sandbox'],
                'watch_extensions': ['.txt', '.doc', '.pdf', '.json', '.csv', '.py']
            }
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    loaded_config = yaml.safe_load(f)
                    if loaded_config:
                        # دمج التكوينات | Fusionner les configurations
                        self._deep_merge(default_config, loaded_config)
                logger.info(f"تم تحميل التكوين من | Configuration chargée de: {config_path}")
            except Exception as e:
                logger.warning(f"خطأ في تحميل التكوين | Erreur de chargement: {e}")
        
        return default_config
    
    def _deep_merge(self, base: Dict, update: Dict):
        """
        دمج عميق للقواميس | Fusion profonde des dictionnaires
        """
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value
    
    def _init_monitors(self):
        """
        إنشاء المراقبين | Créer les moniteurs
        """
        interval = self.config.get('collection_interval', 0.5)
        
        # مراقب العمليات | Moniteur de processus
        proc_config = self.config.get('process_monitor', {})
        if proc_config.get('enabled', True):
            self.process_monitor = ProcessMonitor(
                interval=interval,
                excluded_processes=proc_config.get('excluded_processes', []),
                callback=self._on_process_event
            )
        else:
            self.process_monitor = None
        
        # مراقب الشبكة | Moniteur réseau
        net_config = self.config.get('network_monitor', {})
        if net_config.get('enabled', True):
            self.network_monitor = NetworkMonitor(
                interval=interval,
                excluded_ports=net_config.get('excluded_ports', []),
                callback=self._on_network_event
            )
        else:
            self.network_monitor = None
        
        # مراقب الملفات | Moniteur de fichiers
        file_config = self.config.get('file_monitor', {})
        if file_config.get('enabled', True):
            watch_dirs = file_config.get('watch_directories', [str(self.watch_dir)])
            # تأكد من وجود المجلدات | S'assurer que les répertoires existent
            for d in watch_dirs:
                Path(d).mkdir(parents=True, exist_ok=True)
            
            try:
                self.file_monitor = FileMonitor(
                    watch_directories=watch_dirs,
                    watch_extensions=file_config.get('watch_extensions', []),
                    callback=self._on_file_event,
                    recursive=True
                )
            except ImportError:
                # استخدام المراقب البسيط كبديل
                logger.warning("استخدام مراقب الملفات البسيط | Utilisation du moniteur simple")
                self.file_monitor = SimpleFileMonitor(
                    watch_directories=watch_dirs,
                    interval=interval,
                    callback=self._on_file_event
                )
        else:
            self.file_monitor = None
    
    def _create_unified_event(self, source: str, event_type: str, data: Dict) -> UnifiedEvent:
        """
        إنشاء حدث موحد | Créer un événement unifié
        """
        now = datetime.now()
        return UnifiedEvent(
            timestamp=time.time() * 1000,
            timestamp_iso=now.isoformat(),
            source=source,
            event_type=event_type,
            data=data,
            label=self._current_label
        )
    
    def _add_event(self, event: UnifiedEvent):
        """
        إضافة حدث إلى القائمة | Ajouter un événement à la liste
        """
        with self._lock:
            self._events.append(event)
            self._stats['total_events'] += 1
        
        # كتابة إلى الملف إذا كان مفتوحاً | Écrire dans le fichier si ouvert
        if self._output_file:
            try:
                self._output_file.write(event.to_json() + '\n')
                self._output_file.flush()
            except Exception as e:
                logger.error(f"خطأ في الكتابة | Erreur d'écriture: {e}")
    
    def _on_process_event(self, event: ProcessEvent):
        """
        معالجة حدث العملية | Traiter un événement processus
        """
        unified = self._create_unified_event(
            source="process",
            event_type=event.event_type,
            data=event.to_dict()
        )
        self._add_event(unified)
        self._stats['process_events'] += 1
    
    def _on_network_event(self, event: NetworkEvent):
        """
        معالجة حدث الشبكة | Traiter un événement réseau
        """
        unified = self._create_unified_event(
            source="network",
            event_type=event.event_type,
            data=event.to_dict()
        )
        self._add_event(unified)
        self._stats['network_events'] += 1
    
    def _on_file_event(self, event: FileEvent):
        """
        معالجة حدث الملف | Traiter un événement fichier
        """
        unified = self._create_unified_event(
            source="file",
            event_type=event.event_type,
            data=event.to_dict()
        )
        self._add_event(unified)
        self._stats['file_events'] += 1
    
    def set_label(self, label: str):
        """
        تعيين التسمية الحالية للأحداث | Définir le label actuel
        
        Args:
            label: التسمية (benign/malicious) | Label
        """
        self._current_label = label
        logger.info(f"تم تعيين التسمية | Label défini: {label}")
    
    def start(self, output_file: Optional[str] = None):
        """
        بدء الجمع | Démarrer la collecte
        
        Args:
            output_file: ملف الإخراج (اختياري) | Fichier de sortie (optionnel)
        """
        logger.info("=" * 60)
        logger.info("بدء وكيل جمع السلوك | Démarrage de l'agent de collecte")
        logger.info("=" * 60)
        
        self._stats['start_time'] = time.time()
        
        # فتح ملف الإخراج | Ouvrir le fichier de sortie
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            self._output_file = open(output_path, 'a', encoding='utf-8')
            logger.info(f"الكتابة إلى | Écriture vers: {output_file}")
        
        # بدء المراقبين | Démarrer les moniteurs
        if self.process_monitor:
            self.process_monitor.start()
            logger.info("✅ مراقب العمليات | Moniteur processus")
        
        if self.network_monitor:
            self.network_monitor.start()
            logger.info("✅ مراقب الشبكة | Moniteur réseau")
        
        if self.file_monitor:
            self.file_monitor.start()
            logger.info("✅ مراقب الملفات | Moniteur fichiers")
        
        logger.info("الوكيل يعمل الآن | Agent en cours d'exécution")
    
    def stop(self):
        """
        إيقاف الجمع | Arrêter la collecte
        """
        logger.info("إيقاف وكيل جمع السلوك | Arrêt de l'agent de collecte")
        
        # إيقاف المراقبين | Arrêter les moniteurs
        if self.process_monitor:
            self.process_monitor.stop()
        
        if self.network_monitor:
            self.network_monitor.stop()
        
        if self.file_monitor:
            self.file_monitor.stop()
        
        # إغلاق ملف الإخراج | Fermer le fichier de sortie
        if self._output_file:
            self._output_file.close()
            self._output_file = None
        
        # طباعة الإحصائيات | Afficher les statistiques
        self._print_stats()
    
    def _print_stats(self):
        """
        طباعة الإحصائيات | Afficher les statistiques
        """
        duration = time.time() - self._stats['start_time'] if self._stats['start_time'] else 0
        
        print("\n" + "=" * 60)
        print("📊 إحصائيات الجمع | Statistiques de Collecte")
        print("=" * 60)
        print(f"⏱️  المدة | Durée: {duration:.1f} ثانية | secondes")
        print(f"📦 إجمالي الأحداث | Total événements: {self._stats['total_events']}")
        print(f"   - العمليات | Processus: {self._stats['process_events']}")
        print(f"   - الشبكة | Réseau: {self._stats['network_events']}")
        print(f"   - الملفات | Fichiers: {self._stats['file_events']}")
        if duration > 0:
            rate = self._stats['total_events'] / duration
            print(f"📈 المعدل | Taux: {rate:.1f} حدث/ثانية | événements/s")
        print("=" * 60)
    
    def get_events(self, clear: bool = False) -> List[UnifiedEvent]:
        """
        الحصول على الأحداث المجمعة | Obtenir les événements collectés
        """
        with self._lock:
            events = list(self._events)
            if clear:
                self._events.clear()
        return events
    
    def get_events_as_dicts(self, clear: bool = False) -> List[Dict]:
        """
        الحصول على الأحداث كقواميس | Obtenir les événements comme dicts
        """
        return [e.to_dict() for e in self.get_events(clear)]
    
    def get_stats(self) -> Dict:
        """
        الحصول على الإحصائيات | Obtenir les statistiques
        """
        return self._stats.copy()
    
    def save_to_file(self, filepath: str, format: str = 'jsonl'):
        """
        حفظ الأحداث إلى ملف | Sauvegarder les événements dans un fichier
        
        Args:
            filepath: مسار الملف | Chemin du fichier
            format: التنسيق (jsonl/csv) | Format
        """
        events = self.get_events()
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'jsonl':
            with open(filepath, 'w', encoding='utf-8') as f:
                for event in events:
                    f.write(event.to_json() + '\n')
        
        elif format == 'csv':
            import csv
            if events:
                # استخراج جميع المفاتيح | Extraire toutes les clés
                all_keys = set()
                for event in events:
                    all_keys.update(event.to_dict().keys())
                    all_keys.update(event.data.keys())
                
                with open(filepath, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=sorted(all_keys))
                    writer.writeheader()
                    for event in events:
                        row = event.to_dict()
                        row.update(event.data)
                        writer.writerow(row)
        
        logger.info(f"تم حفظ {len(events)} حدث إلى | {len(events)} événements sauvegardés: {filepath}")


def main():
    """
    الدالة الرئيسية | Fonction principale
    """
    parser = argparse.ArgumentParser(
        description="وكيل جمع السلوك | Agent de Collecte Comportementale"
    )
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='config/config.yaml',
        help='مسار ملف التكوين | Chemin du fichier de configuration'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='ملف الإخراج | Fichier de sortie'
    )
    parser.add_argument(
        '--duration', '-d',
        type=int,
        default=60,
        help='مدة الجمع بالثواني | Durée de collecte en secondes'
    )
    parser.add_argument(
        '--label', '-l',
        type=str,
        default='benign',
        choices=['benign', 'malicious'],
        help='تسمية الأحداث | Label des événements'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='وضع الاختبار | Mode test'
    )
    
    args = parser.parse_args()
    
    # إنشاء الوكيل | Créer l'agent
    collector = BehaviorCollector(config_path=args.config)
    collector.set_label(args.label)
    
    # تحديد ملف الإخراج | Définir le fichier de sortie
    if args.output:
        output_file = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"data/raw/events_{args.label}_{timestamp}.jsonl"
    
    try:
        # بدء الجمع | Démarrer la collecte
        collector.start(output_file=output_file)
        
        if args.test:
            print("\n🧪 وضع الاختبار | Mode Test (5 ثوان)")
            time.sleep(5)
        else:
            print(f"\n⏳ الجمع لمدة {args.duration} ثانية... | Collecte pendant {args.duration} secondes...")
            print("اضغط Ctrl+C للإيقاف | Appuyez sur Ctrl+C pour arrêter")
            time.sleep(args.duration)
    
    except KeyboardInterrupt:
        print("\n\n⚠️ توقف بواسطة المستخدم | Arrêt par l'utilisateur")
    
    finally:
        collector.stop()


if __name__ == "__main__":
    main()
