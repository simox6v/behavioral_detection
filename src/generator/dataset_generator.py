"""
مولد مجموعة البيانات | Générateur de Dataset | Dataset Generator
يجمع بين السيناريوهات الحميدة والمشبوهة لتوليد مجموعة بيانات التدريب
Combine les scénarios bénins et malveillants pour générer le dataset d'entraînement
"""

import os
import sys
import json
import time
import yaml
import argparse
import threading
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from datetime import datetime
import logging

# استيراد السيناريوهات | Importer les scénarios
from .benign_scenarios import BenignScenarios
from .malicious_scenarios import MaliciousScenarios

# استيراد الجامع | Importer le collecteur
try:
    from ..collector.behavior_collector import BehaviorCollector
except ImportError:
    BehaviorCollector = None

# إعداد التسجيل | Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DatasetGenerator:
    """
    مولد مجموعة البيانات | Générateur de Dataset
    يولد مجموعة بيانات متوازنة للتدريب
    Génère un dataset équilibré pour l'entraînement
    """
    
    def __init__(
        self,
        output_dir: Optional[str] = None,
        config_path: Optional[str] = None
    ):
        """
        تهيئة المولد | Initialisation du générateur
        
        Args:
            output_dir: مجلد الإخراج | Répertoire de sortie
            config_path: مسار ملف التكوين | Chemin du fichier de configuration
        """
        self.config = self._load_config(config_path)
        
        # مسارات الإخراج | Chemins de sortie
        self.output_dir = Path(output_dir or self.config.get('output_dir', './data'))
        self.raw_dir = self.output_dir / 'raw'
        self.processed_dir = self.output_dir / 'processed'
        self.sandbox_dir = self.output_dir / 'sandbox'
        
        # إنشاء المجلدات | Créer les répertoires
        for d in [self.raw_dir, self.processed_dir, self.sandbox_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        # إعداد السيناريوهات | Configurer les scénarios
        self.benign_scenarios = BenignScenarios(
            sandbox_dir=str(self.sandbox_dir / 'benign')
        )
        self.malicious_scenarios = MaliciousScenarios(
            sandbox_dir=str(self.sandbox_dir / 'malicious')
        )
        
        # الإحصائيات | Statistiques
        self._stats = {
            'benign_events': 0,
            'malicious_events': 0,
            'generation_time': 0
        }
        
        logger.info(f"تم تهيئة مولد البيانات | Générateur initialisé: {self.output_dir}")
    
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """
        تحميل التكوين | Charger la configuration
        """
        default_config = {
            'output_dir': './data',
            'output_format': 'jsonl',
            'benign_events': 10000,
            'malicious_events': 8000,
            'duration_per_scenario': 30,
            'intensity': 'normal'
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    loaded = yaml.safe_load(f)
                    if loaded and 'dataset' in loaded:
                        default_config.update(loaded['dataset'])
            except Exception as e:
                logger.warning(f"خطأ في تحميل التكوين | Erreur chargement config: {e}")
        
        return default_config
    
    def generate_benign_dataset(
        self,
        target_events: int = 10000,
        duration_per_scenario: float = 30,
        output_file: Optional[str] = None
    ) -> Tuple[str, int]:
        """
        توليد مجموعة البيانات الحميدة
        Générer le dataset bénin
        
        Args:
            target_events: عدد الأحداث المستهدف | Nombre d'événements cible
            duration_per_scenario: مدة كل سيناريو | Durée par scénario
            output_file: ملف الإخراج | Fichier de sortie
            
        Returns:
            مسار الملف وعدد الأحداث | Chemin et nombre d'événements
        """
        logger.info("=" * 60)
        logger.info("🌿 توليد البيانات الحميدة | Génération données bénignes")
        logger.info("=" * 60)
        
        # تحديد ملف الإخراج | Définir le fichier de sortie
        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = str(self.raw_dir / f"benign_{timestamp}.jsonl")
        
        events = []
        event_count = 0
        
        def on_event(scenario_name, count):
            nonlocal event_count
            event_count = count
        
        # تشغيل السيناريوهات | Exécuter les scénarios
        scenarios = [
            ("web_browsing", self.benign_scenarios.simulate_web_browsing),
            ("office_work", self.benign_scenarios.simulate_office_work),
            ("compilation", self.benign_scenarios.simulate_compilation),
            ("file_copy", self.benign_scenarios.simulate_file_copy),
            ("system_update", self.benign_scenarios.simulate_system_update),
        ]
        
        total_events = 0
        
        for name, func in scenarios:
            logger.info(f"▶️ تشغيل | Exécution: {name}")
            
            # حساب المدة بناءً على الأحداث المتبقية
            remaining = target_events - total_events
            if remaining <= 0:
                break
            
            # تقدير المدة | Estimer la durée
            adjusted_duration = min(duration_per_scenario, max(5, remaining / 100))
            
            start_count = event_count
            func(duration=adjusted_duration, intensity='high', callback=on_event)
            scenario_events = event_count - start_count
            
            # توليد أحداث للملف | Générer des événements pour le fichier
            for i in range(scenario_events):
                event = {
                    'timestamp': time.time() * 1000 + i,
                    'timestamp_iso': datetime.now().isoformat(),
                    'source': 'generated',
                    'event_type': name,
                    'scenario': name,
                    'label': 'benign',
                    'data': {
                        'scenario_name': name,
                        'event_index': i
                    }
                }
                events.append(event)
            
            total_events += scenario_events
            logger.info(f"   ✅ {scenario_events} أحداث | événements")
        
        # كتابة الملف | Écrire le fichier
        with open(output_file, 'w', encoding='utf-8') as f:
            for event in events:
                f.write(json.dumps(event, ensure_ascii=False) + '\n')
        
        self._stats['benign_events'] = total_events
        logger.info(f"✅ تم حفظ {total_events} حدث في | Sauvegardé: {output_file}")
        
        return output_file, total_events
    
    def generate_malicious_dataset(
        self,
        target_events: int = 8000,
        duration_per_scenario: float = 30,
        output_file: Optional[str] = None
    ) -> Tuple[str, int]:
        """
        توليد مجموعة البيانات المشبوهة
        Générer le dataset malveillant
        
        Args:
            target_events: عدد الأحداث المستهدف | Nombre d'événements cible
            duration_per_scenario: مدة كل سيناريو | Durée par scénario
            output_file: ملف الإخراج | Fichier de sortie
            
        Returns:
            مسار الملف وعدد الأحداث | Chemin et nombre d'événements
        """
        logger.info("=" * 60)
        logger.info("🔴 توليد البيانات المشبوهة | Génération données malveillantes")
        logger.info("⚠️ هذه محاكاة تعليمية فقط | Simulation éducative uniquement")
        logger.info("=" * 60)
        
        # تحديد ملف الإخراج | Définir le fichier de sortie
        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = str(self.raw_dir / f"malicious_{timestamp}.jsonl")
        
        events = []
        event_count = 0
        
        def on_event(scenario_name, count):
            nonlocal event_count
            event_count = count
        
        # تشغيل السيناريوهات | Exécuter les scénarios
        scenarios = [
            ("file_burst", lambda: self.malicious_scenarios.simulate_file_burst(
                duration=duration_per_scenario, files_count=500, callback=on_event)),
            ("port_scan", lambda: self.malicious_scenarios.simulate_port_scan(
                duration=duration_per_scenario, callback=on_event)),
            ("sensitive_access", lambda: self.malicious_scenarios.simulate_sensitive_file_access(
                duration=duration_per_scenario, callback=on_event)),
            ("ransomware", lambda: self.malicious_scenarios.simulate_ransomware_behavior(
                duration=duration_per_scenario, files_to_encrypt=200, callback=on_event)),
            ("bruteforce", lambda: self.malicious_scenarios.simulate_bruteforce(
                duration=duration_per_scenario, callback=on_event)),
        ]
        
        total_events = 0
        
        for name, func in scenarios:
            logger.info(f"▶️ تشغيل | Exécution: {name}")
            
            start_count = event_count
            func()
            scenario_events = event_count - start_count
            
            # توليد أحداث للملف | Générer des événements pour le fichier
            for i in range(scenario_events):
                event = {
                    'timestamp': time.time() * 1000 + i,
                    'timestamp_iso': datetime.now().isoformat(),
                    'source': 'generated',
                    'event_type': name,
                    'scenario': name,
                    'label': 'malicious',
                    'data': {
                        'scenario_name': name,
                        'event_index': i,
                        'attack_type': name
                    }
                }
                events.append(event)
            
            total_events += scenario_events
            logger.info(f"   ✅ {scenario_events} أحداث | événements")
        
        # كتابة الملف | Écrire le fichier
        with open(output_file, 'w', encoding='utf-8') as f:
            for event in events:
                f.write(json.dumps(event, ensure_ascii=False) + '\n')
        
        self._stats['malicious_events'] = total_events
        logger.info(f"✅ تم حفظ {total_events} حدث في | Sauvegardé: {output_file}")
        
        return output_file, total_events
    
    def generate_combined_dataset(
        self,
        benign_events: int = 10000,
        malicious_events: int = 8000,
        duration_per_scenario: float = 30,
        shuffle: bool = True
    ) -> Tuple[str, Dict]:
        """
        توليد مجموعة بيانات مجمعة
        Générer un dataset combiné
        
        Args:
            benign_events: عدد الأحداث الحميدة | Nombre d'événements bénins
            malicious_events: عدد الأحداث المشبوهة | Nombre d'événements malveillants
            duration_per_scenario: مدة كل سيناريو | Durée par scénario
            shuffle: خلط البيانات | Mélanger les données
            
        Returns:
            مسار الملف والإحصائيات | Chemin et statistiques
        """
        import random
        
        logger.info("=" * 60)
        logger.info("🎯 توليد مجموعة البيانات المجمعة | Génération dataset combiné")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        # توليد البيانات | Générer les données
        benign_file, actual_benign = self.generate_benign_dataset(
            target_events=benign_events,
            duration_per_scenario=duration_per_scenario
        )
        
        malicious_file, actual_malicious = self.generate_malicious_dataset(
            target_events=malicious_events,
            duration_per_scenario=duration_per_scenario
        )
        
        # دمج الملفات | Fusionner les fichiers
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        combined_file = str(self.processed_dir / f"combined_dataset_{timestamp}.jsonl")
        
        all_events = []
        
        # قراءة الأحداث | Lire les événements
        with open(benign_file, 'r', encoding='utf-8') as f:
            for line in f:
                all_events.append(json.loads(line))
        
        with open(malicious_file, 'r', encoding='utf-8') as f:
            for line in f:
                all_events.append(json.loads(line))
        
        # خلط إذا طُلب | Mélanger si demandé
        if shuffle:
            random.shuffle(all_events)
        
        # كتابة الملف المجمع | Écrire le fichier combiné
        with open(combined_file, 'w', encoding='utf-8') as f:
            for event in all_events:
                f.write(json.dumps(event, ensure_ascii=False) + '\n')
        
        generation_time = time.time() - start_time
        self._stats['generation_time'] = generation_time
        
        stats = {
            'benign_events': actual_benign,
            'malicious_events': actual_malicious,
            'total_events': len(all_events),
            'generation_time': generation_time,
            'combined_file': combined_file,
            'benign_file': benign_file,
            'malicious_file': malicious_file
        }
        
        # حفظ الإحصائيات | Sauvegarder les statistiques
        stats_file = str(self.processed_dir / f"dataset_stats_{timestamp}.json")
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        # طباعة الملخص | Afficher le résumé
        self._print_summary(stats)
        
        return combined_file, stats
    
    def _print_summary(self, stats: Dict):
        """
        طباعة ملخص التوليد | Afficher le résumé de génération
        """
        print("\n" + "=" * 60)
        print("📊 ملخص توليد البيانات | Résumé de Génération")
        print("=" * 60)
        print(f"🌿 الأحداث الحميدة | Événements bénins: {stats['benign_events']}")
        print(f"🔴 الأحداث المشبوهة | Événements malveillants: {stats['malicious_events']}")
        print(f"📦 إجمالي الأحداث | Total événements: {stats['total_events']}")
        print(f"⏱️  وقت التوليد | Temps de génération: {stats['generation_time']:.1f}s")
        print(f"📁 الملف المجمع | Fichier combiné: {stats['combined_file']}")
        print("=" * 60)
    
    def cleanup(self):
        """
        تنظيف ملفات المحاكاة | Nettoyer les fichiers de simulation
        """
        self.benign_scenarios.cleanup()
        self.malicious_scenarios.cleanup()
        logger.info("تم التنظيف | Nettoyage effectué")
    
    def validate_dataset(self, filepath: str) -> Dict:
        """
        التحقق من صحة مجموعة البيانات
        Valider le dataset
        
        Args:
            filepath: مسار الملف | Chemin du fichier
            
        Returns:
            نتائج التحقق | Résultats de validation
        """
        logger.info(f"🔍 التحقق من | Validation de: {filepath}")
        
        stats = {
            'total_lines': 0,
            'valid_events': 0,
            'invalid_events': 0,
            'benign_count': 0,
            'malicious_count': 0,
            'scenarios': {},
            'errors': []
        }
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f, 1):
                    stats['total_lines'] += 1
                    try:
                        event = json.loads(line)
                        stats['valid_events'] += 1
                        
                        # عد التصنيفات | Compter les labels
                        label = event.get('label', 'unknown')
                        if label == 'benign':
                            stats['benign_count'] += 1
                        elif label == 'malicious':
                            stats['malicious_count'] += 1
                        
                        # عد السيناريوهات | Compter les scénarios
                        scenario = event.get('scenario', event.get('event_type', 'unknown'))
                        stats['scenarios'][scenario] = stats['scenarios'].get(scenario, 0) + 1
                        
                    except json.JSONDecodeError as e:
                        stats['invalid_events'] += 1
                        stats['errors'].append(f"Line {i}: {str(e)}")
        
        except Exception as e:
            logger.error(f"خطأ في التحقق | Erreur de validation: {e}")
            stats['errors'].append(str(e))
        
        # طباعة النتائج | Afficher les résultats
        print("\n" + "=" * 60)
        print("📋 نتائج التحقق | Résultats de Validation")
        print("=" * 60)
        print(f"📄 إجمالي الأسطر | Total lignes: {stats['total_lines']}")
        print(f"✅ أحداث صالحة | Événements valides: {stats['valid_events']}")
        print(f"❌ أحداث غير صالحة | Événements invalides: {stats['invalid_events']}")
        print(f"🌿 حميدة | Bénins: {stats['benign_count']}")
        print(f"🔴 مشبوهة | Malveillants: {stats['malicious_count']}")
        print("\n📊 السيناريوهات | Scénarios:")
        for scenario, count in sorted(stats['scenarios'].items()):
            print(f"   - {scenario}: {count}")
        print("=" * 60)
        
        return stats


def main():
    """
    الدالة الرئيسية | Fonction principale
    """
    parser = argparse.ArgumentParser(
        description="مولد مجموعة البيانات | Générateur de Dataset"
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
        default='./data',
        help='مجلد الإخراج | Répertoire de sortie'
    )
    parser.add_argument(
        '--benign', '-b',
        type=int,
        default=10000,
        help='عدد الأحداث الحميدة | Nombre d\'événements bénins'
    )
    parser.add_argument(
        '--malicious', '-m',
        type=int,
        default=8000,
        help='عدد الأحداث المشبوهة | Nombre d\'événements malveillants'
    )
    parser.add_argument(
        '--duration', '-d',
        type=float,
        default=30,
        help='مدة كل سيناريو بالثواني | Durée par scénario en secondes'
    )
    parser.add_argument(
        '--validate',
        type=str,
        default=None,
        help='التحقق من ملف | Valider un fichier'
    )
    
    args = parser.parse_args()
    
    generator = DatasetGenerator(
        output_dir=args.output,
        config_path=args.config
    )
    
    try:
        if args.validate:
            # التحقق من ملف موجود | Valider un fichier existant
            generator.validate_dataset(args.validate)
        else:
            # توليد مجموعة بيانات جديدة | Générer un nouveau dataset
            combined_file, stats = generator.generate_combined_dataset(
                benign_events=args.benign,
                malicious_events=args.malicious,
                duration_per_scenario=args.duration
            )
            
            # التحقق من الملف المولد | Valider le fichier généré
            generator.validate_dataset(combined_file)
    
    except KeyboardInterrupt:
        print("\n\n⚠️ توقف بواسطة المستخدم | Arrêt par l'utilisateur")
    
    finally:
        generator.cleanup()


if __name__ == "__main__":
    main()
