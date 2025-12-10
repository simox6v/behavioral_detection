"""
السيناريوهات الحميدة | Scénarios Bénins | Benign Scenarios
محاكاة السلوك العادي للنظام
Simulation du comportement normal du système
"""

import os
import time
import random
import string
import tempfile
import threading
from typing import Callable, Optional, List
from pathlib import Path
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class BenignScenarios:
    """
    مولد السيناريوهات الحميدة | Générateur de Scénarios Bénins
    يحاكي السلوك العادي للنظام لتوليد بيانات التدريب
    Simule le comportement normal du système pour générer des données d'entraînement
    """
    
    def __init__(self, sandbox_dir: Optional[str] = None):
        """
        تهيئة المولد | Initialisation du générateur
        
        Args:
            sandbox_dir: مجلد المحاكاة | Répertoire sandbox
        """
        self.sandbox_dir = Path(sandbox_dir or tempfile.mkdtemp(prefix="benign_"))
        self.sandbox_dir.mkdir(parents=True, exist_ok=True)
        
        self._running = False
        self._threads: List[threading.Thread] = []
        
        logger.info(f"تهيئة السيناريوهات الحميدة في | Scénarios bénins initialisés: {self.sandbox_dir}")
    
    def _random_string(self, length: int = 10) -> str:
        """توليد سلسلة عشوائية | Générer une chaîne aléatoire"""
        return ''.join(random.choices(string.ascii_letters + string.digits, k=length))
    
    def _random_content(self, size: int = 1000) -> str:
        """توليد محتوى عشوائي | Générer un contenu aléatoire"""
        words = ['lorem', 'ipsum', 'dolor', 'sit', 'amet', 'consectetur', 
                 'adipiscing', 'elit', 'sed', 'do', 'eiusmod', 'tempor',
                 'incididunt', 'ut', 'labore', 'et', 'dolore', 'magna', 'aliqua']
        content = []
        while len(' '.join(content)) < size:
            content.append(random.choice(words))
        return ' '.join(content)[:size]
    
    # ==================== السيناريو 1: التصفح العادي ====================
    # ==================== Scénario 1: Navigation Normale ====================
    
    def simulate_web_browsing(
        self,
        duration: float = 30,
        intensity: str = "normal",
        callback: Optional[Callable] = None
    ):
        """
        محاكاة تصفح الويب العادي
        Simuler la navigation web normale
        
        يحاكي: قراءة/كتابة cache، ملفات مؤقتة
        Simule: lecture/écriture cache, fichiers temporaires
        """
        logger.info("🌐 بدء محاكاة التصفح | Démarrage simulation navigation")
        
        cache_dir = self.sandbox_dir / "browser_cache"
        cache_dir.mkdir(exist_ok=True)
        
        intervals = {"low": 2.0, "normal": 0.5, "high": 0.1}
        interval = intervals.get(intensity, 0.5)
        
        end_time = time.time() + duration
        event_count = 0
        
        while time.time() < end_time and self._running:
            try:
                # إنشاء ملف cache | Créer un fichier cache
                cache_file = cache_dir / f"cache_{self._random_string(8)}.tmp"
                content = self._random_content(random.randint(100, 5000))
                cache_file.write_text(content)
                event_count += 1
                
                # قراءة بعض ملفات cache | Lire quelques fichiers cache
                cache_files = list(cache_dir.glob("*.tmp"))
                if cache_files:
                    selected = random.choice(cache_files)
                    _ = selected.read_text()
                    event_count += 1
                
                # حذف ملفات cache قديمة أحياناً | Supprimer parfois les vieux cache
                if random.random() < 0.1 and len(cache_files) > 5:
                    oldest = random.choice(cache_files[:5])
                    if oldest.exists():
                        oldest.unlink()
                        event_count += 1
                
                if callback:
                    callback("web_browsing", event_count)
                
                time.sleep(interval + random.uniform(0, interval))
                
            except Exception as e:
                logger.error(f"خطأ في محاكاة التصفح | Erreur simulation: {e}")
        
        logger.info(f"✅ انتهاء التصفح | Navigation terminée: {event_count} أحداث")
        return event_count
    
    # ==================== السيناريو 2: العمل المكتبي ====================
    # ==================== Scénario 2: Travail Bureautique ====================
    
    def simulate_office_work(
        self,
        duration: float = 30,
        intensity: str = "normal",
        callback: Optional[Callable] = None
    ):
        """
        محاكاة العمل المكتبي
        Simuler le travail bureautique
        
        يحاكي: فتح/حفظ مستندات، تعديل ملفات
        Simule: ouvrir/sauvegarder documents, modifier fichiers
        """
        logger.info("📄 بدء محاكاة العمل المكتبي | Démarrage travail bureautique")
        
        docs_dir = self.sandbox_dir / "documents"
        docs_dir.mkdir(exist_ok=True)
        
        intervals = {"low": 3.0, "normal": 1.0, "high": 0.3}
        interval = intervals.get(intensity, 1.0)
        
        extensions = ['.txt', '.doc', '.csv', '.json']
        end_time = time.time() + duration
        event_count = 0
        
        while time.time() < end_time and self._running:
            try:
                action = random.choice(['create', 'read', 'modify', 'save'])
                
                if action == 'create':
                    # إنشاء مستند جديد | Créer un nouveau document
                    ext = random.choice(extensions)
                    doc_file = docs_dir / f"document_{self._random_string(6)}{ext}"
                    content = self._random_content(random.randint(500, 3000))
                    doc_file.write_text(content)
                    event_count += 1
                
                elif action == 'read':
                    # قراءة مستند | Lire un document
                    doc_files = list(docs_dir.glob("*.*"))
                    if doc_files:
                        selected = random.choice(doc_files)
                        _ = selected.read_text()
                        event_count += 1
                
                elif action == 'modify':
                    # تعديل مستند | Modifier un document
                    doc_files = list(docs_dir.glob("*.*"))
                    if doc_files:
                        selected = random.choice(doc_files)
                        content = selected.read_text()
                        content += f"\n{self._random_content(100)}"
                        selected.write_text(content)
                        event_count += 2
                
                elif action == 'save':
                    # حفظ نسخة | Sauvegarder une copie
                    doc_files = list(docs_dir.glob("*.*"))
                    if doc_files:
                        selected = random.choice(doc_files)
                        backup = docs_dir / f"{selected.stem}_backup{selected.suffix}"
                        backup.write_text(selected.read_text())
                        event_count += 2
                
                if callback:
                    callback("office_work", event_count)
                
                time.sleep(interval + random.uniform(0, interval * 0.5))
                
            except Exception as e:
                logger.error(f"خطأ في العمل المكتبي | Erreur bureautique: {e}")
        
        logger.info(f"✅ انتهاء العمل المكتبي | Bureautique terminé: {event_count} أحداث")
        return event_count
    
    # ==================== السيناريو 3: الترجمة البرمجية ====================
    # ==================== Scénario 3: Compilation ====================
    
    def simulate_compilation(
        self,
        duration: float = 30,
        intensity: str = "normal",
        callback: Optional[Callable] = None
    ):
        """
        محاكاة الترجمة البرمجية
        Simuler la compilation
        
        يحاكي: إنشاء ملفات مؤقتة، عمليات I/O مكثفة
        Simule: création fichiers temporaires, I/O intensives
        """
        logger.info("🔨 بدء محاكاة الترجمة | Démarrage compilation")
        
        build_dir = self.sandbox_dir / "build"
        build_dir.mkdir(exist_ok=True)
        src_dir = self.sandbox_dir / "src"
        src_dir.mkdir(exist_ok=True)
        
        intervals = {"low": 1.0, "normal": 0.2, "high": 0.05}
        interval = intervals.get(intensity, 0.2)
        
        end_time = time.time() + duration
        event_count = 0
        
        # إنشاء ملفات مصدرية | Créer des fichiers sources
        for i in range(10):
            src_file = src_dir / f"module_{i}.py"
            code = f'''"""Module {i}"""
def function_{i}():
    return {i}

class Class{i}:
    def __init__(self):
        self.value = {i}
'''
            src_file.write_text(code)
        
        while time.time() < end_time and self._running:
            try:
                # قراءة ملف مصدري | Lire un fichier source
                src_files = list(src_dir.glob("*.py"))
                if src_files:
                    selected = random.choice(src_files)
                    _ = selected.read_text()
                    event_count += 1
                
                # إنشاء ملف object | Créer un fichier object
                obj_file = build_dir / f"obj_{self._random_string(6)}.o"
                obj_file.write_bytes(os.urandom(random.randint(1000, 10000)))
                event_count += 1
                
                # إنشاء ملفات مؤقتة | Créer des fichiers temporaires
                tmp_file = build_dir / f"tmp_{self._random_string(4)}.tmp"
                tmp_file.write_text(self._random_content(500))
                event_count += 1
                
                # حذف ملفات مؤقتة | Supprimer des fichiers temporaires
                tmp_files = list(build_dir.glob("*.tmp"))
                if len(tmp_files) > 10:
                    for f in tmp_files[:5]:
                        if f.exists():
                            f.unlink()
                            event_count += 1
                
                if callback:
                    callback("compilation", event_count)
                
                time.sleep(interval)
                
            except Exception as e:
                logger.error(f"خطأ في الترجمة | Erreur compilation: {e}")
        
        logger.info(f"✅ انتهاء الترجمة | Compilation terminée: {event_count} أحداث")
        return event_count
    
    # ==================== السيناريو 4: نسخ الملفات ====================
    # ==================== Scénario 4: Copie de Fichiers ====================
    
    def simulate_file_copy(
        self,
        duration: float = 30,
        intensity: str = "normal",
        callback: Optional[Callable] = None
    ):
        """
        محاكاة نسخ الملفات
        Simuler la copie de fichiers
        
        يحاكي: عمليات I/O عادية، نسخ/نقل ملفات
        Simule: opérations I/O normales, copie/déplacement fichiers
        """
        logger.info("📁 بدء محاكاة نسخ الملفات | Démarrage copie fichiers")
        
        source_dir = self.sandbox_dir / "source"
        dest_dir = self.sandbox_dir / "destination"
        source_dir.mkdir(exist_ok=True)
        dest_dir.mkdir(exist_ok=True)
        
        intervals = {"low": 2.0, "normal": 0.5, "high": 0.1}
        interval = intervals.get(intensity, 0.5)
        
        # إنشاء ملفات مصدرية | Créer des fichiers sources
        for i in range(20):
            f = source_dir / f"file_{i}.dat"
            f.write_bytes(os.urandom(random.randint(100, 5000)))
        
        end_time = time.time() + duration
        event_count = 0
        
        while time.time() < end_time and self._running:
            try:
                action = random.choice(['copy', 'read', 'move_back'])
                
                if action == 'copy':
                    # نسخ ملف | Copier un fichier
                    src_files = list(source_dir.glob("*.*"))
                    if src_files:
                        selected = random.choice(src_files)
                        content = selected.read_bytes()
                        dest_file = dest_dir / f"{selected.stem}_copy_{self._random_string(4)}{selected.suffix}"
                        dest_file.write_bytes(content)
                        event_count += 2
                
                elif action == 'read':
                    # قراءة ملفات | Lire des fichiers
                    all_files = list(source_dir.glob("*.*")) + list(dest_dir.glob("*.*"))
                    if all_files:
                        selected = random.choice(all_files)
                        _ = selected.read_bytes()
                        event_count += 1
                
                elif action == 'move_back':
                    # إعادة ملف إلى المصدر | Remettre un fichier à la source
                    dest_files = list(dest_dir.glob("*.*"))
                    if dest_files and len(dest_files) > 5:
                        selected = random.choice(dest_files)
                        if selected.exists():
                            selected.unlink()
                            event_count += 1
                
                if callback:
                    callback("file_copy", event_count)
                
                time.sleep(interval)
                
            except Exception as e:
                logger.error(f"خطأ في نسخ الملفات | Erreur copie: {e}")
        
        logger.info(f"✅ انتهاء نسخ الملفات | Copie terminée: {event_count} أحداث")
        return event_count
    
    # ==================== السيناريو 5: تحديث النظام ====================
    # ==================== Scénario 5: Mise à Jour Système ====================
    
    def simulate_system_update(
        self,
        duration: float = 30,
        intensity: str = "normal",
        callback: Optional[Callable] = None
    ):
        """
        محاكاة تحديث النظام
        Simuler la mise à jour système
        
        يحاكي: تحميل/تثبيت حزم، تحديث ملفات
        Simule: téléchargement/installation paquets, mise à jour fichiers
        """
        logger.info("🔄 بدء محاكاة تحديث النظام | Démarrage mise à jour")
        
        update_dir = self.sandbox_dir / "updates"
        update_dir.mkdir(exist_ok=True)
        install_dir = self.sandbox_dir / "installed"
        install_dir.mkdir(exist_ok=True)
        
        intervals = {"low": 2.0, "normal": 0.8, "high": 0.2}
        interval = intervals.get(intensity, 0.8)
        
        end_time = time.time() + duration
        event_count = 0
        
        while time.time() < end_time and self._running:
            try:
                phase = random.choice(['download', 'extract', 'install', 'cleanup'])
                
                if phase == 'download':
                    # تحميل حزمة | Télécharger un paquet
                    pkg_file = update_dir / f"package_{self._random_string(6)}.pkg"
                    pkg_file.write_bytes(os.urandom(random.randint(1000, 10000)))
                    event_count += 1
                
                elif phase == 'extract':
                    # استخراج حزمة | Extraire un paquet
                    pkg_files = list(update_dir.glob("*.pkg"))
                    if pkg_files:
                        selected = random.choice(pkg_files)
                        extract_dir = update_dir / f"extract_{selected.stem}"
                        extract_dir.mkdir(exist_ok=True)
                        for i in range(random.randint(3, 8)):
                            f = extract_dir / f"file_{i}.bin"
                            f.write_bytes(os.urandom(random.randint(100, 1000)))
                            event_count += 1
                
                elif phase == 'install':
                    # تثبيت | Installer
                    extract_dirs = [d for d in update_dir.iterdir() if d.is_dir()]
                    if extract_dirs:
                        src = random.choice(extract_dirs)
                        for f in src.glob("*.*"):
                            dest = install_dir / f.name
                            dest.write_bytes(f.read_bytes())
                            event_count += 2
                
                elif phase == 'cleanup':
                    # تنظيف | Nettoyage
                    old_pkgs = list(update_dir.glob("*.pkg"))
                    if len(old_pkgs) > 5:
                        for p in old_pkgs[:3]:
                            if p.exists():
                                p.unlink()
                                event_count += 1
                
                if callback:
                    callback("system_update", event_count)
                
                time.sleep(interval)
                
            except Exception as e:
                logger.error(f"خطأ في التحديث | Erreur mise à jour: {e}")
        
        logger.info(f"✅ انتهاء التحديث | Mise à jour terminée: {event_count} أحداث")
        return event_count
    
    # ==================== تشغيل جميع السيناريوهات ====================
    # ==================== Exécuter Tous les Scénarios ====================
    
    def run_all_scenarios(
        self,
        duration_per_scenario: float = 30,
        intensity: str = "normal",
        parallel: bool = True,
        callback: Optional[Callable] = None
    ) -> int:
        """
        تشغيل جميع السيناريوهات الحميدة
        Exécuter tous les scénarios bénins
        
        Args:
            duration_per_scenario: مدة كل سيناريو | Durée par scénario
            intensity: شدة النشاط | Intensité de l'activité
            parallel: تشغيل متوازي | Exécution parallèle
            callback: دالة الاستدعاء | Callback
            
        Returns:
            إجمالي الأحداث | Total événements
        """
        self._running = True
        total_events = 0
        
        scenarios = [
            ("web_browsing", self.simulate_web_browsing),
            ("office_work", self.simulate_office_work),
            ("compilation", self.simulate_compilation),
            ("file_copy", self.simulate_file_copy),
            ("system_update", self.simulate_system_update)
        ]
        
        logger.info(f"🚀 تشغيل {len(scenarios)} سيناريوهات | Exécution de {len(scenarios)} scénarios")
        
        if parallel:
            # تشغيل متوازي | Exécution parallèle
            results = {}
            threads = []
            
            for name, func in scenarios:
                def run_scenario(n, f):
                    results[n] = f(duration=duration_per_scenario, intensity=intensity, callback=callback)
                
                t = threading.Thread(target=run_scenario, args=(name, func))
                threads.append(t)
                t.start()
            
            for t in threads:
                t.join()
            
            total_events = sum(results.values())
        else:
            # تشغيل تسلسلي | Exécution séquentielle
            for name, func in scenarios:
                events = func(duration=duration_per_scenario, intensity=intensity, callback=callback)
                total_events += events
        
        self._running = False
        logger.info(f"✅ انتهت جميع السيناريوهات | Tous les scénarios terminés: {total_events} أحداث")
        return total_events
    
    def stop(self):
        """إيقاف جميع السيناريوهات | Arrêter tous les scénarios"""
        self._running = False
    
    def cleanup(self):
        """تنظيف المجلدات | Nettoyer les répertoires"""
        import shutil
        if self.sandbox_dir.exists():
            shutil.rmtree(self.sandbox_dir)
            logger.info(f"تم تنظيف | Nettoyé: {self.sandbox_dir}")


# اختبار الوحدة | Test du module
if __name__ == "__main__":
    print("=" * 60)
    print("اختبار السيناريوهات الحميدة | Test des Scénarios Bénins")
    print("=" * 60)
    
    scenarios = BenignScenarios()
    
    def on_event(scenario_name, count):
        print(f"  [{scenario_name}] الأحداث | Événements: {count}")
    
    try:
        # اختبار كل سيناريو | Tester chaque scénario
        print("\n🌐 التصفح | Navigation...")
        scenarios.simulate_web_browsing(duration=5, intensity="high", callback=on_event)
        
        print("\n📄 المكتبي | Bureautique...")
        scenarios.simulate_office_work(duration=5, intensity="high", callback=on_event)
        
        print("\n🔨 الترجمة | Compilation...")
        scenarios.simulate_compilation(duration=5, intensity="high", callback=on_event)
        
    finally:
        scenarios.cleanup()
        print("\n✅ تم الانتهاء | Terminé")
