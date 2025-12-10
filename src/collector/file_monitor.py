"""
مراقب الملفات | Moniteur de Fichiers | File Monitor
يراقب عمليات الملفات: إنشاء، حذف، تعديل، نقل
Surveille les opérations fichiers: création, suppression, modification, déplacement
"""

import os
import time
import threading
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import logging

try:
    from watchdog.observers import Observer
    from watchdog.events import (
        FileSystemEventHandler,
        FileCreatedEvent,
        FileDeletedEvent,
        FileModifiedEvent,
        FileMovedEvent,
        DirCreatedEvent,
        DirDeletedEvent,
        DirModifiedEvent,
        DirMovedEvent
    )
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False
    logging.warning("مكتبة watchdog غير متوفرة | watchdog non disponible")

# إعداد التسجيل | Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class FileEvent:
    """
    حدث الملف | Événement fichier
    يمثل عملية واحدة على ملف
    """
    timestamp: float          # الطابع الزمني بالميلي ثانية | Timestamp en ms
    timestamp_iso: str        # الوقت بتنسيق ISO | Temps format ISO
    event_type: str           # نوع الحدث | Type d'événement
    operation: str            # العملية: created, deleted, modified, moved
    path: str                 # مسار الملف | Chemin du fichier
    filename: str             # اسم الملف | Nom du fichier
    extension: str            # امتداد الملف | Extension
    is_directory: bool        # هل هو مجلد | Est un répertoire
    src_path: Optional[str]   # المسار المصدر (للنقل) | Chemin source
    dest_path: Optional[str]  # المسار الهدف (للنقل) | Chemin destination
    file_size: int            # حجم الملف | Taille du fichier
    
    def to_dict(self) -> Dict:
        """تحويل إلى قاموس | Convertir en dictionnaire"""
        return asdict(self)


class FileEventHandler(FileSystemEventHandler):
    """
    معالج أحداث الملفات | Gestionnaire d'événements fichiers
    يعالج الأحداث من watchdog
    """
    
    def __init__(
        self,
        callback: Optional[Callable[[FileEvent], None]] = None,
        watch_extensions: Optional[List[str]] = None,
        excluded_dirs: Optional[List[str]] = None
    ):
        super().__init__()
        self.callback = callback
        self.watch_extensions = set(watch_extensions or [])
        self.excluded_dirs = set(excluded_dirs or ['__pycache__', '.git', 'node_modules', '.venv'])
        self._events: List[FileEvent] = []
        self._lock = threading.Lock()
    
    def _should_process(self, path: str) -> bool:
        """
        التحقق مما إذا كان يجب معالجة الحدث
        Vérifier si l'événement doit être traité
        """
        # استثناء المجلدات | Exclure les répertoires
        path_parts = Path(path).parts
        for excluded in self.excluded_dirs:
            if excluded in path_parts:
                return False
        
        # التحقق من الامتداد | Vérifier l'extension
        if self.watch_extensions:
            ext = Path(path).suffix.lower()
            if ext and ext not in self.watch_extensions and ext[1:] not in self.watch_extensions:
                return False
        
        return True
    
    def _get_file_size(self, path: str) -> int:
        """
        الحصول على حجم الملف | Obtenir la taille du fichier
        """
        try:
            if os.path.exists(path) and os.path.isfile(path):
                return os.path.getsize(path)
        except (OSError, PermissionError):
            pass
        return 0
    
    def _create_event(
        self,
        event_type: str,
        operation: str,
        path: str,
        is_directory: bool,
        src_path: Optional[str] = None,
        dest_path: Optional[str] = None
    ) -> FileEvent:
        """
        إنشاء حدث ملف | Créer un événement fichier
        """
        now = datetime.now()
        file_path = Path(path)
        
        return FileEvent(
            timestamp=time.time() * 1000,
            timestamp_iso=now.isoformat(),
            event_type=event_type,
            operation=operation,
            path=path,
            filename=file_path.name,
            extension=file_path.suffix.lower() if file_path.suffix else "",
            is_directory=is_directory,
            src_path=src_path,
            dest_path=dest_path,
            file_size=self._get_file_size(path) if not is_directory else 0
        )
    
    def _handle_event(self, event, operation: str, is_directory: bool):
        """
        معالجة حدث عام | Traiter un événement général
        """
        path = event.src_path
        
        if not self._should_process(path):
            return
        
        src_path = None
        dest_path = None
        
        if hasattr(event, 'dest_path'):
            src_path = event.src_path
            dest_path = event.dest_path
            path = event.dest_path
        
        file_event = self._create_event(
            event_type="file_operation",
            operation=operation,
            path=path,
            is_directory=is_directory,
            src_path=src_path,
            dest_path=dest_path
        )
        
        with self._lock:
            self._events.append(file_event)
            # الحفاظ على آخر 50000 حدث | Garder les derniers 50000
            if len(self._events) > 50000:
                self._events = self._events[-50000:]
        
        if self.callback:
            self.callback(file_event)
    
    def on_created(self, event):
        """معالجة الإنشاء | Traiter la création"""
        is_dir = isinstance(event, DirCreatedEvent)
        self._handle_event(event, "created", is_dir)
    
    def on_deleted(self, event):
        """معالجة الحذف | Traiter la suppression"""
        is_dir = isinstance(event, DirDeletedEvent)
        self._handle_event(event, "deleted", is_dir)
    
    def on_modified(self, event):
        """معالجة التعديل | Traiter la modification"""
        is_dir = isinstance(event, DirModifiedEvent)
        self._handle_event(event, "modified", is_dir)
    
    def on_moved(self, event):
        """معالجة النقل | Traiter le déplacement"""
        is_dir = isinstance(event, DirMovedEvent)
        self._handle_event(event, "moved", is_dir)
    
    def get_events(self, clear: bool = False) -> List[FileEvent]:
        """
        الحصول على الأحداث المجمعة | Obtenir les événements collectés
        """
        with self._lock:
            events = self._events.copy()
            if clear:
                self._events.clear()
        return events


class FileMonitor:
    """
    مراقب الملفات | Moniteur de Fichiers
    يراقب مجلدات متعددة للتغييرات
    Surveille plusieurs répertoires pour les changements
    """
    
    def __init__(
        self,
        watch_directories: Optional[List[str]] = None,
        watch_extensions: Optional[List[str]] = None,
        excluded_dirs: Optional[List[str]] = None,
        callback: Optional[Callable[[FileEvent], None]] = None,
        recursive: bool = True
    ):
        """
        تهيئة المراقب | Initialisation du moniteur
        
        Args:
            watch_directories: المجلدات المراقبة | Répertoires surveillés
            watch_extensions: الامتدادات المراقبة | Extensions surveillées
            excluded_dirs: المجلدات المستثناة | Répertoires exclus
            callback: دالة الاستدعاء | Callback
            recursive: مراقبة تكرارية | Surveillance récursive
        """
        if not WATCHDOG_AVAILABLE:
            raise ImportError("مكتبة watchdog مطلوبة | watchdog library required")
        
        self.watch_directories = watch_directories or ["."]
        self.watch_extensions = watch_extensions
        self.excluded_dirs = excluded_dirs
        self.callback = callback
        self.recursive = recursive
        
        self._observer: Optional[Observer] = None
        self._handler: Optional[FileEventHandler] = None
        self._running = False
        
        logger.info("تم تهيئة مراقب الملفات | Moniteur de fichiers initialisé")
    
    def start(self):
        """
        بدء المراقبة | Démarrer la surveillance
        """
        if self._running:
            logger.warning("المراقب يعمل بالفعل | Moniteur déjà en cours")
            return
        
        self._handler = FileEventHandler(
            callback=self.callback,
            watch_extensions=self.watch_extensions,
            excluded_dirs=self.excluded_dirs
        )
        
        self._observer = Observer()
        
        for directory in self.watch_directories:
            # إنشاء المجلد إذا لم يكن موجوداً | Créer le répertoire s'il n'existe pas
            dir_path = Path(directory)
            if not dir_path.exists():
                dir_path.mkdir(parents=True, exist_ok=True)
                logger.info(f"تم إنشاء المجلد | Répertoire créé: {directory}")
            
            self._observer.schedule(
                self._handler,
                str(dir_path.absolute()),
                recursive=self.recursive
            )
            logger.info(f"مراقبة | Surveillance: {directory}")
        
        self._observer.start()
        self._running = True
        logger.info("تم بدء مراقب الملفات | Moniteur de fichiers démarré")
    
    def stop(self):
        """
        إيقاف المراقبة | Arrêter la surveillance
        """
        if self._observer:
            self._observer.stop()
            self._observer.join(timeout=2)
        self._running = False
        logger.info("تم إيقاف مراقب الملفات | Moniteur de fichiers arrêté")
    
    def get_events(self, clear: bool = False) -> List[FileEvent]:
        """
        الحصول على الأحداث المجمعة | Obtenir les événements collectés
        """
        if self._handler:
            return self._handler.get_events(clear)
        return []
    
    def is_running(self) -> bool:
        """
        التحقق من حالة المراقب | Vérifier l'état du moniteur
        """
        return self._running


class SimpleFileMonitor:
    """
    مراقب ملفات بسيط (بدون watchdog) | Moniteur simple (sans watchdog)
    يستخدم المسح الدوري للملفات
    Utilise le scan périodique des fichiers
    """
    
    def __init__(
        self,
        watch_directories: Optional[List[str]] = None,
        interval: float = 1.0,
        callback: Optional[Callable[[FileEvent], None]] = None
    ):
        self.watch_directories = watch_directories or ["."]
        self.interval = interval
        self.callback = callback
        
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._events: List[FileEvent] = []
        self._lock = threading.Lock()
        self._prev_state: Dict[str, float] = {}  # path -> mtime
        
        logger.info("تم تهيئة مراقب الملفات البسيط | Moniteur simple initialisé")
    
    def _scan_directory(self, directory: str) -> Dict[str, float]:
        """
        مسح المجلد وجمع أوقات التعديل
        Scanner le répertoire et collecter les mtimes
        """
        state = {}
        try:
            for root, dirs, files in os.walk(directory):
                # استثناء المجلدات المخفية | Exclure les répertoires cachés
                dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'node_modules']]
                
                for filename in files:
                    filepath = os.path.join(root, filename)
                    try:
                        state[filepath] = os.path.getmtime(filepath)
                    except (OSError, PermissionError):
                        pass
        except Exception as e:
            logger.error(f"خطأ في المسح | Erreur de scan: {e}")
        return state
    
    def _check_changes(self):
        """
        التحقق من التغييرات | Vérifier les changements
        """
        current_state = {}
        
        for directory in self.watch_directories:
            if os.path.exists(directory):
                current_state.update(self._scan_directory(directory))
        
        now = datetime.now()
        timestamp = time.time() * 1000
        
        # الملفات الجديدة | Nouveaux fichiers
        for path in current_state:
            if path not in self._prev_state:
                event = FileEvent(
                    timestamp=timestamp,
                    timestamp_iso=now.isoformat(),
                    event_type="file_operation",
                    operation="created",
                    path=path,
                    filename=os.path.basename(path),
                    extension=Path(path).suffix.lower(),
                    is_directory=False,
                    src_path=None,
                    dest_path=None,
                    file_size=os.path.getsize(path) if os.path.exists(path) else 0
                )
                self._add_event(event)
        
        # الملفات المحذوفة | Fichiers supprimés
        for path in self._prev_state:
            if path not in current_state:
                event = FileEvent(
                    timestamp=timestamp,
                    timestamp_iso=now.isoformat(),
                    event_type="file_operation",
                    operation="deleted",
                    path=path,
                    filename=os.path.basename(path),
                    extension=Path(path).suffix.lower(),
                    is_directory=False,
                    src_path=None,
                    dest_path=None,
                    file_size=0
                )
                self._add_event(event)
        
        # الملفات المعدلة | Fichiers modifiés
        for path in current_state:
            if path in self._prev_state and current_state[path] != self._prev_state[path]:
                event = FileEvent(
                    timestamp=timestamp,
                    timestamp_iso=now.isoformat(),
                    event_type="file_operation",
                    operation="modified",
                    path=path,
                    filename=os.path.basename(path),
                    extension=Path(path).suffix.lower(),
                    is_directory=False,
                    src_path=None,
                    dest_path=None,
                    file_size=os.path.getsize(path) if os.path.exists(path) else 0
                )
                self._add_event(event)
        
        self._prev_state = current_state
    
    def _add_event(self, event: FileEvent):
        """إضافة حدث | Ajouter un événement"""
        with self._lock:
            self._events.append(event)
            if len(self._events) > 50000:
                self._events = self._events[-50000:]
        
        if self.callback:
            self.callback(event)
    
    def _monitor_loop(self):
        """حلقة المراقبة | Boucle de surveillance"""
        logger.info("بدء حلقة مراقبة الملفات البسيطة | Démarrage surveillance simple")
        
        # المسح الأولي | Scan initial
        for directory in self.watch_directories:
            if os.path.exists(directory):
                self._prev_state.update(self._scan_directory(directory))
        
        while self._running:
            try:
                self._check_changes()
                time.sleep(self.interval)
            except Exception as e:
                logger.error(f"خطأ في المراقبة | Erreur de surveillance: {e}")
                time.sleep(1)
        
        logger.info("توقف مراقبة الملفات البسيطة | Surveillance simple arrêtée")
    
    def start(self):
        """بدء المراقبة | Démarrer la surveillance"""
        if self._running:
            return
        
        # إنشاء المجلدات | Créer les répertoires
        for directory in self.watch_directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
        
        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        logger.info("تم بدء مراقب الملفات البسيط | Moniteur simple démarré")
    
    def stop(self):
        """إيقاف المراقبة | Arrêter la surveillance"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        logger.info("تم إيقاف مراقب الملفات البسيط | Moniteur simple arrêté")
    
    def get_events(self, clear: bool = False) -> List[FileEvent]:
        """الحصول على الأحداث | Obtenir les événements"""
        with self._lock:
            events = self._events.copy()
            if clear:
                self._events.clear()
        return events


# اختبار الوحدة | Test du module
if __name__ == "__main__":
    import tempfile
    import shutil
    
    print("=" * 60)
    print("اختبار مراقب الملفات | Test du Moniteur de Fichiers")
    print("=" * 60)
    
    # إنشاء مجلد اختبار | Créer un répertoire de test
    test_dir = tempfile.mkdtemp(prefix="file_monitor_test_")
    print(f"\nمجلد الاختبار | Répertoire de test: {test_dir}")
    
    def on_event(event: FileEvent):
        icon = {
            "created": "✅",
            "deleted": "❌",
            "modified": "📝",
            "moved": "📦"
        }.get(event.operation, "❓")
        print(f"  {icon} [{event.operation}] {event.filename}")
    
    # استخدام المراقب البسيط للاختبار | Utiliser le moniteur simple
    monitor = SimpleFileMonitor(
        watch_directories=[test_dir],
        interval=0.5,
        callback=on_event
    )
    
    print("\nبدء المراقبة | Démarrage de la surveillance...")
    monitor.start()
    time.sleep(1)
    
    # محاكاة عمليات الملفات | Simuler des opérations fichiers
    print("\nمحاكاة العمليات | Simulation des opérations:")
    
    # إنشاء ملف | Créer un fichier
    test_file = os.path.join(test_dir, "test_file.txt")
    with open(test_file, "w") as f:
        f.write("Hello, World!")
    time.sleep(1)
    
    # تعديل ملف | Modifier un fichier
    with open(test_file, "a") as f:
        f.write("\nMore content")
    time.sleep(1)
    
    # حذف ملف | Supprimer un fichier
    os.remove(test_file)
    time.sleep(1)
    
    # إيقاف المراقب | Arrêter le moniteur
    monitor.stop()
    
    events = monitor.get_events()
    print(f"\nإجمالي الأحداث المجمعة: {len(events)}")
    
    # تنظيف | Nettoyage
    shutil.rmtree(test_dir)
    print("\nتم تنظيف مجلد الاختبار | Répertoire de test nettoyé")
