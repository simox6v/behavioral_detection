"""
السيناريوهات المشبوهة | Scénarios Malveillants | Malicious Scenarios
محاكاة السلوك المشبوه (محاكاة فقط - لا برامج ضارة حقيقية!)
Simulation du comportement suspect (simulation uniquement - pas de vrai malware!)

⚠️ تحذير: هذه محاكاة تعليمية فقط
⚠️ Avertissement: Ceci est une simulation éducative uniquement
"""

import os
import time
import random
import string
import tempfile
import threading
import socket
from typing import Callable, Optional, List
from pathlib import Path
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class MaliciousScenarios:
    """
    مولد السيناريوهات المشبوهة | Générateur de Scénarios Malveillants
    يحاكي أنماط السلوك المشبوه لتوليد بيانات التدريب
    Simule les patterns de comportement suspect pour les données d'entraînement
    
    ⚠️ تحذير: لا يتم تنفيذ أي برامج ضارة حقيقية
    ⚠️ Avertissement: Aucun malware réel n'est exécuté
    """
    
    def __init__(self, sandbox_dir: Optional[str] = None):
        """
        تهيئة المولد | Initialisation du générateur
        
        Args:
            sandbox_dir: مجلد المحاكاة | Répertoire sandbox
        """
        self.sandbox_dir = Path(sandbox_dir or tempfile.mkdtemp(prefix="malicious_sim_"))
        self.sandbox_dir.mkdir(parents=True, exist_ok=True)
        
        self._running = False
        self._threads: List[threading.Thread] = []
        
        logger.info(f"تهيئة السيناريوهات المشبوهة في | Scénarios malveillants initialisés: {self.sandbox_dir}")
        logger.warning("⚠️ هذه محاكاة تعليمية فقط | Simulation éducative uniquement")
    
    def _random_string(self, length: int = 10) -> str:
        """توليد سلسلة عشوائية | Générer une chaîne aléatoire"""
        return ''.join(random.choices(string.ascii_letters + string.digits, k=length))
    
    def _fake_encrypt(self, data: bytes) -> bytes:
        """
        تشفير وهمي (XOR بسيط) | Chiffrement factice (XOR simple)
        لا يستخدم تشفير حقيقي - للمحاكاة فقط
        """
        key = 0x42
        return bytes([b ^ key for b in data])
    
    # ==================== السيناريو 1: انفجار الملفات ====================
    # ==================== Scénario 1: Burst de Fichiers ====================
    
    def simulate_file_burst(
        self,
        duration: float = 30,
        files_count: int = 1000,
        callback: Optional[Callable] = None
    ):
        """
        محاكاة انفجار إنشاء/حذف الملفات
        Simuler un burst de création/suppression de fichiers
        
        نمط مشبوه: ~10,000 ملف في <30 ثانية
        Pattern suspect: ~10,000 fichiers en <30 secondes
        """
        logger.info(f"💥 بدء انفجار الملفات | Démarrage burst fichiers: {files_count} ملفات")
        
        burst_dir = self.sandbox_dir / "burst_files"
        burst_dir.mkdir(exist_ok=True)
        
        end_time = time.time() + duration
        event_count = 0
        files_created = []
        
        # المرحلة 1: إنشاء سريع | Phase 1: Création rapide
        target_per_second = files_count / (duration * 0.6)  # 60% من الوقت للإنشاء
        interval = 1.0 / target_per_second if target_per_second > 0 else 0.001
        
        while time.time() < end_time * 0.6 + time.time() * 0.4 and self._running and len(files_created) < files_count:
            try:
                # إنشاء ملفات بسرعة عالية | Créer des fichiers rapidement
                batch_size = random.randint(5, 20)
                for _ in range(batch_size):
                    if len(files_created) >= files_count:
                        break
                    
                    filename = f"burst_{self._random_string(8)}.tmp"
                    filepath = burst_dir / filename
                    
                    # محتوى عشوائي صغير | Petit contenu aléatoire
                    content = os.urandom(random.randint(100, 1000))
                    filepath.write_bytes(content)
                    files_created.append(filepath)
                    event_count += 1
                
                if callback:
                    callback("file_burst_create", event_count)
                
                time.sleep(interval)
                
            except Exception as e:
                logger.error(f"خطأ في إنشاء الملفات | Erreur création: {e}")
        
        # المرحلة 2: حذف سريع | Phase 2: Suppression rapide
        logger.info(f"🗑️ بدء الحذف السريع | Démarrage suppression rapide")
        
        while files_created and self._running and time.time() < end_time:
            try:
                batch_size = random.randint(10, 30)
                for _ in range(min(batch_size, len(files_created))):
                    if not files_created:
                        break
                    
                    filepath = files_created.pop()
                    if filepath.exists():
                        filepath.unlink()
                        event_count += 1
                
                if callback:
                    callback("file_burst_delete", event_count)
                
                time.sleep(0.01)  # سريع جداً | Très rapide
                
            except Exception as e:
                logger.error(f"خطأ في الحذف | Erreur suppression: {e}")
        
        logger.info(f"✅ انتهاء انفجار الملفات | Burst terminé: {event_count} أحداث")
        return event_count
    
    # ==================== السيناريو 2: مسح المنافذ ====================
    # ==================== Scénario 2: Scan de Ports ====================
    
    def simulate_port_scan(
        self,
        duration: float = 30,
        target_host: str = "127.0.0.1",
        port_range: tuple = (1, 1024),
        callback: Optional[Callable] = None
    ):
        """
        محاكاة مسح المنافذ
        Simuler un scan de ports
        
        نمط مشبوه: >100 اتصال/ثانية
        Pattern suspect: >100 connexions/seconde
        
        ⚠️ المسح محلي فقط (localhost)
        ⚠️ Scan local uniquement
        """
        logger.info(f"🔍 بدء مسح المنافذ | Démarrage scan ports: {target_host}")
        
        end_time = time.time() + duration
        event_count = 0
        ports_scanned = []
        
        # توليد قائمة المنافذ | Générer la liste des ports
        all_ports = list(range(port_range[0], port_range[1] + 1))
        random.shuffle(all_ports)
        
        while time.time() < end_time and self._running and all_ports:
            try:
                # مسح دفعة من المنافذ | Scanner un lot de ports
                batch_size = random.randint(50, 150)  # معدل عالي | Taux élevé
                
                for _ in range(min(batch_size, len(all_ports))):
                    if not all_ports:
                        break
                    
                    port = all_ports.pop()
                    
                    # محاولة اتصال سريعة | Tentative de connexion rapide
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(0.01)  # timeout قصير جداً
                    
                    try:
                        result = sock.connect_ex((target_host, port))
                        ports_scanned.append((port, result == 0))
                        event_count += 1
                    except:
                        pass
                    finally:
                        sock.close()
                
                if callback:
                    callback("port_scan", event_count)
                
                time.sleep(0.01)  # استمرار بسرعة | Continuer rapidement
                
            except Exception as e:
                logger.error(f"خطأ في مسح المنافذ | Erreur scan: {e}")
        
        open_ports = [p for p, is_open in ports_scanned if is_open]
        logger.info(f"✅ انتهاء المسح | Scan terminé: {event_count} منافذ, {len(open_ports)} مفتوحة")
        return event_count
    
    # ==================== السيناريو 3: قراءة الملفات الحساسة ====================
    # ==================== Scénario 3: Lecture Fichiers Sensibles ====================
    
    def simulate_sensitive_file_access(
        self,
        duration: float = 30,
        callback: Optional[Callable] = None
    ):
        """
        محاكاة قراءة متكررة لملفات حساسة
        Simuler la lecture répétée de fichiers sensibles
        
        نمط مشبوه: قراءة متكررة لنفس الملف الحساس
        Pattern suspect: lecture répétée du même fichier sensible
        """
        logger.info("📖 بدء محاكاة الوصول للملفات الحساسة | Démarrage accès fichiers sensibles")
        
        # إنشاء ملفات حساسة وهمية | Créer des fichiers sensibles factices
        sensitive_dir = self.sandbox_dir / "sensitive"
        sensitive_dir.mkdir(exist_ok=True)
        
        fake_sensitive_files = [
            ("fake_passwd", "root:x:0:0:root:/root:/bin/bash\nuser:x:1000:1000:User:/home/user:/bin/bash"),
            ("fake_shadow", "root:$6$fake$hash:18000:0:99999:7:::\nuser:$6$fake$hash:18000:0:99999:7:::"),
            ("fake_ssh_key", "-----BEGIN RSA PRIVATE KEY-----\nFAKE_KEY_DATA_NOT_REAL\n-----END RSA PRIVATE KEY-----"),
            ("fake_credentials", "username=admin\npassword=not_real_password\napi_key=fake_api_key_12345"),
            ("fake_database.db", "FAKE DATABASE CONTENT - NOT REAL DATA"),
        ]
        
        for filename, content in fake_sensitive_files:
            filepath = sensitive_dir / filename
            filepath.write_text(content)
        
        end_time = time.time() + duration
        event_count = 0
        
        while time.time() < end_time and self._running:
            try:
                # قراءة متكررة للملفات الحساسة | Lecture répétée des fichiers sensibles
                for filename, _ in fake_sensitive_files:
                    if not self._running:
                        break
                    
                    filepath = sensitive_dir / filename
                    
                    # قراءات متعددة سريعة | Multiples lectures rapides
                    for _ in range(random.randint(5, 20)):
                        _ = filepath.read_text()
                        event_count += 1
                    
                    if callback:
                        callback("sensitive_access", event_count)
                    
                    time.sleep(0.05)  # استمرار سريع | Continuer rapidement
                
            except Exception as e:
                logger.error(f"خطأ في القراءة | Erreur lecture: {e}")
        
        logger.info(f"✅ انتهاء الوصول | Accès terminé: {event_count} قراءات")
        return event_count
    
    # ==================== السيناريو 4: محاكاة Ransomware ====================
    # ==================== Scénario 4: Simulation Ransomware ====================
    
    def simulate_ransomware_behavior(
        self,
        duration: float = 30,
        files_to_encrypt: int = 500,
        callback: Optional[Callable] = None
    ):
        """
        محاكاة سلوك Ransomware
        Simuler le comportement ransomware
        
        ⚠️ تحذير: هذا overwrite وهمي فقط - لا تشفير حقيقي
        ⚠️ Avertissement: Overwrite factice uniquement - pas de vrai chiffrement
        
        نمط مشبوه: كتابة مكثفة سريعة مع تغيير الامتدادات
        Pattern suspect: écriture intensive rapide avec changement d'extensions
        """
        logger.info("🔒 بدء محاكاة Ransomware | Démarrage simulation ransomware")
        logger.warning("⚠️ تشفير وهمي فقط | Chiffrement factice uniquement")
        
        # إنشاء ملفات للتشفير الوهمي | Créer des fichiers pour le chiffrement factice
        victim_dir = self.sandbox_dir / "victim_files"
        victim_dir.mkdir(exist_ok=True)
        
        # إنشاء ملفات ضحية | Créer des fichiers victimes
        extensions = ['.txt', '.doc', '.pdf', '.jpg', '.png', '.xlsx']
        created_files = []
        
        for i in range(files_to_encrypt):
            ext = random.choice(extensions)
            filename = f"document_{self._random_string(6)}{ext}"
            filepath = victim_dir / filename
            content = os.urandom(random.randint(100, 2000))
            filepath.write_bytes(content)
            created_files.append(filepath)
        
        end_time = time.time() + duration
        event_count = 0
        encrypted_count = 0
        
        # مرحلة التشفير الوهمي | Phase de chiffrement factice
        while created_files and self._running and time.time() < end_time:
            try:
                # تشفير دفعة | Chiffrer un lot
                batch_size = random.randint(10, 30)
                
                for _ in range(min(batch_size, len(created_files))):
                    if not created_files:
                        break
                    
                    filepath = created_files.pop()
                    
                    if filepath.exists():
                        # قراءة المحتوى | Lire le contenu
                        content = filepath.read_bytes()
                        event_count += 1
                        
                        # تشفير وهمي (XOR بسيط) | Chiffrement factice
                        encrypted = self._fake_encrypt(content)
                        
                        # كتابة الملف المشفر | Écrire le fichier chiffré
                        encrypted_path = filepath.with_suffix(filepath.suffix + '.encrypted')
                        encrypted_path.write_bytes(encrypted)
                        event_count += 1
                        
                        # حذف الأصلي | Supprimer l'original
                        filepath.unlink()
                        event_count += 1
                        
                        encrypted_count += 1
                
                if callback:
                    callback("ransomware_sim", event_count)
                
                time.sleep(0.02)  # سريع جداً | Très rapide
                
            except Exception as e:
                logger.error(f"خطأ في المحاكاة | Erreur simulation: {e}")
        
        # إنشاء ملاحظة فدية وهمية | Créer une fausse note de rançon
        ransom_note = victim_dir / "README_ENCRYPTED.txt"
        ransom_note.write_text("""
⚠️ THIS IS A SIMULATION - NOT REAL RANSOMWARE ⚠️
⚠️ هذه محاكاة - ليست برنامج فدية حقيقي ⚠️
⚠️ CECI EST UNE SIMULATION - PAS UN VRAI RANSOMWARE ⚠️

This is an educational simulation for behavioral detection training.
Your files were NOT actually encrypted.
""")
        
        logger.info(f"✅ انتهاء المحاكاة | Simulation terminée: {encrypted_count} ملفات, {event_count} أحداث")
        return event_count
    
    # ==================== السيناريو 5: Brute-force محاكاة ====================
    # ==================== Scénario 5: Simulation Brute-force ====================
    
    def simulate_bruteforce(
        self,
        duration: float = 30,
        callback: Optional[Callable] = None
    ):
        """
        محاكاة هجوم Brute-force
        Simuler une attaque brute-force
        
        نمط مشبوه: حلقات مكثفة، استخدام CPU عالي
        Pattern suspect: boucles intensives, utilisation CPU élevée
        """
        logger.info("🔐 بدء محاكاة Brute-force | Démarrage simulation brute-force")
        
        # ملف كلمات مرور وهمي | Fichier de mots de passe factice
        wordlist_dir = self.sandbox_dir / "bruteforce"
        wordlist_dir.mkdir(exist_ok=True)
        
        # إنشاء wordlist وهمي | Créer une wordlist factice
        wordlist = wordlist_dir / "wordlist.txt"
        fake_passwords = [f"password{i:04d}" for i in range(10000)]
        wordlist.write_text('\n'.join(fake_passwords))
        
        # ملف هدف وهمي | Fichier cible factice
        target_hash = "fake_hash_5f4dcc3b5aa765d61d8327deb882cf99"  # ليس hash حقيقي
        
        end_time = time.time() + duration
        event_count = 0
        attempts = 0
        
        while time.time() < end_time and self._running:
            try:
                # قراءة wordlist | Lire la wordlist
                passwords = wordlist.read_text().split('\n')
                event_count += 1
                
                # محاكاة محاولات | Simuler des tentatives
                batch = random.sample(passwords, min(100, len(passwords)))
                
                for password in batch:
                    if not self._running:
                        break
                    
                    # محاكاة تجزئة (حساب وهمي) | Simulation hash (calcul factice)
                    fake_hash = ''.join([str(ord(c) % 10) for c in password])
                    
                    # مقارنة وهمية | Comparaison factice
                    if fake_hash == target_hash:
                        pass  # لا شيء - هذا وهمي
                    
                    attempts += 1
                    event_count += 1
                
                if callback:
                    callback("bruteforce_sim", event_count)
                
                # استمرار فوري تقريباً | Continuer presque immédiatement
                time.sleep(0.001)
                
            except Exception as e:
                logger.error(f"خطأ في المحاكاة | Erreur simulation: {e}")
        
        logger.info(f"✅ انتهاء Brute-force | Brute-force terminé: {attempts} محاولات, {event_count} أحداث")
        return event_count
    
    # ==================== تشغيل جميع السيناريوهات ====================
    # ==================== Exécuter Tous les Scénarios ====================
    
    def run_all_scenarios(
        self,
        duration_per_scenario: float = 30,
        parallel: bool = False,
        callback: Optional[Callable] = None
    ) -> int:
        """
        تشغيل جميع السيناريوهات المشبوهة
        Exécuter tous les scénarios malveillants
        
        Args:
            duration_per_scenario: مدة كل سيناريو | Durée par scénario
            parallel: تشغيل متوازي | Exécution parallèle
            callback: دالة الاستدعاء | Callback
            
        Returns:
            إجمالي الأحداث | Total événements
        """
        self._running = True
        total_events = 0
        
        scenarios = [
            ("file_burst", lambda: self.simulate_file_burst(duration=duration_per_scenario, callback=callback)),
            ("port_scan", lambda: self.simulate_port_scan(duration=duration_per_scenario, callback=callback)),
            ("sensitive_access", lambda: self.simulate_sensitive_file_access(duration=duration_per_scenario, callback=callback)),
            ("ransomware", lambda: self.simulate_ransomware_behavior(duration=duration_per_scenario, callback=callback)),
            ("bruteforce", lambda: self.simulate_bruteforce(duration=duration_per_scenario, callback=callback)),
        ]
        
        logger.info(f"🚀 تشغيل {len(scenarios)} سيناريوهات مشبوهة | Exécution de {len(scenarios)} scénarios")
        logger.warning("⚠️ هذه محاكاة تعليمية فقط | Simulation éducative uniquement")
        
        if parallel:
            results = {}
            threads = []
            
            for name, func in scenarios:
                def run_scenario(n, f):
                    results[n] = f()
                
                t = threading.Thread(target=run_scenario, args=(name, func))
                threads.append(t)
                t.start()
            
            for t in threads:
                t.join()
            
            total_events = sum(results.values())
        else:
            for name, func in scenarios:
                logger.info(f"▶️ تشغيل | Exécution: {name}")
                events = func()
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
    print("اختبار السيناريوهات المشبوهة | Test des Scénarios Malveillants")
    print("⚠️ هذه محاكاة تعليمية فقط | Simulation éducative uniquement")
    print("=" * 60)
    
    scenarios = MaliciousScenarios()
    
    def on_event(scenario_name, count):
        print(f"  [{scenario_name}] الأحداث | Événements: {count}")
    
    try:
        # اختبار كل سيناريو | Tester chaque scénario
        print("\n💥 انفجار الملفات | Burst fichiers...")
        scenarios.simulate_file_burst(duration=5, files_count=100, callback=on_event)
        
        print("\n🔍 مسح المنافذ | Scan ports...")
        scenarios.simulate_port_scan(duration=5, callback=on_event)
        
        print("\n📖 الملفات الحساسة | Fichiers sensibles...")
        scenarios.simulate_sensitive_file_access(duration=5, callback=on_event)
        
        print("\n🔒 محاكاة Ransomware...")
        scenarios.simulate_ransomware_behavior(duration=5, files_to_encrypt=50, callback=on_event)
        
    finally:
        scenarios.cleanup()
        print("\n✅ تم الانتهاء | Terminé")
