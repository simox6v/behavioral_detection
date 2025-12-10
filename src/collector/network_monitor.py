"""
مراقب الشبكة | Moniteur Réseau | Network Monitor
يراقب الاتصالات الشبكية: المنافذ، العناوين، البايتات
Surveille les connexions réseau: ports, adresses, bytes
"""

import psutil
import time
import threading
from typing import Dict, List, Optional, Callable, Set
from dataclasses import dataclass, asdict
from datetime import datetime
from collections import defaultdict
import logging

# إعداد التسجيل | Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class NetworkEvent:
    """
    حدث الشبكة | Événement réseau
    يمثل لقطة من حالة الشبكة
    """
    timestamp: float              # الطابع الزمني بالميلي ثانية | Timestamp en ms
    timestamp_iso: str            # الوقت بتنسيق ISO | Temps format ISO
    event_type: str               # نوع الحدث | Type d'événement
    
    # إحصائيات عامة | Statistiques générales
    total_connections: int        # إجمالي الاتصالات | Total connexions
    established_connections: int  # الاتصالات المنشأة | Connexions établies
    listening_ports: int          # المنافذ المستمعة | Ports en écoute
    
    # العناوين والمنافذ | Adresses et ports
    unique_remote_ips: int        # العناوين البعيدة الفريدة | IPs distantes uniques
    unique_remote_ports: int      # المنافذ البعيدة الفريدة | Ports distants uniques
    unique_local_ports: int       # المنافذ المحلية الفريدة | Ports locaux uniques
    
    # حركة البيانات | Trafic de données
    bytes_sent: int               # البايتات المرسلة | Bytes envoyés
    bytes_recv: int               # البايتات المستقبلة | Bytes reçus
    packets_sent: int             # الحزم المرسلة | Paquets envoyés
    packets_recv: int             # الحزم المستقبلة | Paquets reçus
    
    # معدلات (منذ آخر قراءة) | Taux (depuis dernière lecture)
    bytes_sent_rate: float        # معدل الإرسال | Taux d'envoi
    bytes_recv_rate: float        # معدل الاستقبال | Taux de réception
    new_connections: int          # اتصالات جديدة | Nouvelles connexions
    
    # تفاصيل إضافية | Détails supplémentaires
    connection_types: Dict        # أنواع الاتصالات | Types de connexions
    top_remote_ips: List[str]     # أكثر العناوين نشاطاً | IPs les plus actives
    
    def to_dict(self) -> Dict:
        """تحويل إلى قاموس | Convertir en dictionnaire"""
        return asdict(self)


@dataclass
class ConnectionDetail:
    """
    تفاصيل اتصال واحد | Détails d'une connexion
    """
    timestamp: float
    local_address: str
    local_port: int
    remote_address: str
    remote_port: int
    status: str
    pid: int
    process_name: str
    
    def to_dict(self) -> Dict:
        return asdict(self)


class NetworkMonitor:
    """
    مراقب الشبكة | Moniteur Réseau
    يراقب الاتصالات الشبكية ويجمع الإحصائيات
    Surveille les connexions réseau et collecte les statistiques
    """
    
    def __init__(
        self,
        interval: float = 0.5,
        excluded_ports: Optional[List[int]] = None,
        callback: Optional[Callable[[NetworkEvent], None]] = None,
        detailed_callback: Optional[Callable[[ConnectionDetail], None]] = None
    ):
        """
        تهيئة المراقب | Initialisation du moniteur
        
        Args:
            interval: فترة الجمع بالثواني | Intervalle de collecte
            excluded_ports: المنافذ المستثناة | Ports exclus
            callback: دالة الاستدعاء للأحداث | Callback événements
            detailed_callback: دالة الاستدعاء للتفاصيل | Callback détails
        """
        self.interval = interval
        self.excluded_ports = set(excluded_ports or [])
        self.callback = callback
        self.detailed_callback = detailed_callback
        
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._events: List[NetworkEvent] = []
        self._connection_details: List[ConnectionDetail] = []
        self._lock = threading.Lock()
        
        # القراءات السابقة لحساب المعدلات | Lectures précédentes pour les taux
        self._prev_bytes_sent = 0
        self._prev_bytes_recv = 0
        self._prev_connections: Set[tuple] = set()
        self._prev_time = time.time()
        
        logger.info("تم تهيئة مراقب الشبكة | Moniteur réseau initialisé")
    
    def _get_process_name(self, pid: int) -> str:
        """
        الحصول على اسم العملية من PID
        Obtenir le nom du processus à partir du PID
        """
        try:
            return psutil.Process(pid).name()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return "unknown"
    
    def collect_once(self) -> NetworkEvent:
        """
        جمع لقطة واحدة من حالة الشبكة
        Collecter un snapshot de l'état du réseau
        
        Returns:
            حدث الشبكة | Événement réseau
        """
        current_time = time.time()
        now = datetime.now()
        
        # جمع الاتصالات | Collecter les connexions
        connections = psutil.net_connections(kind='inet')
        
        # إحصائيات | Statistiques
        total = 0
        established = 0
        listening = 0
        remote_ips: Set[str] = set()
        remote_ports: Set[int] = set()
        local_ports: Set[int] = set()
        connection_types: Dict[str, int] = defaultdict(int)
        current_connections: Set[tuple] = set()
        
        for conn in connections:
            # تخطي المنافذ المستثناة | Ignorer les ports exclus
            if conn.laddr and conn.laddr.port in self.excluded_ports:
                continue
            
            total += 1
            connection_types[conn.status] += 1
            
            if conn.status == 'ESTABLISHED':
                established += 1
            elif conn.status == 'LISTEN':
                listening += 1
            
            # جمع العناوين والمنافذ | Collecter adresses et ports
            if conn.laddr:
                local_ports.add(conn.laddr.port)
            
            if conn.raddr:
                remote_ips.add(conn.raddr.ip)
                remote_ports.add(conn.raddr.port)
                current_connections.add((conn.raddr.ip, conn.raddr.port))
                
                # تفاصيل الاتصال | Détails de connexion
                if self.detailed_callback:
                    detail = ConnectionDetail(
                        timestamp=current_time * 1000,
                        local_address=conn.laddr.ip if conn.laddr else "",
                        local_port=conn.laddr.port if conn.laddr else 0,
                        remote_address=conn.raddr.ip,
                        remote_port=conn.raddr.port,
                        status=conn.status,
                        pid=conn.pid or 0,
                        process_name=self._get_process_name(conn.pid) if conn.pid else "unknown"
                    )
                    self.detailed_callback(detail)
                    with self._lock:
                        self._connection_details.append(detail)
        
        # إحصائيات حركة البيانات | Statistiques du trafic
        net_io = psutil.net_io_counters()
        bytes_sent = net_io.bytes_sent
        bytes_recv = net_io.bytes_recv
        packets_sent = net_io.packets_sent
        packets_recv = net_io.packets_recv
        
        # حساب المعدلات | Calculer les taux
        time_diff = current_time - self._prev_time
        if time_diff > 0:
            bytes_sent_rate = (bytes_sent - self._prev_bytes_sent) / time_diff
            bytes_recv_rate = (bytes_recv - self._prev_bytes_recv) / time_diff
        else:
            bytes_sent_rate = bytes_recv_rate = 0
        
        # الاتصالات الجديدة | Nouvelles connexions
        new_connections = len(current_connections - self._prev_connections)
        
        # أكثر العناوين نشاطاً | IPs les plus actives
        ip_counts = defaultdict(int)
        for conn in connections:
            if conn.raddr:
                ip_counts[conn.raddr.ip] += 1
        top_ips = sorted(ip_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        top_remote_ips = [ip for ip, _ in top_ips]
        
        # تحديث القراءات السابقة | Mettre à jour les lectures précédentes
        self._prev_bytes_sent = bytes_sent
        self._prev_bytes_recv = bytes_recv
        self._prev_connections = current_connections
        self._prev_time = current_time
        
        # إنشاء الحدث | Créer l'événement
        event = NetworkEvent(
            timestamp=current_time * 1000,
            timestamp_iso=now.isoformat(),
            event_type="network_snapshot",
            total_connections=total,
            established_connections=established,
            listening_ports=listening,
            unique_remote_ips=len(remote_ips),
            unique_remote_ports=len(remote_ports),
            unique_local_ports=len(local_ports),
            bytes_sent=bytes_sent,
            bytes_recv=bytes_recv,
            packets_sent=packets_sent,
            packets_recv=packets_recv,
            bytes_sent_rate=bytes_sent_rate,
            bytes_recv_rate=bytes_recv_rate,
            new_connections=new_connections,
            connection_types=dict(connection_types),
            top_remote_ips=top_remote_ips
        )
        
        if self.callback:
            self.callback(event)
        
        return event
    
    def _monitor_loop(self):
        """
        حلقة المراقبة الرئيسية | Boucle de surveillance principale
        """
        logger.info("بدء حلقة مراقبة الشبكة | Démarrage surveillance réseau")
        
        while self._running:
            try:
                event = self.collect_once()
                
                with self._lock:
                    self._events.append(event)
                    # الحفاظ على آخر 10000 حدث | Garder les derniers 10000
                    if len(self._events) > 10000:
                        self._events = self._events[-10000:]
                    if len(self._connection_details) > 50000:
                        self._connection_details = self._connection_details[-50000:]
                
                time.sleep(self.interval)
                
            except Exception as e:
                logger.error(f"خطأ في مراقبة الشبكة | Erreur surveillance réseau: {e}")
                time.sleep(1)
        
        logger.info("توقف مراقبة الشبكة | Surveillance réseau arrêtée")
    
    def start(self):
        """
        بدء المراقبة في خيط منفصل
        Démarrer la surveillance dans un thread séparé
        """
        if self._running:
            logger.warning("المراقب يعمل بالفعل | Moniteur déjà en cours")
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        logger.info("تم بدء مراقب الشبكة | Moniteur réseau démarré")
    
    def stop(self):
        """
        إيقاف المراقبة | Arrêter la surveillance
        """
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        logger.info("تم إيقاف مراقب الشبكة | Moniteur réseau arrêté")
    
    def get_events(self, clear: bool = False) -> List[NetworkEvent]:
        """
        الحصول على الأحداث المجمعة | Obtenir les événements collectés
        """
        with self._lock:
            events = self._events.copy()
            if clear:
                self._events.clear()
        return events
    
    def get_connection_details(self, clear: bool = False) -> List[ConnectionDetail]:
        """
        الحصول على تفاصيل الاتصالات | Obtenir les détails des connexions
        """
        with self._lock:
            details = self._connection_details.copy()
            if clear:
                self._connection_details.clear()
        return details
    
    def get_current_connections(self) -> List[Dict]:
        """
        الحصول على الاتصالات الحالية | Obtenir les connexions actuelles
        """
        connections = []
        for conn in psutil.net_connections(kind='inet'):
            if conn.raddr:
                connections.append({
                    "local_address": f"{conn.laddr.ip}:{conn.laddr.port}" if conn.laddr else "",
                    "remote_address": f"{conn.raddr.ip}:{conn.raddr.port}",
                    "status": conn.status,
                    "pid": conn.pid,
                    "process": self._get_process_name(conn.pid) if conn.pid else "unknown"
                })
        return connections


# اختبار الوحدة | Test du module
if __name__ == "__main__":
    print("=" * 60)
    print("اختبار مراقب الشبكة | Test du Moniteur Réseau")
    print("=" * 60)
    
    def on_event(event: NetworkEvent):
        print(f"\n📡 لقطة الشبكة | Snapshot Réseau:")
        print(f"   الاتصالات | Connexions: {event.total_connections}")
        print(f"   المنشأة | Établies: {event.established_connections}")
        print(f"   العناوين الفريدة | IPs uniques: {event.unique_remote_ips}")
        print(f"   معدل الإرسال | Taux envoi: {event.bytes_sent_rate/1024:.1f} KB/s")
        print(f"   معدل الاستقبال | Taux réception: {event.bytes_recv_rate/1024:.1f} KB/s")
    
    monitor = NetworkMonitor(interval=1.0, callback=on_event)
    
    # جمع لقطة واحدة | Collecter un snapshot
    print("\nجمع لقطة واحدة | Collecte d'un snapshot...")
    event = monitor.collect_once()
    
    # عرض الاتصالات الحالية | Afficher les connexions actuelles
    print("\nالاتصالات النشطة | Connexions actives:")
    connections = monitor.get_current_connections()
    for i, conn in enumerate(connections[:10]):
        print(f"  {i+1}. {conn['process']} -> {conn['remote_address']} [{conn['status']}]")
    if len(connections) > 10:
        print(f"  ... و {len(connections) - 10} اتصالات أخرى")
    
    # المراقبة المستمرة | Surveillance continue
    print("\nبدء المراقبة المستمرة (5 ثوان)...")
    monitor.start()
    time.sleep(5)
    monitor.stop()
    
    events = monitor.get_events()
    print(f"\nإجمالي الأحداث المجمعة: {len(events)}")
