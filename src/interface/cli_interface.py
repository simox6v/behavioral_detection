"""
واجهة سطر الأوامر | Interface CLI | CLI Interface
واجهة ملونة وتفاعلية للكشف الفوري
Interface colorée et interactive pour la détection en temps réel
"""

import os
import sys
import time
import argparse
from typing import Dict, List, Optional
from pathlib import Path
from datetime import datetime
import logging

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.layout import Layout
    from rich.live import Live
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
    from rich.text import Text
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("⚠️ مكتبة rich غير متوفرة | rich non disponible")

# استيراد الوحدات | Importer les modules
try:
    from ..detector.realtime_detector import RealtimeDetector, DetectionResult
except ImportError:
    RealtimeDetector = None
    DetectionResult = None

# إعداد التسجيل | Configuration du logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


class CLIInterface:
    """
    واجهة سطر الأوامر | Interface CLI
    واجهة ملونة وتفاعلية للكشف الفوري
    Interface colorée et interactive pour la détection
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        scaler_path: Optional[str] = None,
        model_name: str = 'isolation_forest'
    ):
        """
        تهيئة الواجهة | Initialisation de l'interface
        """
        if not RICH_AVAILABLE:
            raise ImportError("مكتبة rich مطلوبة | rich library required")
        
        self.console = Console()
        
        # إنشاء الكاشف | Créer le détecteur
        if RealtimeDetector:
            self.detector = RealtimeDetector(
                model_path=model_path,
                scaler_path=scaler_path,
                model_name=model_name
            )
            self.detector.set_alert_callback(self._on_alert)
        else:
            self.detector = None
            self.console.print("[yellow]⚠️ الكاشف غير متوفر | Détecteur non disponible[/]")
        
        # سجل التنبيهات | Historique des alertes
        self._alerts: List[DetectionResult] = []
        self._max_alerts = 20
        
        self._running = False
    
    def _on_alert(self, result: DetectionResult):
        """
        معالجة التنبيه | Traiter l'alerte
        """
        self._alerts.append(result)
        if len(self._alerts) > self._max_alerts:
            self._alerts.pop(0)
    
    def _get_status_table(self) -> Table:
        """
        إنشاء جدول الحالة | Créer le tableau d'état
        """
        table = Table(
            title="🛡️ حالة النظام | État du Système",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold cyan"
        )
        
        table.add_column("المؤشر | Indicateur", style="cyan", width=30)
        table.add_column("القيمة | Valeur", style="green", width=25)
        
        if self.detector:
            status = self.detector.get_status()
            
            # الحالة | État
            running_text = "[green]✅ يعمل | Running[/]" if status['running'] else "[red]❌ متوقف | Stopped[/]"
            table.add_row("الحالة | État", running_text)
            
            # النموذج | Modèle
            model_status = "[green]✅[/]" if status['model_loaded'] else "[red]❌[/]"
            table.add_row("النموذج | Modèle", f"{model_status} {status['model_name']}")
            
            # الإحصائيات | Statistiques
            table.add_row("إجمالي الكشوفات | Total", str(status['total_detections']))
            table.add_row("حميدة | Bénins", f"[green]{status['benign_count']}[/]")
            table.add_row("مشبوهة | Malveillants", f"[red]{status['malicious_count']}[/]")
            
            # الأداء | Performance
            latency = status['avg_latency_ms']
            latency_color = "green" if latency < 100 else "yellow" if latency < 500 else "red"
            table.add_row("متوسط التأخير | Latence", f"[{latency_color}]{latency:.1f}ms[/]")
            
            memory = status['current_memory_mb']
            memory_color = "green" if memory < 40 else "yellow" if memory < 60 else "red"
            table.add_row("الذاكرة | RAM", f"[{memory_color}]{memory:.1f}MB[/]")
        
        return table
    
    def _get_alerts_table(self) -> Table:
        """
        إنشاء جدول التنبيهات | Créer le tableau des alertes
        """
        table = Table(
            title="🚨 التنبيهات الأخيرة | Alertes Récentes",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold red"
        )
        
        table.add_column("الوقت | Heure", style="dim", width=12)
        table.add_column("المستوى | Niveau", width=10)
        table.add_column("النتيجة | Résultat", width=12)
        table.add_column("الثقة | Confiance", width=10)
        table.add_column("التأخير | Latence", width=10)
        
        for alert in reversed(self._alerts[-10:]):
            time_str = datetime.fromtimestamp(alert.timestamp / 1000).strftime("%H:%M:%S")
            
            level_map = {
                'normal': '[green]🟢 عادي[/]',
                'warning': '[yellow]🟡 تحذير[/]',
                'danger': '[red]🔴 خطر[/]'
            }
            level = level_map.get(alert.alert_level, '❓')
            
            pred_color = 'red' if alert.prediction == 'malicious' else 'green'
            pred_text = f"[{pred_color}]{alert.prediction}[/]"
            
            conf_color = 'red' if alert.confidence > 0.7 else 'yellow' if alert.confidence > 0.4 else 'green'
            conf_text = f"[{conf_color}]{alert.confidence:.1%}[/]"
            
            table.add_row(
                time_str,
                level,
                pred_text,
                conf_text,
                f"{alert.latency_ms:.1f}ms"
            )
        
        return table
    
    def _get_features_panel(self) -> Panel:
        """
        إنشاء لوحة الميزات | Créer le panneau des features
        """
        if self.detector and self.detector.feature_extractor:
            features = self.detector.feature_extractor.get_current_features()
            
            lines = []
            for name, value in list(features.items())[:10]:
                bar_length = int(min(value * 2, 20))
                bar = "█" * bar_length + "░" * (20 - bar_length)
                lines.append(f"{name:<25} {bar} {value:.3f}")
            
            content = "\n".join(lines)
        else:
            content = "لا توجد بيانات | Pas de données"
        
        return Panel(
            content,
            title="📊 الميزات الحالية | Features Actuelles",
            border_style="blue"
        )
    
    def _create_layout(self) -> Layout:
        """
        إنشاء تخطيط الشاشة | Créer la mise en page
        """
        layout = Layout()
        
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="footer", size=3)
        )
        
        layout["body"].split_row(
            Layout(name="left"),
            Layout(name="right")
        )
        
        return layout
    
    def _render(self) -> Layout:
        """
        رسم الواجهة | Rendre l'interface
        """
        layout = self._create_layout()
        
        # الرأس | Header
        header = Panel(
            Text("🛡️ نظام الكشف السلوكي | Système de Détection Comportementale", 
                 justify="center", style="bold white on blue"),
            box=box.DOUBLE
        )
        layout["header"].update(header)
        
        # الجسم - اليسار | Corps - Gauche
        layout["left"].update(self._get_status_table())
        
        # الجسم - اليمين | Corps - Droite
        layout["right"].split_column(
            Layout(self._get_alerts_table(), name="alerts"),
            Layout(self._get_features_panel(), name="features")
        )
        
        # التذييل | Footer
        footer_text = f"⏱️ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | اضغط Ctrl+C للإيقاف | Appuyez sur Ctrl+C pour arrêter"
        footer = Panel(Text(footer_text, justify="center", style="dim"))
        layout["footer"].update(footer)
        
        return layout
    
    def run(self, duration: Optional[int] = None):
        """
        تشغيل الواجهة | Exécuter l'interface
        
        Args:
            duration: مدة التشغيل بالثواني (None = غير محدد) | Durée en secondes
        """
        self._running = True
        
        # بدء الكاشف | Démarrer le détecteur
        if self.detector:
            self.detector.start(interval=1.0)
        
        start_time = time.time()
        
        try:
            with Live(self._render(), refresh_per_second=2, console=self.console) as live:
                while self._running:
                    live.update(self._render())
                    time.sleep(0.5)
                    
                    if duration and (time.time() - start_time) >= duration:
                        break
        
        except KeyboardInterrupt:
            self.console.print("\n[yellow]⚠️ توقف بواسطة المستخدم | Arrêt par l'utilisateur[/]")
        
        finally:
            self._running = False
            if self.detector:
                self.detector.stop()
            
            self._print_summary()
    
    def _print_summary(self):
        """
        طباعة الملخص النهائي | Afficher le résumé final
        """
        self.console.print("\n")
        self.console.print(Panel(
            "[bold]📊 ملخص الجلسة | Résumé de la Session[/]",
            box=box.DOUBLE,
            style="cyan"
        ))
        
        if self.detector:
            self.console.print(self._get_status_table())
            
            if self._alerts:
                self.console.print(f"\n[yellow]⚠️ إجمالي التنبيهات | Total alertes: {len(self._alerts)}[/]")


class SimpleCLI:
    """
    واجهة مبسطة | Interface Simplifiée
    تعمل بدون مكتبة rich
    Fonctionne sans la bibliothèque rich
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        scaler_path: Optional[str] = None,
        model_name: str = 'isolation_forest'
    ):
        """
        تهيئة الواجهة | Initialisation de l'interface
        """
        if RealtimeDetector:
            self.detector = RealtimeDetector(
                model_path=model_path,
                scaler_path=scaler_path,
                model_name=model_name
            )
            self.detector.set_alert_callback(self._on_alert)
        else:
            self.detector = None
        
        self._alerts = []
        self._running = False
    
    def _on_alert(self, result):
        """معالجة التنبيه | Traiter l'alerte"""
        self._alerts.append(result)
        
        level_icons = {'normal': '🟢', 'warning': '🟡', 'danger': '🔴'}
        icon = level_icons.get(result.alert_level, '❓')
        
        print(f"\n{icon} تنبيه | Alerte @ {datetime.now().strftime('%H:%M:%S')}")
        print(f"   النتيجة | Résultat: {result.prediction.upper()}")
        print(f"   الثقة | Confiance: {result.confidence:.1%}")
        print(f"   التأخير | Latence: {result.latency_ms:.1f}ms")
    
    def _print_status(self):
        """طباعة الحالة | Afficher l'état"""
        if self.detector:
            status = self.detector.get_status()
            
            print("\n" + "=" * 50)
            print("🛡️ حالة النظام | État du Système")
            print("=" * 50)
            print(f"   يعمل | Running: {'✅' if status['running'] else '❌'}")
            print(f"   النموذج | Modèle: {status['model_name']}")
            print(f"   إجمالي الكشوفات | Total: {status['total_detections']}")
            print(f"   حميدة | Bénins: {status['benign_count']}")
            print(f"   مشبوهة | Malveillants: {status['malicious_count']}")
            print(f"   متوسط التأخير | Latence: {status['avg_latency_ms']:.1f}ms")
            print(f"   الذاكرة | RAM: {status['current_memory_mb']:.1f}MB")
            print("=" * 50)
    
    def run(self, duration: Optional[int] = None):
        """
        تشغيل الواجهة | Exécuter l'interface
        """
        print("\n" + "=" * 60)
        print("🛡️ نظام الكشف السلوكي | Système de Détection")
        print("=" * 60)
        
        self._running = True
        
        if self.detector:
            self.detector.start(interval=1.0)
        
        start_time = time.time()
        status_interval = 10  # ثوان
        last_status = time.time()
        
        try:
            print("\n⏳ الكشف قيد التشغيل | Détection en cours...")
            print("اضغط Ctrl+C للإيقاف | Appuyez sur Ctrl+C pour arrêter\n")
            
            while self._running:
                time.sleep(1)
                
                # طباعة الحالة كل 10 ثوان
                if time.time() - last_status >= status_interval:
                    self._print_status()
                    last_status = time.time()
                
                if duration and (time.time() - start_time) >= duration:
                    break
        
        except KeyboardInterrupt:
            print("\n\n⚠️ توقف بواسطة المستخدم | Arrêt par l'utilisateur")
        
        finally:
            self._running = False
            if self.detector:
                self.detector.stop()
            
            self._print_status()
            print(f"\n📊 إجمالي التنبيهات | Total alertes: {len(self._alerts)}")


def main():
    """
    الدالة الرئيسية | Fonction principale
    """
    parser = argparse.ArgumentParser(
        description="واجهة CLI للكشف الفوري | Interface CLI de Détection"
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
        help='اسم النموذج | Nom du modèle'
    )
    parser.add_argument(
        '--duration',
        type=int,
        default=None,
        help='مدة التشغيل بالثواني | Durée en secondes'
    )
    parser.add_argument(
        '--simple',
        action='store_true',
        help='استخدام الواجهة المبسطة | Utiliser l\'interface simple'
    )
    
    args = parser.parse_args()
    
    # اختيار الواجهة | Choisir l'interface
    if args.simple or not RICH_AVAILABLE:
        cli = SimpleCLI(
            model_path=args.model,
            scaler_path=args.scaler,
            model_name=args.model_name
        )
    else:
        cli = CLIInterface(
            model_path=args.model,
            scaler_path=args.scaler,
            model_name=args.model_name
        )
    
    cli.run(duration=args.duration)


if __name__ == "__main__":
    main()
