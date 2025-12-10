# 🛡️ نظام الكشف السلوكي للبرامج المشبوهة
# Système de Détection Comportementale de Programmes Suspects
# Behavioral Detection System for Suspicious Programs

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📋 الوصف | Description

نظام كامل للكشف السلوكي عن الشذوذ في Python، قادر على التمييز في الوقت الفعلي بين السلوك الطبيعي والسلوك المشبوه.

Système complet de détection comportementale d'anomalies en Python, capable de distinguer en temps réel un comportement normal d'un comportement suspect.

> ⚠️ **تحذير**: هذا المشروع تعليمي بحت. ممنوع منعاً باتاً تنفيذ أو تحميل برامج ضارة حقيقية.
> 
> ⚠️ **Avertissement**: Ce projet est purement éducatif. Interdiction formelle d'exécuter ou télécharger du vrai malware.

---

## 🏗️ الهيكل | Architecture

```
behavioral_detection/
├── config/config.yaml          # التكوين | Configuration
├── src/
│   ├── collector/              # جمع الأحداث | Collecte
│   ├── generator/              # توليد البيانات | Génération
│   ├── features/               # هندسة الميزات | Features
│   ├── models/                 # نماذج ML | Modèles
│   ├── detector/               # الكشف الفوري | Détection
│   └── interface/              # واجهة المستخدم | Interface
├── data/                       # البيانات | Données
└── tests/                      # الاختبارات | Tests
```

---

## 🚀 التثبيت | Installation

```bash
# استنساخ المشروع | Cloner le projet
git clone <repository>
cd behavioral_detection

# إنشاء بيئة افتراضية | Créer un environnement virtuel
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# أو | ou
.venv\Scripts\activate     # Windows

# تثبيت المتطلبات | Installer les dépendances
pip install -r requirements.txt

# تثبيت المشروع | Installer le projet
pip install -e .
```

---

## 📖 الاستخدام | Utilisation

### 1️⃣ جمع البيانات | Collecte de données

```bash
# تشغيل الجامع | Lancer le collecteur
python -m src.collector.behavior_collector

# توليد البيانات | Générer les données
python -m src.generator.dataset_generator
```

### 2️⃣ تدريب النماذج | Entraînement des modèles

```bash
# تدريب جميع النماذج | Entraîner tous les modèles
python -m src.models.train_models

# تقييم الأداء | Évaluer les performances
python -m src.models.model_evaluation
```

### 3️⃣ الكشف الفوري | Détection en temps réel

```bash
# واجهة Streamlit | Interface Streamlit
streamlit run src/interface/streamlit_app.py

# واجهة CLI | Interface CLI
python -m src.interface.cli_interface
```

---

## 🎯 الميزات | Fonctionnalités

### مراقبة النظام | Surveillance Système
- ✅ مراقبة العمليات (CPU, RAM, I/O, threads)
- ✅ مراقبة الشبكة (اتصالات، منافذ، عناوين IP)
- ✅ مراقبة الملفات (إنشاء، حذف، تعديل، نقل)

### نماذج التعلم الآلي | Modèles ML
- ✅ Isolation Forest
- ✅ One-Class SVM
- ✅ Local Outlier Factor (LOF)
- ✅ Random Forest
- ✅ XGBoost

### الواجهات | Interfaces
- ✅ لوحة معلومات Streamlit تفاعلية
- ✅ واجهة CLI ملونة

---

## 📊 السيناريوهات المحاكاة | Scénarios Simulés

| السيناريو | الوصف |
|-----------|-------|
| 🔥 Burst Files | إنشاء/حذف ملفات بسرعة عالية |
| 🔍 Port Scan | مسح المنافذ السريع |
| 📖 File Sniffing | قراءة متكررة لملفات حساسة |
| 🔒 Ransomware-like | محاكاة تشفير الملفات |
| 🔐 Brute-force | حلقات مكثفة |

---

## 🧪 الاختبارات | Tests

```bash
# تشغيل جميع الاختبارات | Lancer tous les tests
pytest tests/ -v

# مع تغطية الكود | Avec couverture
pytest tests/ -v --cov=src
```

---

## 🐳 Docker

```bash
# بناء الصورة | Construire l'image
docker-compose -f docker/docker-compose.yml build

# تشغيل | Lancer
docker-compose -f docker/docker-compose.yml up
```

---

## 📝 المساهمة | Contribution

نرحب بالمساهمات! يرجى فتح issue أو pull request.

Les contributions sont les bienvenues! Veuillez ouvrir une issue ou une pull request.

---

## 📄 الترخيص | Licence

MIT License - انظر ملف LICENSE للتفاصيل.

---

## 👨‍💻 المؤلف | Auteur

تم التطوير بواسطة نظام الذكاء الاصطناعي.

Développé avec l'aide de l'IA.
