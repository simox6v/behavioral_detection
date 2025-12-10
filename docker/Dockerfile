# نظام الكشف السلوكي | Système de Détection Comportementale
# Dockerfile for Behavioral Detection System

FROM python:3.11-slim

# التسميات | Labels
LABEL maintainer="AI Assistant"
LABEL description="Behavioral Detection System for Suspicious Programs"
LABEL version="1.0.0"

# متغيرات البيئة | Variables d'environnement
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1

# إنشاء مستخدم غير جذري | Créer un utilisateur non-root
RUN useradd --create-home --shell /bin/bash appuser

# مجلد العمل | Répertoire de travail
WORKDIR /app

# تثبيت التبعيات النظام | Installer les dépendances système
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# نسخ ملفات المتطلبات | Copier les fichiers de requirements
COPY requirements.txt .

# تثبيت المتطلبات | Installer les requirements
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# نسخ الكود المصدري | Copier le code source
COPY . .

# تثبيت الحزمة | Installer le package
RUN pip install -e .

# إنشاء المجلدات | Créer les répertoires
RUN mkdir -p /app/data/raw /app/data/processed /app/data/models /app/logs

# تغيير الملكية | Changer la propriété
RUN chown -R appuser:appuser /app

# التبديل للمستخدم | Basculer vers l'utilisateur
USER appuser

# المنفذ للـ Streamlit | Port pour Streamlit
EXPOSE 8501

# الأمر الافتراضي | Commande par défaut
CMD ["python", "-m", "src.interface.cli_interface", "--simple"]

# أوامر بديلة | Commandes alternatives:
# للواجهة التفاعلية | Interface interactive:
# CMD ["streamlit", "run", "src/interface/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
# 
# لتوليد البيانات | Génération de données:
# CMD ["python", "-m", "src.generator.dataset_generator"]
#
# للكشف الفوري | Détection en temps réel:
# CMD ["python", "-m", "src.detector.realtime_detector"]
