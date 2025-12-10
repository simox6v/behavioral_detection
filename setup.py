"""
نظام الكشف السلوكي للبرامج المشبوهة
Système de Détection Comportementale de Programmes Suspects
Behavioral Detection System for Suspicious Programs

ملف التثبيت | Fichier d'installation | Setup file
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="behavioral_detection",
    version="1.0.0",
    author="AI Assistant",
    author_email="",
    description="نظام الكشف السلوكي للبرامج المشبوهة | Système de détection comportementale",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Security",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "behavior-collector=src.collector.behavior_collector:main",
            "dataset-generator=src.generator.dataset_generator:main",
            "train-models=src.models.train_models:main",
            "realtime-detector=src.detector.realtime_detector:main",
            "detection-cli=src.interface.cli_interface:main",
        ],
    },
)
