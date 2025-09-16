"""
Setup script for the RSI-ADX Momentum Trading Strategy.

This script provides installation and setup utilities for the trading system.
"""

from setuptools import setup, find_packages
import os

# Read README for long description
def read_readme():
    """Read README.md file."""
    try:
        with open("README.md", "r", encoding="utf-8") as fh:
            return fh.read()
    except FileNotFoundError:
        return "RSI-ADX Momentum Trading Strategy"

# Read requirements
def read_requirements():
    """Read requirements.txt file."""
    try:
        with open("requirements.txt", "r", encoding="utf-8") as fh:
            return [line.strip() for line in fh if line.strip() and not line.startswith("#")]
    except FileNotFoundError:
        return [
            "pandas>=1.5.0",
            "numpy>=1.21.0",
            "matplotlib>=3.5.0",
            "seaborn>=0.11.0",
            "scipy>=1.9.0"
        ]

setup(
    name="rsi-adx-trading-strategy",
    version="1.0.0",
    author="Trading Bot Developer",
    author_email="developer@example.com",
    description="A professional-grade RSI-ADX momentum trading strategy with backtesting framework",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/rsi-adx-trading-strategy",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
        "jupyter": [
            "jupyter>=1.0.0",
            "ipykernel>=6.15.0",
            "notebook>=6.4.0",
        ],
        "performance": [
            "numba>=0.56.0",
            "cython>=0.29.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "rsi-adx-backtest=main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.yml", "*.yaml"],
        "data": ["*.csv"],
        "config": ["*.py"],
    },
    project_urls={
        "Bug Reports": "https://github.com/your-username/rsi-adx-trading-strategy/issues",
        "Source": "https://github.com/your-username/rsi-adx-trading-strategy",
        "Documentation": "https://github.com/your-username/rsi-adx-trading-strategy#readme",
    },
)