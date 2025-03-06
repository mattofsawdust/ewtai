from setuptools import setup, find_packages

setup(
    name="ewtai",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.20.0",
        "matplotlib>=3.4.0",
        "yfinance>=0.1.70",
        "tensorflow>=2.6.0",
        "scikit-learn>=0.24.0",
        "xgboost>=1.4.0",
        "scipy>=1.7.0",
        "talib-binary>=0.4.0",
        "transformers>=4.8.0",
        "joblib>=1.0.0",
    ],
    author="Matt Hobbs",
    author_email="your.email@example.com",
    description="Enhanced Market Analyzer with AI and Elliott Wave Theory",
    keywords="finance, trading, machine learning, elliott wave",
    url="https://github.com/yourusername/EWTai",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)