from setuptools import setup, find_packages

setup(
    name="trading-core",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=2.0.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.5.0",
        "scipy>=1.14.0",
        "numba>=0.60.0",
        "pydantic>=2.0.0",
        "python-dateutil>=2.8.2",
        "pytz>=2024.1",
    ],
    description="Core Renko Physics and Strategy Gates",
    python_requires=">=3.12",
)

