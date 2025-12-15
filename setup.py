from setuptools import setup, find_packages

setup(
    name="sc-pcgc",
    version="0.1.0",
    description="Scalable Point Cloud Geometry Compression with Oct-Attention",
    author="ESRSchao",
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=[
        "torch>=1.10.0",
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "open3d>=0.13.0",
        "h5py>=3.1.0",
        "pyyaml>=5.4.1",
        "tqdm>=4.62.0",
        "tensorboard>=2.7.0",
        "matplotlib>=3.4.0",
    ],
)
