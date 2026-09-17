from setuptools import setup, find_packages

setup(
    name='aerial-tile-segmentation',
    version='1.0.0',
    description='Semantic Segmentation of High-Resolution Aerial/Satellite Imagery using Residual U-Net',
    author='Abhitesh',
    packages=find_packages(),
    python_requires='>=3.7',
    install_requires=[
        'tensorflow>=2.0.0',
        'numpy>=1.18.0',
        'pandas>=1.0.0',
        'imageio>=2.6.0',
        'matplotlib>=3.1.0',
        'tqdm>=4.40.0',
        'PyYAML>=5.3.0',
    ],
)
