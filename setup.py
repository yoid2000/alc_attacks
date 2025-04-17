from setuptools import setup, find_packages

setup(
    name='alc_attacks',
    version='1.0.4',
    description='A collection of attacks on synthetic data using the anonymity loss coefficient (ALC).',
    author='Paul Francis',
    author_email='paul@francis.com',
    url='https://github.com/yoid2000/alc_attacks',
    packages=find_packages(),
    install_requires=[
        "pandas>=1.0.0",
        "numpy>=1.18.0",
        "matplotlib>=3.0.0",
        "anonymity_loss_coefficient>=1.0.0",
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)