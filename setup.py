from setuptools import setup, find_packages

setup(
    name='unsflow',
    version='1.0.0',
    license='MIT',
    description='Unsflow is a software for compressor instabilities analysis and prediction',
    author='Francesco Neri',
    install_requires=[
        'numpy',
        'scikit-learn',
        'matplotlib',
        'pandas',
        'sympy'
        # other dependencies...
    ],
    packages=find_packages(),
)
