from setuptools import setup, find_packages

setup(
    name='fermi',
    version='1.0.0',
    description='Fermi estimation using Monte Carlo simulation',
    author='ThisCakeIsALie',
    packages=find_packages(),
    py_modules=['fermi'],
    install_requires=[
        'numpy',
        'scipy',
    ],
    entry_points={
        'console_scripts': [
            'fermi=fermi:main',
        ],
    },
)