import os
from setuptools import setup,find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='Weighted Class Tfidf',
    version='1.0.3',
    author='Saurav Pattnaik',
    description='Custom implementation of tfidf for imbalanced datasets',
    long_description=long_description,
    long_description_content_type = "text/markdown",
    url='https://github.com/SauravPattnaikCS60/Weighted-Class-Tfidf',
    python_requires='>=3.6',
    py_modules = ['wcbtfidf'],
    packages=find_packages(exclude=['demos']),
    include_package_data = False,
    classifiers = [
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ]
)