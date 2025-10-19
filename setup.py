#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
GPT-SoVITS-OIE Text Processing Library

A comprehensive text preprocessing library for multilingual speech synthesis,
supporting Chinese, English, Japanese, Korean, and Cantonese languages.
"""

import os
from setuptools import setup, find_packages

# Get the long description from the README file
current_dir = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(current_dir, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Read requirements from requirements.txt
def read_requirements():
    """Read requirements from requirements.txt file"""
    requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    with open(requirements_path, 'r', encoding='utf-8') as f:
        requirements = []
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                requirements.append(line)
    return requirements

# Version information
__version__ = '1.0.0'

setup(
    name='gsv_oie',
    version=__version__,
    description='Multilingual text preprocessing library for GPT-SoVITS',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='GPT-SoVITS-OIE Contributors',
    author_email='',
    url='https://github.com/openGPT-SoVITS/GPT-SoVITS-OIE',
    license='MIT',

    # Package configuration
    packages=find_packages(exclude=['tests*', 'docs*', 'examples*', 'gsv_oie.pretrained_models']),
    include_package_data=True,
    package_data={
        'gsv_oie.text_preprocess': [
            '**/*.py',
            '**/*.txt',
            '**/*.json',
            '**/*.model',
            '**/*.pkl',
            '**/*.pickle',
            '**/*.yaml',
            '**/*.yml',
            '**/*.cfg',
            '**/*.ini',
            '**/*.csv',
            '**/*.tsv',
            '**/*.xml',
            '**/*.html',
            '**/*.md',
            '**/*.rst',
            '**/*.dict',
            '**/*.vocab',
            '**/*.bin',
            '**/*.npz',
            '**/*.npy',
            '**/*.h5',
            '**/*.hdf5',
            '**/*.marisa',
            '**/*.marisa_trie',
            '**/*.trie',
            'g2pw/*',
            'LangSegmenter/*',
            'zh_normalization/*',
            'en_normalization/*',
            'ja_userdic/*',
        ],
    },

    # Python version requirement
    python_requires='>=3.7',

    # Dependencies
    install_requires=read_requirements(),

    # Optional dependencies for different language support
    extras_require={
        'mnn':['mnn'],
        'dev': [
            'pytest>=6.0',
            'black>=21.0',
            'flake8>=3.8',
            'mypy>=0.800',
        ]
    },

    
    # Classifiers for PyPI
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Text Processing :: Linguistic',
        'Topic :: Multimedia :: Sound/Audio :: Speech',
    ],

    # Keywords
    keywords='text-processing, speech-synthesis, nlp, multilingual, gsv-oie, gpt-sovits, bert, phonemes',

    # Project URLs
    project_urls={
        'Bug Reports': 'https://github.com/openGPT-SoVITS/GPT-SoVITS-OIE/issues',
        'Source': 'https://github.com/openGPT-SoVITS/GPT-SoVITS-OIE',
        'Documentation': 'https://github.com/openGPT-SoVITS/GPT-SoVITS-OIE/blob/main/README.md',
    },
)