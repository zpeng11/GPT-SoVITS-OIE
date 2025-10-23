#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
GPT-SoVITS-OIE Text Processing Library

A comprehensive text preprocessing library for multilingual speech synthesis,
supporting Chinese, English, Japanese, Korean, and Cantonese languages.
"""

import os
import sys
import subprocess
from setuptools import setup, find_packages, Extension
from distutils.errors import CompileError
import re

DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(DIR, "extern", "pybind11"))

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

def check_gcc_version():
    cc = os.environ.get('CC', 'gcc')
    cc_path = subprocess.run(['which', cc], capture_output=True, text=True).stdout.strip()
    if not cc_path:
        raise CompileError(f"CC '{cc}' not found in PATH.")
    """检查 CC 是否为 GCC 且版本 > 11"""
    if not os.path.basename(cc_path).endswith('gcc'):
        raise CompileError(f"CC '{cc_path}' is not a GCC compiler (must start with 'gcc').")
    
    try:
        result = subprocess.run([cc_path, '--version'], capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            raise CompileError(f"Failed to get version for {cc_path}: {result.stderr}")
        
        # 匹配 GCC 版本行，如 "gcc (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0"
        version_match = re.search(r'gcc \([^)]+\) (\d+(?:\.\d+)*)', result.stdout)
        if not version_match:
            raise CompileError("Could not parse GCC version from output.")
        
        full_version = version_match.group(1)
        major_version = int(full_version.split('.')[0])
        
        if major_version < 11:
            raise CompileError(f"GCC version {full_version} (<11) is too old. Need >=11.")
        
        print(f"Using GCC {full_version} (>=11): OK.")
        return major_version
    except subprocess.TimeoutExpired:
        raise CompileError("Timeout getting GCC version.")


# C++ extension configuration
def get_include_dirs():
    """Get include directories for the C++ extension."""
    import pybind11
    include_dirs = [
        pybind11.get_include(),
        os.path.join(os.path.dirname(__file__), 'extern', 'MNN', 'include'),
        os.path.join(os.path.dirname(__file__), 'extern', 'onnxruntime', 'include', 'onnxruntime'),
    ]
    return include_dirs

def get_library_dirs():
    """Get library directories for the C++ extension."""
    # Only use local libraries, don't search for external ones
    return [os.path.join(os.path.dirname(__file__), 'extern', 'MNN', 'build')]

def get_libraries():
    """Get libraries to link against."""
    # Only link against system libraries that are always available
    return ['MNN']

def get_gsv_engine_ext():
    from pybind11.setup_helpers import Pybind11Extension
    # Define the C++ extension
    gsv_engine_ext = Pybind11Extension(
        "gsv_oie.gsv_runtime.gsv_engine",
        sources=[
            "gsv_oie/gsv_runtime/src/gsv_engine.cpp",
        ],
        include_dirs=get_include_dirs(),
        library_dirs=get_library_dirs(),
        libraries=get_libraries(),
        cxx_std=17,
        define_macros=[("VERSION_INFO", '"dev"')],
        extra_compile_args=["-O3", "-Wall", "-shared", "-std=c++17"],
        extra_link_args=["-O3"],
    )
    return [gsv_engine_ext]

def get_build_ext():
    from pybind11.setup_helpers import build_ext
    class CoordinatedBuildExt(build_ext):
        def run(self):
            # 第一步：CMake 构建（生成库）
            self._build_cmake()

            # 第二步：pybind11 扩展构建（依赖 CMake 输出）
            super().run()  # 这会按 ext_modules 顺序构建 pybind11 扩展，并链接 CMake 库

        def _build_cmake(self):
            check_gcc_version()
            self.src_dir = os.path.join(os.path.dirname(__file__), 'extern', 'MNN')
            self.build_dir = os.path.join(self.src_dir, 'build')
            # 确保构建目录
            if not os.path.exists(self.build_dir):
                os.makedirs(self.build_dir, exist_ok=True)

            #需要提前配置$CC $CXX 指向gcc11以上
            # CMake 配置
            cmake_cmd = [
                'cmake',
                '-DCMAKE_BUILD_TYPE=Release',
                '-DMNN_BUILD_TOOLS=OFF',
                '-DMNN_BUILD_SHARED_LIBS=OFF',
                '-DMNN_REDUCE_SIZE=ON',
                '-DMNN_LOW_MEMORY=ON',
                '-DMNN_CPU_WEIGHT_DEQUANT_GEMM=ON',
                '-DMNN_USE_SSE=ON',
                '-S', self.src_dir,  # 源目录（项目根，含 CMakeLists.txt）
                '-B', self.build_dir  # 构建目录
            ]
            try:
                subprocess.check_call(cmake_cmd, cwd=self.build_dir)
            except subprocess.CalledProcessError as e:
                raise CompileError(f'CMake config failed: {e}')

            # CMake 构建
            build_cmd = ['cmake', '--build', self.build_dir, '-j8']
            try:
                subprocess.check_call(build_cmd, cwd=self.build_dir)
            except subprocess.CalledProcessError as e:
                raise CompileError(f'CMake build failed: {e}')

            # 可选：打印输出路径
            cmake_lib_path = os.path.join(self.build_dir, 'libMNN.a')
            if os.path.exists(cmake_lib_path):
                print(f"CMake lib built at: {cmake_lib_path}")

        def build_extension(self, ext):
            # 重写以确保 pybind11 扩展能找到 CMake 库（动态注入 library_dirs）
            super().build_extension(ext)

    return CoordinatedBuildExt

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

    # C++ extension modules
    ext_modules=get_gsv_engine_ext(),
    cmdclass={"build_ext": get_build_ext()},


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