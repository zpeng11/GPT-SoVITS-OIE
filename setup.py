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
    ]
    return include_dirs

def get_gsv_engine_ext():
    from pybind11.setup_helpers import Pybind11Extension
    # Define the C++ extension
    gsv_engine_ext = Pybind11Extension(
        "gsv_oie.gsv_runtime.gsv_engine",
        sources=[
            "gsv_oie/gsv_runtime/src/gsv_engine.cpp",
        ],
        include_dirs=get_include_dirs(),
        library_dirs=[],
        libraries=[],
        cxx_std=17,
        define_macros=[("VERSION_INFO", '"dev"')],
        extra_compile_args=["-O3", "-Wall", "-shared", "-std=c++17"],
        extra_link_args=["-O3"],
    )

    tokenizer_ext = Pybind11Extension(
        "gsv_oie.text_preprocess.tokenizers_cpp",
        sources=[
            "gsv_oie/text_preprocess/tokenizers.cpp",
        ],
        include_dirs=get_include_dirs(),
        library_dirs=[],
        libraries=[],
        cxx_std=17,
        define_macros=[("VERSION_INFO", '"dev"')],
        extra_compile_args=["-O3", "-Wall", "-shared", "-std=c++17"],
        extra_link_args=["-O3"],
    )
    return [gsv_engine_ext, tokenizer_ext]

def get_build_ext():
    from pybind11.setup_helpers import build_ext
    class CoordinatedBuildExt(build_ext):
        def run(self):
            # 第一步：CMake 构建（生成库）
            self.build_mnn()

            self.build_tokenizers_cpp()

            self.prebuild_onnxruntime()

            # 第二步：pybind11 扩展构建（依赖 CMake 输出）
            super().run()  # 这会按 ext_modules 顺序构建 pybind11 扩展，并链接 CMake 库

        def build_mnn(self):
            check_gcc_version()
            self.mnn_src_dir = os.path.join(os.path.dirname(__file__), 'extern', 'MNN')
            self.mnn_build_dir = os.path.join(self.mnn_src_dir, 'build')
            # 确保构建目录
            if not os.path.exists(self.mnn_build_dir):
                os.makedirs(self.mnn_build_dir, exist_ok=True)

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
                '-S', self.mnn_src_dir,  # 源目录（项目根，含 CMakeLists.txt）
                '-B', self.mnn_build_dir  # 构建目录
            ]
            try:
                subprocess.check_call(cmake_cmd, cwd=self.mnn_build_dir)
            except subprocess.CalledProcessError as e:
                raise CompileError(f'CMake config failed: {e}')

            # CMake 构建
            build_cmd = ['cmake', '--build', self.mnn_build_dir, '-j8']
            try:
                subprocess.check_call(build_cmd, cwd=self.mnn_build_dir)
            except subprocess.CalledProcessError as e:
                raise CompileError(f'CMake build failed: {e}')

            # 可选：打印输出路径
            lib_path = os.path.join(self.mnn_build_dir, 'libMNN.a')
            if os.path.exists(lib_path):
                print(f"MNN CMake lib built at: {lib_path}")

        def build_tokenizers_cpp(self):
            self.tokenizers_cpp_src_dir = os.path.join(os.path.dirname(__file__), 'extern', 'tokenizers-cpp')
            self.tokenizers_cpp_build_dir = os.path.join(self.tokenizers_cpp_src_dir, 'build')
            if not os.path.exists(self.tokenizers_cpp_build_dir):
                os.makedirs(self.tokenizers_cpp_build_dir, exist_ok=True)
            cmake_cmd = [
                'cmake',
                '-DCMAKE_BUILD_TYPE=Release',
                '-DCMAKE_CXX_FLAGS=-fPIC',
                '-DCMAKE_C_FLAGS=-fPIC',
                '-S', self.tokenizers_cpp_src_dir,
                '-B', self.tokenizers_cpp_build_dir
            ]
            try:
                subprocess.check_call(cmake_cmd, cwd=self.tokenizers_cpp_build_dir)
            except subprocess.CalledProcessError as e:
                raise CompileError(f'Tokenizers CMake config failed: {e}')
            build_cmd = ['cmake', '--build', self.tokenizers_cpp_build_dir, '-j8']
            try:
                subprocess.check_call(build_cmd, cwd=self.tokenizers_cpp_build_dir)
            except subprocess.CalledProcessError as e:
                raise CompileError(f'Tokenizers CMake build failed: {e}')

            lib_path = os.path.join(self.tokenizers_cpp_build_dir, 'libtokenizers_cpp.a')
            if os.path.exists(lib_path):
                print(f"Tokenizers CMake lib built at: {lib_path}")

        def prebuild_onnxruntime(self):
            # 如果需要，可以在这里添加预构建 ONNX Runtime 的逻辑
            pass

        def build_extension(self, ext):
            if ext.name == "gsv_oie.gsv_runtime.gsv_engine":
                ext.library_dirs.extend([self.mnn_build_dir])

                ext.include_dirs.extend([
                    os.path.join(self.mnn_src_dir, 'include'),
                    os.path.join(os.path.dirname(__file__), 'extern', 'onnxruntime', 'include', 'onnxruntime'),
                ])

                ext.libraries.extend(['MNN'])

            elif ext.name == "gsv_oie.text_preprocess.tokenizers_cpp":
                ext.library_dirs.extend([self.tokenizers_cpp_build_dir])

                ext.include_dirs.extend([
                    os.path.join(self.tokenizers_cpp_src_dir, 'include'),
                ])

                ext.libraries.extend(['tokenizers_c','tokenizers_cpp'])

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

    setup_requires=[
        "cmake==3.29.2",  # Minimum version from your CMakeLists.txt
    ],

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