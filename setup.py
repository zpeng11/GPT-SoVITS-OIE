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
import glob
from setuptools import setup, find_packages, Extension
from distutils.errors import CompileError
import shutil
import re
import zipfile

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

IS_WINDOWS = sys.platform.startswith('win')
IS_UNIX = not IS_WINDOWS

GSV_ANDROID_BUILD = False
if os.environ.get('GSV_ANDROID_BUILD', '0') == '1' and IS_UNIX:
    GSV_ANDROID_BUILD = True


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


def get_gsv_engine_ext():
    from pybind11.setup_helpers import Pybind11Extension
    import pybind11

    engine_compiler_args = []
    if IS_WINDOWS:
        engine_compiler_args.extend(['/O2', '/Wall', '/openmp'])
    else:
        engine_compiler_args.extend(['-O3', '-Wall', '-fopenmp'])
        if GSV_ANDROID_BUILD:
            engine_compiler_args.extend(['-mfp16-format=ieee'])
    # Define the C++ extension
    gsv_engine_ext = Pybind11Extension(
        "gsv_oie.gsv_runtime.gsv_engine",
        sources=[
            "gsv_oie/gsv_runtime/cpp/gsv_engine.cpp",
            "gsv_oie/gsv_runtime/cpp/MNNInferenceEngineInterpreter.cpp",
            "gsv_oie/gsv_runtime/cpp/utils.cpp",
        ],
        include_dirs=[pybind11.get_include(), os.path.join(os.path.dirname(__file__), 'gsv_oie', 'gsv_runtime', 'cpp')],
        library_dirs=[],
        libraries=[],
        cxx_std=17,
        define_macros=[("VERSION_INFO", '"release"')],
        extra_compile_args=engine_compiler_args,
        extra_link_args=["/openmp" if IS_WINDOWS else "-fopenmp"],
    )

    tokenizer_compile_args = []
    if IS_WINDOWS:
        tokenizer_compile_args.extend(['/O2', '/Wall'])
    else:
        tokenizer_compile_args.extend(['-O3', '-Wall'])
    tokenizer_ext = Pybind11Extension(
        "gsv_oie.text_preprocess.tokenizers_cpp",
        sources=[
            "gsv_oie/text_preprocess/tokenizers.cpp",
        ],
        include_dirs=[pybind11.get_include()],
        library_dirs=[],
        libraries=[],
        cxx_std=17,
        define_macros=[("VERSION_INFO", '"release"')],
        extra_compile_args=tokenizer_compile_args,
        extra_link_args=[],
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

            self.copy_onnxruntime_to_build()
            self.copy_mnn_to_build()

            # 第二步：pybind11 扩展构建（依赖 CMake 输出）
            super().run()  # 这会按 ext_modules 顺序构建 pybind11 扩展，并链接 CMake 库

        def build_mnn(self):
            if IS_UNIX and not GSV_ANDROID_BUILD:
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
                '-DMNN_REDUCE_SIZE=ON',
                '-DMNN_LOW_MEMORY=ON',
                '-DMNN_CPU_WEIGHT_DEQUANT_GEMM=ON',
                '-DMNN_SEP_BUILD=OFF',
                '-S', self.mnn_src_dir,  # 源目录（项目根，含 CMakeLists.txt）
                '-B', self.mnn_build_dir  # 构建目录
            ]
            if GSV_ANDROID_BUILD:
                android_ndk = os.environ.get('ANDROID_NDK')
                if not android_ndk:
                    raise CompileError("ANDROID_NDK environment variable not set for Android build.")
                cmake_cmd += [
                    f'-DCMAKE_TOOLCHAIN_FILE={android_ndk}/build/cmake/android.toolchain.cmake',
                    '-DANDROID_ABI=arm64-v8a',
                    '-DANDROID_NATIVE_API_LEVEL=21',
                    '-DMNN_USE_SSE=OFF',
                    '-DANDROID_STL=c++_static',
                    '-DMNN_USE_LOGCAT=false',
                    '-DMNN_BUILD_FOR_ANDROID_COMMAND=true',
                    '-DMNN_OPENCL=ON',
                    '-DMNN_VULKAN=ON',
                    '-DMNN_OPENGL=ON',
                ]
            else:
                cmake_cmd += [
                    '-DMNN_USE_SSE=ON',
                    '-DMNN_AVX2=ON',
                    '-DMNN_AVX512=ON',
                ]
                if IS_WINDOWS:
                    cmake_cmd += [
                        '-DMNN_WIN_RUNTIME_MT=ON',
                        '-DMNN_OPENCL=ON',
                        '-DMNN_VULKAN=ON',
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


            self.mnn_dist_dir = os.path.join(self.mnn_src_dir, 'dist')
            os.makedirs(self.mnn_dist_dir, exist_ok=True)
            if IS_UNIX:
                for file in glob.glob(os.path.join(self.mnn_build_dir, '*MNN.so*')):
                    shutil.copy(file, self.mnn_dist_dir)
                for file in glob.glob(os.path.join(self.mnn_build_dir, 'express', '*MNN_Express.so*')):
                    shutil.copy(file, self.mnn_dist_dir)
            else:
                shutil.copy(os.path.join(self.mnn_build_dir,'Debug','MNN.dll'), self.mnn_dist_dir)

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
            if GSV_ANDROID_BUILD:
                android_ndk = os.environ.get('ANDROID_NDK')
                if not android_ndk:
                    raise CompileError("ANDROID_NDK environment variable not set for Android build.")
                cmake_cmd += [
                    f'-DCMAKE_TOOLCHAIN_FILE={android_ndk}/build/cmake/android.toolchain.cmake',
                    '-DANDROID_ABI=arm64-v8a',
                    '-DANDROID_NATIVE_API_LEVEL=21',
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

        def prebuild_onnxruntime(self):
            import requests
            import tarfile
            import tempfile
            import shutil

            # URL for ONNX Runtime Linux x64 package
            ort_version = "1.23.2"
            url = None
            if IS_WINDOWS:
                url = f"https://github.com/microsoft/onnxruntime/releases/download/v{ort_version}/onnxruntime-win-x64-{ort_version}.zip"
            else:
                url = f"https://github.com/microsoft/onnxruntime/releases/download/v{ort_version}/onnxruntime-linux-x64-{ort_version}.tgz"

            # Target directory in extern/onnxruntime
            self.onnxruntime_target_dir = os.path.join(os.path.dirname(__file__), 'extern', 'onnxruntime')
            self.onnxruntime_lib_dir = os.path.join(self.onnxruntime_target_dir, 'lib')
            self.onnxruntime_dist_dir = os.path.join(self.onnxruntime_target_dir, 'dist')

            # Remove existing directory if it exists
            if os.path.exists(self.onnxruntime_target_dir) and \
               (os.path.exists(os.path.join(self.onnxruntime_lib_dir, 'libonnxruntime.so')) or \
                os.path.exists(os.path.join(self.onnxruntime_lib_dir, 'onnxruntime.dll'))):
                return  # 已存在则跳过下载解压

            # Create parent directory if it doesn't exist
            os.makedirs(os.path.dirname(self.onnxruntime_target_dir), exist_ok=True)

            print(f"Downloading ONNX Runtime from {url}...")

            # Download with progress indication
            try:
                response = requests.get(url, stream=True)
                response.raise_for_status()

                # Create temporary file for download
                with tempfile.NamedTemporaryFile(suffix='.tgz' if IS_UNIX else '.zip', delete=False) as temp_file:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            temp_file.write(chunk)
                    temp_file_path = temp_file.name

                print(f"\nDownloaded ONNX Runtime to temporary file: {temp_file_path}")

            except requests.RequestException as e:
                raise CompileError(f"Failed to download ONNX Runtime: {e}")

            # Extract the tarball
            print("Extracting ONNX Runtime...")
            try:
                with tempfile.TemporaryDirectory() as temp_extract_dir:
                    # Extract to temporary directory first
                    if IS_UNIX:
                        with tarfile.open(temp_file_path, 'r:gz') as tar:
                            tar.extractall(temp_extract_dir)
                    else:
                        with zipfile.ZipFile(temp_file_path, 'r') as zip_ref:
                            zip_ref.extractall(temp_extract_dir)

                    # Find the extracted directory (should be 'onnxruntime-linux-x64-1.23.1')
                    extracted_dirs = [d for d in os.listdir(temp_extract_dir)
                                    if os.path.isdir(os.path.join(temp_extract_dir, d)) and d.startswith('onnxruntime')]

                    if not extracted_dirs:
                        raise CompileError("Could not find extracted ONNX Runtime directory")

                    extracted_dir = os.path.join(temp_extract_dir, extracted_dirs[0])

                    # Move to target location as 'onnxruntime'
                    shutil.move(extracted_dir, self.onnxruntime_target_dir)

                    os.makedirs(self.onnxruntime_dist_dir, exist_ok=True)
                    if IS_UNIX:
                        shutil.copy(os.path.join(self.onnxruntime_lib_dir, f'libonnxruntime.so.{ort_version}'), os.path.join(self.onnxruntime_dist_dir, 'libonnxruntime.so'))
                    else:
                        shutil.copy(os.path.join(self.onnxruntime_lib_dir, 'onnxruntime.dll'), self.onnxruntime_dist_dir)

                print(f"ONNX Runtime extracted to: {self.onnxruntime_target_dir}")

            except (tarfile.TarError, OSError, shutil.Error) as e:
                raise CompileError(f"Failed to extract ONNX Runtime: {e}")

            finally:
                # Clean up temporary file
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)

        def build_extension(self, ext):
            if ext.name == "gsv_oie.gsv_runtime.gsv_engine":
                ext.library_dirs.extend([
                    self.mnn_dist_dir if IS_UNIX else os.path.join(self.mnn_build_dir, 'Debug'),
                    self.onnxruntime_lib_dir
                    ])

                ext.include_dirs.extend([
                    os.path.join(self.mnn_src_dir, 'include'),
                    os.path.join(self.onnxruntime_target_dir, 'include'),
                    os.path.join(os.path.dirname(__file__), 'extern', 'fp16', 'include'),
                ])

                ext.libraries.extend(['MNN', 'onnxruntime'])

            elif ext.name == "gsv_oie.text_preprocess.tokenizers_cpp":
                ext.library_dirs.extend([self.tokenizers_cpp_build_dir])
                if IS_WINDOWS:
                    ext.library_dirs.append(os.path.join(self.tokenizers_cpp_build_dir, 'Debug'))

                ext.include_dirs.extend([
                    os.path.join(self.tokenizers_cpp_src_dir, 'include'),
                ])

                ext.libraries.extend(['tokenizers_c','tokenizers_cpp'])
                if IS_WINDOWS:
                    ext.libraries.extend(['ntdll', 'ws2_32', 'wsock32', 'Bcrypt', 'userenv', 'iphlpapi', 'psapi'])

            super().build_extension(ext)

        def copy_onnxruntime_to_build(self):
            """Copy onnxruntime files to the build directory"""
            if not os.path.exists(self.onnxruntime_dist_dir):
                print(f"Warning: {self.onnxruntime_dist_dir} does not exist")
                return

            # Copy to the build directory
            onnxruntime_target = os.path.join(self.build_lib, 'extern', 'onnxruntime', 'dist')

            # Create target directory
            os.makedirs(os.path.dirname(onnxruntime_target), exist_ok=True)

            # Remove existing target if it exists
            if os.path.exists(onnxruntime_target):
                shutil.rmtree(onnxruntime_target)

            # Copy all files and directories
            shutil.copytree(self.onnxruntime_dist_dir, onnxruntime_target)

            print(f"Copied ONNX Runtime to {onnxruntime_target}")
        
        def copy_mnn_to_build(self):
            """Copy MNN files to the build directory"""
            if not os.path.exists(self.mnn_dist_dir):
                print(f"Warning: {self.mnn_dist_dir} does not exist")
                return

            # Copy to the build directory
            mnn_target = os.path.join(self.build_lib, 'extern', 'MNN', 'dist')

            # Create target directory
            os.makedirs(os.path.dirname(mnn_target), exist_ok=True)

            # Remove existing target if it exists
            if os.path.exists(mnn_target):
                shutil.rmtree(mnn_target)

            # Copy all files and directories
            shutil.copytree(self.mnn_dist_dir, mnn_target)

            print(f"Copied MNN to {mnn_target}")

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
    package_data={},

    # Python version requirement
    python_requires='>=3.7',

    setup_requires=[
        "cmake==3.29.2",  # Minimum version from your CMakeLists.txt
        "requests>=2.0.0",
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
    cmdclass={
        "build_ext": get_build_ext()
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