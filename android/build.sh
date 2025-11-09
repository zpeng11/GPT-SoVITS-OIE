#!/bin/bash

set -e  # 退出 on error

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate py310

cp -r /app/gsv_oie_android/android/python-mecab-ko /app/chaquopy/server/pypi/packages
cp -r /app/gsv_oie_android/android/pyopenjtalk /app/chaquopy/server/pypi/packages
cp -r /app/gsv_oie_android/android/fasttext-predict /app/chaquopy/server/pypi/packages
cp -r /app/gsv_oie_android/android/bsdiff4 /app/chaquopy/server/pypi/packages
cp -r /app/gsv_oie_android/android/jieba-fast /app/chaquopy/server/pypi/packages
cp -r /app/gsv_oie_android/android/split-lang /app/chaquopy/server/pypi/packages
cp -r /app/gsv_oie_android/android/inflect /app/chaquopy/server/pypi/packages
cp -r /app/gsv_oie_android/android/gsv_oie /app/chaquopy/server/pypi/packages/gsv-oie

cd /tmp/mecab-0.996-ko-0.9.2
chmod +x config.sub config.guess
(
    export TOOLCHAIN=$ANDROID_NDK/toolchains/llvm/prebuilt/linux-x86_64
    export API_LEVEL=21
    export TARGET=aarch64-linux-android
    export AR=$TOOLCHAIN/bin/llvm-ar
    export CC=$TOOLCHAIN/bin/$TARGET$API_LEVEL-clang
    export CXX=$TOOLCHAIN/bin/$TARGET$API_LEVEL-clang++
    export LD=$TOOLCHAIN/bin/ld
    export RANLIB=$TOOLCHAIN/bin/llvm-ranlib
    export STRIP=$TOOLCHAIN/bin/llvm-strip
    export AM_CXXFLAGS="-std=c++14 -fPIC"
    export AM_CFLAGS="-fPIC"

    # Configure
    ./configure \
        CXXFLAGS="-std=c++14 -fPIC" \
        CFLAGS="-fPIC" \
        --host=$TARGET \
        --prefix=/tmp/mecab-0.996-ko-0.9.2/build \
        --enable-static \
        --disable-shared \
        --with-charset=utf8
    make clean
    make -j$(nproc)
    make install
)
export PATH=/tmp/mecab-0.996-ko-0.9.2/build/bin:$PATH

cd /app/chaquopy/server/pypi
conda activate py310

./build-wheel.py --python 3.10 --abi arm64-v8a --api-level 21 python-mecab-ko || true # Ignore errors here to release files
cp /tmp/mecab-0.996-ko-0.9.2/build/include/mecab.h \
    /app/chaquopy/server/pypi/packages/python-mecab-ko/build/1.3.7/cp310-cp310-android_21_arm64_v8a/env/include
cp /tmp/mecab-0.996-ko-0.9.2/build/lib/libmecab.* \
    /app/chaquopy/server/pypi/packages/python-mecab-ko/build/1.3.7/cp310-cp310-android_21_arm64_v8a/requirements/chaquopy/lib
./build-wheel.py --python 3.10 --abi arm64-v8a --api-level 21 python-mecab-ko --no-unpack
./build-wheel.py --python 3.10 --abi arm64-v8a --api-level 21 pyopenjtalk
./build-wheel.py --python 3.10 --abi arm64-v8a --api-level 21 fasttext-predict
./build-wheel.py --python 3.10 --abi arm64-v8a --api-level 21 bsdiff4
./build-wheel.py --python 3.10 --abi arm64-v8a --api-level 21 jieba-fast
./build-wheel.py --python 3.10 --abi arm64-v8a --api-level 21 split-lang
./build-wheel.py --python 3.10 --abi arm64-v8a --api-level 21 inflect
./build-wheel.py --python 3.10 --abi arm64-v8a --api-level 21 gsv-oie

mkdir -p /app/gsv_oie_android/android/dist
find /app/chaquopy/server/pypi/dist -name "*.whl" -exec cp {} /app/gsv_oie_android/android/dist/ \;
chmod -R a+r+w /app/gsv_oie_android/android/dist

cd /app/gsv_oie_android/android/chaquopy-console
./gradlew assembleDebug && cp app/build/outputs/apk/debug/app-debug.apk /app/gsv_oie_android/build
chmod -R a+r+w /app/gsv_oie_android/android/chaquopy-console