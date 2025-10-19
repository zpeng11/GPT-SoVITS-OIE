#!/bin/bash

set -e  # 退出 on error

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate py310

cp -r /app/gsv_oie_android/android/pyopenjtalk /app/chaquopy/server/pypi/packages/pyopenjtalk
cp -r /app/gsv_oie_android/android/fasttext-predict /app/chaquopy/server/pypi/packages/fasttext-predict

cd /app/chaquopy/server/pypi
# conda activate py39
# ./build-wheel.py --python 3.9 --abi armeabi-v7a pyopenjtalk
# ./build-wheel.py --python 3.9 --abi armeabi-v7a fasttext-predict
# ./build-wheel.py --python 3.9 --abi arm64-v8a pyopenjtalk
# ./build-wheel.py --python 3.9 --abi arm64-v8a fasttext-predict
# ./build-wheel.py --python 3.9 --abi x86 pyopenjtalk
# ./build-wheel.py --python 3.9 --abi x86 fasttext-predict
# ./build-wheel.py --python 3.9 --abi x86_64 pyopenjtalk
# ./build-wheel.py --python 3.9 --abi x86_64 fasttext-predict
conda activate py310
# ./build-wheel.py --python 3.10 --abi armeabi-v7a pyopenjtalk
# ./build-wheel.py --python 3.10 --abi armeabi-v7a fasttext-predict
./build-wheel.py --python 3.10 --abi arm64-v8a pyopenjtalk
./build-wheel.py --python 3.10 --abi arm64-v8a fasttext-predict
# ./build-wheel.py --python 3.10 --abi x86 pyopenjtalk
# ./build-wheel.py --python 3.10 --abi x86 fasttext-predict
# ./build-wheel.py --python 3.10 --abi x86_64 pyopenjtalk
# ./build-wheel.py --python 3.10 --abi x86_64 fasttext-predict
# conda activate py311
# ./build-wheel.py --python 3.11 --abi armeabi-v7a pyopenjtalk
# ./build-wheel.py --python 3.11 --abi armeabi-v7a fasttext-predict
# ./build-wheel.py --python 3.11 --abi arm64-v8a pyopenjtalk
# ./build-wheel.py --python 3.11 --abi arm64-v8a fasttext-predict
# ./build-wheel.py --python 3.11 --abi x86 pyopenjtalk
# ./build-wheel.py --python 3.11 --abi x86 fasttext-predict
# ./build-wheel.py --python 3.11 --abi x86_64 pyopenjtalk
# ./build-wheel.py --python 3.11 --abi x86_64 fasttext-predict

mkdir -p /app/gsv_oie_android/android/build
cp /app/chaquopy/server/pypi/dist/pyopenjtalk/* /app/gsv_oie_android/android/build/
cp /app/chaquopy/server/pypi/dist/fasttext-predict/* /app/gsv_oie_android/android/build/
chmod -R a+r+w /app/gsv_oie_android/android/build