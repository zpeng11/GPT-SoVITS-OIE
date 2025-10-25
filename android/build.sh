#!/bin/bash

set -e  # 退出 on error

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate py310

cp -r /app/gsv_oie_android/android/pyopenjtalk /app/chaquopy/server/pypi/packages
cp -r /app/gsv_oie_android/android/fasttext-predict /app/chaquopy/server/pypi/packages
cp -r /app/gsv_oie_android/android/bsdiff4 /app/chaquopy/server/pypi/packages
cp -r /app/gsv_oie_android/android/soundfile /app/chaquopy/server/pypi/packages
cp -r /app/gsv_oie_android/android/jieba-fast /app/chaquopy/server/pypi/packages

cd /app/chaquopy/server/pypi
conda activate py310

./build-wheel.py --python 3.10 --abi arm64-v8a --api-level 21 pyopenjtalk
./build-wheel.py --python 3.10 --abi arm64-v8a --api-level 21 fasttext-predict
./build-wheel.py --python 3.10 --abi arm64-v8a --api-level 21 bsdiff4
./build-wheel.py --python 3.10 --abi arm64-v8a --api-level 21 numpy
./build-wheel.py --python 3.10 --abi arm64-v8a --api-level 21 soundfile
./build-wheel.py --python 3.10 --abi arm64-v8a --api-level 21 jieba-fast

mkdir -p /app/gsv_oie_android/android/dist
find /app/chaquopy/server/pypi/dist -name "*.whl" -exec cp {} /app/gsv_oie_android/android/dist/ \;
chmod -R a+r+w /app/gsv_oie_android/android/dist