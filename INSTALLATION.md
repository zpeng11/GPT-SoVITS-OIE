# Installation Guide

## Standard Installation (with MNN support)

```bash
pip install gsv_oie
```

This installs all core dependencies and prepares the package for MNN usage.

## Mobile Installation (without MNN dependencies)

```bash
pip install gsv_oie[mobile]
```

This installs only the core dependencies, excluding MNN. Use this for:
- Chaquopy Android development
- iOS Python environments
- When you plan to implement your own MNN functions

## Desktop/Server Installation (with MNN)

```bash
pip install gsv_oie[desktop]
```

This installs core dependencies plus MNN support when available.

## Combined Installation

You can combine extras for specific use cases:

```bash
# For mobile with Chinese language support
pip install gsv_oie[mobile,chinese]

# For desktop with all language support
pip install gsv_oie[desktop,all]

# For development
pip install gsv_oie[mobile,dev]
```

## Mobile Implementation

When using the `[mobile]` extra, you'll need to implement your own MNN prediction functions:

```python
import gsv_oie

def my_mobile_g2pw_predictor(model_path: str, inputs):
    # Your mobile-specific MNN implementation
    # This could use JNI calls to native MNN libraries
    pass

def my_mobile_roberta_predictor(model_path: str, inputs):
    # Your mobile-specific RoBERTa implementation
    pass

# Register your custom functions
gsv_oie.set_g2pw_predict(my_mobile_g2pw_predictor)
gsv_oie.set_roberta_predict(my_mobile_roberta_predictor)

# Use the text processor
processor = gsv_oie.TextPreprocessor()
result = processor.preprocess("Hello world", "en", "cut0")
```