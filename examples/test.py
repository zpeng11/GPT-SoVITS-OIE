from text_preprocess.text_preprocessor import TextPreprocessor
processor = TextPreprocessor()

phones, bert_features, norm_text = processor.get_phones_and_bert("Mygo?まいご？你喜欢卖狗嘛。","auto","v2")

print(phones.shape, bert_features.shape, norm_text)