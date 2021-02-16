import tensorflow as tf
import io
import re
import unicodedata

import jieba


def unicode_to_ascii(s):
    s = s.replace('。', '.')
    s = s.replace('？', '?')
    s = s.replace('！', '!')
    s = s.replace('，', ',')
    s = s.replace('’', '\'')
    s = s.replace('“', '\'')
    s = s.replace('《', '\'')
    s = s.replace('《', '\'')
    s = s.replace('、', ',')
    s = s.replace('：', ':')
    return unicodedata.normalize('NFKC', s)


def preprocess_sentence(w):
    w = w.lower().strip()
    w = unicode_to_ascii(w)
    # 在单词与跟在其后的标点符号之间插入一个空格
    # 例如： "he is a boy." => "he is a boy ."
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)
    # 除了 (a-z, A-Z, ".", "?", "!", ",")，将所有字符替换为空格
    # w = re.sub(r"[\\\t]+", " ", w)
    w = w.rstrip().strip()
    # 给句子加上开始和结束标记
    # 以便模型知道何时开始和结束预测
    #w = '<start> ' + w + ' <end>'
    return w

def create_dataset(path, num_examples=50000):
    lines = io.open(path, encoding='UTF-8').read().strip().split('\n')
    word_pairs = [[sentence for sentence in line.split('\t')][0:2] for line in
                  lines[:num_examples]]
    en, sp = zip(*word_pairs)
    en = ['<start> ' + preprocess_sentence(sentence) + ' <end>' for sentence in en]
    sp = ['<start> ' + preprocess_sentence(' '.join(list(jieba.cut(sentence)))) + ' <end>' for sentence in sp]
    return en, sp


def tokenize(lang):
    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(
      filters='')
    lang_tokenizer.fit_on_texts(lang)
    tensor = lang_tokenizer.texts_to_sequences(lang)
    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor,
                                                         padding='post')
    return tensor, lang_tokenizer

def load_dataset(path, num_examples=None):
    # 创建清理过的输入输出对
    targ_lang, inp_lang = create_dataset(path, num_examples)
    input_tensor, inp_lang_tokenizer = tokenize(inp_lang)
    target_tensor, targ_lang_tokenizer = tokenize(targ_lang)

    return input_tensor, target_tensor, inp_lang_tokenizer, targ_lang_tokenizer

def main():
    en, sp = create_dataset('cmn-eng/cmn.txt')
    print(en[-1])
    print(sp[-1])


if __name__ == '__main__':
    main()
