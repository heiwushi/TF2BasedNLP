import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing

import io
import re
import unicodedata
import pathlib
import tensorflow_text as tf_text

import jieba


def load_data(path):
    with open(path, encoding='utf-8') as f:
        lines = f.readlines()
        pairs = [line.split('\t')[0:2] for line in lines]
        english = [preprocess_sentence(e) for e, c in pairs]
        chinese = [preprocess_sentence(" ".join(list(jieba.cut(c)))) for e, c in pairs]
    return english, chinese


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
    #return unicodedata.normalize('NFKC', s)
    return s

def preprocess_sentence(w):
    w = w.lower().strip()
    w = unicode_to_ascii(w)
    # 在单词与标点符号之间插入一个空格
    w = re.sub(r"([.?!,:'])", r" \1 ", w)
    # 除了 (a-z, 汉字, ".", "?", "!", ",")，将所有字符替换为空格
    w = re.sub(r"([^ a-z.?!,:\u4e00-\u9fa5])", r" ", w)
    # 将多个空格或\t变成一个空格
    w = re.sub(r'[" "]+', " ", w)
    w = re.sub(r"[\\\t]+", " ", w)
    w = w.rstrip().strip()
    # 给句子加上开始和结束标记以便模型知道何时开始和结束预测
    w = '[start] ' + w + ' [end]'
    return w


def main():
    english, chinese = load_data('cmn-eng/cmn.txt')
    print(english[0:5])
    print(chinese[0:5])
    english_processor = preprocessing.TextVectorization(max_tokens=5000)
    chinese_processor = preprocessing.TextVectorization(max_tokens=5000)
    english_processor.adapt(english)
    chinese_processor.adapt(chinese)
    print(english_processor.get_vocabulary()[:10])
    print(chinese_processor.get_vocabulary()[:10])



    # BUFFER_SIZE = len(english)
    # BATCH_SIZE = 2
    # dataset = tf.data.Dataset.from_tensor_slices((source, target)).shuffle(BUFFER_SIZE)
    # dataset = dataset.batch(BATCH_SIZE)
    # #print(dataset)
    # for example_source_batch, example_target_batch in dataset.take(1):
    #   print(example_source_batch)
    #   print(example_target_batch)



    #
    # input_text_processor = preprocessing.TextVectorization(
    #     standardize=tf_lower_and_split_punct,
    #     max_tokens=max_vocab_size)


if __name__ == '__main__':
    main()

    print(re.sub(r"([?.!,¿:'])", r" \1 ", "哈哈!嗯嗯."))
    print(re.sub(r'[" "]+', " ", "哈哈      嗯嗯"))

