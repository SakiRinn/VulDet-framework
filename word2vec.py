import json
import logging
import os
import os.path as osp
from gensim.models import Word2Vec

from tools import code_tokenize


def train_w2v(sentences, epochs=5, min_count=1, embedding_size=128,
              output_dir='outputs/w2v/'):
    words = []
    for sentence in sentences:
        words += code_tokenize(sentence)

    logging.info(f'Total words: {len(words)}')
    logging.info('Training...')

    if not osp.exists(output_dir):
        os.makedirs(output_dir)

    w2v_model = Word2Vec(words, vector_size=embedding_size, min_count=min_count, workers=8)
    for i in range(epochs):
        w2v_model.train(words, total_examples=len(words), epochs=1)
        w2v_model.save(osp.join(output_dir, f'e{i+1}.bin'))

    logging.info('Train completed!')


if __name__ == '__main__':
    with open('data/devign/devign.json', 'r') as f:
        data = json.load(f)
        sentences = [e['func'] for e in data]
    train_w2v(sentences, output_dir='outputs/w2v/')
