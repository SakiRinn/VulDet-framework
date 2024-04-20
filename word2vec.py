import json
import logging
import os.path as osp
from gensim.models import Word2Vec

from tools import code_tokenize


def train_w2v(sentences, epochs=5, min_count=1, embedding_size=128,
              output_dir='outputs/w2v/'):
    sentences = [code_tokenize(sentence) for sentence in sentences]

    logging.info(f'Total sentences: {len(sentences)}')
    logging.info('Training...')

    w2v_model = Word2Vec(sentences, vector_size=embedding_size, min_count=min_count, workers=8)
    for i in range(epochs):
        w2v_model.train(sentences, total_examples=len(sentences), epochs=1)
        w2v_model.save(osp.join(output_dir, 'e1.bin'))

    logging.info('Train completed!')


if __name__ == '__main__':
    with open('data/devign/devign.json', 'r') as f:
        data = json.load(f)
        sentences = [e['func'] for e in data]
    train_w2v(sentences, output_dir='outputs/w2v/devign.bin')
