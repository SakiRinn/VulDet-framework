import logging
from gensim.models import Word2Vec
import json
import re
import os.path as osp


def remove_commit(code):
    pattern = re.compile(r'(/\*([^*]|(\*+[^*/]))*\*+/)|(//.*)')
    code = re.sub(pattern, '', code)
    return code

def to_camelcase(name):
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

def code_tokenize(code):
    # Remove code comments
    code = remove_commit(code)
    # Remove newlines & tabs
    code = re.sub('(\n)|(\\\\n)|(\\\\)|(\\t)|(/)|(\\r)', '', code)
    # Mix split (characters and words)
    splitter = '\"(.*?)\"| +|(;)|(->)|(&)|(\\*)|(\\()|(==)|(~)|(!=)|(<=)|(>=)|(!)|(\\+\\+)|(--)|(\\))|(=)|(\\+)|(\\-)|(\\[)|(\\])|(<)|(>)|(\\.)|({)'
    code = re.split(splitter, code)
    # Remove None type
    code = [item.strip() for item in code if item is not None]
    # snakecase -> camelcase and split camelcase
    code = [to_camelcase(n).split('_') for n in code]
    # filter
    code = [n for n in code if n not in ['{', '}', ';', ':']]
    return code


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
