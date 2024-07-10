import argparse
import json
import logging
import os
import os.path as osp
from gensim.models import Word2Vec


def train_w2v(sentences, epochs=5, min_count=1, embedding_size=128,
              output_dir='outputs/w2v/'):
    words = []
    for sentence in sentences:
        words.append(code_tokenize(sentence))

    print(f'Total words: {len(words)}')
    print('Training...')

    if not osp.exists(output_dir):
        os.makedirs(output_dir)

    w2v_model = Word2Vec(words, vector_size=embedding_size, min_count=min_count, workers=8)
    for i in range(epochs):
        w2v_model.train(words, total_examples=len(words), epochs=1)
        w2v_model.save(osp.join(output_dir, f'e{i+1}.bin'))

    print('Completed!')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', required=True, help='The json file of raw data.')
    parser.add_argument('--data-tag', default='func', help='The item in entry to store code.')
    parser.add_argument('--output-dir', help='The directory to the output w2v models.', default='outputs/w2v/')
    args = parser.parse_args()

    with open(args.data_path, 'r') as f:
        data = json.load(f)
        sentences = [e[args.data_tag] for e in data]
    train_w2v(sentences, output_dir=args.output_dir)


if __name__ == '__main__':
    main()
