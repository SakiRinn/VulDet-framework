from gensim.models import Word2Vec
import json


def train_w2v(json_paths, embedding_size=64, epochs=5, output_path='outputs/w2v/w2v.bin'):
    sentences = []
    for path in json_paths:
        with open(path, 'r') as f:
            data = json.load(f)
            for entry in data:
                code = entry['code']    # EDIT
                sentences.append([token.strip() for token in code.split()])

    print(f'Total sentences: {len(sentences)}')
    print('Training...')

    w2v_model = Word2Vec(sentences, vector_size=embedding_size, min_count=1, workers=8)
    # for i in range(epochs):
    #     w2v_model.train(sentences, total_examples=len(sentences), epochs=1)
    w2v_model.train(sentences, total_examples=len(sentences), epochs=epochs)
    w2v_model.save(output_path)


if __name__ == '__main__':
    train_w2v([''])
