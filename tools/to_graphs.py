import os
import sys
import argparse
import json

from gensim.models import Word2Vec
import numpy as np

root_dir = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
os.chdir(root_dir)
sys.path.append(root_dir)
from dataloaders.graphs import json_to_graphs


def save_graphs(graphs, save_dir, filename='graphs'):
    node_features = []
    for i, graph in enumerate(graphs):
        node_features.append(graph['nodes'])
        graphs[i]['nodes'] = f'arr_{i}'
    with open(os.path.join(save_dir, f'{filename}.json'), 'w') as f:
        json.dump(graphs, f, indent=4)
    with open(os.path.join(save_dir, f'{filename}_nodes.npz'), 'wb') as f:
        np.savez(f, *node_features)

def load_graphs(load_dir, filename='graphs'):
    with open(os.path.join(load_dir, f'{filename}.json'), 'r') as f:
            graphs = json.load(f)
    with open(os.path.join(load_dir, f'{filename}_nodes.npz'), 'rb') as f:
        node_features = np.load(f)
        for i, graph in enumerate(graphs):
            arr_idx = graph['nodes']
            graphs[i]['nodes'] = node_features[arr_idx]
    return graphs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', required=True, help='The .json file of raw data.')
    parser.add_argument('--w2v', required=True, help='The .bin file of Word2vec model.')
    parser.add_argument('--output-dir', help='The directory to the output all files.', default='outputs/graph/')
    args = parser.parse_args()

    w2v_model = Word2Vec.load(args.w2v)
    print("Success to load w2v model, start processing...")

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    graphs = json_to_graphs(w2v_model, args.data_path, args.output_dir)

    data_name = os.path.split(args.data_path)[-1].split('.')[0]
    save_graphs(graphs, args.output_dir, data_name)
    print('Completed!')


if __name__ == '__main__':
    main()
