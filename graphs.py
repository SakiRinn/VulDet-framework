import argparse
import json
import os
import os.path as osp

from gensim.models import Word2Vec
import numpy as np
import pandas as pd

from utils.graph import df2csv, csv2graph


def generate_graphs(csv_dir, raw_data, w2v_model):
    graphs = []
    for i, entry in enumerate(raw_data):
        filename = f'{i}.c'
        node_csv = osp.join(csv_dir, filename, 'nodes.csv')
        edge_csv = osp.join(csv_dir, filename, 'edges.csv')
        label = int(entry['target'])
        if not osp.exists(node_csv) or not osp.exists(edge_csv):
            continue
        graph = csv2graph(node_csv, edge_csv, w2v_model)    # main
        graph.update({'index': i, 'label': label})
        graphs.append(graph)
    return graphs

def save_graphs(graphs, save_dir, filename='graphs'):
    node_features = []
    for i, graph in enumerate(graphs):
        node_features.append(graph['nodes'])
        graphs[i]['nodes'] = f'arr_{i}'
    with open(osp.join(save_dir, f'{filename}.json'), 'w') as f:
        json.dump(graphs, f, indent=4)
    with open(osp.join(save_dir, f'{filename}_nodes.npz'), 'wb') as f:
        np.savez(f, *node_features)

def load_graphs(load_dir, filename='graphs'):
    with open(osp.join(load_dir, f'{filename}.json'), 'r') as f:
            graphs = json.load(f)
    with open(osp.join(load_dir, f'{filename}_nodes.npz'), 'rb') as f:
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

    with open(args.data_path, 'r') as f:
        raw_data = json.load(f)
        df_data = pd.DataFrame.from_records(raw_data)
    w2v_model = Word2Vec.load(args.w2v)

    print("Processing...")

    if not osp.exists(args.output_dir):
        os.mkdir(args.output_dir)

    csv_dir = osp.join(args.output_dir, 'csv')
    df2csv(df_data, csv_dir)
    graphs = generate_graphs(csv_dir, raw_data, w2v_model)

    data_name = osp.split(args.data_path)[-1].split('.')[0]
    save_graphs(graphs, args.output_dir, data_name)

    print('Completed!')


if __name__ == '__main__':
    main()
