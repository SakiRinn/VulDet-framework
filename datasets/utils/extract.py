import csv
import json
import os
from tokenize import tokenize
from tqdm import tqdm

from datasets.utils.graph import (combine_adjacents, create_adjacency_list, create_backward_slice,
                                  create_forward_slice, reformat_code_line_graph)
from datasets.utils.read import read_csv, read_code, read_text


def extract_line_number(idx, nodes):
    while idx >= 0:
        c_node = nodes[idx]
        if 'location' in c_node.keys():
            location = c_node['location']
            if location.strip() != '':
                try:
                    ln = int(location.split(':')[0])
                    return ln
                except:
                    pass
        idx -= 1
    return -1

def extract_slices():
    all_data = []
    ggnn_json_data = json.load(open('../data/ggnn_input/devign_cfg_full_text_files.json'))
    files = [d['file_name'] for d in ggnn_json_data]
    print(len(files))

    for i, file_name  in enumerate(files):
        label = file_name.strip()[:-2].split('_')[-1]
        code_text = read_text(split_dir + file_name.strip())

        nodes_file_path = parsed + file_name.strip() + '/nodes.csv'
        edges_file_path = parsed + file_name.strip() + '/edges.csv'
        nc = open(nodes_file_path)
        nodes_file = csv.DictReader(nc, delimiter='\t')
        nodes = [node for node in nodes_file]
        call_lines = set()
        array_lines = set()
        ptr_lines = set()
        arithmatic_lines = set()

        if len(nodes) == 0:
            continue

        for node_idx, node in enumerate(nodes):
            ntype = node['type'].strip()
            if ntype == 'CallExpression':
                function_name = nodes[node_idx + 1]['code']
                if function_name  is None or function_name.strip() == '':
                    continue
                if function_name.strip() in l_funcs:
                    line_no = extract_line_number(node_idx, nodes)
                    if line_no > 0:
                        call_lines.add(line_no)
            elif ntype == 'ArrayIndexing':
                line_no = extract_line_number(node_idx, nodes)
                if line_no > 0:
                    array_lines.add(line_no)
            elif ntype == 'PtrMemberAccess':
                line_no = extract_line_number(node_idx, nodes)
                if line_no > 0:
                    ptr_lines.add(line_no)
            elif node['operator'].strip() in ['+', '-', '*', '/']:
                line_no = extract_line_number(node_idx, nodes)
                if line_no > 0:
                    arithmatic_lines.add(line_no)

        nodes = read_csv(nodes_file_path)
        edges = read_csv(edges_file_path)
        node_indices, node_ids, line_numbers, node_id_to_ln = extract_nodes_with_location_info(nodes)
        adjacency_list = create_adjacency_list(line_numbers, node_id_to_ln, edges, False)
        combined_graph = combine_adjacents(adjacency_list)

        array_slices = []
        array_slices_bdir = []
        call_slices = []
        call_slices_bdir = []
        arith_slices = []
        arith_slices_bdir = []
        ptr_slices = []
        ptr_slices_bdir = []
        all_slices = []


        all_keys = set()
        _keys = set()
        for slice_ln in call_lines:
            forward_sliced_lines = create_forward_slice(combined_graph, slice_ln)
            backward_sliced_lines = create_backward_slice(combined_graph, slice_ln)
            all_slice_lines = forward_sliced_lines
            all_slice_lines.extend(backward_sliced_lines)
            all_slice_lines = sorted(list(set(all_slice_lines)))
            key = ' '.join([str(i) for i in all_slice_lines])
            if key not in _keys:
                call_slices.append(backward_sliced_lines)
                call_slices_bdir.append(all_slice_lines)
                _keys.add(key)
            if key not in all_keys:
                all_slices.append(all_slice_lines)
                all_keys.add(key)

        _keys = set()
        for slice_ln in array_lines:
            forward_sliced_lines = create_forward_slice(combined_graph, slice_ln)
            backward_sliced_lines = create_backward_slice(combined_graph, slice_ln)
            all_slice_lines = forward_sliced_lines
            all_slice_lines.extend(backward_sliced_lines)
            all_slice_lines = sorted(list(set(all_slice_lines)))
            key = ' '.join([str(i) for i in all_slice_lines])
            if key not in _keys:
                array_slices.append(backward_sliced_lines)
                array_slices_bdir.append(all_slice_lines)
                _keys.add(key)
            if key not in all_keys:
                all_slices.append(all_slice_lines)
                all_keys.add(key)

        _keys = set()
        for slice_ln in arithmatic_lines:
            forward_sliced_lines = create_forward_slice(combined_graph, slice_ln)
            backward_sliced_lines = create_backward_slice(combined_graph, slice_ln)
            all_slice_lines = forward_sliced_lines
            all_slice_lines.extend(backward_sliced_lines)
            all_slice_lines = sorted(list(set(all_slice_lines)))
            key = ' '.join([str(i) for i in all_slice_lines])
            if key not in _keys:
                arith_slices.append(backward_sliced_lines)
                arith_slices_bdir.append(all_slice_lines)
                _keys.add(key)
            if key not in all_keys:
                all_slices.append(all_slice_lines)
                all_keys.add(key)

        _keys = set()
        for slice_ln in ptr_lines:
            forward_sliced_lines = create_forward_slice(combined_graph, slice_ln)
            backward_sliced_lines = create_backward_slice(combined_graph, slice_ln)
            all_slice_lines = forward_sliced_lines
            all_slice_lines.extend(backward_sliced_lines)
            all_slice_lines = sorted(list(set(all_slice_lines)))
            key = ' '.join([str(i) for i in all_slice_lines])
            if key not in _keys:
                ptr_slices.append(backward_sliced_lines)
                ptr_slices_bdir.append(all_slice_lines)
                _keys.add(key)
            if key not in all_keys:
                all_slices.append(all_slice_lines)
                all_keys.add(key)

        t_code = tokenize(code_text)
        if t_code is None:
            continue
        data_instance = {
            'file_path': split_dir + file_name.strip(),
            'code' : code_text,
            'tokenized': t_code,
            'call_slices_vd': call_slices,
            'call_slices_sy': call_slices_bdir,
            'array_slices_vd': array_slices,
            'array_slices_sy': array_slices_bdir,
            'arith_slices_vd': arith_slices,
            'arith_slices_sy': arith_slices_bdir,
            'ptr_slices_vd': ptr_slices,
            'ptr_slices_sy': ptr_slices_bdir,
            'label': int(label)
        }
        all_data.append(data_instance)

        if i % 1000 == 0:
            print(i, len(call_slices), len(call_slices_bdir),
                len(array_slices), len(array_slices_bdir),
                len(arith_slices), len(arith_slices_bdir), sep='\t')

def extract_nodes_with_location_info(nodes: dict):
    # Will return an array identifying the indices of those nodes in nodes array,
    # another array identifying the node_id of those nodes
    # another array indicating the line numbers
    # all 3 return arrays should have same length indicating 1-to-1 matching.
    node_indices = []
    node_ids = []
    line_numbers = []
    node_id_to_line_number = {}
    for i, node in enumerate(nodes):
        if 'location' in node.keys():
            location = node['location']
            if location == '':
                continue
            node_indices.append(i)
            node_id = node['key'].strip()
            node_ids.append(node_id)
            line_number = int(location.split(':')[0])
            line_numbers.append(line_number)
            node_id_to_line_number[node_id] = line_number
    return node_indices, node_ids, line_numbers, node_id_to_line_number

def extract_graph_data(dataset_type, portion='full_graph',
                       input_dir='../../data/full_experiment_real_data/',
                       output_dir='../../data/full_experiment_real_data_processed/'):
    assert portion in ['full_graph', 'cgraph', 'dgraph', 'cdgraph']
    shards = os.listdir(os.path.join(input_dir, dataset_type))
    shard_count = len(shards)
    total_functions, in_scope_function = set(), set()
    vnt, nvnt = 0, 0
    graphs = []
    for sc in range(1, shard_count + 1):
        with open(os.path.join(input_dir, dataset_type, dataset_type + '.json.shard' + str(sc))) as sf:
            shard_data = json.load(sf)
            for data in tqdm(shard_data):
                id = data['id']
                total_functions.add(id)
                code_graph = data[portion]
                if data[portion] is not None:
                    code_graph['id'] = id
                    code_graph['code'] = data['code']
                    code_graph['filename'] = data['file_name']
                    code_graph['path'] = data['path']
                    graphs.append(code_graph)
                    in_scope_function.add(id)
                else:
                    label = int(data['label'])
                    if label == 1:
                        vnt += 1
                    else:
                        nvnt += 1
    # Save
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    with open(os.path.join(output_dir, f'{dataset_type}-{portion}.json'), 'w') as of:
        json.dump(graphs, of)

def extract_line_graph_data(dataset_type,
                            input_dir='../../data/full_experiment_real_data/',
                            output_dir='../../data/full_experiment_real_data_processed/'):
    if dataset_type == 'devign':
        split_dir = '../data/neurips_parsed/neurips_data/'
        parsed = '../data/neurips_parsed/parsed_results/'
        wv_path = '../data/neurips_parsed/raw_code_neurips.100'
        wv_model_original = Word2Vec.load(wv_path)
    else:
        split_dir = '../data/chrome_debian/raw_code/'
        parsed = '../data/chrome_debian/parsed/'
        wv_path = '../data/chrome_debian/raw_code_deb_chro.100'
        wv_model_original = Word2Vec.load(wv_path)
    shards = os.listdir(os.path.join(input_dir, dataset_type))
    shard_count = len(shards)
    total_functions, in_scope_function = set(), set()
    vnt, nvnt = 0, 0
    graphs = []
    for sc in range(1, shard_count + 1):
        shard_file = open(os.path.join(input_dir, dataset_type, dataset_type + '.json.shard' + str(sc)))
        shard_data = json.load(shard_file)
        try:
            for data in tqdm(shard_data):
                file_name = data['file_name']
                label = int(file_name.strip()[:-2].split('_')[-1])
                code_text = read_code(split_dir + file_name.strip())
                nodes_file_path = parsed + file_name.strip() + '/nodes.csv'
                edges_file_path = parsed + file_name.strip() + '/edges.csv'
                nc = open(nodes_file_path)
                nodes_file = csv.DictReader(nc, delimiter='\t')
                nodes = [node for node in nodes_file]
                if len(nodes) == 0:
                    continue
                nodes = read_csv(nodes_file_path)
                edges = read_csv(edges_file_path)
                node_indices, node_ids, line_numbers, node_id_to_ln = extract_nodes_with_location_info(nodes)
                adjacency_list = create_adjacency_list(line_numbers, node_id_to_ln, edges, False)
                combined_graph = combine_adjacents(adjacency_list)
                data_point = reformat_code_line_graph(
                    code_text, adjacency_list, label, wv_model_original, wv_model_li, label)
                graphs.append(data_point)
        finally:
            pass
        del shard_data
    # Save
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    with open(os.path.join(output_dir, f'{dataset_type}-line-ggnn.json'), 'w') as of:
        json.dump(graphs, of)
    return graphs
