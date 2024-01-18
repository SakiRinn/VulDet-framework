import nltk
import numpy as np
from graphviz import Digraph

from datasets.utils.tokenize import symbolic_tokenize


def combine_adjacents(adjacency_list):
    cgraph = {}
    for ln in adjacency_list:
        cgraph[ln] = set(adjacency_list[ln][0]).union(adjacency_list[ln][1])
    return cgraph

def invert_graph(adjacency_list):
    inverted_graph = {}
    for ln in adjacency_list.keys():
        inverted_graph[ln] = set()
    for ln in adjacency_list:
        adj = adjacency_list[ln]
        for node in adj:
            inverted_graph[node].add(ln)
    return inverted_graph

def create_visual_graph(adjacency_list, code, filename='test_graph', verbose=False):
    graph = Digraph('Code Property Graph')
    for ln in adjacency_list:
        graph.node(str(ln), str(ln) + '\t' + code[ln], shape='box')
        control_dependency, data_dependency = adjacency_list[ln]
        for anode in control_dependency:
            graph.edge(str(ln), str(anode), color='red')
        for anode in data_dependency:
            graph.edge(str(ln), str(anode), color='blue')
    graph.render(filename, view=verbose)

def create_adjacency_list(line_numbers, node_id_to_line_numbers, edges, data_dependency_only=False):
    adjacency_list = {}
    for ln in set(line_numbers):
        adjacency_list[ln] = [set(), set()]
    for edge in edges:
        edge_type = edge['type'].strip()
        if True:  # edge_type in ['IS_AST_PARENT', 'FLOWS_TO']:
            start_node_id = edge['start'].strip()
            end_node_id = edge['end'].strip()
            if start_node_id not in node_id_to_line_numbers.keys() or end_node_id not in node_id_to_line_numbers.keys():
                continue
            start_ln = node_id_to_line_numbers[start_node_id]
            end_ln = node_id_to_line_numbers[end_node_id]
            if not data_dependency_only:
                if edge_type == 'CONTROLS':     # Control flow edges
                    adjacency_list[start_ln][0].add(end_ln)
            if edge_type == 'REACHES':          # Data flow edges
                adjacency_list[start_ln][1].add(end_ln)
    return adjacency_list

def create_forward_slice(adjacency_list, line_number):
    sliced_lines = set()
    sliced_lines.add(line_number)
    stack = [line_number]
    while len(stack) != 0:
        cur = stack.pop()
        if cur not in sliced_lines:
            sliced_lines.add(cur)
        for node in adjacency_list[cur]:
            if node not in sliced_lines:
                stack.append(node)
    return sorted(sliced_lines)

def create_backward_slice(adjacency_list, line_number):
    return create_forward_slice(invert_graph(adjacency_list), line_number)

def reformat_code_line_graph(code_lines, adjacency_lists, w2v_model_original, w2v_model_li, label):
    actual_lines = []
    for ln in adjacency_lists.keys():
        cd, dd = adjacency_lists[ln]
        new_cd = [l for l in cd]
        new_dd = [l for l in dd]
        actual_lines.extend(new_cd)
        actual_lines.extend(new_dd)
        actual_lines.append(ln)
    actual_lines = sorted(list(set(actual_lines)))
    line_no_to_idx = {}
    idx_to_line_no = {}
    for idx, ln in enumerate(actual_lines):
        line_no_to_idx[ln] = idx
        idx_to_line_no[idx] = ln
    data_point = {}
    graph = []
    for src in adjacency_lists.keys():
        cd, dd = adjacency_lists[src]
        for dest in cd:
            graph.append([line_no_to_idx[src], 0, line_no_to_idx[dest]])
            graph.append([line_no_to_idx[dest], 1, line_no_to_idx[src]])
        for dest in dd:
            graph.append([line_no_to_idx[src], 2, line_no_to_idx[dest]])
            graph.append([line_no_to_idx[dest], 3, line_no_to_idx[src]])
    original_tokens = []
    symbolic_tokens = []
    line_features_wv = []
    sym_line_features_wv = []

    for lidx in range(len(idx_to_line_no.keys())):
        actual_code_line = code_lines[idx_to_line_no[lidx]]
        actual_line_tokens = nltk.wordpunct_tokenize(actual_code_line)
        symbolic_line_tokens = symbolic_tokenize(actual_code_line).split()
        original_tokens.append(actual_line_tokens)
        symbolic_tokens.append(symbolic_line_tokens)

        nrp = np.zeros(100)
        for token in actual_line_tokens:
            try:
                embedding = w2v_model_original.wv[token]
            except:
                embedding = np.zeros(100)
            nrp = np.add(nrp, embedding)
        if len(actual_line_tokens) > 0:
            fNrp = np.divide(nrp, len(actual_line_tokens))
        else:
            fNrp = nrp
        line_features_wv.append(fNrp.tolist())

        nrp = np.zeros(64)
        for token in symbolic_line_tokens:
            try:
                embedding = w2v_model_li.wv[token]
            except:
                embedding = np.zeros(64)
            nrp = np.add(nrp, embedding)
        if len(actual_line_tokens) > 0:
            fNrp = np.divide(nrp, len(symbolic_line_tokens))
        else:
            fNrp = nrp
        sym_line_features_wv.append(fNrp.tolist())
    data_point = {
        'node_features': line_features_wv,
        'node_features_sym': sym_line_features_wv,
        'graph': graph,
        'original_tokens': original_tokens,
        'symbolic_tokens': symbolic_tokens,
        'targets': [[label]]
    }
    return data_point
