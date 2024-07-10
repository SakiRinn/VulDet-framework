import csv
import datetime
import os
import os.path as osp
import shutil
import signal
import subprocess
import time

import numpy as np
import pandas as pd


# Require joern v0.2.5

SCRIPT_DIR = osp.dirname(osp.realpath(__file__))
PROJECT_DIR = osp.join(SCRIPT_DIR, '..')
JOERN_DIR = osp.join(PROJECT_DIR, 'joern')
TIMEOUT = 3600

NODE_TYPES = [
    'AndExpression',
    'Sizeof',
    'Identifier',
    'ForInit',
    'ReturnStatement',
    'SizeofOperand',
    'InclusiveOrExpression',
    'PtrMemberAccess',
    'AssignmentExpression',
    'ParameterList',
    'IdentifierDeclType',
    'SizeofExpression',
    'SwitchStatement',
    'IncDec',
    'Function',
    'BitAndExpression',
    'UnaryExpression',
    'DoStatement',
    'GotoStatement',
    'Callee',
    'OrExpression',
    'ShiftExpression',
    'Decl',
    'CFGErrorNode',
    'WhileStatement',
    'InfiniteForNode',
    'RelationalExpression',
    'CFGExitNode',
    'Condition',
    'BreakStatement',
    'CompoundStatement',
    'UnaryOperator',
    'CallExpression',
    'CastExpression',
    'ConditionalExpression',
    'ArrayIndexing',
    'PostIncDecOperationExpression',
    'Label',
    'ArgumentList',
    'EqualityExpression',
    'ReturnType',
    'Parameter',
    'Argument',
    'Symbol',
    'ParameterType',
    'Statement',
    'AdditiveExpression',
    'PrimaryExpression',
    'DeclStmt',
    'CastTarget',
    'IdentifierDeclStatement',
    'IdentifierDecl',
    'CFGEntryNode',
    'TryStatement',
    'Expression',
    'ExclusiveOrExpression',
    'ClassDef',
    'File',
    'UnaryOperationExpression',
    'ClassDefStatement',
    'FunctionDef',
    'IfStatement',
    'MultiplicativeExpression',
    'ContinueStatement',
    'MemberAccess',
    'ExpressionStatement',
    'ForStatement',
    'InitializerList',
    'ElseStatement'
]

NODE_TYPES_TO_IDS = {
    'AndExpression': 1,
    'Sizeof': 2,
    'Identifier': 3,
    'ForInit': 4,
    'ReturnStatement': 5,
    'SizeofOperand': 6,
    'InclusiveOrExpression': 7,
    'PtrMemberAccess': 8,
    'AssignmentExpression': 9,
    'ParameterList': 10,
    'IdentifierDeclType': 11,
    'SizeofExpression': 12,
    'SwitchStatement': 13,
    'IncDec': 14,
    'Function': 15,
    'BitAndExpression': 16,
    'UnaryExpression': 17,
    'DoStatement': 18,
    'GotoStatement': 19,
    'Callee': 20,
    'OrExpression': 21,
    'ShiftExpression': 22,
    'Decl': 23,
    'CFGErrorNode': 24,
    'WhileStatement': 25,
    'InfiniteForNode': 26,
    'RelationalExpression': 27,
    'CFGExitNode': 28,
    'Condition': 29,
    'BreakStatement': 30,
    'CompoundStatement': 31,
    'UnaryOperator': 32,
    'CallExpression': 33,
    'CastExpression': 34,
    'ConditionalExpression': 35,
    'ArrayIndexing': 36,
    'PostIncDecOperationExpression': 37,
    'Label': 38,
    'ArgumentList': 39,
    'EqualityExpression': 40,
    'ReturnType': 41,
    'Parameter': 42,
    'Argument': 43,
    'Symbol': 44,
    'ParameterType': 45,
    'Statement': 46,
    'AdditiveExpression': 47,
    'PrimaryExpression': 48,
    'DeclStmt': 49,
    'CastTarget': 50,
    'IdentifierDeclStatement': 51,
    'IdentifierDecl': 52,
    'CFGEntryNode': 53,
    'TryStatement': 54,
    'Expression': 55,
    'ExclusiveOrExpression': 56,
    'ClassDef': 57,
    'File': 58,
    'UnaryOperationExpression': 59,
    'ClassDefStatement': 60,
    'FunctionDef': 61,
    'IfStatement': 62,
    'MultiplicativeExpression': 63,
    'ContinueStatement': 64,
    'MemberAccess': 65,
    'ExpressionStatement': 66,
    'ForStatement': 67,
    'InitializerList': 68,
    'ElseStatement': 69
}

EDGE_TYPES = [
    'IS_AST_PARENT',
    'IS_CLASS_OF',
    'FLOWS_TO',
    'EDF',
    'USE',
    'REACHES',
    'CONTROLS',
    'DECLARES',
    'DOM',
    'POST_DOM',
    'IS_FUNCTION_OF_AST',
    'IS_FUNCTION_OF_CFG'
]

EDGE_TYPES_TO_IDS = {
    'IS_AST_PARENT': 1,
    'IS_CLASS_OF': 2,
    'FLOWS_TO': 3,
    'DEF': 4,
    'USE': 5,
    'REACHES': 6,
    'CONTROLS': 7,
    'DECLARES': 8,
    'DOM': 9,
    'POST_DOM': 10,
    'IS_FUNCTION_OF_AST': 11,
    'IS_FUNCTION_OF_CFG': 12
}


def df2code(dataframe, output_dir, code_tag='func'):
    if not osp.exists(output_dir):
        os.makedirs(output_dir)
    for idx, row in dataframe.iterrows():
        with open(osp.join(output_dir, f"{idx}.c"), 'w') as f:
            f.write(row[code_tag])


def code2csv(code_dir, output_dir):
    def terminate(pid):
        os.kill(pid, signal.SIGKILL)
        os.waitpid(pid, os.WNOHANG)
        shutil.rmtree('tmp/')

    joern_parse = osp.join(JOERN_DIR, 'joern-parse')
    command = f"{joern_parse} {code_dir} tmp"
    start_date = datetime.datetime.now()
    process = subprocess.Popen(command, shell=True)
    try:
        while process.poll() is None:
            time.sleep(0.5)
            end_date = datetime.datetime.now()
            if (end_date - start_date).seconds > TIMEOUT:
                terminate(process.pid)
    except BaseException as e:
        terminate(process.pid)
        raise e
    os.rename(f'tmp/{code_dir}', output_dir)
    shutil.rmtree('tmp/')


def df2csv(dataframe, output_dir, code_tag='func'):
    df2code(dataframe, 'raw_code/', code_tag)
    print('Success to convert dataframe to a set of code files.')
    try:
        code2csv('raw_code/', output_dir)
    except BaseException as e:
        shutil.rmtree('raw_code/')
        raise e
    print('Success to convert code files to graph files.')
    shutil.rmtree('raw_code/')


def csv2graph(node_csv, edge_csv, w2v_model, embed_dim=128,
              node_types_to_ids=NODE_TYPES_TO_IDS, edge_types_to_ids=EDGE_TYPES_TO_IDS):

    with open(node_csv, 'r') as nc:
        nodes = csv.DictReader(nc, delimiter='\t')

        nodes_to_ids, node_features = {}, {}
        for i, node in enumerate(nodes):
            is_cfg_node = node['isCFGNode'].strip()
            if is_cfg_node != 'True':
                continue

            node_key, node_type = node['key'], node['type']
            if node_type == 'File':
                continue
            node_content = node['code'].strip()
            node_split = code_tokenize(node_content)

            nrp = np.zeros(embed_dim)
            for token in node_split:
                embedding = w2v_model.wv[token] if token in w2v_model.wv else np.zeros(embed_dim)
                nrp += embedding
            f_nrp = np.divide(nrp, len(node_split)) if len(node_split) > 0 else nrp
            type_onehot = np.eye(len(node_types_to_ids))[node_types_to_ids[node_type] - 1]
            node_feature = np.concatenate([type_onehot, f_nrp], axis=0)

            node_features[node_key] = node_feature
            nodes_to_ids[node_key] = i

    edges = []
    nodes_on_edge = set()
    with open(edge_csv, 'r') as ec:
        raw_edges = csv.DictReader(ec, delimiter='\t')
        for edge in raw_edges:
            start, end, edge_type = edge["start"], edge["end"], edge["type"]
            if edge_type == "IS_FILE_OF" or edge_type not in edge_types_to_ids:
                continue
            if start not in nodes_to_ids or end not in nodes_to_ids:
                continue
            edges.append([start, edge_types_to_ids[edge_type], end])
            nodes_on_edge.update({start, end})

    noes_to_ids = {node: i for i, node in enumerate(nodes_on_edge)}
    nodes = [node_features[node_key] for node_key in nodes_on_edge]
    edges = [[noes_to_ids[start], edge_type_id, noes_to_ids[end]]
             for start, edge_type_id, end in edges]

    try:
        graph = {'nodes': np.stack(nodes, axis=0), 'edges': edges}
    except ValueError:
        graph = {'nodes': np.array([]), 'edges': edges}
    return graph
