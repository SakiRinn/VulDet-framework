import json
import h5py
import clang.cindex
import clang.enumerations
import csv
import numpy as np
import os


train_file_name = '../data/draper/raw/VDISC_train.hdf5'
train_file = h5py.File(train_file_name)


list(train_file)

num_vul = 0
num_non_vul = 0
vul_indices = []

for idx, (a, b, c, d, e) in  enumerate(zip(
    train_file['CWE-119'], train_file['CWE-120'], train_file['CWE-469'],
    train_file['CWE-476'], train_file['CWE-other']
)):
    if a or b or c or d or e:
        num_vul += 1
        vul_indices.append(idx)
    else:
        num_non_vul += 1

print(num_vul, num_non_vul, len(vul_indices))

print(tokenize("int main(){\n\tint *a = new int[10];\n\treturn 50;\n}\n"))
ratio = 65907 / float(953567)
print(ratio)

sources = []
v, nv = 0, 0

for idx, func in enumerate(train_file['functionSource']):
    if idx % 10000 == 0:
        print(idx, v, nv)
    if idx in vul_indices:
        tokenized = tokenize(func.strip())
        if tokenize is None:
            continue
        sources.append({'code': func.strip(), 'label': 1, 'tokenized': tokenized})
        v += 1
    else:
        r = np.random.uniform()
        if r <= 1.00:
            tokenized = tokenize(func.strip())
            if tokenize is None:
                continue
            sources.append({'code': func.strip(), 'label': 0, 'tokenized': tokenized})
            nv += 1

len(sources)

train_file_name = open('../data/draper/train_full.json', 'w')
json.dump(sources, train_file_name)
train_file_name.close()
print(sources[0])

def get_all(file_path):
    _file = h5py.File(file_path)
    v = 0
    nv = 0
    sources = []
    for idx, (a, b, c, d, e, f) in  enumerate(zip(
        _file['CWE-119'], _file['CWE-120'], _file['CWE-469'],
        _file['CWE-476'], _file['CWE-other'], _file['functionSource']
    )):
        if idx % 10000 == 0:
            print(idx)
        tokenized = tokenize(f)
        if tokenized == None:
            continue
        if a or b or c or d or e:
            sources.append({
                'code': f.strip(),
                'label': 1,
                'tokenized': tokenized
            })
            v += 1
        else:
            sources.append({
                'code': f.strip(),
                'label': 0,
                'tokenized': tokenized
            })
            nv += 1
    return sources, v, nv

valid_file_name = '../data/draper/VDISC_validate.hdf5'
valid_data, v, nv = get_all(valid_file_name)
print(v, nv, len(valid_data), valid_data[0])

json_file_name = open('../data/draper/valid.json', 'w')
json.dump(valid_data, json_file_name)
json_file_name.close()

test_file_name = '../data/draper/VDISC_test.hdf5'
test_data, v, nv = get_all(test_file_name)
print(v, nv, len(test_data))
json_file_name = open('../data/draper/test.json', 'w')

json.dump(test_data, json_file_name)
json_file_name.close()