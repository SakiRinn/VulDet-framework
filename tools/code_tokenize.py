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
