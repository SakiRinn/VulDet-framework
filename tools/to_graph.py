import datetime
import os
import os.path as osp
import shutil
import signal
import subprocess
import time

import pandas as pd

# Require joern v0.2.5
JOERN_DIR = './joern/'
TIMEOUT = 3600


def df2code(dataframe, output_dir, code_column='func'):
    if not osp.exists(output_dir):
        os.makedirs(output_dir)
    for idx, row in dataframe.iterrows():
        with open(osp.join(output_dir, f"{idx}.c"), 'w') as f:
            f.write(row[code_column])

def code2graph(code_dir, output_dir):
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

def df2graph(dataframe, output_dir):
    df2code(dataframe, 'raw_code/')
    print('Success to convert dataframe to a set of code files.')
    try:
        code2graph('raw_code/', output_dir)
    except BaseException as e:
        shutil.rmtree('raw_code/')
        raise e
    print('Success to convert code files to graph files.')
    shutil.rmtree('raw_code/')


if __name__ == '__main__':
    raw = pd.read_json('data/devign/devign.json')
    df2graph(raw, 'data/graph_data')
