import sys
import json
import os.path as osp

sys.path.append(osp.realpath(osp.join(osp.dirname(__file__), '..')))
from dataloaders.records import pkl_to_records


def d2a_to_json(pkl_gz_path):
    data = pkl_to_records(pkl_gz_path, with_gz=True)
    data_dir, filename = osp.split(pkl_gz_path)
    filename = filename.split(".")[0]
    with open(f'{data_dir}/{filename}.json', 'w') as f:
        json.dump(data, f, indent=4)


if __name__ == "__main__":
    d2a_to_json('data/ffmpeg_labeler_0.pickle.gz')
    d2a_to_json('data/ffmpeg_labeler_1.pickle.gz')