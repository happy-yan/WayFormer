from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from typing import List, Tuple, Dict

import math
import numpy as np
import sys
import os
import time

eps = 1e-5


def assert_(satisfied, info=None):
    if not satisfied:
        if info is not None:
            print(info)
        print(sys._getframe().f_code.co_filename, sys._getframe().f_back.f_lineno)
    assert satisfied


def rotate(x, y, angle):
    res_x = x * math.cos(angle) - y * math.sin(angle)
    res_y = x * math.sin(angle) + y * math.cos(angle)
    return res_x, res_y


def get_unit_vector(point_a, point_b):
    der_x = point_b[0] - point_a[0]
    der_y = point_b[1] - point_a[1]
    scale = 1 / math.sqrt(der_x ** 2 + der_y ** 2)
    der_x *= scale
    der_y *= scale
    return (der_x, der_y)


def larger(a, b):
    return a > b + eps


def get_dis(points: np.ndarray, point_label):
    return np.sqrt(np.square((points[:, 0] - point_label[0])) + np.square((points[:, 1] - point_label[1])))


def get_angle(x, y):
    return math.atan2(y, x)


def get_time():
    return time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())


time_begin = get_time()


def get_name(name: str = '', mode: str = 'train', append_time=False):
    if name.endswith(time_begin):
        return name
    if mode == 'test' or mode == 'val':
        prefix = f'{mode}.'
    elif mode == 'train':
        prefix = ''
    else:
        raise NotImplementedError
    # prefix = 'debug.' + prefix if args.debug else prefix
    # prefix = args.add_prefix + '.' + prefix if args.add_prefix is not None else prefix
    suffix = '.' + time_begin if append_time else ''
    return prefix + str(name) + suffix


def get_from_mapping(mappings: List[Dict], key: str):
    return [each_map[key] for each_map in mappings]


def load_config(config_path: str):
    with open(config_path, 'r') as f:
        config = Args.from_json(f.read())
    return config


def batch_list_to_batch_tensors(batch):
    return [each for each in batch]


@dataclass
class Metrics:
    k: int
    minADE: float
    minFDE: float
    missed: int


@dataclass_json
@dataclass
class BlockConfig:
    option: str
    dim_T: int
    dim_S: int
    dim_D: int
    n_head: int
    dim_feedforward: int
    dropout: float = 0.1
    latent_query: bool = False
    lq_ratio: float = 0.5


@dataclass_json
@dataclass
class Args:
    exp_name: str

    # data preprocessing
    direction: bool = True
    train_data_dir: str = os.path.expanduser('~/dataset/argo/train/data')
    val_data_dir: str = os.path.expanduser('~/dataset/argo/val/data')
    temp_file_dir: str = os.path.expanduser('~/dataset/argo/temp')
    core_num: int = 8
    reuse_temp_file: bool = False
    compress: str = 'zlib'

    # training
    batch_size: int = 32
    do_test: bool = False
    do_eval: bool = False
    do_train: bool = True
    learning_rate: float = 2e-4
    num_gpu: int = 2
    max_epochs: int = 20
    data_workers: int = 8
    log_period: int = 100

    # road info
    semantic_lane: bool = True
    max_distance: float = 50.0

    # encoding
    option: str = 'factorized'  # option = 'multi_axis' or 'factorized'
    hidden_size: int = 256  # dim_D
    dim_T: int = 20
    S_h: int = 1
    S_i: int = 31
    S_r: int = 64
    n_head: int = 2
    dim_feedforward: int = 512
    dropout: float = 0.1
    latent_query: bool = False
    lq_ratio: float = 0.5
    factorized: str = 'interleaved'
    num_blocks: int = 6

    # road embedding
    sub_graph_hidden: int = 128
    sub_graph_depth: int = 2
    sub_graph_heads: int = 2
    sub_graph_dim_ffn: int = 256

    # history embedding
    dim_D_h: int = 3

    # interaction embedding
    dim_D_i: int = 6

    # decoding
    k_components: int = 16
    num_decoder_blocks: int = 4
    pred_horizon: int = 30

    # output
    output_dir: str = 'output'

    def __post_init__(self):
        # assert os.path.exists(self.train_data_dir), f'Cannot find train data dir {self.train_data_dir}'
        # assert os.path.exists(self.val_data_dir), f'Cannot find val data dir {self.val_data_dir}'
        output_dir = os.path.join(self.output_dir, self.exp_name)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if not os.path.exists(self.temp_file_dir):
            os.makedirs(self.temp_file_dir)

    def get_block_config(self) -> List[BlockConfig]:
        if self.option == 'multi_axis':
            blocks, lqs = self.get_multi_axis_blocks()
        elif self.option == 'factorized':
            blocks, lqs = self.get_factorized_blocks()
        result = [BlockConfig(
            option=opt,
            dim_T=self.dim_T,
            dim_S=self.S_i + self.S_r + self.S_h,
            dim_D=self.hidden_size,
            n_head=self.n_head,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            latent_query=lq,
            lq_ratio=self.lq_ratio
        ) for opt, lq in zip(blocks, lqs)]

        return result

    def get_factorized_blocks(self) -> Tuple[List[str], List[bool]]:
        assert self.num_blocks > 1 and self.num_blocks % 2 == 0
        repeat = self.num_blocks // 2
        if self.factorized == 'sequential':
            blocks = ['temporal'] * repeat + ['spatial'] * repeat
        elif self.factorized == 'interleaved':
            blocks = ['temporal', 'spatial'] * repeat
        else:
            raise NotImplementedError(f'Unknown option {self.factorized}')

        if self.latent_query:
            if self.factorized == 'sequential':
                lqs = ([True] + [False] * (repeat - 1)) * 2
            elif self.factorized == 'interleaved':
                lqs = [True] * 2 + [False] * (self.num_blocks - 2)
            else:
                raise NotImplementedError(f'Unknown option {self.factorized}')
        else:
            lqs = [False] * self.num_blocks
        return blocks, lqs

    def get_multi_axis_blocks(self) -> Tuple[List[str], List[bool]]:
        assert self.num_blocks > 0
        blocks = ['multi_axis'] * self.num_blocks
        if self.latent_query:
            lqs = [True] + [False] * (self.num_blocks - 1)
        else:
            lqs = [False] * self.num_blocks
        return blocks, lqs

    def save_config(self, save_dir: str):
        with open(os.path.join(save_dir, f'{self.exp_name}.json'), 'w') as f:
            f.write(self.to_json(indent=2))


if __name__ == '__main__':
    # config = EncoderConfig(
    #     option='multi_axis',
    #     dim_T=16,
    #     dim_S=8,
    #     dim_D=32,
    #     n_head=2,
    #     dim_feedforward=64,
    #     dropout=0.1,
    #     latent_query=True,
    #     lq_ratio=0.5,
    #     factorized='sequential',
    #     num_blocks=6
    # )
    # print(config.get_block_config())
    args = Args('wayformer.1')
    print(args)