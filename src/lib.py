from typing import List, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor

from einops import parse_shape, rearrange, repeat, reduce
# from einops.layers.torch import Repeat

from utils import Args, BlockConfig


def merge_tensors(tensors: List[Tensor], device, max_length: int = None) -> Tuple[Tensor, float, List]:
    lengths = [t.shape[0] for t in tensors]
    if max_length is None:
        max_length = max(lengths)
    res = torch.zeros((len(tensors), max_length, *tensors[0].shape[1:]), device=device)
    for i, tensor in enumerate(tensors):
        res[i, :tensor.shape[0]] = tensor
    # res: (batch_size, max_length, embedding_dim)
    return res, max_length, lengths


def get_1D_padding_mask(lengths: List[int], max_length: int, device) -> Tensor:
    mask = torch.zeros((len(lengths), max_length), dtype=torch.bool, device=device)
    for i, length in enumerate(lengths):
        mask[i, length:] = True
    return mask


def get_2D_padding_mask(lengths: List[List[int]], max_length1: int, max_length2: int, device) -> Tensor:
    mask = torch.zeros((len(lengths), max_length1, max_length2), dtype=torch.bool, device=device)
    for i, len_list in enumerate(lengths):
        length1 = len(len_list)
        mask[i, length1:, :] = True
        for j, length2 in enumerate(len_list):
            mask[i, j, length2:] = True
    return mask


def get_src_mask(src_key_padding_mask: Tensor, num_heads: int) -> Tensor:
    batch_size, src_len = src_key_padding_mask.shape
    src_mask = torch.zeros((batch_size, src_len, src_len), device=src_key_padding_mask.device)
    src_mask.masked_fill_(src_key_padding_mask.unsqueeze(1), -1e5)
    src_mask = src_mask.repeat_interleave(num_heads, dim=0)
    return src_mask


class LayerNorm(nn.Module):
    r"""
    Layer normalization.
    """

    def __init__(self, hidden_size: int, eps=1e-5):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x: Tensor):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class MLP(nn.Module):
    def __init__(self, hidden_size: int, out_features=None):
        super(MLP, self).__init__()
        if out_features is None:
            out_features = hidden_size
        self.linear = nn.Linear(hidden_size, out_features)
        self.layer_norm = LayerNorm(out_features)

    def forward(self, hidden_states: Tensor):
        hidden_states = self.linear(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = torch.nn.functional.relu(hidden_states)
        return hidden_states


class Projection(nn.Module):

    def __init__(self, dim_in: int, dim_out: int) -> None:
        super(Projection, self).__init__()
        self.linear = nn.Linear(dim_in, dim_out)

    def forward(self, x: Tensor) -> Tensor:
        x = self.linear(x)
        x = F.relu(x)
        return x


class RoadGraph(nn.Module):

    def __init__(self, config: Args) -> None:
        super(RoadGraph, self).__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=config.sub_graph_hidden,
            nhead=config.sub_graph_heads,
            dim_feedforward=config.sub_graph_dim_ffn,
            dropout=config.dropout,
            batch_first=True
        )
        self.model = nn.TransformerEncoder(layer, num_layers=config.sub_graph_depth)
        self.projeciton = Projection(config.sub_graph_hidden, config.hidden_size)

    def forward(self, x: Tensor) -> Tensor:
        # x: (S, V, D)   V - num of vector
        x = self.model(x)
        x = torch.max(x, dim=1)[0]  # (S, D)
        return self.projeciton(x)  # (S, hidden_size)


class LatentQuery(nn.Module):

    def __init__(self, config: Args) -> None:
        super(LatentQuery, self).__init__()
        raise NotImplementedError


class EncoderBlock(nn.Module):

    def __init__(self, config: BlockConfig) -> None:
        super(EncoderBlock, self).__init__()

        option = config.option
        dim_D = config.dim_D
        dim_T = config.dim_T
        dim_S = config.dim_S
        n_head = config.n_head
        dim_feedforward = config.dim_feedforward
        latent_query = config.latent_query
        lq_ratio = config.lq_ratio
        dropout = config.dropout

        # [A, T, S, D]
        assert not latent_query
        if option == 'temporal':
            self.in_pattern = 'A T S D -> (A S) T D'
            self.in_pattern_mask = 'A T S -> (A S) T'
            self.out_pattern = '(A S) T D -> A T S D'
            self.L_in = dim_T
        elif option == 'spatial':
            self.in_pattern = 'A T S D -> (A T) S D'
            self.in_pattern_mask = 'A T S -> (A T) S'
            self.out_pattern = '(A T) S D -> A T S D'
            self.L_in = dim_S
        elif option == 'multi_axis':
            self.in_pattern = 'A T S D -> A (T S) D'
            self.in_pattern_mask = 'A T S -> A (T S)'
            self.out_pattern = 'A (T S) D -> A T S D'
            self.L_in = dim_T * dim_S
        else:
            raise NotImplementedError(f'EncoderBlock with {option} not implemented')
        self.apply_out_rearrange = (option == 'spatial' or option == 'temporal')
        self.latent_query = latent_query
        self.lq_ratio = lq_ratio
        if latent_query:
            self.L_out = int(self.L_in * lq_ratio)
            self.latent_pattern = 'A L D -> A D L'
            self.latent_mapping = nn.Linear(self.L_in, self.L_out)

        self.tfm_layer = nn.TransformerEncoderLayer(
            d_model=dim_D,
            nhead=n_head,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.n_head = n_head

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        if x.dim() == 4:
            # for factorized blocks or first block of multi-axis
            axes_lengths = parse_shape(x, 'A T S D')
            x = rearrange(x, self.in_pattern)
        else:
            assert x.dim() == 3
        if mask is not None:
            mask = rearrange(mask, self.in_pattern_mask)
            src_mask = get_src_mask(mask, self.n_head)
        if self.latent_query:
            x = rearrange(x, self.latent_pattern)
            x = self.latent_mapping(x)
            x = rearrange(x, self.latent_pattern)
            axes_lengths = {}
        # x: [A, L, D]

        # print('mask', mask.max().item(), mask.min().item(), mask.dtype)
        x = self.tfm_layer(x, src_mask=src_mask)
        # print(f'{x.max().item()}')
        if self.apply_out_rearrange:
            x = rearrange(x, self.out_pattern, **axes_lengths)
        return x


class Encoder(nn.Module):

    def __init__(self, config: Args) -> None:
        super(Encoder, self).__init__()
        block_configs = config.get_block_config()
        self.blocks = nn.ModuleList([EncoderBlock(block_cfg) for block_cfg in block_configs])
        self.num_blocks = len(self.blocks)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        for i, block in enumerate(self.blocks):
            # print(f'encoder layer{i}: {x.max().item()}')
            x = block(x, mask)
            # assert False
            if (i == self.num_blocks - 1) and (x.dim() == 4):
                x = rearrange(x, 'A T S D -> A (T S) D')
        return x


class Decoder(nn.Module):

    def __init__(self, config: Args) -> None:
        super(Decoder, self).__init__()
        self.config = config
        self.initial_qs = nn.Parameter(torch.randn(config.k_components, config.hidden_size))
        layer = nn.TransformerDecoderLayer(
            d_model=config.hidden_size,
            nhead=config.n_head,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True
        )
        self.model = nn.TransformerDecoder(layer, num_layers=config.num_decoder_blocks)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.regressor = nn.Linear(config.hidden_size, config.pred_horizon * 4)

    def forward(self, memory: Tensor, mask: Tensor = None) -> Tuple[Tensor]:
        target = repeat(self.initial_qs, 'k d -> b k d', b=memory.shape[0])
        if mask is not None:
            src_mask = get_src_mask(mask, self.config.n_head)
        embeddings = self.model(target, memory, memory_key_padding_mask=mask)
        cls_head = self.classifier(embeddings)
        reg_head = self.regressor(embeddings)
        cls_head = cls_head.squeeze(-1)
        reg_head = rearrange(reg_head, 'b k (t c) -> b k t c', c=4)
        return cls_head, reg_head


class EarlyFusion(nn.Module):

    def __init__(self, config: Args) -> None:
        super(EarlyFusion, self).__init__()
        self.config = config
        self.projection1 = Projection(config.dim_D_h, config.hidden_size)
        self.projection2 = Projection(config.dim_D_i, config.hidden_size)
        self.road_graph = RoadGraph(config)
        self.encoder = Encoder(config)

    def forward_road_embedding(self, matrix: List[np.ndarray], device) -> Tuple[Tensor]:
        batch_size = len(matrix)
        batch_embedding = []
        for i in range(batch_size):
            # tensor = torch.from_numpy(matrix[i]).to(device)
            tensor = torch.tensor(matrix[i], device=device)
            embedding = self.road_graph(tensor)
            batch_embedding.append(embedding)
        embeddings, max_len, lengths = merge_tensors(batch_embedding, device)
        padding_mask = get_1D_padding_mask(lengths, max_len, device)

        embeddings = repeat(embeddings, 'A S D -> A T S D', T=self.config.dim_T)
        padding_mask = repeat(padding_mask, 'A S -> A T S', T=self.config.dim_T)
        return embeddings, padding_mask

    def forward_interact_embedding(self, agents: List[List[np.ndarray]], device) -> Tuple[Tensor]:
        length_list = list()
        embedding_list = list()
        for agent in agents:
            input_list = [torch.tensor(a, device=device) for a in agent]
            embeddings, _, lengths = \
                merge_tensors(input_list, device, max_length=self.config.dim_T)
            length_list.append(lengths)
            embedding_list.append(embeddings)
        embeddings, max_len, lengths = merge_tensors(embedding_list, device)
        embeddings = self.projection2(embeddings)
        padding_mask = get_2D_padding_mask(length_list, max_len, self.config.dim_T, device)

        embeddings = rearrange(embeddings, 'A S T D -> A T S D')
        padding_mask = rearrange(padding_mask, 'A S T -> A T S')
        return embeddings, padding_mask

    def forward_history_embedding(self, agents: List[List[np.ndarray]], device) -> Tuple[Tensor]:
        batch_size = len(agents)
        histories = [agent[0][:, :3] for agent in agents]
        histories = torch.tensor(rearrange(histories, 'A T D -> A T D'), device=device)
        embeddings = self.projection1(histories)
        embeddings = repeat(embeddings, 'A T D -> A T 1 D')
        padding_mask = torch.zeros(batch_size, self.config.dim_T, 1, dtype=torch.bool, device=device)
        return embeddings, padding_mask

    def forward(self, agents: List[List[np.ndarray]], matrix: List[np.ndarray], device) -> Tensor:
        road, road_mask = self.forward_road_embedding(matrix, device)
        interact, interact_mask = self.forward_interact_embedding(agents, device)
        history, history_mask = self.forward_history_embedding(agents, device)
        embedding = torch.cat([road, interact, history], dim=2)
        padding_mask = torch.cat([road_mask, interact_mask, history_mask], dim=2)
        assert embedding.shape[:3] == padding_mask.shape
        memory_mask = rearrange(padding_mask, 'A T S -> A (T S)')
        return self.encoder(embedding, padding_mask), memory_mask
        # return embedding, padding_mask