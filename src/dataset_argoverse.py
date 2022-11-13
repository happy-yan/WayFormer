import math
import multiprocessing
import os
import pickle
from typing import Dict, List
import zlib
import lz4.frame
from multiprocessing import Process
from random import choice
from einops import rearrange

import numpy as np
import torch
from argoverse.map_representation.map_api import ArgoverseMap
from tqdm import tqdm

from utils import rotate, larger, assert_, get_dis, get_angle, get_name, Args

TIMESTAMP = 0
TRACK_ID = 1
OBJECT_TYPE = 2
X = 3
Y = 4
CITY_NAME = 5

type2index = {}
type2index["OTHERS"] = 0
type2index["AGENT"] = 1
type2index["AV"] = 2

max_vector_num = 0

VECTOR_PRE_X = 0
VECTOR_PRE_Y = 1
VECTOR_X = 2
VECTOR_Y = 3

# am = ArgoverseMap()

def get_min_distance(polygon: np.ndarray) -> float:
    # polygon: [N, 2]
    dists = polygon[:, 0] ** 2 + polygon[:, 1] ** 2
    return np.min(dists)


#TODO: args check!
def get_sub_map(args: Args, x, y, city_name, mapping: Dict=None) -> np.ndarray:
    """
    Calculate lanes which are close to (x, y) on map.
    Only take lanes which are no more than args.max_distance away from (x, y).
    """
    vectors = list()
    polyline_spans = list()
    assert isinstance(am, ArgoverseMap)
    # Add more lane attributes, such as 'has_traffic_control', 'is_intersection' etc.
    if args.semantic_lane:
        lane_ids = am.get_lane_ids_in_xy_bbox(x, y, city_name, query_search_range_manhattan=args.max_distance)
        local_lane_centerlines = [am.get_lane_segment_centerline(lane_id, city_name) for lane_id in lane_ids]
        polygons = local_lane_centerlines

    #     if args.visualize:
    #         angle = mapping['angle']
    #         vis_lanes = [am.get_lane_segment_polygon(lane_id, city_name)[:, :2] for lane_id in lane_ids]
    #         t = []
    #         for each in vis_lanes:
    #             for point in each:
    #                 point[0], point[1] = rotate(point[0] - x, point[1] - y, angle)
    #             num = len(each) // 2
    #             t.append(each[:num].copy())
    #             t.append(each[num:num * 2].copy())
    #         vis_lanes = t
    #         mapping['vis_lanes'] = vis_lanes
    else:
        polygons = am.find_local_lane_centerlines(x, y, city_name, args.max_distance)
    polygons = [polygon[:, :2].copy() for polygon in polygons]

    angle = mapping['angle']
    #TODO check 'scale'
    for index_polygon, polygon in enumerate(polygons):
        for i, point in enumerate(polygon):
            point[0], point[1] = rotate(point[0] - x, point[1] - y, angle)
            if 'scale' in mapping:
                # assert 'enhance_rep_4' in args.other_params
                scale = mapping['scale']
                point[0] *= scale
                point[1] *= scale

    # sort polygons by distance
    polygons = sorted(polygons, key=get_min_distance)

    count_poly = 0
    for index_polygon, polygon in enumerate(polygons):
        assert_(2 <= len(polygon) <= 10, info=len(polygon))

        start = len(vectors)
        if args.semantic_lane:
            assert len(lane_ids) == len(polygons)
            lane_id = lane_ids[index_polygon]
            lane_segment = am.city_lane_centerlines_dict[city_name][lane_id]

        if len(polygon) < 10: # filter some short lanelets
            continue
        else:
            count_poly += 1
        for i, point in enumerate(polygon):
            if i > 0:
                vector = [0] * args.sub_graph_hidden
                vector[-1 - VECTOR_PRE_X], vector[-1 - VECTOR_PRE_Y] = point_pre[0], point_pre[1]
                vector[-1 - VECTOR_X], vector[-1 - VECTOR_Y] = point[0], point[1]
                vector[-5] = 1
                vector[-6] = i

                vector[-7] = len(polyline_spans)

                if args.semantic_lane:
                    vector[-8] = 1 if lane_segment.has_traffic_control else -1
                    vector[-9] = 1 if lane_segment.turn_direction == 'RIGHT' else \
                        -1 if lane_segment.turn_direction == 'LEFT' else 0
                    vector[-10] = 1 if lane_segment.is_intersection else -1
                point_pre_pre = (2 * point_pre[0] - point[0], 2 * point_pre[1] - point[1])
                if i >= 2:
                    point_pre_pre = polygon[i - 2]
                vector[-17] = point_pre_pre[0]
                vector[-18] = point_pre_pre[1]

                vectors.append(vector)
            point_pre = point

        end = len(vectors)
        if start < end:
            polyline_spans.append([start, end])
        if count_poly >= args.S_r:
            break
    vectors = np.array(vectors, dtype=np.float32)
    return rearrange(vectors, '(S V) D -> S V D', V=9)


def preprocess(args: Args, id2info: Dict, mapping: Dict):
    """
    This function calculates matrix based on information from get_instance.
    """
    # polyline_spans = []
    keys = list(id2info.keys())
    assert 'AV' in keys
    assert 'AGENT' in keys
    keys.remove('AV')
    keys.remove('AGENT')
    keys = ['AGENT', 'AV'] + keys
    # vectors = []
    two_seconds = mapping['two_seconds']
    # mapping['trajs'] = []
    agents = list()

    for id in keys:
        info = id2info[id]
        agent = []
        for _, line in enumerate(info):
            if larger(line[TIMESTAMP], two_seconds):
                break
            agent.append((line[X], line[Y], line[TIMESTAMP],\
                          line[OBJECT_TYPE] == 'AV', line[OBJECT_TYPE] == 'AGENT', line[OBJECT_TYPE] == 'OTHERS'))
        if len(agent) > 1:
            agents.append(np.array(agent, dtype=np.float32))

    if len(agents) > args.S_i:
        agents = sorted(agents, key=get_min_distance)
        agents = agents[:args.S_i]
    mapping['agents'] = agents

    matrix = get_sub_map(args, mapping['cent_x'], mapping['cent_y'], mapping['city_name'], mapping)
    assert_(matrix.shape[0] <= args.S_r, info=matrix.shape)

    labels = []
    info = id2info['AGENT']
    info = info[mapping['agent_pred_index']:]
    if not args.do_test:
        assert len(info) == 30
    for line in info:
        labels.append(line[X])
        labels.append(line[Y])

    mapping.update(dict(
        matrix=matrix,
        labels=np.array(labels, dtype=np.float32).reshape([30, 2]),
        # polyline_spans=[slice(each[0], each[1]) for each in polyline_spans],
        # labels_is_valid=np.ones(args.future_frame_num, dtype=np.int64),
        eval_time=30,
    ))

    return mapping

def argoverse_get_instance(lines: List[str], file_name: str, args: Args):
    """
    Extract polylines from one example file content.
    """

    global max_vector_num
    vector_num = 0
    id2info = {}
    mapping = {}
    mapping['file_name'] = file_name

    for i, line in enumerate(lines):

        line = line.strip().split(',')
        if i == 0:
            mapping['start_time'] = float(line[TIMESTAMP])
            mapping['city_name'] = line[CITY_NAME]

        line[TIMESTAMP] = float(line[TIMESTAMP]) - mapping['start_time']
        line[X] = float(line[X])
        line[Y] = float(line[Y])
        id = line[TRACK_ID]

        if line[OBJECT_TYPE] == 'AV' or line[OBJECT_TYPE] == 'AGENT':
            line[TRACK_ID] = line[OBJECT_TYPE]

        if line[TRACK_ID] in id2info:
            id2info[line[TRACK_ID]].append(line)
            vector_num += 1
        else:
            id2info[line[TRACK_ID]] = [line]

        if line[OBJECT_TYPE] == 'AGENT' and len(id2info['AGENT']) == 20:
            assert 'AV' in id2info
            assert 'cent_x' not in mapping
            agent_lines = id2info['AGENT']
            mapping['cent_x'] = agent_lines[-1][X]
            mapping['cent_y'] = agent_lines[-1][Y]
            mapping['agent_pred_index'] = len(agent_lines)
            mapping['two_seconds'] = line[TIMESTAMP]

            # Smooth the direction of agent. Only taking the direction of the last frame is not accurate due to label error.
            if args.direction:
                span = agent_lines[-6:]
                intervals = [2]
                angles = []
                for interval in intervals:
                    for j in range(len(span)):
                        if j + interval < len(span):
                            der_x, der_y = span[j + interval][X] - span[j][X], span[j + interval][Y] - span[j][Y]
                            angles.append([der_x, der_y])

            der_x, der_y = agent_lines[-1][X] - agent_lines[-2][X], agent_lines[-1][Y] - agent_lines[-2][Y]

    if not args.do_test:
        # if 'set_predict' in args.other_params:
        #     pass
        # else:
        assert len(id2info['AGENT']) == 50

    if vector_num > max_vector_num:
        max_vector_num = vector_num

    if 'cent_x' not in mapping:
        return None

    if args.do_eval:
        origin_labels = np.zeros([30, 2])
        for i, line in enumerate(id2info['AGENT'][20:]):
            origin_labels[i][0], origin_labels[i][1] = line[X], line[Y]
        mapping['origin_labels'] = origin_labels

    angle = -get_angle(der_x, der_y) + math.radians(90)

    # Smooth the direction of agent. Only taking the direction of the last frame is not accurate due to label error.
    if args.direction:
        angles = np.array(angles)
        der_x, der_y = np.mean(angles, axis=0)
        angle = -get_angle(der_x, der_y) + math.radians(90)

    mapping['angle'] = angle
    for id in id2info:
        info = id2info[id]
        for line in info:
            line[X], line[Y] = rotate(line[X] - mapping['cent_x'], line[Y] - mapping['cent_y'], angle)
        if 'scale' in mapping:
            scale = mapping['scale']
            line[X] *= scale
            line[Y] *= scale
    return preprocess(args, id2info, mapping)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, args: Args, mode: str='train', to_screen=True):

        if mode == 'train':
            data_dir = args.train_data_dir # data_dir - a list
        elif mode == 'val':
            data_dir = args.val_data_dir
        elif mode == 'test':
            raise NotImplementedError
            data_dir = args.test_data_dir
        else:
            raise NotImplementedError
        self.ex_list = []
        self.args = args

        if args.compress == 'zlib':
            file_name = 'ex_list'
            self.compress = zlib.compress
            self.decompress = zlib.decompress
        elif args.compress == 'lz4':
            file_name = 'ex_list_lz4'
            self.compress = lz4.frame.compress
            self.decompress = lz4.frame.decompress
        else:
            raise NotImplementedError("compress method not implemented")

        if args.reuse_temp_file:
            pickle_file = open(os.path.join(args.temp_file_dir, get_name(file_name, mode)), 'rb')
            self.ex_list = pickle.load(pickle_file)
            pickle_file.close()
        else:
            global am
            am = ArgoverseMap()
            if args.core_num >= 1:
                # extract all file names and store them in a list
                files = []
                # for each_dir in data_dir:
                each_dir = data_dir
                _, _, cur_files = os.walk(each_dir).__next__()
                files.extend([os.path.join(each_dir, file) for file in cur_files if
                                file.endswith("csv") and not file.startswith('.')])
                print(files[:5], files[-5:])

                pbar = tqdm(total=len(files))
                queue = multiprocessing.Queue(args.core_num)
                queue_res = multiprocessing.Queue()

                def calc_ex_list(queue, queue_res, args):
                    # res = []
                    dis_list = []
                    while True:
                        file = queue.get()
                        if file is None:
                            break
                        if file.endswith("csv"):
                            with open(file, "r", encoding='utf-8') as fin:
                                lines = fin.readlines()[1:]
                            instance = argoverse_get_instance(lines, file, args)
                            if instance is not None:
                                # data_compress = zlib.compress(pickle.dumps(instance))
                                data_compress = self.compress(pickle.dumps(instance))
                                # data_compress = pickle.dumps(instance)
                                # res.append(data_compress)
                                queue_res.put(data_compress)
                            else:
                                queue_res.put(None)

                processes = [Process(target=calc_ex_list, args=(queue, queue_res, args,)) for _ in range(args.core_num)]
                for each in processes:
                    each.start()
                # res = pool.map_async(calc_ex_list, [queue for i in range(args.core_num)])
                for file in files:
                    assert file is not None
                    queue.put(file)
                    pbar.update(1)

                # necessary because queue is out-of-order
                while not queue.empty():
                    pass
                pbar.close()

                self.ex_list = []
                pbar = tqdm(total=len(files))
                for i in range(len(files)):
                    t = queue_res.get()
                    if t is not None:
                        self.ex_list.append(t)
                    pbar.update(1)
                pbar.close()
                for _ in range(args.core_num):
                    queue.put(None)
                for each in processes:
                    each.join()

            else:
                assert False

            # cache the ex_list
            pickle_file = open(os.path.join(args.temp_file_dir, get_name(file_name, mode)), 'wb')
            pickle.dump(self.ex_list, pickle_file)
            pickle_file.close()
        assert len(self.ex_list) > 0
        if to_screen:
            print("valid data size is", len(self.ex_list))
            # logging('max_vector_num', max_vector_num)
        self.batch_size = args.batch_size

    def __len__(self):
        return len(self.ex_list)

    def __getitem__(self, idx):

        data_compress = self.ex_list[idx]
        while True:
            try:
                instance = pickle.loads(self.decompress(data_compress))
                break
            except:
                # print(f"error {idx}")
                data_compress = self.ex_list[idx + 1]
        return instance

if __name__ == "__main__":
    args = Args('test')
    args.compress = 'lz4'
    args.reuse_temp_file = True
    dataset = Dataset(args, 'train')
    for _ in range(4):
        for i in tqdm(range(len(dataset))):
            _ = dataset[i]