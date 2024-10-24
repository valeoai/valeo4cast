from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# This is parallelized version of tracking code, independant from the used tracking method. #
import os
import sys 
import json
import numpy as np
import time
import copy
from copy import deepcopy
import argparse
import copy
import json
import os
import numpy as np
from ab3dmot import AB3DMOT
import json 
import time
from tqdm import tqdm
from pathlib import Path

import pickle
import math
from collections import defaultdict
from av2.utils.io import read_feather
from av2.geometry.geometry import quat_to_mat
from av2.datasets.sensor.av2_sensor_dataloader import AV2SensorDataLoader
from av2.datasets.sensor.sensor_dataloader import read_city_SE3_ego
from typing import Any, Dict, Iterable, List, Optional, Union
from itertools import chain

from multiprocessing import Pool


Frame = Dict[str, Any]
Frames = List[Frame]
Sequences = Dict[str, Frames]
ForecastSequences = Dict[str, Dict[int, List[Frame]]]


def group_frames(frames_list: List[Dict]) -> Dict[str, List[Dict]]:
    """
    Parameters
    ----------
    frames_list: list
        list of frames, each containing a detections snapshot for a timestamp
    """
    frames_by_seq_id = defaultdict(list)
    frames_list = sorted(frames_list, key=lambda f: f["timestamp_ns"])
    for frame in frames_list:
        frames_by_seq_id[frame["seq_id"]].append(frame)
    return dict(frames_by_seq_id)

def ungroup_frames(frames_by_seq_id: Sequences) -> Frames:
    """Ungroup dictionary of frames into a list of frames.

    Args:
        frames_by_seq_id: dictionary of frames

    Returns:
        List of frames
    """
    return list(chain.from_iterable(frames_by_seq_id.values()))

def wrap_pi(theta) :
    theta = np.remainder(theta, 2 * np.pi)
    theta[theta > np.pi] -= 2 * np.pi
    return theta

def concatenate_array_values(array_dicts: List[Dict]) -> Dict[str, np.ndarray]:
    """
    Concatenates numpy arrays in list of dictionaries
    Handles inconsistent keys (will skip missing keys)
    Does not concatenate non-numpy values (int, str), sets to value if all values are equal
    """
    combined = defaultdict(list)
    for array_dict in array_dicts:
        for k, v in array_dict.items():
            combined[k].append(v)
    concatenated = {}
    for k, vs in combined.items():
        if all(isinstance(v, np.ndarray) for v in vs):
            if any(v.size > 0 for v in vs):
                concatenated[k] = np.concatenate([v for v in vs if v.size > 0])
            else:
                concatenated[k] = vs[0]
        elif all(vs[0] == v for v in vs):
            concatenated[k] = vs[0]
    return concatenated

def parse_args():
    parser = argparse.ArgumentParser(description="Tracking Evaluation")
    parser.add_argument("--work_dir", help="the dir to save logs and tracking results")
    parser.add_argument(
        "--checkpoint", help="the dir to checkpoint which the model read from"
    )
    # parser.add_argument(
    #     "--raw_res", help="the dir to checkpoint which the model read from"
    # )
    parser.add_argument("--hungarian", action='store_true')
    parser.add_argument("--ego_coord", action='store_true')
    parser.add_argument("--root", type=str, default="/datasets_master/argoverse2_sensor/")
    parser.add_argument("--version", type=str, default='val')
    parser.add_argument("--max_age", type=int, default=3)
    parser.add_argument("--worker", type=int, default=0)
    parser.add_argument("--chunks", type=int, default=35)
    parser.add_argument("--trainset", action='store_true')
    

    args = parser.parse_args()

    return args

def average_scores_across_track(tracks):
    score_avg_tracks = deepcopy(tracks)
    for seq_id in tracks:
        scores_by_id = defaultdict(list)
        for frame in tracks[seq_id]:
            # print("frame ", frame)
            for id, score in zip(frame["track_id"], frame["score"]):
                scores_by_id[id].append(score)
        score_by_id = {id: np.mean(scores) for id, scores in scores_by_id.items()}
        for frame in score_avg_tracks[seq_id]:
            frame["detection_score"] = frame["score"]
            frame["score"] = np.array([score_by_id[id] for id in frame["track_id"]])
    return score_avg_tracks


def save_first_frame():
    args = parse_args()
    if os.path.exists(f'./av2_frame_meta/frames_meta_{args.version}.json'):
        return
    av2_data_directory = args.root + f'/{args.version}'
    # print(f"av2 path:{av2_data_directory}")
    all_scenario_files = sorted(Path(av2_data_directory).rglob("city_SE3_egovehicle.feather"))
    av2_dataloader = AV2SensorDataLoader(Path(av2_data_directory), Path(av2_data_directory))
    all_log_ids = av2_dataloader.get_log_ids()
    # print("all_log_ids ", all_log_ids)
    # print("len(all_log_ids) ", len(all_log_ids))
    

    frames = []
    # for loop scenes
    for scenario_path in all_scenario_files:
        scenario_id = str(scenario_path.parents[0]).split('/')[-1]
        assert scenario_id in all_log_ids 
        # print("scenario_id: ", scenario_id)
        
        # load city_name, scenario_id, timestamps_ns (list of start_timestamp, end_timestamp, num_timestamps), tracks, map_id, slice_id
        lidar_paths = str(scenario_path.parents[0]) + "/sensors/lidar/"
        all_timestamps = sorted([int(filename.split(".")[0]) for filename in os.listdir(lidar_paths)])
    
    
        for idx, sample in enumerate(all_timestamps):
            # print(sample)
            scene_name = scenario_id
            timestamp = sample * 1e-9
            token = sample
            frame = {}
            frame['token'] = token
            frame['timestamp'] = timestamp 
            frame['scene_name'] = scene_name

            # start of a sequence
            if idx ==0:
                frame['first'] = True 
            else:
                frame['first'] = False 
            frames.append(frame)
        # print()
    
    with open(f'./av2_frame_meta/frames_meta_{args.version}.json', "w", encoding="utf-8") as f:
        json.dump({'frames': frames}, f)


class Tracking():
    def __init__(self, predictions, all_frames_by_seq, all_seq_names, args, class_names, sample_freq = 1, process_num=10):
        self.predictions = predictions
        self.max_age = args.max_age
        self.class_names = class_names
        self.ego_coord = args.ego_coord
        self.all_seq_names = all_seq_names
        self.all_frames_by_seq = all_frames_by_seq
        self.sample_freq = sample_freq
        self.args = args
        self.process_num = process_num
    
    def tracking(self):
        # Parallelize data preprocessing
        print("Running in multi-process mode...")
        results = self.tracking_chunk(self.args.worker)
        if self.args.chunks == 1:
            # concatenate all the results
            all_av2_annos = {}
            all_av2_annos.update(results)
        
            res_dir = os.path.join(self.args.work_dir)
            if not os.path.exists(res_dir):
                os.makedirs(res_dir)
            
            all_av2_annos = ungroup_frames(all_av2_annos)
            all_av2_annos =  group_frames(all_av2_annos)
            # formating, same ordering as 
            print(f"Save av2 tracking res to {os.path.join(self.args.work_dir, f'{self.args.version}_tracking.pkl')}.")
            with open(os.path.join(self.args.work_dir, f'{self.args.version}_tracking.pkl'), "wb") as f:
                pickle.dump(all_av2_annos, f)
    
    def tracking_chunk(self, worker_index):
        # collect files for this worker 
        all_seq_names_chunk = np.array_split(sorted(self.all_seq_names), self.process_num)[worker_index]
        frames = []
        for seq_name in all_seq_names_chunk:
            frames += self.all_frames_by_seq[seq_name]
        
        print(f"Worker {worker_index} has {len(frames)} frames to predict.")
        tracker = AB3DMOT(classes=self.class_names, max_age=self.max_age, ego_coord=self.ego_coord) # one tracker per worker
        av2_annos = {}
        frame_size = len(frames)
        print("Begin Tracking\n")
        start = time.time()
        frame_counter = 0
        for i in tqdm(range(frame_size)):
            token = frames[i]['token'] # = timestamp 
            scene_name = frames[i]['scene_name'] # = log_id
            if os.path.exists(os.path.join(self.args.work_dir, f'tmp/{self.args.version}/{scene_name}.pkl')):
                
                if scene_name not in av2_annos:
                    with open(os.path.join(self.args.work_dir, f'tmp/{self.args.version}/{scene_name}.pkl'), "rb") as f:
                        print(os.path.join(self.args.work_dir, f'tmp/{self.args.version}/{scene_name}.pkl'), ' exists, skip.')
                        tmp = pickle.load(f)
                        av2_annos[scene_name] = tmp[scene_name]
                frame_counter +=1
                continue 
            
            # reset tracking after one video sequence
            if frames[i]['first']: # frame_meta.json should be ordered
                # use this for sanity check to ensure your token order is correct
                print("new seqence reset ")
                tracker.reset()
                last_time_stamp = frames[i]['timestamp'] # = timestamp *1e-9
                timestamp_city_SE3_ego_dict = read_city_SE3_ego(Path(os.path.join(self.args.root, self.args.version, scene_name)))
                frame_counter = 0
                

            time_lag = (frames[i]['timestamp'] - last_time_stamp) 
            last_time_stamp = frames[i]['timestamp']

            if token in self.predictions[scene_name]:
                preds = self.predictions[scene_name][token]
            else: # empty dets for this timestamp
                preds = []
                print(f"empty dets for {scene_name} of timestamp {token}")
            
            if scene_name not in av2_annos:
                av2_annos[scene_name] = []
            
            ego_to_city_SE3 = timestamp_city_SE3_ego_dict[token]
            outputs = tracker.step(preds, time_lag, ego_to_city_SE3)
            annos = [] # save results for this timestamp of the scene

            empty = True
            cur_track_ids = []
            for item in outputs: # save results for this timestamps of the scene
                
                if item['active'] == 0:
                    # print("it happens, empty frame.") #TODO
                    continue
                else:
                    yaw = item['yaw']
                    # to global
                    empty=False
                    
                    cur_track_ids.append(int(item['track_id']))
                    
                    av2_anno = {
                        "timestamp_ns": token,
                        'seq_id': scene_name,
                        "track_id": np.array([int(item['track_id'])], np.int32),
                        "score": np.array([float(item['detection_score'])], np.float32),
                        "label": np.array([self.class_names.index(item['detection_name'])], np.int32),
                        "name": np.array([str(item['detection_name'])]),
                        "translation_m": np.array([item['translation']], dtype=np.float32),
                        "size": np.array([item['size']], dtype=np.float32), #l,w,h
                        "yaw": wrap_pi(np.array([float(yaw)],np.float32)),
                        "velocity_m_per_s": np.array([item['velocity']], np.float32)

                    }
                    if frame_counter %self.sample_freq == 0:
                        annos.append(av2_anno) # all tracks in this frame
                        
                    
            if frame_counter %self.sample_freq == 0:
                print(f"{scene_name}, {token}")
                print(frames[i]['first'])
                print("frame_counter ", frame_counter)
                print()          
                assert len(set(cur_track_ids)) == len(cur_track_ids)
                annos = concatenate_array_values(annos)
                # print("annos: ", annos)
                assert annos['velocity_m_per_s'].shape[0] == annos['yaw'].shape[0]
                assert annos['yaw'].shape[0] == annos['size'].shape[0]
                assert np.unique(annos['track_id']).shape == annos['track_id'].shape
                assert annos['size'].shape[0] == annos['translation_m'].shape[0]
                assert annos['translation_m'].shape[0] == annos['name'].shape[0]
                assert annos['label'].shape[0] == annos['name'].shape[0]
                assert annos['score'].shape[0] == annos['name'].shape[0]
                assert annos['track_id'].shape[0] == annos['name'].shape[0]
                # assert False
                if empty == True:
                    print("it happens, empty frame.")
                av2_annos[scene_name].append(annos) # all tracks all timestamps in this scene
                # break
            frame_counter +=1

        # Finished tracking this worker #
        # avg track score
        av2_annos = average_scores_across_track(av2_annos)
        
        # save for this scene:
        os.makedirs(os.path.join(self.args.work_dir, f'tmp/{self.args.version}'), exist_ok=True)
        for sn in av2_annos.keys():
            if not os.path.exists(os.path.join(self.args.work_dir, f'tmp/{self.args.version}/{sn}.pkl')):
                print("saving: ", os.path.join(self.args.work_dir, f'tmp/{self.args.version}/{sn}.pkl'))
                with open(os.path.join(self.args.work_dir, f'tmp/{self.args.version}/{sn}.pkl'), "wb") as f:
                    pickle.dump({sn:av2_annos[sn]}, f)
        
        
        # print(av2_annos)
        
        end = time.time()

        second = (end-start) 

        speed=frame_size / second
        print("The speed is {} FPS".format(speed))
        
        return av2_annos


def main():
    args = parse_args()
    print('Deploy OK')
    
    with open(f'./av2_frame_meta/frames_meta_{args.version}.json', 'r', encoding="utf-8") as f:
        frames=json.load(f)['frames']
    
    if os.path.exists(os.path.join(args.work_dir, f'{args.version}_det.pkl')):
        with open(os.path.join(args.work_dir, f'{args.version}_det.pkl'), "rb") as f:
            print("loading detection results: ")
            predictions = pickle.load(f)
            
    else:
        predictions_pd = read_feather(Path(args.checkpoint))
        predictions = {}
        
        len_pd = len(predictions_pd)
        print(len_pd)
        for i in tqdm(range(len_pd)):
            info = predictions_pd.iloc[i]
            
            # 'index', 'timestamp_ns', 'track_uuid', 'category', 'length_m',
            #'width_m', 'height_m', 'qw', 'qx', 'qy', 'qz', 'tx_m', 'ty_m', 'tz_m','log_id'
            
            if info['log_id'] not in predictions.keys():
                predictions[info['log_id']] = {}
            
            if int(info['timestamp_ns']) not in predictions[info['log_id']]:
                predictions[info['log_id']][int(info['timestamp_ns'])] = []
            # print("velocity is not implemented!! All zeros!!")
            predictions[info['log_id']][int(info['timestamp_ns'])].append(
                {
                    "translation": [info['tx_m'], info['ty_m'], info['tz_m']],
                    "rotation": [info['qw'], info['qx'], info['qy'], info['qz']],
                    "size":  [info['length_m'], info['width_m'],  info['height_m']], #l,w,h
                    "velocity": [0.0, 0.0, 0.0], # todo missing in .feather from dets.
                    "detection_score": float(info['score']),
                    "detection_name": info["category"]
                }
            )
        
        # predictions:
        '''
        { log_id:
            {
                'timestamp':
                    [
                        {
                            'sample_token':
                            'translation: [x,y,z]
                            'size': [w,l,h]
                            'rotation': quaternion
                            'velocity': []
                            'detection_name': string
                            'detection_score': float
                            'attrubte_name': string
                        }
                        ...
                    ]
                }
                ...
            }
        }    
        '''
        
        with open(os.path.join(args.work_dir, f'{args.version}_det.pkl'), "wb") as f:
            pickle.dump(predictions, f)
            
            
    # target tracking output format:
    '''
    {
        <log_id>: [
                { #all info in the same timestamp are concatenated.
                    "timestamp_ns": <timestamp_ns>,
                    "track_id":array([<track_id>, <track_id>, <track_id>], dtype=int32),
                    "score": array([<score>,..], dtype=float32),
                    "label": array([<label>, <label>, <label>], dtype=int32),
                    "name": array([<name>, <name>, <name>], dtype='<U31'),
                    "translation_m": array([[x,y,z], [x,y,z], [x,y,z]...],dtype=float32),
                    "size": array([[l,w,h], [l,w,h], [l,w,h]...],dtype=float32),
                    "yaw": array([<yaw>, <yaw>, <yaw>], dtype=float32),
                    "velocity_m_per_s": array([[vx,vy,vz], [vx,vy,vz], [vx,vy,vz]...],dtype=float32)
                }
                ...
        ]
    }
    '''
    
    class_names = ('REGULAR_VEHICLE', 'PEDESTRIAN', 'BICYCLIST', 'MOTORCYCLIST', 
               'WHEELED_RIDER', 'BOLLARD', 'CONSTRUCTION_CONE', 'SIGN', 
               'CONSTRUCTION_BARREL', 'STOP_SIGN', 
               'MOBILE_PEDESTRIAN_CROSSING_SIGN','LARGE_VEHICLE', 
               'BUS', 'BOX_TRUCK', 'TRUCK', 'VEHICULAR_TRAILER', 'TRUCK_CAB', 
               'SCHOOL_BUS', 'ARTICULATED_BUS','MESSAGE_BOARD_TRAILER', 'BICYCLE', 
               'MOTORCYCLE', 'WHEELED_DEVICE', 'WHEELCHAIR', 'STROLLER', 'DOG')
    
    SAMPLE_FREQ = 1
    
    # organize all_predictions_files by scenario to parallelization
    all_frames_by_seq = {}
    all_seq_names = [] # for spliting by worker_id
    for frame in frames:
        seq_name = frame['scene_name'] # = log_id
        if seq_name not in all_seq_names:
            all_seq_names.append(seq_name)
            all_frames_by_seq[seq_name] = []
        all_frames_by_seq[seq_name].append(frame)

    print("len(set(all_seq_names)) ", len(set(all_seq_names)))
    # print("len(all_frames_by_seq['02678d04-cc9f-3148-9f95-1ba66347dff9']) ", len(all_frames_by_seq['02678d04-cc9f-3148-9f95-1ba66347dff9']))
    tracking_process = Tracking(predictions, all_frames_by_seq, all_seq_names, args, class_names, sample_freq=SAMPLE_FREQ, process_num=args.chunks)
    tracking_process.tracking()



if __name__ == '__main__':
    save_first_frame()
    main()
