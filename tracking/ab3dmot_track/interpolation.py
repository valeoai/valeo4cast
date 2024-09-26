import os
import numpy as np
import pickle
import copy
from tqdm import tqdm

def dti(pkl_path, n_min=25, n_dti=20, split='val', sub_sample_ratio=5):
    with open(pkl_path, 'rb') as f:
        all_seqs = pickle.load(f)
    with open(f"./scenario_timestamps_{split}.pkl", "rb") as f:
        scenario_timestamps = pickle.load(f)
    
    interpolated_outputs = {}
    total_interpolated = 0
    for seq_name in tqdm(all_seqs.keys()):
        seq_data = all_seqs[seq_name]
        # collect track info per track {trackid: [{"score": , "label":, "name":, "translation_m": , "size", "yaw", "velocity_m_per_s"}...], ...}
        tracks_info = {}
        all_timestamps = sorted(scenario_timestamps[seq_name])
        print("len(all_timestamps) ", len(all_timestamps))
        all_timestamps_sub_samped = all_timestamps[::sub_sample_ratio]
        print("len(all_timestamps_sub_samped) ", len(all_timestamps_sub_samped))
        for ts_info in seq_data:
            curr_timestamp = ts_info["timestamp_ns"]
            curr_seq_id = ts_info["seq_id"]
            assert seq_name == curr_seq_id
            assert ts_info['track_id'].shape[0]==ts_info['score'].shape[0]==ts_info['label'].shape[0]== \
            ts_info['name'].shape[0]==ts_info['translation_m'].shape[0]==ts_info['size'].shape[0]==ts_info['yaw'].shape[0]==ts_info['velocity_m_per_s'].shape[0]
            for track_id, score, label, name, translation_m, size, yaw, velocity_m_per_s in zip(ts_info['track_id'], ts_info['score'], ts_info['label'], 
                                                                                                ts_info['name'], ts_info['translation_m'], ts_info['size'],
                                                                                                ts_info['yaw'], ts_info['velocity_m_per_s']):
                # print("track_id ", track_id)
                # print("score ", score)
                # print("label ", label)
                # print("name ", name)
                # print("translation_m ", translation_m.shape)
                # print("size ", size)
                # print("yaw ", yaw)
                # print("velocity_m_per_s ", velocity_m_per_s)
                # print()
                
                if track_id not in tracks_info:
                    tracks_info[track_id] = []
                tracks_info[track_id].append({
                    "track_id": track_id,
                    "timestamp_ns": curr_timestamp,
                    "seq_id": curr_seq_id,
                    "score": score,
                    "label": label,
                    "translation_m": translation_m,
                    "size": size,
                    "name": name,
                    "yaw": yaw,
                    "velocity_m_per_s": velocity_m_per_s
                })
        
        # ordered list by timestamps
        ordered_tracks_info = {}
        for k,v in tracks_info.items():
            tmp = [vv["timestamp_ns"] for vv in v]
            sorted_index = np.argsort(tmp)
            # print("sorted_index ", sorted_index)
            new_v = [v[idx] for idx in sorted_index]
            ordered_tracks_info[k] = new_v
        
        tracks_info = ordered_tracks_info
        tracks_info_dti = copy.deepcopy(tracks_info)
            
        # results to insert in the vanilla result {'timestamp_ns': {'seq_id':, "score": , "label":, "name":, "translation_m": , "size", "yaw", "velocity_m_per_s", "track_id"}}
        for track_id, tracklet in tracks_info.items():
            tracklet_dti = copy.deepcopy(tracklet)
            if len(tracklet) == 0:
                continue
            n_frame = len(tracklet)
            
            n_conf = []
            timestamp_for_track = []
            frames = []
            for tt in tracklet:
                n_conf.append(tt['score'])
                timestamp_for_track.append(tt['timestamp_ns'])
                frames.append(all_timestamps.index(tt['timestamp_ns'])+1) #TODO starts with 1
            
            # print("before interpolation: ", len(timestamp_for_track))
            if n_frame > n_min and np.mean(n_conf)>0.0: # confidence >0.5
                frames_dti = []
                for i in range(0, n_frame):
                    right_frame = frames[i]
                    if i > 0:
                        left_frame = frames[i - 1]
                    else:
                        left_frame = frames[i]
                    # disconnected track interpolation
                    if 1 < right_frame - left_frame < n_dti:
                        num_bi = int(right_frame - left_frame - 1)
                        right_bbox = tracklet[i]["translation_m"]
                        left_bbox = tracklet[i - 1]["translation_m"]
                        left_yaw = tracklet[i]["yaw"]
                        right_yaw = tracklet[i - 1]["yaw"]
                        for j in range(1, num_bi + 1):
                            curr_frame = j + left_frame
                            curr_bbox = (curr_frame - left_frame) * (right_bbox - left_bbox) / \
                                        (right_frame - left_frame) + left_bbox
                            curr_yaw = (curr_frame - left_frame) * (right_yaw - left_yaw) / \
                                        (right_frame - left_frame) + left_yaw
                            curr_info = copy.deepcopy(tracklet[i - 1])
                            curr_info["yaw"] = wrap_pi(curr_yaw[np.newaxis])[0]
                            curr_info["translation_m"] = curr_bbox
                            curr_info["timestamp_ns"] = all_timestamps[curr_frame-1]
                            frames_dti.append(curr_info)
                tracklet_dti = tracklet + frames_dti
            tracks_info_dti[track_id] = tracklet_dti
            check_timestamps = [tt['timestamp_ns'] for tt in tracklet_dti]
            assert len(check_timestamps) == len(set(check_timestamps))
            # print("after interpolation: ", len(check_timestamps))
            total_interpolated += len(check_timestamps) - len(timestamp_for_track)
        
        
        # organize the tracks by timestamp as key
        frames_info = {}
        for k,v in tracks_info_dti.items():
            for vv in v:
                # sub sampling:
                if vv['timestamp_ns'] not in all_timestamps_sub_samped:
                    continue
                if vv['timestamp_ns'] not in frames_info:
                    frames_info[vv['timestamp_ns']] = []
                vv["translation_m"] =vv["translation_m"][np.newaxis]
                vv["size"] = vv["size"][np.newaxis]
                
                vv["velocity_m_per_s"] = vv["velocity_m_per_s"][np.newaxis]
                vv["score"] = np.array([float(vv["score"])], np.float32)
                vv["label"] = np.array([int(vv["label"])], np.int32)
                vv["name"] = np.array([str(vv["name"])])
                vv["yaw"]  = np.array([float(vv["yaw"])], np.float32)
                vv['track_id'] = np.array([float(vv['track_id'])], np.int64)
                frames_info[vv['timestamp_ns']].append(vv)
        
        # merge all info with the same timestamps
        interpolated_output = []
        for k,v in frames_info.items():
            concatenated = concatenate_array_values(v)
            # print("concatenated ", concatenated.keys())
            # for k,v in concatenated.items():
            #     if k not in ["seq_id", "timestamp_ns"]:
            #         print(k)
            #         print(k, v.shape)
            # print("concatenated['translation_m'].shape ", concatenated['translation_m'].shape)
            # print("concatenated['size'].shape ", concatenated['size'].shape)
            # print("concatenated['velocity_m_per_s'].shape ", concatenated['velocity_m_per_s'].shape)
            interpolated_output.append(concatenated)
        interpolated_outputs[seq_name] = interpolated_output
    
    print("total_interpolated ", total_interpolated)
    if sub_sample_ratio == 1:
        save_path = pkl_path.replace('.pkl', '_interpolated.pkl')
    else:
        save_path = pkl_path.replace('.pkl', '_interpolated_sub_sampled.pkl')
    with open(save_path, 'wb') as f:
        print("interpolated results saved to ", save_path)
        pickle.dump(interpolated_outputs, f)
    

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Union
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


# dti(pkl_path="cbevfusion_54_ab3dmot_greedy/10Hz/test_tracking_result.pkl",
#     n_min=25, n_dti=20, split='test', sub_sample_ratio=1)
    
dti(pkl_path="PathToTrackingResult", 
    n_min=25, n_dti=20, split='test', sub_sample_ratio=1)