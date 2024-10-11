import os
import pickle
import shutil
from collections import defaultdict
from multiprocessing import Pool
from uuid import UUID
import numpy as np
import torch
from metadrive.scenario.scenario_description import MetaDriveType
from scenarionet.common_utils import read_scenario, read_dataset_summary
from torch.utils.data import Dataset
from tqdm import tqdm

from unitraj.datasets import common_utils
from unitraj.datasets.common_utils import get_polyline_dir, find_true_segments, generate_mask, is_ddp, \
    get_kalman_difficulty, get_trajectory_type, interpolate_polyline
from unitraj.datasets.dataset_types import object_type, polyline_type
from unitraj.utils.visualization import check_loaded_data
import copy
import lap

default_value = 0
object_type = defaultdict(lambda: default_value, object_type)
polyline_type = defaultdict(lambda: default_value, polyline_type)

CLASS_NAMES = ('REGULAR_VEHICLE', 'PEDESTRIAN', 'BICYCLIST', 'MOTORCYCLIST', 'WHEELED_RIDER', 
               'BOLLARD', 'CONSTRUCTION_CONE', 'SIGN', 'CONSTRUCTION_BARREL', 'STOP_SIGN', 
               'MOBILE_PEDESTRIAN_CROSSING_SIGN','LARGE_VEHICLE', 'BUS', 'BOX_TRUCK', 'TRUCK',
               'VEHICULAR_TRAILER', 'TRUCK_CAB', 'SCHOOL_BUS', 'ARTICULATED_BUS','MESSAGE_BOARD_TRAILER', 
               'BICYCLE', 'MOTORCYCLE', 'WHEELED_DEVICE', 'WHEELCHAIR', 'STROLLER', 'DOG')

CATEGORY_TO_VELOCITY_M_PER_S = {
    "ARTICULATED_BUS": 4.58,
    "BICYCLE": 0.97,
    "BICYCLIST": 3.61,
    "BOLLARD": 0.02,
    "BOX_TRUCK": 2.59,
    "BUS": 3.10,
    "CONSTRUCTION_BARREL": 0.03,
    "CONSTRUCTION_CONE": 0.02,
    "DOG": 0.72,
    "LARGE_VEHICLE": 1.56,
    "MESSAGE_BOARD_TRAILER": 0.41,
    "MOBILE_PEDESTRIAN_CROSSING_SIGN": 0.03,
    "MOTORCYCLE": 1.58,
    "MOTORCYCLIST": 4.08,
    "PEDESTRIAN": 0.80,
    "REGULAR_VEHICLE": 2.36,
    "SCHOOL_BUS": 4.44,
    "SIGN": 0.05,
    "STOP_SIGN": 0.09,
    "STROLLER": 0.91,
    "TRUCK": 2.76,
    "TRUCK_CAB": 2.36,
    "VEHICULAR_TRAILER": 1.72,
    "WHEELCHAIR": 1.50,
    "WHEELED_DEVICE": 0.37,
    "WHEELED_RIDER": 2.03,
}


class BaseDataset(Dataset):

    def __init__(self, config=None, is_validation=False):
        if is_validation:
            self.data_path = config['val_data_path']
        else:
            self.data_path = config['train_data_path']
        self.is_validation = is_validation
        self.config = config
        if self.config["drop_no_future"] == True:
            print("!!!!!!!!!!!!!!Warning: you drop all objects THAT DOES NOT EXIST AT T+1 (GOOD CHOICE FOR NOISY INPUTS???). Might cause FN in forecasting with noisy inputs.!!!!!!!!!!!!!!!!")
        self.data_loaded_memory = []
        self.data_chunk_size = 8
        self.load_data()

    def load_data(self):
        self.data_loaded = {}
        if self.is_validation:
            print('Loading validation data...')
        else:
            print('Loading training data...')
        # refine past collection #
        if "refine_past_res_train" in self.config  or "refine_past_res_val" in self.config :
            if '/val/' in self.data_path[0] or '/test/' in self.data_path[0]:
                print("loading ", self.config["refine_past_res_val"])
                with open(self.config["refine_past_res_val"], 'rb') as f:
                    self.refined_pasts = pickle.load(f)
            else:
                print("loading ", self.config["refine_past_res_train"])
                with open(self.config["refine_past_res_train"], 'rb') as f:
                    self.refined_pasts = pickle.load(f)
        else:   
            self.refined_pasts = None
        for cnt, data_path in enumerate(self.data_path):
            dataset_name = data_path.split('/')[-1]
            # data_path = 
           
            self.cache_path = os.path.join(data_path, f'cache_test_{self.config.method.model_name}')
            print('cache to: ', self.cache_path)
            data_usage_this_dataset = self.config['max_data_num'][cnt]
            data_usage_this_dataset = int(data_usage_this_dataset / self.data_chunk_size)

            _, summary_list, mapping = read_dataset_summary(data_path)

            if self.config["overwrite_cache"]:
                shutil.rmtree(self.cache_path)

            # Create the cache directory if it doesn't exist
            if not os.path.exists(self.cache_path):
                os.makedirs(self.cache_path, exist_ok=True)

            process_num = 8 #os.cpu_count() - 1
            #process_num = 1 #os.cpu_count() - 1
            if self.config["debug_challenge"]:
                process_num = 1
            print('Using {} processes to load data...'.format(process_num))

            # If needed, cut the summary_list to make quick tests.
            # The length is the number of scenarios (e.g. 150 for the val set)
            if self.config["debug_max_nb_scenarios"] > 0:
                summary_list = summary_list[:self.config["debug_max_nb_scenarios"]]

            data_splits = np.array_split(summary_list, process_num)

            data_splits = [(data_path, mapping, list(data_splits[i]), dataset_name) for i in range(process_num)]
            # save the data_splits in a tmp directory
            self.tmp_directory = os.path.join(self.cache_path, "tmp")
            os.makedirs(self.tmp_directory, exist_ok=True)
            if len(self.config.devices) == 1: #TODO update xyh is it correct?
                for i in range(process_num):
                    with open(os.path.join(self.tmp_directory, '{}.pkl'.format(i)), 'wb') as f:
                        print("dump")
                        pickle.dump(data_splits[i], f)

            if self.config["debug_challenge"] or len(self.config.devices) > 1:
                # Just one thread to process the data, just one worker
                
                # 
                file_list = {}
                for fake_worker_index in range(process_num): #8 because the cache was generated by 8 processes.
                    res = self.process_data_chunk(fake_worker_index)
                    file_list.update(res)
                    print("len(file_list) ", len(file_list))
            else:
                # Parallelize data preprocessing
                with Pool(processes=process_num) as pool:
                    results = pool.map(self.process_data_chunk, list(range(process_num)))
                    # concatenate all the results
                    file_list = {}
                    for result in results:
                        file_list.update(result)
            if len(self.config.devices) == 1: #TODO update xyh is it correct?
                with open(os.path.join(self.cache_path, 'file_list.pkl'), 'wb') as f:
                    print("dump file_list.pkl")
                    pickle.dump(file_list, f)

            data_list = list(file_list.items())
            # Shuffle the list
            np.random.shuffle(data_list)
            # Take a subset of the training data
            if not self.is_validation:
                # randomly sample data_usage number of data
                file_list = dict(data_list[:data_usage_this_dataset])

            print('Loaded {} samples from {}'.format(len(file_list) * self.data_chunk_size, data_path))
            self.data_loaded.update(file_list)

        self.data_loaded_keys = list(self.data_loaded.keys())
        print('Data loaded')

    def process_data_chunk(self, worker_index):

        # Load the tmp dispatch of the files to the workers
        with open(os.path.join(self.tmp_directory, '{}.pkl'.format(worker_index)), 'rb') as f:
            data_chunk = pickle.load(f)
        file_list = {}
        # data_path is the raw path for scenarionet folder
        # mapping is a dictionary {scenario_id: <worker_index>} for all scenarios
        # data_list if a list of all scenarios to be processed by the current worker_index
        # dataset_name is the split (e.g., "val")
        data_path, mapping, data_list, dataset_name = data_chunk

        for cnt, file_name in enumerate(data_list):
            if self.config["debug_challenge"]:
                if "87ca3d9f-f317-3efb-b1cb-aaaf525227e5" not in file_name:
                    continue
                print("processing:", file_name)
            if worker_index == 0 and cnt % max(int(len(data_list) / 10), 1) == 0:
                # This info is not very reliable with the new caching mechanism as the first worker will process all the data
                print(f'{cnt}/{len(data_list)} data processed', flush=True)

            # Check if the scenario has been full prepocessed in the cache
            scenario_summary_path = os.path.join(self.cache_path, f'{file_name}___scenario_summary.pkl')
            if os.path.exists(scenario_summary_path):
                print(f"scenario already processed: {file_name}")
                with open(scenario_summary_path, "rb") as f:
                    scenario_summary = pickle.load(f)
                file_list.update(scenario_summary)
                continue

            # If not cache #
            if self.config["debug_challenge"]:
                print("reading scenario...")
            # print("data_path: ", data_path)
            # print("mapping: ", mapping)
            # print("file_name: ", file_name)
            # print()
            scenario = read_scenario(data_path, mapping, file_name)

            """
            Preprocess the scenario, especially the map, dynamic map and trajectories
            Scenario-centric view
            Returns the scenario processed as dictionary
            """
            if self.config["debug_challenge"]:
                print("preprocessing...")
            output = self.preprocess(scenario)
            
            # update xyh
            if "need_match_with_gt" in self.config and self.config['need_match_with_gt']:
                if '/val/' in data_path and 'merge' not in data_path:
                    #print("in GT val.")
                    data_path_GT = "/Path/TO/argoverse2_sensor_scenarios/val/"
                    _, _, mapping_GT = read_dataset_summary(data_path_GT)
                elif '/train/' in data_path and 'merge' not in data_path:
                    data_path_GT = "/Path/TO/argoverse2_sensor_scenarios/train/"
                    #print("in GT train.")
                    _, _, mapping_GT = read_dataset_summary(data_path_GT)
                else:
                    #print("in GT merge.")
                    data_path_GT = "/Path/TO/argoverse2_sensor_scenarios/merge/"
                    mapping_GT = {}
                    _, _, mapping_GT_train = read_dataset_summary("/Path/TO/argoverse2_sensor_scenarios/train/")
                    for k,v in mapping_GT_train.items():
                        assert k not in mapping_GT
                        mapping_GT[k] = 'train/' + v + '/'
                        
                    _, _, mapping_GT_val = read_dataset_summary("/Path/TO/argoverse2_sensor_scenarios/val/")
                    for k,v in mapping_GT_val.items():
                        assert k not in mapping_GT
                        mapping_GT[k] = 'val/' + v + '/'
                #print("len(mapping_GT) ", len(set(mapping_GT.keys())))
                scenario_gt = read_scenario(data_path_GT, mapping_GT, file_name)
                # print(scenario_gt)
                output_GT = self.preprocess(scenario_gt)
            else:
                output_GT = None

            """
            Process the scenario to convert it to the right coordinate system
            Put it in agent-centric view, at the right timestamp to be predicted
            Agent-centric view
            returns a list, where each element is a trajectory to forecast (along with all its context)
            """
            if self.config["debug_challenge"]:
                print("processing...")
            """
            # refine past collection #
            if "refine_past_res_train" in self.config  or "refine_past_res_val" in self.config :
                if '/val/' in data_path or '/test/' in data_path:
                    print("loading ", self.config["refine_past_res_val"])
                    with open(self.config["refine_past_res_val"], 'rb') as f:
                        refined_pasts = pickle.load(f)
                else:
                    print("loading ", self.config["refine_past_res_train"])
                    with open(self.config["refine_past_res_train"], 'rb') as f:
                        refined_pasts = pickle.load(f)
                    
            else:
                refined_pasts = None
            """    
            output = self.process(output, output_GT, self.refined_pasts)

            """
            Add the kalman difficulty and trajectory type info
            Returns the scenario processed as list
            """
            if self.config["debug_challenge"]:
                print("postprocessing...")
            output = self.postprocess(output)

            if output is None: continue
            nb_forecasts_to_make = len(output)
            scenario_summary = {}


            # Cache the scenario, save by chunk_size of 8
            save_cnt = 0
            while len(output) >= self.data_chunk_size:
                save_path = os.path.join(self.cache_path, f'{file_name}___{save_cnt}.pkl')
                to_save = output[:self.data_chunk_size]
                # print("chunk traj tpye: ", [chunk_elem['center_objects_traj_type'] for chunk_elem in to_save])
                output = output[self.data_chunk_size:]
                kalman_difficulty = np.stack([x['kalman_difficulty'] for x in to_save])
                file_info = {}
                file_info['kalman_difficulty'] = kalman_difficulty
                file_info['sample_num'] = len(to_save)
                scenario_summary[save_path] = file_info
                with open(save_path, 'wb') as f:
                    pickle.dump(to_save, f)
                save_cnt += 1

            # Save the last cut from it
            if len(output) > 0:
                save_path = os.path.join(self.cache_path, f'{file_name}___{save_cnt}.pkl')
                kalman_difficulty = np.stack([x['kalman_difficulty'] for x in output])
                file_info = {}
                file_info['kalman_difficulty'] = kalman_difficulty
                file_info['sample_num'] = len(output)
                scenario_summary[save_path] = file_info
                with open(save_path, 'wb') as f:
                    pickle.dump(output, f)

            # Save that the scenario is well processed
            with open(scenario_summary_path, "wb") as f:
                pickle.dump(scenario_summary, f)
            print(f"scenario processed with {nb_forecasts_to_make} forecasts to make: {file_name}")

        return file_list

    def preprocess_static_map(self, scenario):

        # x,y,z,type
        map_infos = {
            'lane': [],
            'road_line': [],
            'road_edge': [],
            'stop_sign': [],
            'crosswalk': [],
            'speed_bump': [],
        }
        polylines = []
        point_cnt = 0
        for k, v in scenario['map_features'].items():
            type_poly = polyline_type[v['type']]
            if type_poly == 0:
                continue
            #TODO update xyh no polyline, skip
            if 'polyline' not in v:
                # print("noployline")   
                continue
            #TODO update xyh nan value in polyline, skip
            if np.isnan(v['polyline']).any():
                # print("isnan")
                continue
            
            cur_info = {'id': k}
            cur_info['type'] = v['type']
            if type_poly in [1, 2, 3]:
                cur_info['speed_limit_mph'] = v.get('speed_limit_mph', None)
                cur_info['interpolating'] = v.get('interpolating', None)
                cur_info['entry_lanes'] = v.get('entry_lanes', None)
                try:
                    cur_info['left_boundary'] = [{
                        'start_index': x['self_start_index'], 'end_index': x['self_end_index'],
                        'feature_id': x['feature_id'],
                        'boundary_type': 'UNKNOWN'  # roadline type
                    } for x in v['left_neighbor']
                    ]
                    cur_info['right_boundary'] = [{
                        'start_index': x['self_start_index'], 'end_index': x['self_end_index'],
                        'feature_id': x['feature_id'],
                        'boundary_type': 'UNKNOWN'  # roadline type
                    } for x in v['right_neighbor']
                    ]
                except:
                    cur_info['left_boundary'] = []
                    cur_info['right_boundary'] = []
                polyline = v['polyline']
                polyline = interpolate_polyline(polyline)
                map_infos['lane'].append(cur_info)
            elif type_poly in [6, 7, 8, 9, 10, 11, 12, 13]:
                polyline = v['polyline']
                polyline = interpolate_polyline(polyline)
                map_infos['road_line'].append(cur_info)
            elif type_poly in [15, 16]:
                polyline = v['polyline']
                polyline = interpolate_polyline(polyline)
                cur_info['type'] = 7
                map_infos['road_line'].append(cur_info)
            elif type_poly in [17]:
                cur_info['lane_ids'] = v['lane']
                cur_info['position'] = v['position']
                map_infos['stop_sign'].append(cur_info)
                polyline = v['position'][np.newaxis]
            elif type_poly in [18]:
                map_infos['crosswalk'].append(cur_info)
                polyline = v['polygon']
            elif type_poly in [19]:
                map_infos['crosswalk'].append(cur_info)
                polyline = v['polygon']
            if polyline.shape[-1] == 2:
                polyline = np.concatenate((polyline, np.zeros((polyline.shape[0], 1))), axis=-1)
            try:
                cur_polyline_dir = get_polyline_dir(polyline)
                type_array = np.zeros([polyline.shape[0], 1])
                type_array[:] = type_poly
                cur_polyline = np.concatenate((polyline, cur_polyline_dir, type_array), axis=-1)
            except:
                cur_polyline = np.zeros((0, 7), dtype=np.float32)
            polylines.append(cur_polyline)
            cur_info['polyline_index'] = (point_cnt, point_cnt + len(cur_polyline))
            point_cnt += len(cur_polyline)

        try:
            polylines = np.concatenate(polylines, axis=0).astype(np.float32)
        except:
            polylines = np.zeros((0, 7), dtype=np.float32)
        map_infos['all_polylines'] = polylines

        return map_infos

    def preprocess_track_info(self, scenario):

        track_infos = {
            'object_id': [],
            'object_type': [],  # {0: unset, 1: vehicle, 2: pedestrian, 3: cyclist, 4: others}
            'trajs': [],
            'timestamps': [],
            #TODO update xyh, add REAL object class label
            'object_class_label': [],
            'tracking_scores': []
        }

        timestamps = []
        for k, v in scenario['tracks'].items():

            state = v['state']
            
            timestamps += state["timestamps"].tolist()
            
            
            for key, value in state.items():
                if len(value.shape) == 1 and key != 'tracking_scores':
                    state[key] = np.expand_dims(value, axis=-1)
            all_state = [state['position'], state['length'], state['width'], state['height'], state['heading'],
                         state['velocity'], state['valid']]
            # type, x,y,z,l,w,h,heading,vx,vy,valid
            all_state = np.concatenate(all_state, axis=-1)

            # Unitraj original code use to pad and cut to the cocrrect time length
            # we do that in the process() function instead
            # so that we can loop on the frames to be predicted in the same scenario

            track_infos['object_id'].append(k)
            # print(v['type'])
            track_infos['object_type'].append(object_type[v['type']]) # UniTraj object_type
            #TODO update xyh, add REAL object class label
            track_infos['object_class_label'].append(v['object_label'])
            # print(state.keys())
            if 'tracking_scores' not in state:
                track_infos['tracking_scores'].append(np.zeros((state['position'].shape[0],)))   
            else: 
                track_infos['tracking_scores'].append(state['tracking_scores'])
            
            track_infos['trajs'].append(all_state)


        track_infos['tracking_scores'] = np.stack(track_infos['tracking_scores'], axis=0)
        # print("track_infos['tracking_scores'].shape ", track_infos['tracking_scores'].shape)
        track_infos['trajs'] = np.stack(track_infos['trajs'], axis=0)
        timestamps = list(set(timestamps))
        timestamps = sorted([ts for ts in timestamps if ts !=-1])
        track_infos['timestamps'] = timestamps
        return track_infos

    def preprocess_dynamic_map(self, scenario):
        dynamic_map_infos = {
            'lane_id': [],
            'state': [],
            'stop_point': []
        }
        for k, v in scenario['dynamic_map_states'].items():  # (num_timestamp)
            lane_id, state, stop_point = [], [], []
            for cur_signal in v['state']['object_state']:  # (num_observed_signals)
                lane_id.append(str(v['lane']))
                state.append(cur_signal)
                stop_point.append(v['stop_point'].tolist())
            # lane_id = lane_id[::sample_inverval]
            # state = state[::sample_inverval]
            # stop_point = stop_point[::sample_inverval]

            # The dynamic state are cut here, while we shouldn't
            assert False

#            lane_id = lane_id[:total_steps]
#            state = state[:total_steps]
#            stop_point = stop_point[:total_steps]
            dynamic_map_infos['lane_id'].append(np.array([lane_id]))
            dynamic_map_infos['state'].append(np.array([state]))
            dynamic_map_infos['stop_point'].append(np.array([stop_point]))

        return dynamic_map_infos


    def preprocess(self, scenario):
        """
        heavy modifications from the original unitraj
        especially trajectories are not cut here now
        """

        # Process the static map
        map_infos = self.preprocess_static_map(scenario)

        #total_steps = self.config["past_len"] + self.config["future_len"]
        assert self.config["trajectory_sample_interval"] == 1
        #frequency_mask = generate_mask(self.config["past_len"] - 1, total_steps, self.config["trajectory_sample_interval"])

        # Process track info
        #track_infos = self.preprocess_track_info(scenario, total_steps, frequency_mask)
        track_infos = self.preprocess_track_info(scenario) #, total_steps)

        # Process the dynamic map
        dynamic_map_infos = self.preprocess_dynamic_map(scenario) #, total_steps)

        ret = {
            'track_infos': track_infos,
            'dynamic_map_infos': dynamic_map_infos,
            'map_infos': map_infos
        }

        #scenario['metadata']['ts'] = scenario['metadata']['ts'][:total_steps]
        ret.update(scenario['metadata'])

        ret['timestamps_seconds'] = ret.pop('ts')
        ret['current_time_index'] = self.config['past_len'] - 1
        ret['sdc_track_index'] = track_infos['object_id'].index(ret['sdc_id']) # The ego-vehicle track id
        if self.config['only_train_on_ego'] or ret.get('tracks_to_predict', None) is None:
            tracks_to_predict = {
                'track_index': [ret['sdc_track_index']],
                'difficulty': [0],
                'object_type': [MetaDriveType.VEHICLE]
            }
        else:
            sample_list = list(ret['tracks_to_predict'].keys())  # + ret.get('objects_of_interest', [])
            sample_list = list(set(sample_list))

            tracks_to_predict = {
                'track_index': [track_infos['object_id'].index(id) for id in sample_list if id in track_infos['object_id']],
                'object_type': [track_infos['object_type'][track_infos['object_id'].index(id)] for id in sample_list if id in track_infos['object_id']],
            }
        # print("tracks_to_predict: ", tracks_to_predict['object_type'])
        ret['tracks_to_predict'] = tracks_to_predict

        ret['map_center'] = scenario['metadata'].get('map_center', np.zeros(3))[np.newaxis]
        return ret

    def process(self, internal_format, internal_format_GT, refined_pasts=None):

        info = internal_format
        scene_id = info['scenario_id']
        
        if "need_match_with_gt" in self.config and self.config["need_match_with_gt"]:
            assert scene_id == internal_format_GT['scenario_id']
            track_infos_GT = internal_format_GT['track_infos']
            obj_trajs_full_GT = track_infos_GT['trajs']
            assert len(track_infos_GT['object_class_label']) == obj_trajs_full_GT.shape[0]
            
            # print("obj_trajs_full_GT.shape ", obj_trajs_full_GT.shape) # (116, 157, 10)

        sdc_track_index = info['sdc_track_index']
        track_infos = info['track_infos']
        track_index = np.array(info['tracks_to_predict']['track_index'])
        obj_types = np.array(track_infos['object_type'])
        # print("obj_types ", obj_types)
        obj_trajs_full = track_infos['trajs']  # (num_objects, all_timestamp, 10)
        
        
            
        
        
        assert len(track_infos['object_class_label']) == obj_trajs_full.shape[0]
        nb_all_agent, nb_all_ts, feat_dim = obj_trajs_full.shape

        past_len = self.config["past_len"]
        future_len = self.config["future_len"]

        # WARNING hard coded 10Hz
        timestamps = np.arange(0, past_len/10, 0.1, dtype=np.float32)

        center_objects_final = []
        center_objects_traj_type = []
        center_objects_label = []
        track_index_to_predict_final = []
        obj_trajs_data_final = []
        obj_trajs_mask_final = []
        obj_trajs_pos_final = []
        obj_trajs_last_pos_final = []
        obj_trajs_future_state_final = []
        obj_trajs_future_mask_final = []
        center_gt_trajs_final = []
        center_gt_trajs_mask_final = []
        center_gt_final_valid_idx_final = []
        track_index_to_predict_new_final = []
        center_gt_trajs_src_final = []
        current_time_indices = []
        
        
        #TODO xyh update:
        track_scores = track_infos['tracking_scores']
        assert len(track_scores)  == obj_trajs_full.shape[0]
        # print("track_scores.shape ", track_scores.shape) #[#objects, timesteps]
        assert len(track_infos['object_id']) == obj_trajs_full.shape[0]
        
        
        
       
        assert len(track_infos['timestamps']) == nb_all_ts
        # Looping over all the valid indices for the challenge Important, sliding window implementation
        all_time_indices = np.arange(0, nb_all_ts, 5) # All 5 frames (i.e. 2Hz)
        #all_time_indices = [20]
        # sliding window thing
        # if self.config["debug_challenge"]:
        #     print("len(track_infos['timestamps']) ", len(track_infos['timestamps']))
        #     print("obj_trajs_full.shape ", obj_trajs_full.shape)
        #     print("nb_all_ts ", nb_all_ts)
        #     print("obj_trajs_full shape is", obj_trajs_full.shape)
        #     print("object ids: ",track_infos["object_id"])
        #     print("target pos: ", obj_trajs_full[20, 35]) # correct here.
        for time_idx, current_time_index in enumerate(all_time_indices):
            # print("current_time_index : ", current_time_index)
            #TODO update xyh: drop last frame which surely have no future
            if time_idx == len(all_time_indices)-1: # be careful, drop last frame = last frame in SUB SAMPLED timesteps. see forecasting eval.
                if self.config["debug_challenge"]:
                    print("drop last frame.")
                continue
            # if self.config["debug_challenge"]:
            #     print("current_time_index ", current_time_index)
            #     print("current timestamps: ", track_infos['timestamps'][current_time_index])
            
                
            # if self.config["debug_challenge"]:
            #     print("time index:", current_time_index)

            # center_objects is (num_objects_of_interest, 10) # t=0 status
            # track_index_to_predict is (num_objects_of_interest,)
            # Get the agents of interest, if they are valid at the given timestep
            center_objects, track_index_to_predict = self.get_interested_agents(
                track_index_to_predict=track_index,
                obj_trajs_full=obj_trajs_full,
                current_time_index=current_time_index,
                obj_types=obj_types, scene_id=scene_id
            )

            #print("%d objects of interest at timestep %d for scene %s" % (center_objects.shape[0], current_time_index, scene_id))

            # Extract the available past and future
            # TODO check padding value
            starting_past_ts = max(0, current_time_index+1-past_len)
            # TODO modify here for training with noisy past and matched GT future
            ending_future_ts = min(nb_all_ts, current_time_index+1+future_len)
            
            if refined_pasts is not None:
                #TODO need to insert the predicted past here:
                refined_past_timestep = refined_pasts[scene_id][track_infos['timestamps'][current_time_index]]
                forecast_by_object_id = {}
                for forecast_info in refined_past_timestep:
                    
                    # print("forecast_info['prediction_m'].shape", forecast_info['prediction_m'].shape)
                    # print("forecast_info['score'] ", forecast_info['score'])
                    #TODO refine past: currently select only the best performed forecast of the past
                    index_best_forecast = np.argmax(forecast_info['score'])
                    forecast_by_object_id[forecast_info["instance_id"]] = forecast_info["prediction_m"][index_best_forecast] # [60, 2]
                #print("obj_trajs_full.shape ", obj_trajs_full.shape)
            else:
                forecast_by_object_id = None

            #print("forecast_by_object_id.keys() ", forecast_by_object_id.keys())
            # 10: x,y,z,l,w,h,heading,vx,vy,valid
            
            obj_trajs_past_at_current_time = copy.deepcopy(obj_trajs_full[:, starting_past_ts:current_time_index+1])
            obj_trajs_future_at_current_time = copy.deepcopy(obj_trajs_full[:, current_time_index+1:ending_future_ts])
            
            
            
            
            #TODO update xyh
            # save pred traj types: 0=static, 1=linear, 2=non linear, -1=unknown
            obj_pred_traj_types = np.zeros((obj_trajs_full.shape[0],), dtype=np.int32) - 1
            
            object_class_label_at_current_time = np.array(track_infos['object_class_label'], dtype=np.int32)
            track_scores_past = copy.deepcopy(track_scores[:, starting_past_ts:current_time_index+1])
            # print(object_class_label_at_current_time)
            # replace by refined past before matching
            current_object_ids = np.array(track_infos['object_id'])
            if refined_pasts is not None:
                #update past trajs with forecast past results
                for current_object_idx, current_object_id in enumerate(current_object_ids):
                    if current_object_id in forecast_by_object_id:
                        #pad
                        #last_available_index = int(obj_trajs_past_at_current_time[current_object_idx, :, -1].sum())
                        #print("before: ", obj_trajs_past_at_current_time[current_object_idx, :, :])
                        obj_trajs_past_at_current_time[current_object_idx, :-1, :2] = forecast_by_object_id[current_object_id][:obj_trajs_past_at_current_time.shape[1]-1,:]
                        obj_trajs_past_at_current_time[current_object_idx, :, -1] = 1

             
            # print("before ops: ", obj_trajs_full.shape[0])
            if "need_match_with_gt" in self.config and self.config["need_match_with_gt"]:
                assert obj_trajs_full_GT.shape[1] == obj_trajs_full.shape[1]
                obj_trajs_past_at_current_time_GT = copy.deepcopy(obj_trajs_full_GT[:, starting_past_ts:current_time_index+1])
                obj_trajs_future_at_current_time_GT = copy.deepcopy(obj_trajs_full_GT[:, current_time_index+1:ending_future_ts])
                
                # matching here#
                # matched on the past:
                # one to many matching:
                # print("obj_trajs_past_at_current_time.shape ", obj_trajs_past_at_current_time.shape)
                # print("obj_trajs_past_at_current_time_GT.shape ", obj_trajs_past_at_current_time_GT.shape)
                
                # get preds valid at current time stamps
                valid_past_preds_idx = np.where(obj_trajs_past_at_current_time[:,-1, -1]==1)[0] # indices in the origianl object_trajs
                # print("valid_past_preds_idx: ", len(valid_past_preds_idx))
                valid_past_preds = obj_trajs_past_at_current_time[valid_past_preds_idx,:,:]
                
                # get GTs valid at current time stamps
                valid_past_gt_idx = np.where(obj_trajs_past_at_current_time_GT[:,-1, -1]==1)[0] # indices in the origianl GT_object_trajs
                valid_past_gts =  obj_trajs_past_at_current_time_GT[valid_past_gt_idx,:,:]
                # print("valid_past_gts: ", valid_past_gts.shape)
                # print("valid_past_preds: ", valid_past_preds.shape)
                
                validity_past_gts = valid_past_gts[:, :, -1].astype(np.float32) # [#gts, #past]
                assert validity_past_gts.sum() > 0.0 # at least available at current timestep
                

                if "one_to_one" in self.config and self.config["one_to_one"]:
                    past_dists = []
                    # get distance 
                    for valid_idx, valid_past_pred in enumerate(valid_past_preds): 
                        # valid_past_pred of shape [#past, 10]
                        validity_past_pred = valid_past_pred[:, -1].astype(np.float32) # [#past]
                        
                        # print("validity_past_gts.shape ", validity_past_gts.shape)
                        # print('validity_past_pred.shape', validity_past_pred.shape)
                        # print("valid_past_gts.shape ", valid_past_gts.shape)
                        # print('valid_past_pred.shape', valid_past_pred.shape)
                        # print()
                        assert validity_past_gts.shape[-1] == validity_past_pred.shape[-1]
                        
                        # always available at t=0
                        assert validity_past_pred.sum() > 0.0
                        
                        # only calculate the dist for timestamps where both are available
                        joint_mask = validity_past_pred[np.newaxis, :] * validity_past_gts
                        
                        if 'starting_point' in self.config and self.config['starting_point']: # only consider t=0
                            joint_mask[:, :-1] = 0.0
                            #print("joint_mask.shape ", joint_mask.shape)
                            #print("joint_mask.sum(-1) ", joint_mask.sum(-1))
                        assert joint_mask.shape[0] == valid_past_gts.shape[0]
                        assert any(joint_mask.sum(-1)>0.0) == True
                        # :2 => (x,y)
                        past_dist = np.linalg.norm((valid_past_pred[None, :, :2]*joint_mask[:, :, None]) - valid_past_gts[:,:,:2]*joint_mask[:,:,None], axis=-1).sum(-1)/joint_mask.sum(-1)
                        past_dists.append(past_dist)
                    past_dists = np.array(past_dists).astype(np.float32)
                    #print("past_dists.shape ", past_dists.shape)
                    #print("len(valid_past_preds) ", len(valid_past_preds))
                    #print("len(validity_past_gts) ", len(validity_past_gts))
                    #print()
                    assert past_dists.shape[0] == len(valid_past_preds) and past_dists.shape[1] == len(validity_past_gts)
                    matches = []
                    cost, x, y = lap.lapjv(past_dists, extend_cost=True, cost_limit=2)
                    for ix, mx in enumerate(x):
                        if mx >= 0: # matched
                            matches.append([ix, mx])
                            best_pred_idx, best_gt_idx = ix, mx

                            GT_index_in_full = valid_past_gt_idx[best_gt_idx]
                            pred_index_in_full = valid_past_preds_idx[best_pred_idx]
                            # update pred future with GT future for training
                            if "train_past" in self.config and self.config["train_past"]:
                                # print("obj_trajs_past_at_current_time_GT.shape ", obj_trajs_past_at_current_time_GT.shape)
                                selected_past = copy.deepcopy(obj_trajs_past_at_current_time_GT[GT_index_in_full,:,:])
                                # print("selected_past.shape: ", selected_past.shape)

                                fake_future = np.zeros_like(obj_trajs_future_at_current_time_GT[GT_index_in_full,:,:])
                                # print("fake_future.shape: ", fake_future.shape)
                                # print()
                                # fill fake future with matched GT past traj
                                fake_future[:min(fake_future.shape[0],selected_past.shape[0]),:] = selected_past[:min(fake_future.shape[0],selected_past.shape[0])]
                                # Predt (t=-2, t=-1, t=0) => GT (t=-2, t=-1, t=0)
                                obj_trajs_future_at_current_time[pred_index_in_full,:,:] = fake_future
                            else:
                                obj_trajs_future_at_current_time[pred_index_in_full,:,:] = copy.deepcopy(obj_trajs_future_at_current_time_GT[GT_index_in_full,:,:])
                        else: # not matched
                            best_pred_idx = ix 
                            pred_index_in_full = valid_past_preds_idx[best_pred_idx]
                            #xyh update 03/06/2024, all future invalide
                            obj_trajs_future_at_current_time[pred_index_in_full,:,-1] = 0
                    assert len(matches) >0 # has at least one match
                else:    
                    # matching one pred with all GTs, one to many matching 
                    for valid_idx, valid_past_pred in enumerate(valid_past_preds): 
                        # valid_past_pred of shape [#past, 10]
                        validity_past_pred = valid_past_pred[:, -1].astype(np.float32) # [#past]
                        
                        # print("validity_past_gts.shape ", validity_past_gts.shape)
                        # print('validity_past_pred.shape', validity_past_pred.shape)
                        # print("valid_past_gts.shape ", valid_past_gts.shape)
                        # print('valid_past_pred.shape', valid_past_pred.shape)
                        # print()
                        assert validity_past_gts.shape[-1] == validity_past_pred.shape[-1]
                        
                        # always available at t=0
                        assert validity_past_pred.sum() > 0.0
                        
                        # only calculate the dist for timestamps where both are available
                        joint_mask = validity_past_pred[np.newaxis, :] * validity_past_gts
                        
                        if 'starting_point' in self.config and self.config['starting_point']: # only consider t=0
                            joint_mask[:, :-1] = 0.0
                            #print("joint_mask.shape ", joint_mask.shape)
                            #print("joint_mask t=0 ", joint_mask)



                        assert joint_mask.shape[0] == valid_past_gts.shape[0]
                        assert any(joint_mask.sum(-1)>0.0) == True
                        # :2 => (x,y)
                        past_dist = np.linalg.norm((valid_past_pred[None, :, :2]*joint_mask[:, :, None]) - valid_past_gts[:,:,:2]*joint_mask[:,:,None], axis=-1).sum(-1)/joint_mask.sum(-1)
                        # print(past_dist.shape)
                        min_past_dist = np.min(past_dist)
                        min_past_dist_idx = np.argmin(past_dist)
                        # print(min_past_dist)
                        GT_index_in_full = valid_past_gt_idx[min_past_dist_idx]
                        pred_index_in_full = valid_past_preds_idx[valid_idx]
                        
                        # min_dist and GT has future at t+1
                        if min_past_dist <= 2.0:
                            # print("matched: ", min_past_dist)
                            # update noisy future with GT future
                            if "train_past" in self.config and self.config["train_past"]:
                                # print("obj_trajs_past_at_current_time_GT.shape ", obj_trajs_past_at_current_time_GT.shape)
                                selected_past = copy.deepcopy(obj_trajs_past_at_current_time_GT[GT_index_in_full,:,:])
                                # print("selected_past.shape: ", selected_past.shape)

                                fake_future = np.zeros_like(obj_trajs_future_at_current_time_GT[GT_index_in_full,:,:])
                                # print("fake_future.shape: ", fake_future.shape)
                                # print()
                                # fill fake future with matched GT past traj
                                fake_future[:min(fake_future.shape[0],selected_past.shape[0]),:] = selected_past[:min(fake_future.shape[0],selected_past.shape[0])]
                                # Predt (t=-2, t=-1, t=0) => GT (t=-2, t=-1, t=0)
                                obj_trajs_future_at_current_time[pred_index_in_full,:,:] = fake_future
                            else:
                                obj_trajs_future_at_current_time[pred_index_in_full,:,:] = copy.deepcopy(obj_trajs_future_at_current_time_GT[GT_index_in_full,:,:])
                            
                            #TODO update, insert infomation about GT traj type to this pred
                            # 1) determine GT traj type:
                            time = obj_trajs_future_at_current_time_GT[GT_index_in_full, 4::5, :][:6, :2].shape[0] * 0.5
                            # static end point
                            static_target = obj_trajs_past_at_current_time_GT[GT_index_in_full, -1, :2]
                            # linear end point
                            # (Pt+1 -Pt) / 0.5
                            GT_velocity_m_per_s = (obj_trajs_future_at_current_time_GT[GT_index_in_full, 4::5, :][0, :2]- obj_trajs_past_at_current_time_GT[GT_index_in_full, -1,:2])/0.5
                            # print("obj_trajs_future_at_current_time_GT[GT_index_in_full, 4::5, :][:, -1] ", obj_trajs_future_at_current_time_GT[GT_index_in_full, 4::5, :][:, -1])
                            # print("obj_trajs_past_at_current_time_GT[GT_index_in_full, -1,:2] ", obj_trajs_past_at_current_time_GT[GT_index_in_full, -1,:2])
                            # print("obj_trajs_past_at_current_time_GT.shape ", obj_trajs_past_at_current_time_GT.shape)
                            # print("obj_trajs_future_at_current_time_GT.shape ", obj_trajs_future_at_current_time_GT.shape)
                            # print("GT_velocity_m_per_s: ", GT_velocity_m_per_s)
                            linear_target = (
                                obj_trajs_past_at_current_time_GT[GT_index_in_full, -1,:2] + time * GT_velocity_m_per_s
                            )
                            # print("linear_target at Final end point: ", linear_target)
                            # GT end point
                            final_position = obj_trajs_future_at_current_time_GT[GT_index_in_full, 4::5, :][-1, :2]
                            # threshold
                            forecast_scalar = np.linspace(0, 1, 6 + 1)
                            # print("track_infos_GT['object_class_label'][GT_index_in_full]", track_infos_GT['object_class_label'][GT_index_in_full])
                            GT_class_name = CLASS_NAMES[int(track_infos_GT['object_class_label'][GT_index_in_full])]
                            # print("GT_class_name: ", GT_class_name)
                            threshold = 1 + forecast_scalar[
                                len(obj_trajs_future_at_current_time_GT[GT_index_in_full, 4::5, :][:6, :2])
                            ] * CATEGORY_TO_VELOCITY_M_PER_S.get(GT_class_name, 0)
                            # compare
                            if np.linalg.norm(final_position - static_target) < threshold:
                                obj_pred_traj_types[pred_index_in_full] = 0
                                # print('static')
                            elif np.linalg.norm(final_position - linear_target) < threshold:
                                obj_pred_traj_types[pred_index_in_full] = 1
                                # print('linear')
                            else:
                                obj_pred_traj_types[pred_index_in_full] = 2
                                # print('non linear')
                            
                        elif track_infos['object_id'][valid_past_preds_idx[valid_idx]] != 'AV':
                            # set this object to invalid, will be filtered out in condition_1
                            pred_index_in_full = valid_past_preds_idx[valid_idx]
                            #xyh update 03/06/2024
                            obj_trajs_future_at_current_time[pred_index_in_full,:,-1] = 0
                            #obj_trajs_past_at_current_time[pred_index_in_full, -1,-1] = 0
                        
                        # print()
                
                # print()
                
                

            #TODO update xyh: remove no future agents (because they cause ade = NaN) 
            # OPTIMIZATION HERE, 
            # With noisy inputs, (and clean inputs to a lesser extent), lots of lines of this matrix will be empty
            # The idea is to remove the lines
            # condition_1 = np.max(obj_trajs_past_at_current_time[:, :, 0], axis=1) != 0 
            #TODO update xyh, conditon 1 changed, we have negative coordiantes, if np.max => will remove track of interest. 
            condition_1 = obj_trajs_past_at_current_time[:, -1, -1] == 1 # keep valid objects at current timestamp
            if self.config["drop_no_future"]:
                assert obj_trajs_past_at_current_time.shape[1] > 0 # at least we have t=0
                assert obj_trajs_future_at_current_time.shape[1] > 0 # no last frame
            # 10 info: [state['position'], state['length'], state['width'], state['height'], state['heading'], state['velocity'], state['valid']]
            # print("obj_trajs_past_at_current_time.shape ", obj_trajs_past_at_current_time.shape) # [#of objects, past time steps, 10]
            # print("obj_trajs_future_at_current_time.shape ", obj_trajs_future_at_current_time.shape) # [#of objects, future time steps, 10]
            # print("condition_1.shape ", condition_1.shape) # [#of objects,]
            # print()
            # remove no future objects
            # print(obj_trajs_future_at_current_time.shape)
            # print("future to predict: ", track_infos['timestamps'][current_time_index+1:ending_future_ts][4::5][:6]) # 5 => every 5 frames, 4 = 5-1 (shift), 6 = only 3 seconds.
            # print(obj_trajs_future_at_current_time[:, 4::5, -1].shape)
            
            
            # if self.config["debug_challenge"] and track_infos['timestamps'][current_time_index] == 315969186260008000:
                
            #     uuid_version = []
            #     for idd in track_infos['object_id']:
            #         if idd != 'AV':
            #             uuid_version.append(UUID(idd).int)
            #         else:
            #             uuid_version.append('AV')
            #     assert len(uuid_version) == len(set(uuid_version))
                # print("target future:", obj_trajs_future_at_current_time[uuid_version.index(208133934815086590781149785702168011766), 4::5][:6])
                
            #TODO update xyh warning: filter out if t+1 does not exist in SUBSAMPLED space, also we only look at 3s. following forecasting eval code.
            # condition_2 = np.sum(obj_trajs_future_at_current_time[:, 4::5, :][:, :6, -1], axis=1) != 0 # softer conditon, having not at all future.
            # print("condition_2.shape ", condition_2.shape)
            if self.config["drop_no_future"]:
                if "train_past" in self.config and self.config["train_past"]: # even softer condition
                    #print("here.")
                    condition_2 = np.sum(obj_trajs_future_at_current_time[:, :, -1], axis=1) != 0 
                    
                elif "need_match_with_gt" in self.config and self.config["need_match_with_gt"]: # softer condition for noisy inputs
                    # print("here.")
                    condition_2 = np.sum(obj_trajs_future_at_current_time[:, 4::5, :][:, :6, -1], axis=1) != 0 
                else:
                    condition_2 = obj_trajs_future_at_current_time[:, 4::5, :][:, 0, -1] != 0 # more strict conditon: if track does not exist at t+1, then drop.
                condition = np.logical_and(condition_1, condition_2)
                # print("condition_2[sdc_track_index] ", condition_2[sdc_track_index])
                # print("condition_1[sdc_track_index] ", condition_1[sdc_track_index])
                # print()
            else:
                condition = condition_1
            non_null_lines = np.where(condition)[0]
            # print(non_null_lines)

            # update to new index
            # print("before: ", obj_trajs_past_at_current_time.shape)
            assert obj_trajs_past_at_current_time.shape[0] == len(track_infos['object_id'])
            obj_trajs_past_at_current_time = obj_trajs_past_at_current_time[non_null_lines, :, :]
            # print("after: ", obj_trajs_past_at_current_time.shape)
            obj_trajs_future_at_current_time = obj_trajs_future_at_current_time[non_null_lines, :, :]
            obj_types_at_current_time = obj_types[non_null_lines]
            obj_pred_traj_types = obj_pred_traj_types[non_null_lines]
            object_class_label_at_current_time = object_class_label_at_current_time[non_null_lines]
            track_scores_past = track_scores_past[non_null_lines, :]
            current_object_ids = np.array(track_infos['object_id'])[non_null_lines]
            
            
            if refined_pasts is not None:
                #update past trajs with forecast past results
                for current_object_idx, current_object_id in enumerate(current_object_ids):
                    #print(current_object_id)
                    #print(type(current_object_id))

                    if current_object_id in forecast_by_object_id:
                        #print("current_object_id ", current_object_id)
                        #print("obj_trajs_past_at_current_time[current_object_idx].shape ", obj_trajs_past_at_current_time[current_object_idx].shape)
                        #print('before past: ', obj_trajs_past_at_current_time[current_object_idx, :, :2])
                        #print("forecast_by_object_id[current_object_id].shape", forecast_by_object_id[current_object_id].shape)
                        #print("now past: ", forecast_by_object_id[current_object_id][:obj_trajs_past_at_current_time.shape[1], :])
                        #print()center_gt_trajs_src
                        # pad
                        last_available_index = int(obj_trajs_past_at_current_time[current_object_idx, :, -1].sum()) 
                        #print("before: ", obj_trajs_past_at_current_time[current_object_idx, :, :])
                        obj_trajs_past_at_current_time[current_object_idx, :obj_trajs_past_at_current_time.shape[1]-last_available_index, :] =  obj_trajs_past_at_current_time[current_object_idx, -last_available_index, :]
                        #print("after: ", obj_trajs_past_at_current_time[current_object_idx, :, :])
                        #print()

                        obj_trajs_past_at_current_time[current_object_idx, :-1, :2] = forecast_by_object_id[current_object_id][:obj_trajs_past_at_current_time.shape[1]-1,:]
                        #obj_trajs_past_at_current_time[current_object_idx, :, :2] = forecast_by_object_id[current_object_id][:obj_trajs_past_at_current_time.shape[1],:]
                        # # all as valid past
                        obj_trajs_past_at_current_time[current_object_idx, :, -1] = 1
            

            
            
            # The matrix will be reduced by keeping only positive lines.
            # eg (120000, 157, 10) --> (200, 157, 10)
            # So we have to reindex for the track_index_to_predict
            new_reindexing = np.zeros(obj_trajs_full.shape[0], dtype=np.int32)-1 # {old index: new index}
            for i, non_null_line in enumerate(non_null_lines):
                new_reindexing[non_null_line] = i
            # print("after op: ", len(non_null_lines))
            # Pad
            if starting_past_ts <= 0:
                # Pad at the beginning
                valid_past_ts = obj_trajs_past_at_current_time.shape[1]
                obj_trajs_past_at_current_time = np.pad(obj_trajs_past_at_current_time, ((0, 0), (past_len - valid_past_ts, 0), (0, 0)))
                for current_object_idx in range(obj_trajs_past_at_current_time.shape[0]):
                    last_available_index = int(obj_trajs_past_at_current_time[current_object_idx, :, -1].sum())
                    #TODO pad here
                    #print("before: ", obj_trajs_past_at_current_time[current_object_idx, :, :])
                    obj_trajs_past_at_current_time[current_object_idx, :obj_trajs_past_at_current_time.shape[1]-last_available_index, :] =  obj_trajs_past_at_current_time[current_object_idx, -last_available_index, :]
                #print("after: ", obj_trajs_past_at_current_time[current_object_idx, :, :])
                #print()
                track_scores_past = np.pad(track_scores_past, ((0,0), (past_len - valid_past_ts, 0)))
            if ending_future_ts >= nb_all_ts:
                # Pad at the end
                obj_trajs_future_at_current_time = np.pad(obj_trajs_future_at_current_time, ((0, 0), (0, future_len - obj_trajs_future_at_current_time.shape[1]), (0, 0)))

            #print("obj_trajs_past_at_current_time.shape ", obj_trajs_past_at_current_time.shape)

            # print("obj_types_at_current_time.shape ", obj_types_at_current_time.shape)
            # print("track_scores_past.shape ", track_scores_past.shape)
            # print("obj_trajs_past_at_current_time.shape ", obj_trajs_past_at_current_time.shape)
            # Concatenate 
            # first 21 are past + cur, last 60 are future
            obj_trajs_full_at_current_time = np.concatenate((obj_trajs_past_at_current_time, obj_trajs_future_at_current_time), axis=1)
            ####
            # Change the coordinate system, pad to max nb of agent, compute various features
            ####
            # WARNING: this is the preprocessing bottleneck, as the total number of agents may be big!
            # The thing is that most of obj_trajs_past is empty
            

            #     print("new_reindexing[track_index_to_predict] ", new_reindexing[track_index_to_predict])
            
            assert center_objects.shape[0] == track_index_to_predict.shape[0]
            #TODO update xyh, just remove not valid track of interests
            new_center_objects = []
            new_track_index_to_predict = [] # still old index
            center_traj_types = []
            center_object_labels = []
            for kk, iid in enumerate(new_reindexing[track_index_to_predict]):
                if iid == -1:
                    # if self.config["debug_challenge"] and track_infos['timestamps'][current_time_index] == 315969186260008000:
                    #     print(f"{track_index_to_predict[kk]} is removed. ")
                    #     print(f'because: ', obj_trajs_full[track_index_to_predict[kk], current_time_index])
                    #     print(obj_trajs_full[track_index_to_predict[kk], current_time_index+1:ending_future_ts].sum())
                    #     assert obj_trajs_full[track_index_to_predict[kk], current_time_index+1:ending_future_ts].sum() == 0.0
                    continue
                else:
                    #if "need_match_with_gt" in self.config and self.config["need_match_with_gt"]:
                    #    assert obj_pred_traj_types[iid] != -1 # no -1 type
                    
                    center_traj_types.append(obj_pred_traj_types[iid])
                    center_object_labels.append(object_class_label_at_current_time[iid])
                    new_center_objects.append(center_objects[kk])
                    new_track_index_to_predict.append(track_index_to_predict[kk])
            center_objects = np.array(new_center_objects)
            track_index_to_predict = np.array(new_track_index_to_predict)
            center_traj_types = np.array(center_traj_types, dtype=np.float32)
            center_object_labels = np.array(center_object_labels, dtype=np.int32)
            
            # has at least one valid target?
            assert track_index_to_predict.shape[0] > 0
            
            # if self.config["debug_challenge"]:
            #     print(track_infos['timestamps'][current_time_index], track_index_to_predict.shape[0])
            
                # print("After track_index_to_predict: ", sorted([UUID(idd).int for idd in np.array(track_infos['object_id'])[track_index_to_predict].tolist()]))
            #TODO update xyh sdc index:
            # 'AV' is always in valid list.
            assert sdc_track_index in non_null_lines.tolist()
            sdc_track_index = new_reindexing[sdc_track_index]
            

            assert -1 not in track_index_to_predict.tolist()   
            assert center_objects.shape[0] == track_index_to_predict.shape[0]
            (obj_trajs_data, obj_trajs_mask, obj_trajs_pos, obj_trajs_last_pos, obj_trajs_future_state,
             obj_trajs_future_mask, center_gt_trajs, center_gt_trajs_mask, center_gt_final_valid_idx,
             track_index_to_predict_new) = self.get_agent_data(
                center_objects=center_objects, obj_trajs_past=obj_trajs_past_at_current_time, obj_trajs_future=obj_trajs_future_at_current_time,
                track_index_to_predict=new_reindexing[track_index_to_predict], sdc_track_index=sdc_track_index,
                timestamps=timestamps, obj_types=obj_types_at_current_time, object_scores= track_scores_past
            )

            #TODO update xyh, record traj type
            center_objects_traj_type.append(center_traj_types[:, np.newaxis])
            center_objects_label.append(center_object_labels[:, np.newaxis])
            
            center_objects_final.append(center_objects)
            track_index_to_predict_final.append(track_index_to_predict)
            obj_trajs_data_final.append(obj_trajs_data)
            obj_trajs_mask_final.append(obj_trajs_mask)
            obj_trajs_pos_final.append(obj_trajs_pos)
            obj_trajs_last_pos_final.append(obj_trajs_last_pos)
            obj_trajs_future_state_final.append(obj_trajs_future_state)
            obj_trajs_future_mask_final.append(obj_trajs_future_mask)
            center_gt_trajs_final.append(center_gt_trajs)
            center_gt_trajs_mask_final.append(center_gt_trajs_mask)
            center_gt_final_valid_idx_final.append(center_gt_final_valid_idx)
            track_index_to_predict_new_final.append(track_index_to_predict_new)
            center_gt_trajs_src_final.append(obj_trajs_full_at_current_time[new_reindexing[track_index_to_predict]])
            current_time_indices.append(np.array([current_time_index]*track_index_to_predict_new.shape[0]))


        #TODO update xyh, record traj type
        center_objects_traj_type = np.vstack(center_objects_traj_type)
        center_objects_label = np.vstack(center_objects_label)
        
        center_objects = np.vstack(center_objects_final)
        
        assert center_objects_traj_type.shape[0] == center_objects.shape[0] == center_objects_label.shape[0]
        # print("center_objects_traj_type.shape ", center_objects_traj_type.shape)
        track_index_to_predict = np.concatenate(track_index_to_predict_final)
        obj_trajs_data = np.vstack(obj_trajs_data_final)
        obj_trajs_mask = np.vstack(obj_trajs_mask_final)
        obj_trajs_pos = np.vstack(obj_trajs_pos_final)
        obj_trajs_last_pos = np.vstack(obj_trajs_last_pos_final)
        obj_trajs_future_state = np.vstack(obj_trajs_future_state_final)
        obj_trajs_future_mask = np.vstack(obj_trajs_future_mask_final)
        center_gt_trajs = np.vstack(center_gt_trajs_final)
        center_gt_trajs_mask = np.vstack(center_gt_trajs_mask_final)
        center_gt_final_valid_idx = np.concatenate(center_gt_final_valid_idx_final)
        track_index_to_predict_new = np.concatenate(track_index_to_predict_new_final)
        center_gt_trajs_src = np.vstack(center_gt_trajs_src_final)
        current_time_indices = np.concatenate(current_time_indices)


        if center_objects is None: return None
        sample_num = center_objects.shape[0]

        ret_dict = {
            'scenario_id': np.array([scene_id] * len(track_index_to_predict)),
            'obj_trajs': obj_trajs_data,
            'obj_trajs_mask': obj_trajs_mask,
            'track_index_to_predict': track_index_to_predict_new,  # used to select center-features # usually it's always 0, 0, 0...????? #TODO why??
            'obj_trajs_pos': obj_trajs_pos,
            'obj_trajs_last_pos': obj_trajs_last_pos,

            'center_objects_world': center_objects,
            
            #TODO update xyh
            'center_objects_traj_type': center_objects_traj_type,
            'center_objects_label': center_objects_label,
            
            'center_objects_id': np.array(track_infos['object_id'])[track_index_to_predict],
            'center_objects_type': np.array(track_infos['object_type'])[track_index_to_predict],
            'map_center': info['map_center'],

            'obj_trajs_future_state': obj_trajs_future_state,
            'obj_trajs_future_mask': obj_trajs_future_mask,
            'center_gt_trajs': center_gt_trajs,
            'center_gt_trajs_mask': center_gt_trajs_mask,
            'center_gt_final_valid_idx': center_gt_final_valid_idx,
            'center_gt_trajs_src': center_gt_trajs_src,
            'current_time_indices': current_time_indices,
        }

        if info['map_infos']['all_polylines'].__len__() == 0:
            info['map_infos']['all_polylines'] = np.zeros((2, 7), dtype=np.float32)
            print(f'Warning: empty HDMap {scene_id}')

        if self.config.manually_split_lane:
            map_polylines_data, map_polylines_mask, map_polylines_center = self.get_manually_split_map_data(
                center_objects=center_objects, map_infos=info['map_infos'])
        else:
            map_polylines_data, map_polylines_mask, map_polylines_center = self.get_map_data(
                center_objects=center_objects, map_infos=info['map_infos'])

        ret_dict['map_polylines'] = map_polylines_data
        ret_dict['map_polylines_mask'] = map_polylines_mask.astype(bool)
        ret_dict['map_polylines_center'] = map_polylines_center


        # masking out unused attributes to Zero
        masked_attributes = self.config['masked_attributes']
        if 'z_axis' in masked_attributes:
            ret_dict['obj_trajs'][..., 2] = 0
            ret_dict['map_polylines'][..., 2] = 0
        if 'size' in masked_attributes:
            ret_dict['obj_trajs'][..., 3:6] = 0
        if 'velocity' in masked_attributes:
            ret_dict['obj_trajs'][..., 25:27] = 0
        if 'acceleration' in masked_attributes:
            ret_dict['obj_trajs'][..., 27:29] = 0
        if 'heading' in masked_attributes:
            ret_dict['obj_trajs'][..., 23:25] = 0

        # change every thing to float32
        for k, v in ret_dict.items():
            if isinstance(v, np.ndarray) and v.dtype == np.float64:
                ret_dict[k] = v.astype(np.float32)

        ret_dict['map_center'] = ret_dict['map_center'].repeat(sample_num, axis=0)
        ret_dict['dataset_name'] = [info['dataset']] * sample_num

        # Convert the dictionary to a list
        ret_list = []
        for i in range(sample_num):
            ret_dict_i = {}
            for k, v in ret_dict.items():
                ret_dict_i[k] = v[i]
            ret_list.append(ret_dict_i)

        return ret_list

    def postprocess(self, output):

        # Add the trajectory difficulty
        get_kalman_difficulty(output)

        # Add the trajectory type (stationary, straight, right turn...)
        get_trajectory_type(output)

        return output

    def collate_fn(self, data_list):
        batch_list = []
        for batch in data_list:
            batch_list += batch

        batch_size = len(batch_list)
        key_to_list = {}
        for key in batch_list[0].keys():
            key_to_list[key] = [batch_list[bs_idx][key] for bs_idx in range(batch_size)]

        input_dict = {}
        for key, val_list in key_to_list.items():
            # if val_list is str:
            try:
                input_dict[key] = torch.from_numpy(np.stack(val_list, axis=0))
            except:
                input_dict[key] = val_list

        input_dict['center_objects_type'] = input_dict['center_objects_type'].numpy()

        batch_dict = {'batch_size': batch_size, 'input_dict': input_dict, 'batch_sample_count': batch_size}
        return batch_dict

    def __len__(self):
        return len(self.data_loaded)

    def __getitem__(self, idx):
#        if self.config['store_data_in_memory']:
#            return self.data_loaded_memory[idx]
#        else:
        # Actually loads multiple examples simultaneously, number is data_chunk_size (or less for the final batch)
        with open(self.data_loaded_keys[idx], 'rb') as f:
            return pickle.load(f)

    def get_data_list(self, data_usage):
        file_list_path = os.path.join(self.cache_path, 'file_list.pkl')
        if os.path.exists(file_list_path):
            data_loaded = pickle.load(open(file_list_path, 'rb'))
        else:
            raise ValueError('Error: file_list.pkl not found')

        data_list = list(data_loaded.items())
        np.random.shuffle(data_list)

        if not self.is_validation:
            # randomly sample data_usage number of data
            data_loaded = dict(data_list[:data_usage])
        else:
            data_loaded = dict(data_list)
        return data_loaded

    def get_agent_data(
            self, center_objects, obj_trajs_past, obj_trajs_future, track_index_to_predict, sdc_track_index, timestamps,
            obj_types, object_scores
    ):

        num_center_objects = center_objects.shape[0]
        num_objects, num_timestamps, box_dim = obj_trajs_past.shape
        # all_state = [state['position'], state['length'], state['width'], state['height'], state['heading'],
                        #  state['velocity'], state['valid']]
        # centered to then rot to agent at t=0
        obj_trajs = self.transform_trajs_to_center_coords(
            obj_trajs=obj_trajs_past,
            center_xyz=center_objects[:, 0:3],
            center_heading=center_objects[:, 6],
            heading_index=6, rot_vel_index=[7, 8]
        )
        # print("obj_types 2: ", obj_types)
        object_onehot_mask = np.zeros((num_center_objects, num_objects, num_timestamps, 5))
        
        
        #TODO print("Should I add OTHER class obj_types==True?")
        if "nei_weight_by_score" in self.config and self.config["nei_weight_by_score"]:
            # object_onehot_mask[:, obj_types == 4, :, 2] = 1
            assert len(object_scores) == num_objects == len(obj_types)
            # print("object_onehot_mask.shape ", object_onehot_mask.shape)
            
            # print("object_onehot_mask[:, obj_types == 1, :, :][:,:,:,0].shape ", object_onehot_mask[:, obj_types == 1, :, :][:,:,:,0].shape)
            # print("object_scores[obj_types == 1].shape ", object_scores[obj_types == 1].shape)
            # print()
            # print("object_scores.shape ", object_scores.shape)
            # print("obj_types.shape ", obj_types.shape)
            # print("object_onehot_mask.shape ", object_onehot_mask.shape)
            # print()
            object_onehot_mask[:, (obj_types == 1)[:,np.newaxis] * object_scores[:, :]>0.2, :][...,0] = 1
            object_onehot_mask[:, (obj_types == 2)[:,np.newaxis] * object_scores[:, :]>0.2, :][...,1] = 1
            object_onehot_mask[:, (obj_types == 3)[:,np.newaxis] * object_scores[:, :]>0.2, :][...,2] = 1
            object_onehot_mask[:, (obj_types == 4)[:,np.newaxis] * object_scores[:, :]>0.2, :][...,2] = 1
        else:
            object_onehot_mask[:, obj_types == 1, :, 0] = 1
            object_onehot_mask[:, obj_types == 2, :, 1] = 1
            object_onehot_mask[:, obj_types == 3, :, 2] = 1
            
            
            
        object_onehot_mask[np.arange(num_center_objects), track_index_to_predict, :, 3] = 1
        object_onehot_mask[:, sdc_track_index, :, 4] = 1
        # print("object_onehot_mask.shape ", object_onehot_mask.shape)
        # print("sdc_track_index ", sdc_track_index)
        # print()

        object_time_embedding = np.zeros((num_center_objects, num_objects, num_timestamps, num_timestamps + 1))
        for i in range(num_timestamps):
            object_time_embedding[:, :, i, i] = 1
        object_time_embedding[:, :, :, -1] = timestamps

        object_heading_embedding = np.zeros((num_center_objects, num_objects, num_timestamps, 2))
        object_heading_embedding[:, :, :, 0] = np.sin(obj_trajs[:, :, :, 6])
        object_heading_embedding[:, :, :, 1] = np.cos(obj_trajs[:, :, :, 6])

        vel = obj_trajs[:, :, :, 7:9]
        vel_pre = np.roll(vel, shift=1, axis=2)
        acce = (vel - vel_pre) / 0.1
        acce[:, :, 0, :] = acce[:, :, 1, :]

        obj_trajs_data = np.concatenate([
            obj_trajs[:, :, :, 0:6],
            object_onehot_mask,
            object_time_embedding,
            object_heading_embedding,
            obj_trajs[:, :, :, 7:9],
            acce,
        ], axis=-1)

        obj_trajs_mask = obj_trajs[:, :, :, -1]
        obj_trajs_data[obj_trajs_mask == 0] = 0

        obj_trajs_future = obj_trajs_future.astype(np.float32)
        obj_trajs_future = self.transform_trajs_to_center_coords(
            obj_trajs=obj_trajs_future,
            center_xyz=center_objects[:, 0:3],
            center_heading=center_objects[:, 6],
            heading_index=6, rot_vel_index=[7, 8]
        )
        obj_trajs_future_state = obj_trajs_future[:, :, :, [0, 1, 7, 8]]  # (x, y, vx, vy)
        obj_trajs_future_mask = obj_trajs_future[:, :, :, -1]
        obj_trajs_future_state[obj_trajs_future_mask == 0] = 0

        center_obj_idxs = np.arange(len(track_index_to_predict))
        center_gt_trajs = obj_trajs_future_state[center_obj_idxs, track_index_to_predict]
        center_gt_trajs_mask = obj_trajs_future_mask[center_obj_idxs, track_index_to_predict]
        center_gt_trajs[center_gt_trajs_mask == 0] = 0

        assert obj_trajs_past.__len__() == obj_trajs_data.shape[1]
        valid_past_mask = np.logical_not(obj_trajs_past[:, :, -1].sum(axis=-1) == 0)

        obj_trajs_mask = obj_trajs_mask[:, valid_past_mask]
        obj_trajs_data = obj_trajs_data[:, valid_past_mask]
        obj_trajs_future_state = obj_trajs_future_state[:, valid_past_mask]
        obj_trajs_future_mask = obj_trajs_future_mask[:, valid_past_mask]

        obj_trajs_pos = obj_trajs_data[:, :, :, 0:3]
        num_center_objects, num_objects, num_timestamps, _ = obj_trajs_pos.shape
        obj_trajs_last_pos = np.zeros((num_center_objects, num_objects, 3), dtype=np.float32)
        for k in range(num_timestamps):
            cur_valid_mask = obj_trajs_mask[:, :, k] > 0
            obj_trajs_last_pos[cur_valid_mask] = obj_trajs_pos[:, :, k, :][cur_valid_mask]

        center_gt_final_valid_idx = np.zeros((num_center_objects), dtype=np.float32)
        for k in range(center_gt_trajs_mask.shape[1]):
            cur_valid_mask = center_gt_trajs_mask[:, k] > 0
            center_gt_final_valid_idx[cur_valid_mask] = k

        max_num_agents = self.config['max_num_agents']
        object_dist_to_center = np.linalg.norm(obj_trajs_data[:, :, -1, 0:2], axis=-1)

        object_dist_to_center[obj_trajs_mask[..., -1] == 0] = 1e10
        topk_idxs = np.argsort(object_dist_to_center, axis=-1)[:, :max_num_agents]

        topk_idxs = np.expand_dims(topk_idxs, axis=-1)
        topk_idxs = np.expand_dims(topk_idxs, axis=-1)

        obj_trajs_data = np.take_along_axis(obj_trajs_data, topk_idxs, axis=1)
        obj_trajs_mask = np.take_along_axis(obj_trajs_mask, topk_idxs[..., 0], axis=1)
        obj_trajs_pos = np.take_along_axis(obj_trajs_pos, topk_idxs, axis=1)
        obj_trajs_last_pos = np.take_along_axis(obj_trajs_last_pos, topk_idxs[..., 0], axis=1)
        obj_trajs_future_state = np.take_along_axis(obj_trajs_future_state, topk_idxs, axis=1)
        obj_trajs_future_mask = np.take_along_axis(obj_trajs_future_mask, topk_idxs[..., 0], axis=1)
        track_index_to_predict_new = np.zeros(len(track_index_to_predict), dtype=np.int64)

        obj_trajs_data = np.pad(obj_trajs_data, ((0, 0), (0, max_num_agents - obj_trajs_data.shape[1]), (0, 0), (0, 0)))
        obj_trajs_mask = np.pad(obj_trajs_mask, ((0, 0), (0, max_num_agents - obj_trajs_mask.shape[1]), (0, 0)))
        obj_trajs_pos = np.pad(obj_trajs_pos, ((0, 0), (0, max_num_agents - obj_trajs_pos.shape[1]), (0, 0), (0, 0)))
        obj_trajs_last_pos = np.pad(obj_trajs_last_pos,
                                    ((0, 0), (0, max_num_agents - obj_trajs_last_pos.shape[1]), (0, 0)))
        obj_trajs_future_state = np.pad(obj_trajs_future_state,
                                        ((0, 0), (0, max_num_agents - obj_trajs_future_state.shape[1]), (0, 0), (0, 0)))
        obj_trajs_future_mask = np.pad(obj_trajs_future_mask,
                                       ((0, 0), (0, max_num_agents - obj_trajs_future_mask.shape[1]), (0, 0)))

        return (obj_trajs_data, obj_trajs_mask.astype(bool), obj_trajs_pos, obj_trajs_last_pos,
                obj_trajs_future_state, obj_trajs_future_mask, center_gt_trajs, center_gt_trajs_mask,
                center_gt_final_valid_idx,
                track_index_to_predict_new)

    def get_interested_agents(self, track_index_to_predict, obj_trajs_full, current_time_index, obj_types, scene_id):
        center_objects_list = []
        track_index_to_predict_selected = []
        selected_type = self.config['object_type']
        selected_type = [object_type[x] for x in selected_type]

        for obj_idx in track_index_to_predict:
            # Check validity of agent at current_time_index
            if obj_trajs_full[obj_idx, current_time_index, -1] == 0:
                #print(f'Warning: obj_idx={obj_idx} is not valid at time step {current_time_index}, scene_id={scene_id}')
                continue
            # Check validity of object_type
            if obj_types[obj_idx] not in selected_type:
                continue

            center_objects_list.append(obj_trajs_full[obj_idx, current_time_index])
            track_index_to_predict_selected.append(obj_idx)

        if len(center_objects_list) == 0:
            print(f'Warning: no center objects at time step {current_time_index}, scene_id={scene_id}')
            return None, []
        center_objects = np.stack(center_objects_list, axis=0)  # (num_center_objects, num_attrs)
        track_index_to_predict = np.array(track_index_to_predict_selected)

        return center_objects, track_index_to_predict

    def transform_trajs_to_center_coords(self, obj_trajs, center_xyz, center_heading, heading_index,
                                         rot_vel_index=None):
        """
        Args:
            obj_trajs (num_objects, num_timestamps, num_attrs):
                first three values of num_attrs are [x, y, z] or [x, y]
            center_xyz (num_center_objects, 3 or 2): [x, y, z] or [x, y]
            center_heading (num_center_objects):
            heading_index: the index of heading angle in the num_attr-axis of obj_trajs
        """
        num_objects, num_timestamps, num_attrs = obj_trajs.shape
        num_center_objects = center_xyz.shape[0]
        assert center_xyz.shape[0] == center_heading.shape[0]
        assert center_xyz.shape[1] in [3, 2]

        obj_trajs = np.tile(obj_trajs[None, :, :, :], (num_center_objects, 1, 1, 1))
        obj_trajs[:, :, :, 0:center_xyz.shape[1]] -= center_xyz[:, None, None, :]
        obj_trajs[:, :, :, 0:2] = common_utils.rotate_points_along_z(
            points=obj_trajs[:, :, :, 0:2].reshape(num_center_objects, -1, 2),
            angle=-center_heading
        ).reshape(num_center_objects, num_objects, num_timestamps, 2)

        obj_trajs[:, :, :, heading_index] -= center_heading[:, None, None]

        # rotate direction of velocity
        if rot_vel_index is not None:
            assert len(rot_vel_index) == 2
            obj_trajs[:, :, :, rot_vel_index] = common_utils.rotate_points_along_z(
                points=obj_trajs[:, :, :, rot_vel_index].reshape(num_center_objects, -1, 2),
                angle=-center_heading
            ).reshape(num_center_objects, num_objects, num_timestamps, 2)

        return obj_trajs

    def get_map_data(self, center_objects, map_infos):

        num_center_objects = center_objects.shape[0]

        def transform_to_center_coordinates(neighboring_polylines):
            neighboring_polylines[:, :, 0:3] -= center_objects[:, None, 0:3]
            neighboring_polylines[:, :, 0:2] = common_utils.rotate_points_along_z(
                points=neighboring_polylines[:, :, 0:2],
                angle=-center_objects[:, 6]
            )
            neighboring_polylines[:, :, 3:5] = common_utils.rotate_points_along_z(
                points=neighboring_polylines[:, :, 3:5],
                angle=-center_objects[:, 6]
            )

            return neighboring_polylines

        polylines = np.expand_dims(map_infos['all_polylines'].copy(), axis=0).repeat(num_center_objects, axis=0)

        map_polylines = transform_to_center_coordinates(neighboring_polylines=polylines)
        num_of_src_polylines = self.config['max_num_roads']
        map_infos['polyline_transformed'] = map_polylines

        all_polylines = map_infos['polyline_transformed']
        max_points_per_lane = self.config.get('max_points_per_lane', 20)
        line_type = self.config.get('line_type', [])
        map_range = self.config.get('map_range', None)
        center_offset = self.config.get('center_offset_of_map', (30.0, 0))
        num_agents = all_polylines.shape[0]
        polyline_list = []
        polyline_mask_list = []

        for k, v in map_infos.items():
            if k == 'all_polylines' or k not in line_type:
                continue
            if len(v) == 0:
                continue
            for polyline_dict in v:
                polyline_index = polyline_dict.get('polyline_index', None)
                polyline_segment = all_polylines[:, polyline_index[0]:polyline_index[1]]
                polyline_segment_x = polyline_segment[:, :, 0] - center_offset[0]
                polyline_segment_y = polyline_segment[:, :, 1] - center_offset[1]
                in_range_mask = (abs(polyline_segment_x) < map_range) * (abs(polyline_segment_y) < map_range)

                segment_index_list = []
                for i in range(polyline_segment.shape[0]):
                    segment_index_list.append(find_true_segments(in_range_mask[i]))
                max_segments = max([len(x) for x in segment_index_list])

                segment_list = np.zeros([num_agents, max_segments, max_points_per_lane, 7], dtype=np.float32)
                segment_mask_list = np.zeros([num_agents, max_segments, max_points_per_lane], dtype=np.int32)

                for i in range(polyline_segment.shape[0]):
                    if in_range_mask[i].sum() == 0:
                        continue
                    segment_i = polyline_segment[i]
                    segment_index = segment_index_list[i]
                    for num, seg_index in enumerate(segment_index):
                        segment = segment_i[seg_index]
                        if segment.shape[0] > max_points_per_lane:
                            segment_list[i, num] = segment[
                                np.linspace(0, segment.shape[0] - 1, max_points_per_lane, dtype=int)]
                            segment_mask_list[i, num] = 1
                        else:
                            segment_list[i, num, :segment.shape[0]] = segment
                            segment_mask_list[i, num, :segment.shape[0]] = 1

                polyline_list.append(segment_list)
                polyline_mask_list.append(segment_mask_list)
        if len(polyline_list) == 0: return np.zeros((num_agents, 0, max_points_per_lane, 7)), np.zeros(
            (num_agents, 0, max_points_per_lane))
        batch_polylines = np.concatenate(polyline_list, axis=1)
        batch_polylines_mask = np.concatenate(polyline_mask_list, axis=1)

        polyline_xy_offsetted = batch_polylines[:, :, :, 0:2] - np.reshape(center_offset, (1, 1, 1, 2))
        polyline_center_dist = np.linalg.norm(polyline_xy_offsetted, axis=-1).sum(-1) / np.clip(
            batch_polylines_mask.sum(axis=-1).astype(float), a_min=1.0, a_max=None)
        polyline_center_dist[batch_polylines_mask.sum(-1) == 0] = 1e10
        topk_idxs = np.argsort(polyline_center_dist, axis=-1)[:, :num_of_src_polylines]

        # Ensure topk_idxs has the correct shape for indexing
        topk_idxs = np.expand_dims(topk_idxs, axis=-1)
        topk_idxs = np.expand_dims(topk_idxs, axis=-1)
        map_polylines = np.take_along_axis(batch_polylines, topk_idxs, axis=1)
        map_polylines_mask = np.take_along_axis(batch_polylines_mask, topk_idxs[..., 0], axis=1)

        # pad map_polylines and map_polylines_mask to num_of_src_polylines
        map_polylines = np.pad(map_polylines,
                               ((0, 0), (0, num_of_src_polylines - map_polylines.shape[1]), (0, 0), (0, 0)))
        map_polylines_mask = np.pad(map_polylines_mask,
                                    ((0, 0), (0, num_of_src_polylines - map_polylines_mask.shape[1]), (0, 0)))

        temp_sum = (map_polylines[:, :, :, 0:3] * map_polylines_mask[:, :, :, None].astype(float)).sum(
            axis=-2)  # (num_center_objects, num_polylines, 3)
        map_polylines_center = temp_sum / np.clip(map_polylines_mask.sum(axis=-1).astype(float)[:, :, None], a_min=1.0,
                                                  a_max=None)  # (num_center_objects, num_polylines, 3)

        xy_pos_pre = map_polylines[:, :, :, 0:2]
        xy_pos_pre = np.roll(xy_pos_pre, shift=1, axis=-2)
        xy_pos_pre[:, :, 0, :] = xy_pos_pre[:, :, 1, :]
        map_polylines = np.concatenate((map_polylines, xy_pos_pre), axis=-1)
        map_polylines[map_polylines_mask == 0] = 0

        return map_polylines, map_polylines_mask, map_polylines_center

    def get_manually_split_map_data(self, center_objects, map_infos):
        """
        Args:
            center_objects (num_center_objects, 10): [cx, cy, cz, dx, dy, dz, heading, vel_x, vel_y, valid]
            map_infos (dict):
                all_polylines (num_points, 7): [x, y, z, dir_x, dir_y, dir_z, global_type]
            center_offset (2):, [offset_x, offset_y]
        Returns:
            map_polylines (num_center_objects, num_topk_polylines, num_points_each_polyline, 9): [x, y, z, dir_x, dir_y, dir_z, global_type, pre_x, pre_y]
            map_polylines_mask (num_center_objects, num_topk_polylines, num_points_each_polyline)
        """
        num_center_objects = center_objects.shape[0]
        center_offset = self.config.get('center_offset_of_map', (30.0, 0))

        # transform object coordinates by center objects
        def transform_to_center_coordinates(neighboring_polylines, neighboring_polyline_valid_mask):
            neighboring_polylines[:, :, :, 0:3] -= center_objects[:, None, None, 0:3]
            neighboring_polylines[:, :, :, 0:2] = common_utils.rotate_points_along_z(
                points=neighboring_polylines[:, :, :, 0:2].reshape(num_center_objects, -1, 2),
                angle=-center_objects[:, 6]
            ).reshape(num_center_objects, -1, batch_polylines.shape[1], 2)
            neighboring_polylines[:, :, :, 3:5] = common_utils.rotate_points_along_z(
                points=neighboring_polylines[:, :, :, 3:5].reshape(num_center_objects, -1, 2),
                angle=-center_objects[:, 6]
            ).reshape(num_center_objects, -1, batch_polylines.shape[1], 2)

            # use pre points to map
            # (num_center_objects, num_polylines, num_points_each_polyline, num_feat)
            xy_pos_pre = neighboring_polylines[:, :, :, 0:2]
            xy_pos_pre = np.roll(xy_pos_pre, shift=1, axis=-2)
            xy_pos_pre[:, :, 0, :] = xy_pos_pre[:, :, 1, :]
            neighboring_polylines = np.concatenate((neighboring_polylines, xy_pos_pre), axis=-1)

            neighboring_polylines[neighboring_polyline_valid_mask == 0] = 0
            return neighboring_polylines, neighboring_polyline_valid_mask

        polylines = map_infos['all_polylines'].copy()
        center_objects = center_objects

        point_dim = polylines.shape[-1]

        point_sampled_interval = self.config['point_sampled_interval']
        vector_break_dist_thresh = self.config['vector_break_dist_thresh']
        num_points_each_polyline = self.config['num_points_each_polyline']

        sampled_points = polylines[::point_sampled_interval]
        sampled_points_shift = np.roll(sampled_points, shift=1, axis=0)
        buffer_points = np.concatenate((sampled_points[:, 0:2], sampled_points_shift[:, 0:2]),
                                       axis=-1)  # [ed_x, ed_y, st_x, st_y]
        buffer_points[0, 2:4] = buffer_points[0, 0:2]

        break_idxs = \
            (np.linalg.norm(buffer_points[:, 0:2] - buffer_points[:, 2:4],
                            axis=-1) > vector_break_dist_thresh).nonzero()[0]
        polyline_list = np.array_split(sampled_points, break_idxs, axis=0)
        ret_polylines = []
        ret_polylines_mask = []

        def append_single_polyline(new_polyline):
            cur_polyline = np.zeros((num_points_each_polyline, point_dim), dtype=np.float32)
            cur_valid_mask = np.zeros((num_points_each_polyline), dtype=np.int32)
            cur_polyline[:len(new_polyline)] = new_polyline
            cur_valid_mask[:len(new_polyline)] = 1
            ret_polylines.append(cur_polyline)
            ret_polylines_mask.append(cur_valid_mask)

        for k in range(len(polyline_list)):
            if polyline_list[k].__len__() <= 0:
                continue
            for idx in range(0, len(polyline_list[k]), num_points_each_polyline):
                append_single_polyline(polyline_list[k][idx: idx + num_points_each_polyline])

        batch_polylines = np.stack(ret_polylines, axis=0)
        batch_polylines_mask = np.stack(ret_polylines_mask, axis=0)

        # collect a number of closest polylines for each center objects
        num_of_src_polylines = self.config['max_num_roads']

        if len(batch_polylines) > num_of_src_polylines:
            # Sum along a specific axis and divide by the minimum clamped sum
            polyline_center = np.sum(batch_polylines[:, :, 0:2], axis=1) / np.clip(
                np.sum(batch_polylines_mask, axis=1)[:, None].astype(float), a_min=1.0, a_max=None)
            # Convert the center_offset to a numpy array and repeat it for each object
            center_offset_rot = np.tile(np.array(center_offset, dtype=np.float32)[None, :], (num_center_objects, 1))

            center_offset_rot = common_utils.rotate_points_along_z(
                points=center_offset_rot[:, None, :],
                angle=center_objects[:, 6]
            )

            pos_of_map_centers = center_objects[:, 0:2] + center_offset_rot[:, 0]

            dist = np.linalg.norm(pos_of_map_centers[:, None, :] - polyline_center[None, :, :], axis=-1)

            # Getting the top-k smallest distances and their indices
            topk_idxs = np.argsort(dist, axis=1)[:, :num_of_src_polylines]
            map_polylines = batch_polylines[
                topk_idxs]  # (num_center_objects, num_topk_polylines, num_points_each_polyline, 7)
            map_polylines_mask = batch_polylines_mask[
                topk_idxs]  # (num_center_objects, num_topk_polylines, num_points_each_polyline)

        else:
            map_polylines = batch_polylines[None, :, :, :].repeat(num_center_objects, 0)
            map_polylines_mask = batch_polylines_mask[None, :, :].repeat(num_center_objects, 0)

            map_polylines = np.pad(map_polylines,
                                   ((0, 0), (0, num_of_src_polylines - map_polylines.shape[1]), (0, 0), (0, 0)))
            map_polylines_mask = np.pad(map_polylines_mask,
                                        ((0, 0), (0, num_of_src_polylines - map_polylines_mask.shape[1]), (0, 0)))

        map_polylines, map_polylines_mask = transform_to_center_coordinates(
            neighboring_polylines=map_polylines,
            neighboring_polyline_valid_mask=map_polylines_mask
        )

        temp_sum = (map_polylines[:, :, :, 0:3] * map_polylines_mask[:, :, :, None].astype(np.float32)).sum(
            axis=-2)  # (num_center_objects, num_polylines, 3)
        map_polylines_center = temp_sum / np.clip(map_polylines_mask.sum(axis=-1)[:, :, np.newaxis].astype(float),
                                                  a_min=1.0, a_max=None)

        return map_polylines, map_polylines_mask, map_polylines_center

    def sample_from_distribution(self, original_array, m=100):
        distribution = [
            ("-10,0", 0),
            ("0,10", 23.952629169758517),
            ("10,20", 24.611144221251667),
            ("20,30.0", 21.142773679220554),
            ("30,40.0", 15.996653629820514),
            ("40,50.0", 9.446714336574939),
            ("50,60.0", 3.7812939732733786),
            ("60,70", 0.8821063091988663),
            ("70,80.0", 0.1533644322320915),
            ("80,90.0", 0.027777741552241064),
            ("90,100.0", 0.005542507117231198),
        ]

        # Define bins and calculate sample sizes for each bin
        bins = np.array([float(range_.split(',')[1]) for range_, _ in distribution])
        sample_sizes = np.array([round(perc / 100 * m) for _, perc in distribution])

        # Digitize the original array into bins
        bin_indices = np.digitize(original_array, bins)

        # Sample from each bin
        sampled_indices = []
        for i, size in enumerate(sample_sizes):
            # Find indices of original array that fall into current bin
            indices_in_bin = np.where(bin_indices == i)[0]
            # Sample without replacement to avoid duplicates
            sampled_indices_in_bin = np.random.choice(indices_in_bin, size=min(size, len(indices_in_bin)),
                                                      replace=False)
            sampled_indices.extend(sampled_indices_in_bin)

        # Extract the sampled elements and their original indices
        sampled_array = original_array[sampled_indices]
        print('total sample:', len(sampled_indices))
        # Verify distribution (optional, for demonstration)
        for i, (range_, _) in enumerate(distribution):
            print(
                f"Bin {range_}: Expected {distribution[i][1]}%, Actual {len(np.where(bin_indices[sampled_indices] == i)[0]) / len(sampled_indices) * 100}%")

        return sampled_array, sampled_indices


import hydra
from omegaconf import OmegaConf


#@hydra.main(version_base=None, config_path="../configs", config_name="config")
@hydra.main(version_base=None, config_path="../configs", config_name="challenge_config")
def draw_figures(cfg):
    set_seed(cfg.seed)
    OmegaConf.set_struct(cfg, False)  # Open the struct
    cfg = OmegaConf.merge(cfg, cfg.method)
    train_set = build_dataset(cfg)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=True, num_workers=4,
                                               collate_fn=train_set.collate_fn)
    # for data in train_loader:
    #     inp = data['input_dict']
    #     plt = check_loaded_data(inp, 0)
    #     plt.show()

    #concat_list = [4, 4, 4, 4, 4, 4, 4, 4]
    concat_list = [2, 2]
    images = []
    for n, data in tqdm(enumerate(train_loader)):
        for i in range(data['batch_size']):
            plt = check_loaded_data(data['input_dict'], i)
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
            buf.seek(0)
            img = Image.open(buf)
            images.append(img)
        if len(images) >= sum(concat_list):
            break
    final_image = concatenate_varying(images, concat_list)

    # kalman_dict = {}
    # # create 10 buckets with length 10 as the key
    # for i in range(10):
    #     kalman_dict[i] = {}
    #
    # data_list = []
    # for data in train_loader:
    #     inp = data['input_dict']
    #     kalman_diff = inp['kalman_difficulty']
    #     for idx,k in enumerate(kalman_diff):
    #         k6 = np.floor(k[2]/10)
    #         if k6 in kalman_dict and len(kalman_dict[k6]) == 0:
    #             kalman_dict[k6]['kalman'] = k[2]
    #             kalman_dict[k6]['data'] = inp
    #             check_loaded_data()
    #


#@hydra.main(version_base=None, config_path="../configs", config_name="config")
@hydra.main(version_base=None, config_path="../configs", config_name="challenge_config")
def split_data(cfg):
    set_seed(cfg.seed)
    OmegaConf.set_struct(cfg, False)  # Open the struct
    cfg = OmegaConf.merge(cfg, cfg.method)
    train_set = build_dataset(cfg)

    copy_dir = ''
    for data in tqdm(train_set.data_loaded_keys):
        shutil.copy(data, copy_dir)


if __name__ == '__main__':
    from unitraj.datasets import build_dataset
    from unitraj.utils.utils import set_seed
    import io
    from PIL import Image
    from unitraj.utils.visualization import concatenate_varying

    #split_data()
    draw_figures()
