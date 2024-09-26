### This is a parallelized version to prepare forecasting submission files ###
import os
import pickle
import torch
import numpy as np
from tqdm import tqdm
import glob
from multiprocessing import Pool

class PrepareSubForecasting():
    
    def __init__(self, split, folder_root, folder_of_interest, scenario_paths, class_names, saved_predictions_folder, all_predictions_files_by_seq, all_seq_names, process_num=16):
        
        # unitraj_modes = 6
        # unitraj_future_len = 60 # 6 seconds at 10Hz
        self.av_modes = 5
        self.downsampling_factor = 5 # from 10Hz to 2Hz
        self.av_future_len = 6 # 3 seconds at 2Hz
        self.split = split
        self.folder_root = folder_root
        self.folder_of_interest = folder_of_interest
        self.scenario_paths = scenario_paths
        self.class_names = class_names
        self.saved_predictions_folder = saved_predictions_folder
        self.all_predictions_files_by_seq = all_predictions_files_by_seq
        self.process_num = process_num
        self.all_seq_names = all_seq_names

    def get_trajs_and_scores(self, unitraj_proba, unitraj_trajectory):
        # Sort the unitraj trajectories and keep only self.av_modes based on the score
        sorted_indices = np.argsort(unitraj_proba, axis=0)[::-1]

        top_trajectories = unitraj_trajectory[sorted_indices[:self.av_modes]]
        top_scores = unitraj_proba[sorted_indices[:self.av_modes]]

        # Downsample from 10Hz to 2Hz
        top_trajectories = top_trajectories[:, (self.downsampling_factor-1)::self.downsampling_factor]

        # Cut to 3s
        top_trajectories = top_trajectories[:, :self.av_future_len]

        return top_scores, top_trajectories


    def prepare_results(self):

        # Parallelize data preprocessing
        with Pool(processes=self.process_num) as pool:
            results = pool.map(self.prepare_results_chunk, list(range(self.process_num)))
            # concatenate all the results
            all_output = {}
            for result in results:
                all_output.update(result)

        # sanity check, contain all scenarios and all frames?
        print(len(all_output.keys()))
        print(len(self.scenario_paths))
        print(set(self.scenario_paths)^set(all_output.keys()))
        assert len(all_output.keys()) == len(self.scenario_paths)
        # Save the file for the challenge
        print("submission file saved to: ", os.path.join(self.folder_root, self.folder_of_interest + "_submission_final.pkl"))
        with open(os.path.join(self.folder_root, self.folder_of_interest + "_submission_final.pkl"), "wb") as f:
            pickle.dump(all_output, f)
        


    def prepare_results_chunk(self, worker_index):
        
        output = {}
        # to speed up, save all timestamps per scenario once
        with open(f"./scenario_timestamps_{self.split}.pkl", "rb") as f:
            scenario_timestamps = pickle.load(f)

        #print("len(scenario_timestamps) ", len(scenario_timestamps))

        # # to speed up, save all scenarios once
        scenario_vanilla = {} 
        
        # collect files for this worker 
        all_seq_names_chunk = np.array_split(sorted(self.all_seq_names), self.process_num)[worker_index]
        all_predictions_files = []
        for seq_name in all_seq_names_chunk:
            all_predictions_files += self.all_predictions_files_by_seq[seq_name]
        
        print(f"worker {worker_index} has {len(all_predictions_files)} pred files.")
        
        for filename in tqdm(all_predictions_files):
            if '.pt' not in filename:
                continue
            
            pred = torch.load(os.path.join(self.saved_predictions_folder, filename))
            

            scenario_id = pred["scenario_id"]
            if scenario_id not in output:
                output[scenario_id] = {}
                if scenario_id not in scenario_vanilla.keys():
                    with open(self.scenario_paths[scenario_id], "rb") as f:
                        print(f"loading: {self.scenario_paths[scenario_id]} for {scenario_id}...")
                        scenario_vanilla[scenario_id] = pickle.load(f)
                    
                    

            
            
            unitraj_proba = pred["pred_scores"] # should be already sorted by score
            unitraj_trajectory = pred["pred_trajs"]
            

            # Process trajectories
            top_scores, top_trajectories = self.get_trajs_and_scores(unitraj_proba, unitraj_trajectory)

            # Recover the timestamp
            all_timestamps = scenario_timestamps[scenario_id]
            timestamp = all_timestamps[int(pred["current_time_index"])]
            
            if timestamp not in output[scenario_id]:
                output[scenario_id][timestamp] = []

            # get scenario
            object_id = pred["object_id"]
            # print("scenario_vanilla[scenario_id]['tracks']: ", scenario_vanilla[scenario_id]["tracks"].keys())
            full_track_info = scenario_vanilla[scenario_id]["tracks"][object_id]
            assert object_id != 'AV'
            # dict_keys(['type', 'state', 'metadata', 'object_class', 'object_label'])
            # print("full_track_info['state'].keys() ", full_track_info['state'].keys())
                
            # label
            label = self.class_names.index(full_track_info['object_class'])
            assert full_track_info['object_class'] == self.class_names[label]

            # if label !=0:
            #     print("full_track_info['object_class'] ", full_track_info['object_class'])
            # Prepare the output
            
            # get score
            if 'tracking_scores' in full_track_info['state'].keys():
                detection_score = full_track_info['state']['tracking_scores'][int(pred["current_time_index"])]
            else:
                detection_score = np.array([1.0], dtype=np.float32)
                
            # get velocity
            my_velocity = full_track_info['state']['velocity'][int(pred["current_time_index"])]
            formatted_info = {"current_translation_m": pred["gt_trajs"][20, 0:2], # The current detection of the agent of interst
                            "detection_score": detection_score,
                            "size": np.array([full_track_info['state']['length'][int(pred["current_time_index"])], full_track_info['state']['width'][int(pred["current_time_index"])], full_track_info['state']['height'][int(pred["current_time_index"])]], dtype=np.float32), #
                            "label": label, 
                            "name": class_names[label],
                            "prediction_m": top_trajectories,
                            "score": top_scores,
                            "my_velocity": my_velocity,
                            #   "instance_id": len(output[scenario_id][timestamp]), # Incremental counter
                            "instance_id": str(object_id),
                            "yaw": full_track_info['state']['heading'][int(pred["current_time_index"])]
                            }
            # print()

            # Add to the output
            output[scenario_id][timestamp].append(formatted_info)
            
            # rm file
            os.remove(os.path.join(self.saved_predictions_folder, filename))

        return output


split = "val"
tracker = "combine_75_54_scalrV2_full"
base_root = "" #path to scenarionet files
folder_root = f"{base_root}/{tracker}/{split}/"
folder_of_interest = "output_MTR"



# vanilla scenarionet file (for filling data)
scenarios_path = ""#path to scenarionet files
scenarios = glob.glob(f'{scenarios_path}/_*/*.pkl',recursive = True)

# collect scenario_paths
scenario_paths = {}
for scenario_path in scenarios:
    if "sd_av2_v2_" in scenario_path and "cache_test_MTR" not in scenario_path and "output_MTR" not in scenario_path:
        scenario_paths[scenario_path.split('/')[-1].replace("sd_av2_v2_", "").replace(".pkl", "")] = scenario_path

# print(scenario_paths.keys())
print("len(scenario_paths) ", len(scenario_paths))
# Key data
class_names = ['REGULAR_VEHICLE', 'PEDESTRIAN', 'BICYCLIST', 'MOTORCYCLIST', 
                'WHEELED_RIDER', 'BOLLARD', 'CONSTRUCTION_CONE', 'SIGN', 
                'CONSTRUCTION_BARREL', 'STOP_SIGN', 'MOBILE_PEDESTRIAN_CROSSING_SIGN', 
                'LARGE_VEHICLE', 'BUS', 'BOX_TRUCK', 'TRUCK', 'VEHICULAR_TRAILER', 
                'TRUCK_CAB', 'SCHOOL_BUS', 'ARTICULATED_BUS', 'MESSAGE_BOARD_TRAILER', 
                'BICYCLE', 'MOTORCYCLE', 'WHEELED_DEVICE', 'WHEELCHAIR', 'STROLLER', 'DOG']


# Load a UniTraj output prediction
saved_predictions_folder = os.path.join(folder_root, folder_of_interest)
print(f"Loading all files to process in {saved_predictions_folder}...")
all_predictions_files = os.listdir(saved_predictions_folder)
print(f'There are {len(all_predictions_files)} files to process')

# organize all_predictions_files by scenario to parallelization
all_predictions_files_by_seq = {}
all_seq_names = [] # for spliting by worker_id
for pred_file in all_predictions_files:
    seq_name = pred_file.split("___")[0]
    if seq_name not in all_seq_names:
        all_seq_names.append(seq_name)
        all_predictions_files_by_seq[seq_name] = []
    all_predictions_files_by_seq[seq_name].append(pred_file)

print("len(set(all_seq_names)) ", len(set(all_seq_names)))


p_sub = PrepareSubForecasting(split, folder_root, folder_of_interest, scenario_paths, class_names, saved_predictions_folder, all_predictions_files_by_seq, set(all_seq_names))
p_sub.prepare_results()


        
        


