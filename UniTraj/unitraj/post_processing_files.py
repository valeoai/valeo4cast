import os
import pickle
import argparse

import numpy as np

from tqdm import tqdm
from av2.evaluation.forecasting.utils import (agent_velocity_m_per_s,
                                              trajectory_type)

AV2_CLASS_NAMES = ['REGULAR_VEHICLE', 'PEDESTRIAN', 'BOLLARD', 'CONSTRUCTION_CONE', 'CONSTRUCTION_BARREL', 'STOP_SIGN', 'BICYCLE', 'WHEELED_DEVICE', 'SIGN', 'BOX_TRUCK', 'VEHICULAR_TRAILER', 'BUS', 'LARGE_VEHICLE', 'TRUCK',  'MOTORCYCLE', 'TRUCK_CAB', 'WHEELED_RIDER', 'BICYCLIST', 'MOTORCYCLIST', 'SCHOOL_BUS',  'DOG', 'STROLLER', 'ARTICULATED_BUS', 'WHEELCHAIR', 'MOBILE_PEDESTRIAN_CROSSING_SIGN', 'MESSAGE_BOARD_TRAILER']

static_only_classes = set(['BOLLARD', 'CONSTRUCTION_CONE', 'CONSTRUCTION_BARREL', 'SIGN', 'MOBILE_PEDESTRIAN_CROSSING_SIGN', 'MESSAGE_BOARD_TRAILER'])
no_linear_classes = set(['BOLLARD', 'CONSTRUCTION_CONE', 'CONSTRUCTION_BARREL', 'SIGN', 'MOBILE_PEDESTRIAN_CROSSING_SIGN', 'MESSAGE_BOARD_TRAILER'])
no_non_linear_classes = set(['BOLLARD', 'CONSTRUCTION_CONE', 'CONSTRUCTION_BARREL', 'SIGN', 'MOBILE_PEDESTRIAN_CROSSING_SIGN', 'MESSAGE_BOARD_TRAILER', "WHEELED_RIDER", "BICYCLIST", "SCHOOL_BUS", "DOG", "STROLLER", "ARTICULATED_BUS", "WHEELCHAIR"])

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

TRAJ_TYPES = ["static", "linear", "non-linear"]

classes_to_hack = set(AV2_CLASS_NAMES)

def hack_the_pickle(predictions):
    time = np.linspace(0.5, 3, 6)

    predictions_hacked = {}
    for scenario_id, prediction in tqdm(list(predictions.items())):
        predictions_hacked[scenario_id] = {}
        for timestamp, prediction_t in prediction.items():
            predictions_hacked[scenario_id][timestamp] = []
            for prediction_t_i in prediction_t:
                prediction_t_i_hacked = prediction_t_i

                prediction_t_i["velocity_m_per_s"] = agent_velocity_m_per_s(prediction_t_i)
                traj_type = trajectory_type(
                        prediction_t_i, CATEGORY_TO_VELOCITY_M_PER_S
                    )

                ######
                # Insert a static trajectory with probability 1.
                ######
                # Get position of the static traj
                try:
                    static_index = traj_type.index("static")
                except:
                    static_index = -1
                prediction_t_i_hacked["prediction_m"][static_index] = np.tile(prediction_t_i_hacked["current_translation_m"], (6, 1))

                prediction_t_i_hacked["score"][static_index] = 1.

                ######
                # For static only objects classes, only have a single static trajectory
                ######
                if prediction_t_i["name"] in static_only_classes:
                    for i, tt in enumerate(traj_type):
                        if i != static_index:
                            prediction_t_i_hacked["score"][i] = 0.

                predictions_hacked[scenario_id][timestamp].append(prediction_t_i)
    return predictions_hacked

def parse_args():
    parser = argparse.ArgumentParser(description="post_processing")
    parser.add_argument("--pickle_file", type=str, default="")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    prediction_file = args.pickle_file
    prediction_file_hacked = prediction_file.replace(".pkl", "_processed.pkl")

    # Load the pickle
    print("loading pickled predictions...")
    with open(prediction_file, "rb") as f:
        predictions = pickle.load(f)
    print("done")

    print("processing the pickle")
    predictions_hacked = hack_the_pickle(predictions)

    # Dump into the new pickle
    print("pickling hacked predictions...")
    with open(prediction_file_hacked, "wb") as f:
        pickle.dump(predictions_hacked, f)
    print("done")
