import logging

import tqdm

from scenarionet.converter.utils import mph_to_kmh
import geopandas as gpd
from shapely.ops import unary_union
logger = logging.getLogger(__name__)
import numpy as np

from pathlib import Path
from tqdm import tqdm
from metadrive.scenario import ScenarioDescription as SD
from metadrive.type import MetaDriveType

from scenarionet.converter.argoverse2.type import get_traffic_obj_type, get_lane_type, get_lane_mark_type
from av2.datasets.motion_forecasting import scenario_serialization
from av2.map.map_api import ArgoverseStaticMap
from typing import Final
from shapely.geometry import Point, Polygon

#xyh#
import pickle
from av2.utils.io import read_feather
import os
from av2.datasets.sensor.av2_sensor_dataloader import AV2SensorDataLoader
from av2.datasets.motion_forecasting.data_schema import (
    ArgoverseScenario,
    ObjectState,
    ObjectType,
    Track,
    TrackCategory,
)
from av2.geometry.geometry import quat_to_mat
import math
from av2.datasets.sensor.sensor_dataloader import read_city_SE3_ego
VELOCITY_SAMPLING_RATE = 5

_ESTIMATED_VEHICLE_LENGTH_M: Final[float] = 4.0
_ESTIMATED_VEHICLE_WIDTH_M: Final[float] = 2.0
# _ESTIMATED_CYCLIST_LENGTH_M: Final[float] = 2.0
# _ESTIMATED_CYCLIST_WIDTH_M: Final[float] = 0.7
# _ESTIMATED_PEDESTRIAN_LENGTH_M: Final[float] = 0.5
# _ESTIMATED_PEDESTRIAN_WIDTH_M: Final[float] = 0.5
# _ESTIMATED_BUS_LENGTH_M: Final[float] = 12.0
# _ESTIMATED_BUS_WIDTH_M: Final[float] = 2.5

_HIGHWAY_SPEED_LIMIT_MPH: Final[float] = 85.0

class_names = ('REGULAR_VEHICLE', 'PEDESTRIAN', 'BICYCLIST', 'MOTORCYCLIST', 'WHEELED_RIDER', 
               'BOLLARD', 'CONSTRUCTION_CONE', 'SIGN', 'CONSTRUCTION_BARREL', 'STOP_SIGN', 
               'MOBILE_PEDESTRIAN_CROSSING_SIGN','LARGE_VEHICLE', 'BUS', 'BOX_TRUCK', 'TRUCK',
               'VEHICULAR_TRAILER', 'TRUCK_CAB', 'SCHOOL_BUS', 'ARTICULATED_BUS','MESSAGE_BOARD_TRAILER', 
               'BICYCLE', 'MOTORCYCLE', 'WHEELED_DEVICE', 'WHEELCHAIR', 'STROLLER', 'DOG')

def extract_tracks(tracks, sdc_idx, track_length):
    ret = dict()

    def _object_state_template(object_id):
        return dict(type=None, state=dict(# Never add extra dim if the value is scalar.
            position=np.zeros([track_length, 3], dtype=np.float32), length=np.zeros([track_length], dtype=np.float32),
            width=np.zeros([track_length], dtype=np.float32), height=np.zeros([track_length], dtype=np.float32),
            #xyh added to record timestamp
            timestamps=np.zeros([track_length], dtype=np.int64)-1,
            # tracking scores
            tracking_scores=np.zeros([track_length], dtype=np.float32),
            heading=np.zeros([track_length], dtype=np.float32), velocity=np.zeros([track_length, 2], dtype=np.float32),
            valid=np.zeros([track_length], dtype=bool), ),
            metadata=dict(track_length=track_length, type=None, object_id=object_id, dataset="av2", object_class=None))

    track_category = []
    check_uniqueness_track_ids = []
    for obj in tracks: #for loop Track
        object_id = obj.track_id # str value
        
        check_uniqueness_track_ids.append(object_id)
        # break
        track_category.append(obj.category)
        # empty past is zero padded
        obj_state = _object_state_template(object_id)
        # Transform it to Waymo type string
        # TODO replace with real whl done
        
        #TODO xyh added
        # converted type under MetaDriveType format        
        obj_state["type"] = get_traffic_obj_type(obj.object_type)
        # real (detailed) object class
        obj_state['object_class'] = obj.object_type
        #TODO xyh added
        obj_state['object_label'] = obj.label
        for _, state in enumerate(obj.object_states):
            step_count = state.timestep
            
            # record timestamp
            obj_state["state"]["timestamps"][step_count] = state.timestamp
            
            # record position
            # print(state.position)
            obj_state["state"]["position"][step_count][0] = state.position[0]
            obj_state["state"]["position"][step_count][1] = state.position[1]
            obj_state["state"]["position"][step_count][2] = 0
            
            # record scores
            obj_state["state"]['tracking_scores'][step_count] = state.score
            # if object_id != 'AV':
            #     print(object_id)
            #     print("obj_state['state']['tracking_scores'][step_count] ", obj_state['state']['tracking_scores'][step_count])

            # l = [state.length for state in obj.states]
            # w = [state.width for state in obj.states]
            # h = [state.height for state in obj.states]
            # obj_state["state"]["size"] = np.stack([l, w, h], 1).astype("float32")
            length = state.lwh[0]
            width = state.lwh[1]
            height = state.lwh[2]
            obj_state["state"]["length"][step_count] = length
            obj_state["state"]["width"][step_count] = width
            obj_state["state"]["height"][step_count] = height

            # heading = [state.heading for state in obj.states]
            obj_state["state"]["heading"][step_count] = state.heading

            obj_state["state"]["velocity"][step_count][0] = state.velocity[0]
            obj_state["state"]["velocity"][step_count][1] = state.velocity[1]

            obj_state["state"]["valid"][step_count] = True

        obj_state["metadata"]["type"] = obj_state["type"]
        # print("obj_state['state']['position'] ", obj_state['state']['position'])
        # print()
        ret[object_id] = obj_state
    
    assert len(set(check_uniqueness_track_ids)) == len(check_uniqueness_track_ids)
    return ret, track_category


def extract_lane_mark(lane_mark):
    line = dict()
    line["type"] = get_lane_mark_type(lane_mark.mark_type)
    line["polyline"] = lane_mark.polyline.astype(np.float32)
    return line


def extract_map_features(map_features):
    # with open(
    #         "/Users/fenglan/Desktop/vita-group/code/mdsn/scenarionet/data_sample/waymo_converted_0/sd_waymo_v1.2_7e8422433c66cc13.pkl",
    #         'rb') as f:
    #     waymo_sample = pickle.load(f)
    ret = {}
    vector_lane_segments = map_features.get_scenario_lane_segments()
    vector_drivable_areas = map_features.get_scenario_vector_drivable_areas()
    ped_crossings = map_features.get_scenario_ped_crossings()

    ids = map_features.get_scenario_lane_segment_ids()

    max_id = max(ids)
    for seg in vector_lane_segments:
        center = {}
        lane_id = str(seg.id)

        left_id = str(seg.id + max_id + 1)
        right_id = str(seg.id + max_id + 2)
        left_marking = extract_lane_mark(seg.left_lane_marking)
        right_marking = extract_lane_mark(seg.right_lane_marking)

        ret[left_id] = left_marking
        ret[right_id] = right_marking

        center["speed_limit_mph"] = _HIGHWAY_SPEED_LIMIT_MPH

        center["speed_limit_kmh"] = mph_to_kmh(_HIGHWAY_SPEED_LIMIT_MPH)

        center["type"] = get_lane_type(seg.lane_type)

        polyline = map_features.get_lane_segment_centerline(seg.id)
        center["polyline"] = polyline.astype(np.float32)

        center["interpolating"] = True

        center["entry_lanes"] = [str(id) for id in seg.predecessors]

        center["exit_lanes"] = [str(id) for id in seg.successors]

        center["left_boundaries"] = []

        center["right_boundaries"] = []

        center["left_neighbor"] = []

        center["right_neighbor"] = []
        center['width'] = np.zeros([len(polyline), 2], dtype=np.float32)

        ret[lane_id] = center

    polygons = []
    for polygon in vector_drivable_areas:
        # convert to shapely polygon
        points = polygon.area_boundary
        polygons.append(Polygon([(p.x, p.y) for p in points]))

    polygons = [geom if geom.is_valid else geom.buffer(0) for geom in polygons]
    boundaries = gpd.GeoSeries(unary_union(polygons)).boundary.explode(index_parts=True)
    for idx, boundary in enumerate(boundaries[0]):
        block_points = np.array(list(i for i in zip(boundary.coords.xy[0], boundary.coords.xy[1])))
        for i in range(0, len(block_points), 20):
            id = f'boundary_{idx}{i}'
            ret[id] = {SD.TYPE: MetaDriveType.LINE_SOLID_SINGLE_WHITE, SD.POLYLINE: block_points[i:i + 20]}

    for cross in ped_crossings:
        bound = dict()
        bound["type"] = MetaDriveType.CROSSWALK
        bound["polygon"] = cross.polygon.astype(np.float32)
        ret[str(cross.id)] = bound

    return ret


def get_av2_scenarios(av2_data_directory, start_index, num):
    # parse raw data from input path to output path,
    # there is 1000 raw data in google cloud, each of them produce about 500 pkl file
    logger.info("\nReading raw data")
    print("av2_data_directory ", av2_data_directory)
    #TODO file all scenarios_path
    # done
    if '/merge/' in av2_data_directory:
        all_scenario_files = sorted(Path(av2_data_directory).rglob("*/*/city_SE3_egovehicle.feather"))
    else:
        all_scenario_files = sorted(Path(av2_data_directory).rglob("city_SE3_egovehicle.feather"))
    print("len(all_scenario_files) ", len(all_scenario_files))
    return all_scenario_files


def convert_av2_scenario(scenario, version):
    md_scenario = SD()

    md_scenario[SD.ID] = scenario.scenario_id
    md_scenario[SD.VERSION] = version

    # Please note that SDC track index is not identical to sdc_id.
    # sdc_id is a unique indicator to a track, while sdc_track_index is only the index of the sdc track
    # in the tracks datastructure.

    track_length = scenario.timestamps_ns.shape[0]
    print("track_length ", track_length)

    tracks, category = extract_tracks(scenario.tracks, scenario.focal_track_id, track_length)

    md_scenario[SD.LENGTH] = track_length

    md_scenario[SD.TRACKS] = tracks

    md_scenario[SD.DYNAMIC_MAP_STATES] = {}

    #TODO check map features compatibility
    map_features = extract_map_features(scenario.static_map)
    md_scenario[SD.MAP_FEATURES] = map_features

    # compute_width(md_scenario[SD.MAP_FEATURES])

    md_scenario[SD.METADATA] = {}
    md_scenario[SD.METADATA][SD.ID] = md_scenario[SD.ID]
    md_scenario[SD.METADATA][SD.COORDINATE] = MetaDriveType.COORDINATE_WAYMO
    md_scenario[SD.METADATA][SD.TIMESTEP] = np.array(list(range(track_length))) / 10 #???
    md_scenario[SD.METADATA][SD.METADRIVE_PROCESSED] = False
    md_scenario[SD.METADATA][SD.SDC_ID] = 'AV'
    md_scenario[SD.METADATA]["dataset"] = "av2"
    md_scenario[SD.METADATA]["scenario_id"] = scenario.scenario_id
    md_scenario[SD.METADATA]["source_file"] = scenario.scenario_id
    md_scenario[SD.METADATA]["track_length"] = track_length

    # === Waymo specific data. Storing them here ===
    md_scenario[SD.METADATA]["current_time_index"] = 49

    # obj id
    obj_keys = list(tracks.keys())
    
    #TODO
    md_scenario[SD.METADATA]["objects_of_interest"] = [obj_keys[idx] for idx, cat in enumerate(category) if cat == 2]

    # the egocar has track_id: 'AV'
    md_scenario[SD.METADATA]["sdc_track_index"] = obj_keys.index('AV')
    # #TODO xyh added
    # md_scenario[SD.METADATA]["sdc_track_index"] = 0

    track_id = scenario.focal_track_id
    print("len(tracks) ", len(tracks))
    
    md_scenario[SD.METADATA]["tracks_to_predict"] = {
        id: {
            "track_index": obj_keys.index(id),
            "track_id": id,
            "difficulty": 0,
            "object_type": tracks[id]["type"]
        }
        for count, id in enumerate(track_id)
    }
    # clean memory
    print(len(md_scenario[SD.METADATA]["tracks_to_predict"].keys()))
    del scenario
    return md_scenario


def preprocess_av2_scenarios(files, worker_index):
    """
    Convert the waymo files into scenario_pb2. This happens in each worker.
    :param files: a list of file path
    :param worker_index, the index for the worker
    :return: a list of scenario_pb2
    """
    '''
    av2 sensor data structure: #scenario = scene
    |--train
        |--scenario_1
            |-- calibration # ignored
            |-- sensors # ignored 
                |-- lidar
                |-- cameras
            |-- annotations.feather #all annoatations within the scenario
            |-- city_SE3_egovehicle.feather # egocar information
            |-- map
                |-- [scenario_1]___img_Sim2_city.json #?
                |-- [scenario_1]_ground_height_surface____MIA.npy #?
                |-- log_map_archive_[scenario_1]_ground_height_surface____MIA_city_47894.json # HD map?
        |--scenario_2
            |-- calibration # ignored
            |-- sensors # ignored 
            |-- annotations.feather #all annoatations within the scenario
            |-- city_SE3_egovehicle.feather # egocar information
            |-- map
                |-- [scenario_2]___img_Sim2_city.json #?
                |-- [scenario_2]_ground_height_surface____MIA.npy #?
                |-- log_map_archive_[scenario_2]____MIA_city_47894.json # HD map?
        ...
    |--val
    ...
    |--test   
    '''
    
    av2_dataloader = None
    all_log_ids = None
    #TODO read .feather, map
    for scenario_path in tqdm(files, desc="Process av2 SENSOR scenarios for worker {}".format(worker_index)):
        if av2_dataloader is None: # dataloader for all seqeunces in the train/val/seq
            av2_dataloader = AV2SensorDataLoader(Path("/".join(str(scenario_path.parents[0]).split('/')[:-1])), Path("/".join(str(scenario_path.parents[0]).split('/')[:-1])))
            all_log_ids = av2_dataloader.get_log_ids()
            print("all_log_ids ", all_log_ids)
            
        scenario_id = str(scenario_path.parents[0]).split('/')[-1]
        if scenario_id not in all_log_ids: # change from train to val
            av2_dataloader = AV2SensorDataLoader(Path("/".join(str(scenario_path.parents[0]).split('/')[:-1])), Path("/".join(str(scenario_path.parents[0]).split('/')[:-1])))
            all_log_ids = av2_dataloader.get_log_ids()
        
        assert scenario_id in all_log_ids
             
        print("scenario_id: ", scenario_id)
        
        # find static map:
        tmp_path = os.listdir(scenario_path.parents[0] / 'map')
        map_name = [pth for pth in tmp_path if 'log_map_archive' in pth]
        assert len(map_name) == 1
        static_map_path = (scenario_path.parents[0] / f'map/{map_name[-1]}')
        print("static_map_path: ", static_map_path)
        print("Line 363, please input noisy tracking result input path: ")
        print("str(scenario_path.parents[0])", str(scenario_path.parents[0]))
        if "/test/" in str(scenario_path.parents[0]):
            annotation_pth = "./test_tracking.pkl"
        elif "/val/" in str(scenario_path.parents[0]):
            annotation_pth = "./val_tracking.pkl"
        elif '/train/' in str(scenario_path.parents[0]):
            annotation_pth = "./train_tracking.pkl"
        else:
            print("tracking result is missing.")
            assert False
        assert os.path.exists(annotation_pth)
        #get panda file of annoations
        if '/merge/' in str(scenario_path.parents[0]):
            all_raw_annotations = {}
            for ann_path in annotation_pth:
                print("Reading tracking results from: ", ann_path)
                with open(ann_path, "rb") as f:
                    a = pickle.load(f) # dict of a list of info about all frames in the scene/log_id
                all_raw_annotations.update(a)
            print("len(all_raw_annotations): ", len(all_raw_annotations))
            assert len(all_raw_annotations) == 850
        else:
            print("reading tracking results from: ", annotation_pth)
            with open(annotation_pth, "rb") as f:
                all_raw_annotations = pickle.load(f) # dict of a list of info about all frames in the scene/log_id
        
        # load city_name, scenario_id, timestamps_ns (list of start_timestamp, end_timestamp, num_timestamps), tracks, map_id, slice_id
        scenario = load_argoverse_scenario(av2_dataloader, scenario_id, str(scenario_path.parents[0]), all_raw_annotations)
        
        #TODO does it work?
        static_map = ArgoverseStaticMap.from_json(static_map_path)
        scenario.static_map = static_map
        
        
        yield scenario

    # logger.info("Worker {}: Process {} waymo scenarios".format(worker_index, len(scenarios)))  # return scenarios
    # for testset: all_timestamps = sorted([int(filename.split(".")[0]) for filename in os.listdir(lidar_paths)])

#xyh
# load city_name, scenario_id, timestamps_ns (list of timestamps), tracks, map_id, slice_id
def load_argoverse_scenario(av2_dataloader, scenario_id, scenario_path, all_raw_annotations):
    lidar_paths = str(scenario_path + "/sensors/lidar/")

    #TODO we can subsample time stamps here, for example VAL_SAMPLE_RATE = 5 => 2Hz if i % VAL_SAMPLE_RATE != 0: continue 
    all_timestamps = sorted([int(filename.split(".")[0]) for filename in os.listdir(lidar_paths)])
    city_name = av2_dataloader.get_city_name(scenario_id)
    timestamps_ns = all_timestamps
    
    # print("scenario_id: ", scenario_id)
    # print("city_name: ", city_name)
    # print('seq length: ', len(all_timestamps))
    
    # 1) generate Noisy tracks for train, val, test
    
    tracks_by_id = dict() # collect all trajectories for each object indexed by id {track_id: []} for this scenario/scene
    focal_track_id = []
    if all_raw_annotations is not None:
        #get panda file of annoations
        raw_annotations = all_raw_annotations[scenario_id] # a list of info about all frames in the scene/log_id
        # print("scenario_id: ", scenario_id)
        annotations = {} # {timestamps: info, timestamps: info, ...}
        for raw_ann in raw_annotations:
            assert raw_ann['timestamp_ns'] not in annotations.keys()
            annotations[raw_ann['timestamp_ns']] = raw_ann
        
        tracks_by_id["AV"] = [] # list of ordered ObjectState for ego car
        for timestep, ts in enumerate(all_timestamps): #ordered
            curr_annotations = annotations[ts] # anns for current timestamp
            curr_track_ids = curr_annotations['track_id'].tolist()
            # print("cur time: ", ts)
            # print("#objects: ", len(curr_track_ids))
            assert len(set(curr_track_ids)) == len(curr_track_ids) #unique
            
            # collect ego car info this timestamp
            observed = True
            timestamp_city_SE3_ego_dict = read_city_SE3_ego(Path(scenario_path))
            if ts in timestamp_city_SE3_ego_dict:
                ego_to_city_SE3 = timestamp_city_SE3_ego_dict[ts]
                ego_position = ego_to_city_SE3.translation.tolist()[:2]
                rot = ego_to_city_SE3.rotation
                ego_heading = wrap_pi(np.array([math.atan2(rot[1, 0], rot[0, 0])], dtype=np.float32))
                ego_velocity = box_velocity_ego(ts, all_timestamps, timestamp_city_SE3_ego_dict)
                # print("ego_position ", ego_position)
                # print("ego_heading ", ego_heading)
                # print("ego_velocity ", ego_velocity)
                # save info in ObjectState and push to tracks_by_id[track_id]
                obj = ObjectState(
                    observed=observed,
                    timestep=timestep,
                    position=ego_position, # only record (x, y)
                    heading=ego_heading[0], #yaw, float
                    velocity=ego_velocity.tolist()[:2], # (vx, vy)
                    )
                # extra info
                obj.lwh = np.array([_ESTIMATED_VEHICLE_LENGTH_M, _ESTIMATED_VEHICLE_WIDTH_M, 1.0]) # fake lwh, because no lwh info for ego
                obj.timestamp = ts
                obj.score = 1.0
                tracks_by_id["AV"].append(obj)
                
                
                # ego_pos = 
                # ego_heading =
                # velocity =
                # timestep = 
            else:
                print("ts ", ts)
                print("timestamp_city_SE3_ego_dict.keys() ", timestamp_city_SE3_ego_dict.keys())
                print(f"no ego info for timestamp {ts}.")
                assert False
                       
            
            assert len(curr_annotations['translation_m']) == len(curr_annotations['score']) 
            assert len(curr_annotations['label']) == len(curr_annotations['score'])
            assert len(curr_annotations['name']) == len(curr_annotations['score'])
            assert len(curr_annotations['yaw']) == len(curr_annotations['score'])
            assert len(curr_annotations['velocity_m_per_s']) == len(curr_annotations['score'])
            # assert len(curr_annotations['detection_score']) == len(curr_annotations['score'])
            
            # collecting normala agents
            for record_idx, track_id in enumerate(curr_track_ids): #record track info for cur timestamp
                # print("track_id ", track_id)
                
                # prepare elems for ObjectState (info for an object at a timestep)#
                observed = True
                
                # get object type:
                # print(ann_track['category'])
                object_type = curr_annotations['name'][record_idx]
                if object_type not in class_names:
                    print(f'{object_type} not in class_names, skip.')
                    continue
                
                if track_id not in tracks_by_id.keys():
                    tracks_by_id[track_id] = [] # list of ordered ObjectState
                
                focal_track_id.append(str(track_id))
                
                # all info are in global reference frame
                position = curr_annotations['translation_m'][record_idx][np.newaxis, :]           
                lwh = curr_annotations['size'][record_idx]
                
                heading = wrap_pi(np.array([curr_annotations['yaw'][record_idx]],dtype=np.float32))
                # print("track_id before:", track_id)
                #TODO to check 
                velocity = np.array(box_velocity(curr_annotations, ts, record_idx, all_timestamps, annotations)[:2])
                # save info in ObjectState and push to tracks_by_id[track_id]
                obj = ObjectState(
                    observed=observed,
                    timestep=timestep,
                    position=position[0].tolist()[:2], # only record (x, y)
                    heading=float(heading[0]), #yaw, float
                    velocity=velocity.tolist()[:2], # (vx, vy)
                    )
                # extra info
                obj.lwh = lwh
                obj.timestamp = ts
                obj.classname = curr_annotations['name'][record_idx]
                obj.score = curr_annotations['score'][record_idx]
                tracks_by_id[track_id].append(obj)
                # break
                
        # record each track (pos from all timestamps) as Track and save it to list
        #TODO handle object class!!!
        tracks: List[Track] = []
        for t_id, obj in tracks_by_id.items():
            object_category = 2 # https://github.com/argoverse/av2-api/blob/bcde90f2b66fbe1f7fb550bf358660bf0c7ed65a/src/av2/datasets/motion_forecasting/data_schema.py#L18
            # get object type:
            if t_id == 'AV': # all ego car is regular vehicle
                object_type = 'REGULAR_VEHICLE'
                class_label = 0
            else:
                
                # print(ann_track['category'])
                object_type = obj[0].classname
                assert object_type in class_names
                class_label = class_names.index(object_type)
            
            # save tracks
            track = Track(
                track_id=str(t_id),
                object_states=obj,
                object_type=object_type,
                category=object_category,
                )
            track.label = class_label
            tracks.append(track)
    else: # testset, only ego
        tracks_by_id = dict() # collect all trajectories for each object indexed by id {track_id: []}
        tracks_by_id["AV"] = [] # list of ordered ObjectState for ego car
        for timestep, ts in enumerate(all_timestamps): #ordered            
            # collect ego car info this timestamp
            observed = True
            timestamp_city_SE3_ego_dict = read_city_SE3_ego(Path(scenario_path))
            if ts in timestamp_city_SE3_ego_dict:
                ego_to_city_SE3 = timestamp_city_SE3_ego_dict[ts]
                ego_position = ego_to_city_SE3.translation.tolist()[:2]
                rot = ego_to_city_SE3.rotation
                ego_heading = wrap_pi(np.array([math.atan2(rot[1, 0], rot[0, 0])], dtype=np.float32))
                ego_velocity = box_velocity_ego(ts, all_timestamps, timestamp_city_SE3_ego_dict)
                # print("ego_position ", ego_position)
                # print("ego_heading ", ego_heading)
                # print("ego_velocity ", ego_velocity)
                # save info in ObjectState and push to tracks_by_id[track_id]
                obj = ObjectState(
                    observed=observed,
                    timestep=timestep,
                    position=ego_position, # only record (x, y)
                    heading=ego_heading[0], #yaw, float
                    velocity=ego_velocity.tolist()[:2], # (vx, vy)
                    )
                # extra info
                obj.lwh = np.array([_ESTIMATED_VEHICLE_LENGTH_M, _ESTIMATED_VEHICLE_WIDTH_M, 1.0]) # fake lwh, because no lwh info for ego
                obj.timestamp = ts
                tracks_by_id["AV"].append(obj)
                obj.score = 1.0
                
                # ego_pos = 
                # ego_heading =
                # velocity =
                # timestep = 
            else:
                print("ts ", ts)
                print("timestamp_city_SE3_ego_dict.keys() ", timestamp_city_SE3_ego_dict.keys())
                
                print(f"no ego info for timestamp {ts}.")
                assert False
        
        #TODO handle object class!!!
        tracks: List[Track] = []
        for t_id, obj in tracks_by_id.items():
            object_category = 2 # https://github.com/argoverse/av2-api/blob/bcde90f2b66fbe1f7fb550bf358660bf0c7ed65a/src/av2/datasets/motion_forecasting/data_schema.py#L18
            # get object type:
            assert t_id == "AV"
            object_type = 'REGULAR_VEHICLE'
            class_label = 0
            
            # save tracks
            track = Track(
                track_id=t_id,
                object_states=obj,
                object_type=object_type,
                category=object_category,
                )
            track.label = class_label
            tracks.append(track)
        
        
        
    
    
    # 2) additional info
    map_id = None
    slice_id = None
    
    
    # print("all_timestamps ", all_timestamps)
    return ArgoverseScenario(
        scenario_id=scenario_id,
        timestamps_ns=np.array(timestamps_ns),
        tracks=tracks,
        focal_track_id=list(set(focal_track_id)),
        city_name=city_name,
        map_id=map_id,
        slice_id=slice_id,
    )
    

def wrap_pi(theta) :
    theta = np.remainder(theta, 2 * np.pi)
    theta[theta > np.pi] -= 2 * np.pi
    return theta

def transform_to_global_reference(log_dir, current_timestamp_ns, box, velocity, yaw):
    # load ego car info for current timestamp 
    timestamp_city_SE3_ego_dict = read_city_SE3_ego(log_dir)
    ego_to_city_SE3 = timestamp_city_SE3_ego_dict[current_timestamp_ns]
    
    # print(box.shape)
    # print(box)
    # transform xyz, velocity and yaw to city reference frame
    box = ego_to_city_SE3.transform_from(
        box
    )  # I (number of instances), 3

    rotation = ego_to_city_SE3.rotation.astype(np.float32)
    velocity_3d = np.pad(
        velocity, [(0, 0), (0, 1)]
    )  # pad last dimension -> [x, y, 0]
    # print("velocity_3d ", velocity_3d)
    velocity = velocity_3d @ rotation.T  # I, 3
    
    ego_to_city_yaw = math.atan2(rotation[1, 0], rotation[0, 0])
    yaw = wrap_pi(yaw + ego_to_city_yaw)  # I

    return box, velocity, yaw

def box_velocity(current_annotation, current_timestamp_ns, record_idx, all_timestamps, annotations):
    curr_index = all_timestamps.index(current_timestamp_ns)
    prev_index = curr_index - VELOCITY_SAMPLING_RATE
    next_index = curr_index + VELOCITY_SAMPLING_RATE

    track_uuid = current_annotation['track_id'][record_idx]

    if prev_index > 0:
        prev_timestamp_ns = all_timestamps[prev_index]

        #get annotation in prev timestamp
        prev_annotations = annotations[prev_timestamp_ns]
        prev_annotation = prev_annotations

        if track_uuid not in prev_annotations['track_id'].tolist():
            prev_annotation = None
    else:
        prev_annotation = None 

    if next_index < len(all_timestamps):
        next_timestamp_ns = all_timestamps[next_index]

        #get annotation in next timestamp
        next_annotations = annotations[next_timestamp_ns]
        next_annotation = next_annotations

        if track_uuid not in next_annotations['track_id'].tolist():
            next_annotation = None
    else:
        next_annotation = None 

    if prev_annotation is None and next_annotation is None:
        return np.array([0, 0, 0])

    # take centered average of displacement for velocity
    if prev_annotation is not None and next_annotation is not None:

        prev_translation = prev_annotations['translation_m'][prev_annotations['track_id'].tolist().index(track_uuid)]
        next_translation = next_annotations['translation_m'][next_annotations['track_id'].tolist().index(track_uuid)]


        delta_t = (next_timestamp_ns - prev_timestamp_ns) * 1e-9
        return (next_translation - prev_translation) / delta_t

    # take one-sided average of displacement for velocity
    else:
        if prev_annotation is not None:

            prev_translation = prev_annotations['translation_m'][prev_annotations['track_id'].tolist().index(track_uuid)]
            current_translation = current_annotation['translation_m'][record_idx]


            delta_t = (current_timestamp_ns - prev_timestamp_ns) * 1e-9
            return (current_translation - prev_translation) / delta_t

        if next_annotation is not None:

            current_translation = current_annotation['translation_m'][record_idx]
            next_translation = next_annotations['translation_m'][next_annotations['track_id'].tolist().index(track_uuid)]

            
            delta_t = (next_timestamp_ns - current_timestamp_ns) * 1e-9
            return (next_translation - current_translation) / delta_t



def box_velocity_ego(current_timestamp_ns, all_timestamps, timestamp_city_SE3_ego_dict):
    city_SE3_ego_reference = timestamp_city_SE3_ego_dict[current_timestamp_ns]

    curr_index = all_timestamps.index(current_timestamp_ns)
    prev_index = curr_index - VELOCITY_SAMPLING_RATE
    next_index = curr_index + VELOCITY_SAMPLING_RATE

    if prev_index > 0:
        prev_timestamp_ns = all_timestamps[prev_index]

        #get annotation in prev timestamp
        if prev_timestamp_ns in timestamp_city_SE3_ego_dict:
            prev_annotation = timestamp_city_SE3_ego_dict[prev_timestamp_ns]
        else:
            prev_annotation = None
    else:
        prev_annotation = None 

    if next_index < len(all_timestamps):
        next_timestamp_ns = all_timestamps[next_index]

        #get annotation in next timestamp
        if next_timestamp_ns in timestamp_city_SE3_ego_dict:
            next_annotation = timestamp_city_SE3_ego_dict[next_timestamp_ns]
        else:
            next_annotation = None
    else:
        next_annotation = None 

    if prev_annotation is None and next_annotation is None:
        return np.array([0, 0, 0])

    # take centered average of displacement for velocity
    if prev_annotation is not None and next_annotation is not None:

        prev_translation = prev_annotation.translation   
        next_translation = next_annotation.translation  


        delta_t = (next_timestamp_ns - prev_timestamp_ns) * 1e-9
        return (next_translation - prev_translation) / delta_t

    # take one-sided average of displacement for velocity
    else:
        if prev_annotation is not None:

            prev_translation = prev_annotation.translation  
            current_translation = city_SE3_ego_reference.translation  


            delta_t = (current_timestamp_ns - prev_timestamp_ns) * 1e-9
            return (current_translation - prev_translation) / delta_t

        if next_annotation is not None:

            current_translation = city_SE3_ego_reference.translation   
            next_translation = next_annotation.translation

            
            delta_t = (next_timestamp_ns - current_timestamp_ns) * 1e-9
            return (next_translation - current_translation) / delta_t
