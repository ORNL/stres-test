import os
import sys
import re
import json
import h5py
import time
import random
import argparse

import numpy as np
import pickle as pkl

from PIL import Image
from pathlib import Path
from scipy.signal import butter, filtfilt
from statistics import mode as stat_mode

META_DATA_LOADING_WAIT_TIME = 600 # seconds # 60 FOR DEBUGGING PURPOSES

INITIAL_RAND_SEED = 123
SHUFFLE_SEED_1 = 1
SHUFFLE_SEED_2 = 2
SHUFFLE_SEED_3 = 3
SHUFFLE_SEED_4 = 4
SHUFFLE_SEED_5 = 5
SHUFFLE_SEED_6 = 6

UNUSABLE_EVENTS = ['us7000jvl3', ] # events to be excluded from the dataset, usually because of the large amount of nan values in time series recordings 

LOWPASS_CUTOFF_FREQ = 49#50#hz
ORIGINAL_NEIC_SAMPLED_FREQ = 100#hz
PWAVE_ONSET_SECOND_MARK = 30
TIME_STEP_FROM_WHICH_TO_SAMPLE_WINDOWS_IN_ORIG_RECORDINGS = PWAVE_ONSET_SECOND_MARK * ORIGINAL_NEIC_SAMPLED_FREQ
BUTTER_FILTER_ORDER = 4
BUTTER_FILTER_B, BUTTER_FILTER_A = butter(BUTTER_FILTER_ORDER, LOWPASS_CUTOFF_FREQ, btype='low', fs=ORIGINAL_NEIC_SAMPLED_FREQ)
DOWN_SAMPLE_STEP_SIZE = 2

ORIG_USGS_MULTIMODAL_FOLDER_NAME = "version_3_appendshakemap"
ORIG_USGS_IMAGE_MODALITY_PATH = ORIG_USGS_MULTIMODAL_FOLDER_NAME + os.sep + "images"
ORIG_USGS_ONEPAGER_MODALITY_PATH = ORIG_USGS_MULTIMODAL_FOLDER_NAME + os.sep + "onepager_images"
IMAGE_SUBMODALITIES = [
    "intensity.jpg", "pga.jpg", "pgv.jpg", "psa0p3_regr.png", "psa0p3.jpg", 
    "psa1p0_regr.png", "psa1p0.jpg", "psa3p0_regr.png", "psa3p0.jpg", 
]
SENSOR_SEP_STR = "::"
STATION_SENSOR_SEP_STR = "<<"
STANDARD_USGS_REC_NUM_TIME_STEPS = 100000#66000
US_TERRITORY_FULL_NAME_TO_ABBREV_MAP = {
    'Alabama': 'AL', 'Kentucky': 'KY', 'Alaska': 'AK', 'Louisiana': 'LA', 'Arizona': 'AZ', 'Maine': 'ME', 'Arkansas': 'AR', 'Maryland': 'MD', 'American Samoa': 'AS', 'Massachusetts': 'MA',
    'California': 'CA', 'Michigan': 'MI', 'Colorado': 'CO', 'Minnesota': 'MN', 'Connecticut': 'CT', 'Mississippi': 'MS', 'Delaware': 'DE', 'Missouri': 'MO', 'District of Columbia': 'DC',
    'Montana': 'MT', 'Florida': 'FL', 'Nebraska': 'NE', 'Georgia': 'GA', 'Nevada': 'NV', 'Guam': 'GU', 'New Hampshire': 'NH', 'Hawaii': 'HI', 'New Jersey': 'NJ', 'Idaho': 'ID',
    'New Mexico': 'NM', 'Illinois': 'IL', 'New York': 'NY', 'Indiana': 'IN', 'North Carolina': 'NC', 'Iowa': 'IA', 'North Dakota': 'ND', 'Kansas': 'KS', 'Northern Mariana Islands': 'MP',
    'Ohio': 'OH', 'Oklahoma': 'OK', 'Oregon': 'OR', 'Pennsylvania': 'PA', 'Puerto Rico': 'PR', 'Rhode Island': 'RI', 'South Carolina': 'SC', 'South Dakota': 'SD', 'Tennessee': 'TN',
    'Texas': 'TX', 'Trust Territories': 'TT', 'Utah': 'UT', 'Vermont': 'VT', 'Virginia': 'VA', 'Virgin Islands': 'VI', 'Washington': 'WA', 'West Virginia': 'WV', 'Wisconsin': 'WI', 'Wyoming': 'WY',
}
US_TERRITORY_ABBREV_TO_FULL_NAME_MAP = {
    'AL': 'Alabama', 'KY': 'Kentucky', 'AK': 'Alaska', 'LA': 'Louisiana', 'AZ': 'Arizona', 'ME': 'Maine', 'AR': 'Arkansas', 'MD': 'Maryland', 'AS': 'American Samoa', 'MA': 'Massachusetts', 
    'CA': 'California', 'MI': 'Michigan', 'CO': 'Colorado', 'MN': 'Minnesota', 'CT': 'Connecticut', 'MS': 'Mississippi', 'DE': 'Delaware', 'MO': 'Missouri', 'DC': 'District of Columbia', 
    'MT': 'Montana', 'FL': 'Florida', 'NE': 'Nebraska', 'GA': 'Georgia', 'NV': 'Nevada', 'GU': 'Guam', 'NH': 'New Hampshire', 'HI': 'Hawaii', 'NJ': 'New Jersey', 'ID': 'Idaho', 
    'NM': 'New Mexico', 'IL': 'Illinois', 'NY': 'New York', 'IN': 'Indiana', 'NC': 'North Carolina', 'IA': 'Iowa', 'ND': 'North Dakota', 'KS': 'Kansas', 'MP': 'Northern Mariana Islands', 
    'OH': 'Ohio', 'OK': 'Oklahoma', 'OR': 'Oregon', 'PA': 'Pennsylvania', 'PR': 'Puerto Rico', 'RI': 'Rhode Island', 'SC': 'South Carolina', 'SD': 'South Dakota', 'TN': 'Tennessee', 
    'TX': 'Texas', 'TT': 'Trust Territories', 'UT': 'Utah', 'VT': 'Vermont', 'VA': 'Virginia', 'VI': 'Virgin Islands', 'WA': 'Washington', 'WV': 'West Virginia', 'WI': 'Wisconsin', 'WY': 'Wyoming',
}
PLACE_NAME_INDEX = 0
PLACE_DIST_TO_EVENT_INDEX = 1
PLACE_POPULATION_INDEX = 2
NEARBY_PLACES_KEY_TO_INDEX_MAP = {
    "name": PLACE_NAME_INDEX, 
    "distance": PLACE_DIST_TO_EVENT_INDEX, 
    "population": PLACE_POPULATION_INDEX, 
}

EVENT_ID_TO_REGION_NAME_MAP_FILE_NAME = os.path.expanduser("~/results/partition_and_plot_usgs_data_by_lat_an_lon_v02162026/EVENT_ID_TO_REGION_NAME_MAP.pkl") # os.path.expanduser("~/usgs-multimodal-models/usgs_data/EVENT_ID_TO_REGION_NAME_MAP.pkl") # REGION_NAME_TO_EVENT_ID_MAP_FILE_NAME = "~/usgs-multimodal-models/usgs_data/REGION_NAME_TO_EVENT_ID_MAP.pkl"
EVENT_ID_TO_STRES_ID_MAP_FILE_NAME = os.path.expanduser("~/results/partition_and_plot_usgs_data_by_lat_an_lon_v02162026/EVENT_ID_TO_STRES_ID_MAP.pkl")
EVENT_ID_TO_STRES_ID_MAP = pkl.load(open(EVENT_ID_TO_STRES_ID_MAP_FILE_NAME, 'rb'))

def get_event_id_to_modal_files_map(events_of_interest_to_neic_file_map, orig_TxtImg_data_path):
    event_ids_to_modal_file_paths_map = dict()
    
    for e, eid in enumerate(events_of_interest_to_neic_file_map.keys()): 
        one_pager_path = orig_TxtImg_data_path+os.sep+ORIG_USGS_ONEPAGER_MODALITY_PATH+os.sep+eid+"_onepager"
        
        if os.path.exists(one_pager_path+os.sep+'figure-1-6.jpg'):
            one_pager_path = one_pager_path+os.sep+'figure-1-6.jpg'
        elif os.path.exists(one_pager_path+os.sep+'figure-1-4.jpg'):
            one_pager_path = one_pager_path+os.sep+'figure-1-4.jpg'
        else:
            one_pager_path = None

        if os.path.exists(orig_TxtImg_data_path+os.sep+ORIG_USGS_MULTIMODAL_FOLDER_NAME+os.sep+eid+".json"): # only include events that have text descriptions associated with them
            event_ids_to_modal_file_paths_map[eid] = {
                "time series": events_of_interest_to_neic_file_map[eid], 
                "text description": orig_TxtImg_data_path+os.sep+ORIG_USGS_MULTIMODAL_FOLDER_NAME+os.sep+eid+".json", 
                "onepager image": one_pager_path, 
            }
            for sm in IMAGE_SUBMODALITIES: 
                event_ids_to_modal_file_paths_map[eid][sm[:-1*len(".jpg")]] = orig_TxtImg_data_path+os.sep+ORIG_USGS_IMAGE_MODALITY_PATH+os.sep+eid+os.sep+sm
        else:
            print("get_event_id_to_modal_files_map: WARNING - omitting event with eid == ", eid, " due to missing .json data", flush=True)
        # if e == 2: # FOR DEBUGGING PURPOSES
        #     print("get_event_id_to_modal_files_map: WARNING - BREAKING FOR DEBUGGING PURPOSES", flush=True) # FOR DEBUGGING PURPOSES
        #     break # FOR DEBUGGING PURPOSES
    
    return event_ids_to_modal_file_paths_map


# METHODS FOR USGS TIME SERIES -----------------------------------------------------------------------------------------------------------------------
def determine_reporting_stations_by_event_signal_combo(event_id, h5_file_path):
    usgs_ts_data = h5py.File(h5_file_path, 'r')
    station_sensors_to_rec_len_map = dict()#set()
    combined_rec_len = 0.
    num_recs = 0.
    avg_rec_len = None
    mode_rec_len = None
    for station_ind, station_key in enumerate(usgs_ts_data[event_id].keys()): 
        station_sensor_keys = sorted(list(usgs_ts_data[event_id][station_key].keys()))
        for sensor_ind, sensor_key in enumerate(station_sensor_keys): 
            curr_rec_len = len(usgs_ts_data[event_id][station_key][sensor_key][:])
            combined_rec_len += curr_rec_len
            num_recs += 1.
            station_sensors_to_rec_len_map[station_key+STATION_SENSOR_SEP_STR+sensor_key] = curr_rec_len #station_sensors.add(station_key+STATION_SENSOR_SEP_STR+sensor_key)
            # if sensor_ind == 2: # FOR DEBUGGING PURPOSES
            #     print("determine_reporting_stations_by_event_signal_combo: WARNING - BREAKING FOR DEBUGGING PURPOSES", flush=True) # FOR DEBUGGING PURPOSES
            #     break # FOR DEBUGGING PURPOSES
        # if station_ind == 2: # FOR DEBUGGING PURPOSES
        #     print("determine_reporting_stations_by_event_signal_combo: WARNING - BREAKING FOR DEBUGGING PURPOSES", flush=True) # FOR DEBUGGING PURPOSES
        #     break # FOR DEBUGGING PURPOSES
    try:
        avg_rec_len = combined_rec_len/num_recs
    except:
        print("determine_reporting_stations_by_event_signal_combo: WARNING -- could not compute avg_rec_len for event_id == ", event_id, ", likely due to the fact that num_recs == ", num_recs, flush=True)
    mode_rec_len = stat_mode([station_sensors_to_rec_len_map[k] for k in station_sensors_to_rec_len_map.keys()])
    station_sensors = [s for s in station_sensors_to_rec_len_map.keys() if station_sensors_to_rec_len_map[s] == mode_rec_len]
    print("determine_reporting_stations_by_event_signal_combo: event_id ", event_id, " has num_recs == ", num_recs, " with avg_rec_len == ", avg_rec_len, " and mode_rec_len == ", mode_rec_len, "; filtered len(station_sensors) == ", len(station_sensors), flush=True)
    return set(station_sensors), mode_rec_len

def explore_usgs_event_time_series_files(events_to_modal_file_paths_map, event_ids_to_focus_on, save_dir):
    print("explore_usgs_event_time_series_files: START ", flush=True)
    usgs_stations_sensors_shared_across_events = set()
    usgs_station_sensors_total_across_events = set()
    usgs_station_sensors_to_events_map = dict()
    station_sensor_ids_initialized = False
    events_with_no_standard_recordings = []
    standardized_event_ids_to_rec_len_map = dict()
    for e, eid in enumerate(event_ids_to_focus_on):
        if e % 50 == 0:
            print("explore_usgs_event_time_series_files: inspecting sensors associated with event e == ", e, " of ", len(event_ids_to_focus_on), ": -- eid == ", eid, flush=True)
        for modality_key in events_to_modal_file_paths_map[eid].keys():
            if modality_key == "time series" and events_to_modal_file_paths_map[eid][modality_key] is not None:
                curr_event_station_sensors, rec_len = determine_reporting_stations_by_event_signal_combo(eid, events_to_modal_file_paths_map[eid][modality_key])
                if len(list(curr_event_station_sensors)) == 0:
                    events_with_no_standard_recordings.append(eid)
                else:
                    standardized_event_ids_to_rec_len_map[eid] = rec_len
                    if not station_sensor_ids_initialized:
                        usgs_stations_sensors_shared_across_events = curr_event_station_sensors
                        station_sensor_ids_initialized = True
                    else:
                        usgs_stations_sensors_shared_across_events = usgs_stations_sensors_shared_across_events & curr_event_station_sensors
                    usgs_station_sensors_total_across_events = usgs_station_sensors_total_across_events | curr_event_station_sensors
                    for sensor_ind, sensor in enumerate(curr_event_station_sensors):
                        if sensor not in usgs_station_sensors_to_events_map.keys():
                            usgs_station_sensors_to_events_map[sensor] = []
                        usgs_station_sensors_to_events_map[sensor].append(eid)
                        # if sensor_ind == 2: # FOR DEBUGGING PURPOSES
                        #     print("explore_usgs_event_time_series_files: WARNING - BREAKING FOR DEBUGGING PURPOSES", flush=True) # FOR DEBUGGING PURPOSES
                        #     break # FOR DEBUGGING PURPOSES
        # if e == 2: # FOR DEBUGGING PURPOSES
        #     print("explore_usgs_event_time_series_files: WARNING - BREAKING FOR DEBUGGING PURPOSES", flush=True) # FOR DEBUGGING PURPOSES
        #     break # FOR DEBUGGING PURPOSES
    events_with_no_standard_recordings = sorted(events_with_no_standard_recordings)
    with open(save_dir+os.sep+"events_with_no_standard_recordings.pkl", 'wb') as outfile:
        pkl.dump(events_with_no_standard_recordings, outfile)
    standardized_event_ids_to_focus_on = sorted([e for e in event_ids_to_focus_on if e not in events_with_no_standard_recordings])
    for e1, e2 in zip(standardized_event_ids_to_focus_on, sorted(list(standardized_event_ids_to_rec_len_map.keys()))):
        assert e1 == e2
    with open(save_dir+os.sep+"standardized_event_ids_to_focus_on.pkl", 'wb') as outfile:
        pkl.dump(standardized_event_ids_to_focus_on, outfile)
    usgs_stations_sensors_shared_across_events = sorted(list(usgs_stations_sensors_shared_across_events))
    with open(save_dir+os.sep+"usgs_stations_sensors_shared_across_events.pkl", 'wb') as outfile:
        pkl.dump(usgs_stations_sensors_shared_across_events, outfile)
    usgs_station_sensors_total_across_events = sorted(list(usgs_station_sensors_total_across_events))
    with open(save_dir+os.sep+"usgs_station_sensors_total_across_events.pkl", 'wb') as outfile:
        pkl.dump(usgs_station_sensors_total_across_events, outfile)
    with open(save_dir+os.sep+"usgs_station_sensors_to_events_map.pkl", 'wb') as outfile:
        pkl.dump(usgs_station_sensors_to_events_map, outfile)    
    plot_sensor_event_histogram(None, save_dir+os.sep+"usgs_station_sensors_to_events_map.png", usgs_station_sensors_to_events_map)
    print("explore_usgs_event_time_series_files: len(usgs_stations_sensors_shared_across_events) == ", len(usgs_stations_sensors_shared_across_events))
    print("explore_usgs_event_time_series_files: len(usgs_station_sensors_total_across_events) == ", len(usgs_station_sensors_total_across_events), flush=True)
    print("explore_usgs_event_time_series_files: STOP ", flush=True)
    return usgs_stations_sensors_shared_across_events, usgs_station_sensors_total_across_events, standardized_event_ids_to_focus_on, standardized_event_ids_to_rec_len_map
# METHODS FOR USGS TIME SERIES -----------------------------------------------------------------------------------------------------------------------

# METHODS FOR USGS TEXT DATA -------------------------------------------------------------------------------------------------------------------------
def extract_relationships_from_json(data, parent="Root"): # Recursive Function to Explore the JSON and Extract Relationships:
    relationships = []
    if isinstance(data, dict):
        for key, value in data.items():
            relationships.append((parent, "has_attribute", f"{parent} -- {key}")) # Node-to-Key Relationship
            relationships.extend(extract_relationships_from_json(value, f"{parent} -- {key}")) # Recurse Deeper
    elif isinstance(data, list):
        for index, item in enumerate(data):
            relationships.append((parent, "has_item", f"{parent}_item_{index}")) # Node-to-List Relationship
            relationships.extend(extract_relationships_from_json(item, f"{parent}_item_{index}")) # Recurse into each list item
    else: # Base Case: Leaf Node
        relationships.append((parent, "has_value", str(data)))
    return relationships

#""" # FOR DEBUGGING PURPOSES
def extract_kg_relationships_from_tectonic_summary(text):
    #print("extract_kg_relationships_from_tectonic_summary: START", flush=True)
    relationships = set()
    # Helper function to create a relationship
    def add_relationship(node1, relationship, node2):
        relationships.add((str(node1).strip(), str(relationship).strip(), str(node2).strip()))
    # Pattern matching for tectonic-related relationships
    # 1. Plate convergence
    plate_relation_pattern = r"((collision|convergence)(\s)+of(\s)+the(\s)+[a-zA-Z]*(\s)+and(\s)+[a-zA-Z]*(\s)+plates|the(\s)+[a-zA-Z]*(\s)+and(\s)+[a-zA-Z]*(\s)+plates(\s)+(collide|converge))(,|\s|[a-zA-z])*rate(,|\s|[a-zA-z])*(\s)+[0-9]+(-[0-9]+)*(\s)+mm\/yr"#r"((collision|convergence)(\s)+of(\s)+the(\s)+[a-zA-Z]*(\s)+and(\s)+[a-zA-Z]*(\s)+plates|the(\s)+[a-zA-Z]*(\s)+and(\s)+[a-zA-Z]*(\s)+plates(\s)+(collide|converge))"#r".*? the .*? and .*? plates .*?"
    for match in re.finditer(plate_relation_pattern, text, re.IGNORECASE):
        plate1 = match.group().split("and")[0].strip().split(' ')[-1]
        plate2 = match.group().split("and")[1].strip().split(' ')[0]
        add_relationship(plate1, "is a", "tectonic plate")
        add_relationship(plate2, "is a", "tectonic plate")
        conv_rt = match.group().split("mm/yr")[0].strip().split(' ')[-1]
        plate_boundary_name1 = plate1+"-"+plate2
        plate_boundary_name2 = plate2+"-"+plate1
        add_relationship(plate_boundary_name1, "is a", "plate boundary")
        add_relationship(plate_boundary_name2, "is a", "plate boundary")
        add_relationship(plate1, "converges at a rate of {} with".format(conv_rt), plate2)
        add_relationship(plate2, "converges at a rate of {} with".format(conv_rt), plate1)
        boundary_plate_relation_pattern = fr"({plate1}|{plate2})(\s)+(above|over|below|beneath|across|along|against)(\s)+({plate1}|{plate2})"
        for bndry_match in re.finditer(boundary_plate_relation_pattern, text, re.IGNORECASE):
            split_kg_relation = bndry_match.group().strip().split(' ')
            add_relationship(split_kg_relation[0], "moves "+split_kg_relation[1], split_kg_relation[2])
    # 2. Significant earthquakes
    #print("extract_kg_relationships_from_tectonic_summary: \t earthquake_pattern START", flush=True)
    earthquake_pattern = r"[0-9][0-9][0-9][0-9](,)*(\s)+M[0-9].[0-9](\s)([A-Z]+[a-z]*)+"
    for match in re.finditer(earthquake_pattern, text):
        #date, magnitude, earthquake_name, region = match.groups()
        year = match.group().strip().split(' ')[0].strip()
        if year[-1] in [',',]:
            year = year[:-1]
        magnitude = float(match.group().strip().split(' ')[1].strip()[len('M'):])
        region_name = match.group().strip().split(' ')[2].strip()
        if region_name != 'earthquake': # band-aid because 'earthquake' kept passing through regex, but could not determine why
            entity_name = year+" "+region_name+" STRES-TS Earthquake"
            entity_name = " ".join([t[0].upper()+t[1:] for t in entity_name.split(' ')])
            add_relationship(entity_name, "is an", "earthquake")
            # add_relationship(entity_name, "occured in the year", year) # WE EXCLUDE TIME-RELATED QUESTIONS IN SECOND ITERATION OF STRES DATA CURATION 02/06/2026
            add_relationship(entity_name, "has a magnitude of", magnitude)
            add_relationship(entity_name, "struck the region referred to as the", region_name)
    # 3. Fault systems and types
    #print("extract_kg_relationships_from_tectonic_summary: \t fault_system_pattern START", flush=True)
    fault_system_pattern = r"(.*?Fault).*?(strike-slip|thrust|normal|oblique).*?fault"
    for match in re.finditer(fault_system_pattern, text, re.IGNORECASE):
        fault_name, fault_type = match.groups()
        add_relationship(fault_name.strip(), "has the fault type of", fault_type.strip())
    # 4. Region-Fault relationships
    #print("extract_kg_relationships_from_tectonic_summary: \t region_fault_pattern START", flush=True)
    tail_str_strike_slip_fault_pattern = r"((R|r)ight|(L|l)eft)-lateral, strike-slip fault(\s|system|s|,|)*([a-zA-Z]|\s|,|;|\(|\)|-)*"
    for match in re.finditer(tail_str_strike_slip_fault_pattern, text, re.IGNORECASE):
        fault_type = "strike-slip"
        split_match = match.group().strip().split('-lateral, strike-slip fault')
        fault_orientation = split_match[0].split(' ')[-1].lower()+"-lateral"
        tail_match_str = split_match[1].strip()
        split_tail = tail_match_str.split(', ')
        split_tail = [x.strip() for elem in split_tail for x in elem.split('and') if len(x.strip()) > 0]
        for elem in split_tail:
            name = ' '.join([s for s in elem.split(' ') if s[0] != s[0].lower()])
            if len(name) > 0:
                add_relationship(name, 'is a', 'fault')
                add_relationship(name, 'has the fault type of', fault_type)
                add_relationship(name, 'is oriented', fault_orientation)
    # 5. Gather general factoids
    #print("extract_kg_relationships_from_tectonic_summary: \t geo_context_pattern START", flush=True)
    text_sentences = [x.replace("\n", "") for x in text.split('.')]
    geo_context_pattern = r"([0-9][0-9][0-9][0-9]*(\s|\n|-)*)*(\(*[A-Z][a-z]*\)*(\s|\n|-)*)*(((R|r)egion(s)*|(M|m)ountain(s)*|(B|b)asi(s)*|(T|t)hrust(s)*|(F|f)ault(s)*|(E|e)arthquake(s)*|(T|t)sunami(s)*|(P|p)late(s)*|(P|p)lateau(s)*|(R|r)ange(s)*|(A|a)rchipelago(es)*|(A|a)rch(es)*|(A|a)rc(s)*|(F|f)ront(s)*|(S|s)uture(s)*|(R|r)iver(s)*|(B|b)elt(s)*|(Z|z)one(s)*|(M|m)argin(s)*|(P|p)eriod(s)*|(A|a)rea(s)*|(A|a)ge(s)*|(C|c)hain(s)*|(O|o)cean(s)*|(S|s)ea(s)*|(P|p)assage(s)*|(T|t)rough(s)*|(T|t)rench(es)*|(I|i)sland(s)*|(V|v)olcano(es)*|(of )*[0-9][0-9][0-9][0-9])(\s|\n|-)*)+"
    for match in re.finditer(geo_context_pattern, text): 
        #print("extract_kg_relationships_from_tectonic_summary: \t\t match == ", match, flush=True)
        match_text = match.group().strip().replace("\n", "")
        for sentence in text_sentences:
            if match_text in sentence:
                add_relationship(match_text, 'is contextualized by the statement:', sentence)
    #for i,rel in enumerate(list(relationships)):
    #    print("extract_kg_relationships_from_tectonic_summary: i == ", i, " rel == ", rel)
    #print("extract_kg_relationships_from_tectonic_summary: STOP", flush=True)
    return relationships

def extract_kg_relationships_from_event_onepager_summary(text, event_id, debugging_json_kg=None):
    # print("extract_kg_relationships_from_event_onepager_summary: START", flush=True)
    EXPOSURE_LEVEL_NUMERAL_TO_DIGIT_MAP = {'I': '1', 'II': '2', 'III': '3', 'IV': '4', 'V': '5', }
    endline_split_text = text.split('\n')
    relationships = set()
    # Helper function to create a relationship
    def add_relationship(node1, relationship, node2):
        relationships.add((str(node1).strip(), str(relationship).strip(), str(node2).strip()))
    event_city_country_str = None
    event_city = None
    event_date = None
    event_year = None
    event_kg_node_name = None
    for line_ind, line in enumerate(endline_split_text):
        if " Origin Time" in line:
            # 1. Pattern matching for city relationships
            city_country_pattern = r"([A-Z][a-z]+\s*(of )*)+(, ([A-Z][a-z]+\s*(of )*)+)*"
            city_country_match = re.search(city_country_pattern, line.split(' Origin Time')[0].strip())
            if city_country_match is None:
                city_country_match = str(event_id)+"_UNK_CITY, "+str(event_id)+"_UNK_COUNTRY"
            else:
                city_country_match = city_country_match.group().strip()
            event_city_country_str = city_country_match
            event_city = city_country_match.split(', ')[0]
            event_date = line.split("Origin Time: ")[1].strip().split(' ')[0]
            event_year = event_date.split('-')[0]
            assert len(event_year) == 4
            assert event_id.split(' ')[1] in EVENT_ID_TO_STRES_ID_MAP.keys()
            event_kg_node_name = event_year+' '+event_city+' STRES'+str(EVENT_ID_TO_STRES_ID_MAP[event_id.split(' ')[1]])+" Earthquake" #event_year+' '+event_city+" Earthquake"
            add_relationship(event_kg_node_name, 'has Event ID:', event_id)
            add_relationship(event_id, 'is referred to as the', event_kg_node_name)
            event_area_kg_node_name = event_city+" Area"
            add_relationship(event_kg_node_name, 'occured nearest to major population of', event_city_country_str)
            add_relationship(event_city_country_str, 'was the closest major population to', event_kg_node_name)
            #add_relationship(event_kg_node_name, 'occured on the date of', event_date) # WE EXCLUDE TIME-RELATED QUESTIONS IN SECOND ITERATION OF STRES DATA CURATION 02/06/2026
            event_approx_magnitude = line.split(',')[0].strip().split(' ')[-1].strip()
            add_relationship(event_kg_node_name, 'had approximate magnitude of', event_approx_magnitude)
            if " of " in line.split(',')[1].strip() and "state of " not in line.split(',')[1].strip().lower() and "gulf of " not in line.split(',')[1].strip().lower() and "island of " not in line.split(',')[1].strip().lower():
                event_origin_rel_to_city_country_str = line.split(',')[1].strip().split(' of ')[0]
                add_relationship(event_kg_node_name, 'had a point of origin from the nearest major population that was approximately', event_origin_rel_to_city_country_str)
                event_origin_rel_to_city_country_str_reverse_dir = ' '.join(event_origin_rel_to_city_country_str.split(' ')[:-1])+' '+reverse_cardinal_direction(event_origin_rel_to_city_country_str.split(' ')[-1])
                event_origin_rel_to_city_country_str_reverse_dir = event_origin_rel_to_city_country_str_reverse_dir.strip()
                add_relationship(event_city, 'is located {} of'.format(event_origin_rel_to_city_country_str_reverse_dir), event_kg_node_name)
                add_relationship(event_area_kg_node_name, 'is located {} of'.format(event_origin_rel_to_city_country_str), event_city)
            event_utc_time = line.split(" UTC")[0].strip().split(' ')[-1]
            # add_relationship(event_kg_node_name, 'occured at UTC time', event_utc_time) # WE EXCLUDE TIME-RELATED QUESTIONS IN SECOND ITERATION OF STRES DATA CURATION 02/06/2026
            event_loc_coords = ' '.join(line.split(" Location: ")[1].strip().split(' ')[:4])
            add_relationship(event_kg_node_name, 'originated from GPS coordinates', event_loc_coords)
            event_depth = ' '.join(line.split(" Depth: ")[1].strip().split(' ')[:2])
            assert event_depth[-2:] == 'km' # sanity check
            add_relationship(event_kg_node_name, 'originated from a depth of', event_depth)
        elif "Estimated Fatalities" in line:
            est_fatalities = line[len('Estimated Fatalities'):].strip()
            if len(est_fatalities) > 0:
                raise NotImplementedError("split line of estimated fatalities not handled currently, with est_fatalities == "+str(est_fatalities))
        elif " alert for shaking-related fatalities. " in line:
            alert_level_for_shaking_rel_fatalities_str = ': '.join([s.strip() for s in line.split(" alert for shaking-related fatalities. ")])
            add_relationship(event_kg_node_name, 'triggered a shaking related fatalities alert level of', alert_level_for_shaking_rel_fatalities_str)
        elif " alert for economic losses. " in line:
            alert_level_for_econ_loss_str = ': '.join([s.strip() for s in line.split(" alert for economic losses. ")])
            add_relationship(event_kg_node_name, 'triggered an economical loss alert level of', alert_level_for_econ_loss_str)
        elif "Population Exposure " in line:
            est_pop_exposure_to_shaking = line.strip()[len("Population Exposure "):]
            add_relationship(event_kg_node_name, 'exposed the population to shaking with a rated level of', est_pop_exposure_to_shaking)
        elif "Estimated Population Exposed to Earthquake Shaking" in line:
            est_shaken_pop = line[len('Estimated Population Exposed to Earthquake Shaking'):].strip()
            add_relationship(event_kg_node_name, 'exposed an estimated population to earthquake shaking of around', est_shaken_pop)
        elif "population per 1 sq. km from Landscan " in line:
            est_pop_density_per_sqr_km = line.strip()[len("population per 1 sq. km from Landscan "):]
            add_relationship(event_kg_node_name, 'occured in an area with estimated population density (per square km) of', est_pop_density_per_sqr_km)
        elif "Structures Overall, " in line:
            area_structures_summary = line.strip()[len("Structures Overall, "):]
            add_relationship(event_area_kg_node_name, 'has structures characterized by the statement that', area_structures_summary)
        # elif "Historical Earthquakes Date (UTC) " in line: # THESE DATA POINTS ARE EXCLUDED FROM STRES DATA DUE TO OVERLAPS ACROSS EVENTS, LEADING TO RISKS OF DATA LEAKAGE
        #     split_eq_hist_line = line.strip()[len("Historical Earthquakes Date (UTC) "):].split(' ')
        #     recent_preceding_erthqk_dates_in_area = [x for x in split_eq_hist_line if len(x)==10 and len(x.split('-'))==3]
        #     for edate in recent_preceding_erthqk_dates_in_area+[event_date]:
        #         add_relationship(event_area_kg_node_name, 'experienced an earthquake on the date of', edate)
        #     recent_preceding_erthqk_hist_summary_for_area = ' '.join([x for x in split_eq_hist_line if len(x.split('-'))!=3])
        #     add_relationship(event_area_kg_node_name, 'has experienced earthquakes characterized by the statement that', recent_preceding_erthqk_hist_summary_for_area)
        elif "Selected City Exposure from GeoNames.org MMI City " in line and " bold cities appear on map." in line:
            alternating_area_and_exposure_levels = line.strip()[len("Selected City Exposure from GeoNames.org MMI City "):-1*len(" bold cities appear on map.")].split(' ')
            area_exposure_levels_by_city = dict()
            temp_city_name = ""
            for elem in alternating_area_and_exposure_levels:
                if elem in ['I','II','III','IV','V',]:
                    area_exposure_levels_by_city[temp_city_name] = EXPOSURE_LEVEL_NUMERAL_TO_DIGIT_MAP[elem]
                    temp_city_name = ""
                else:
                    temp_city_name = temp_city_name + ' ' + elem
            for city_name in area_exposure_levels_by_city.keys():
                add_relationship(event_area_kg_node_name, 'encompasses a city named', city_name)
                add_relationship(city_name, 'is located in the', event_area_kg_node_name)
                add_relationship(city_name, 'experienced shaking level of {} caused by'.format(area_exposure_levels_by_city[city_name]), event_kg_node_name)
                add_relationship(event_kg_node_name, 'caused shaking level of {} in'.format(area_exposure_levels_by_city[city_name]), city_name)
        elif "Event ID: " in line:
            try:
                if normalize_string(event_id) != normalize_string("Event ak023qqz70l") or normalize_string('Event '+line.strip()[len('Event ID: '):]) != normalize_string("Event us7000j4q9"):
                    assert normalize_string('Event '+line.strip()[len('Event ID: '):]) == normalize_string(event_id)
            except:
                print("extract_kg_relationships_from_event_onepager_summary: EVENT ID MISMATCH DETECTED !!!")
                print("extract_kg_relationships_from_event_onepager_summary: text == ", text)
                print("extract_kg_relationships_from_event_onepager_summary: normalize_string('Event '+line.strip()[len('Event ID: '):]) == ", normalize_string('Event '+line.strip()[len('Event ID: '):]))
                print("extract_kg_relationships_from_event_onepager_summary: normalize_string(event_id) == ", normalize_string(event_id), flush=True)
                print("extract_kg_relationships_from_event_onepager_summary: debugging_json_kg == ", debugging_json_kg, flush=True)
                raise ValueError()
    #for i,rel in enumerate(list(relationships)):
    #    print("extract_kg_relationships_from_event_onepager_summary: i == ", i, " rel == ", rel)
    # print("extract_kg_relationships_from_event_onepager_summary: STOP", flush=True)
    return relationships, event_kg_node_name

def extract_kg_relationships_from_event_moment_tensor_description(text, text_meta_node, node_tied_to_event):
    split_meta_node = text_meta_node.split(' -- ')
    tensor_element_value_name = split_meta_node[-1]
    kg_rel = (node_tied_to_event, "has a W-phase Moment Tensor with a {} element value of".format(tensor_element_value_name), text)
    return {kg_rel}

def extract_kg_relationships_from_event_origin_details(text, text_meta_node, node_tied_to_event):
    split_meta_node = text_meta_node.split(' -- ')
    tensor_element_value_name = split_meta_node[-1]
    kg_rel = (node_tied_to_event, "has an origin characterized by a {} with a value of".format(tensor_element_value_name), text)
    return {kg_rel}

def extract_kg_relationships_from_nearby_places_info(place_to_info_map, node_tied_to_event):
    #print("extract_kg_relationships_from_nearby_places_info: START", flush=True)
    relationships = set()
    # Helper function to create a relationship
    def add_relationship(node1, relationship, node2):
        relationships.add((str(node1).strip(), str(relationship).strip(), str(node2).strip()))
    # collect relationships
    for place_meta_key in place_to_info_map.keys():
        city, region, country = None, None, None
        split_place_name = [elem.strip() for elem in place_to_info_map[place_meta_key][PLACE_NAME_INDEX].strip().split(',')]
        if len(split_place_name) == 2 and (split_place_name[-1] in US_TERRITORY_ABBREV_TO_FULL_NAME_MAP.keys() or split_place_name[-1] in US_TERRITORY_FULL_NAME_TO_ABBREV_MAP.keys()):
            [city, region] = split_place_name
            if region in US_TERRITORY_ABBREV_TO_FULL_NAME_MAP.keys():
                region = US_TERRITORY_ABBREV_TO_FULL_NAME_MAP[region]
            country = "United States"
        elif len(split_place_name) == 3:
            [city, region, country] = split_place_name
        else:
            raise ValueError("Un-handled number of elements in place name, with split_place_name == "+str(split_place_name))
        city_country_str = city+', '+country
        add_relationship(city_country_str, 'is located in the region referred to as the', region+' Region')
        add_relationship(region+' Region', 'encompasses a city named', city_country_str)
        place_dist_from_event_orig = place_to_info_map[place_meta_key][PLACE_DIST_TO_EVENT_INDEX]
        add_relationship(city_country_str, 'is located {} of'.format(place_dist_from_event_orig), node_tied_to_event)
        event_orig_dist_from_place = ' '.join(place_dist_from_event_orig.split(' ')[:-1])+' '+reverse_cardinal_direction(place_dist_from_event_orig.split(' ')[-1])
        add_relationship(node_tied_to_event, 'was located {} of'.format(event_orig_dist_from_place), city_country_str)
        add_relationship(city_country_str, 'has a population of', place_to_info_map[place_meta_key][PLACE_POPULATION_INDEX])
    #for i,rel in enumerate(list(relationships)):
    #    print("extract_kg_relationships_from_nearby_places_info: i == ", i, " rel == ", rel)
    #print("extract_kg_relationships_from_nearby_places_info: STOP", flush=True)
    return relationships

def get_knowledge_graph_from_text_json(usgs_text_data, root_event, UNIVERSAL_GENERAL_INFO_KNOWLEDGE_GRAPH):
    #print("get_knowledge_graph_from_text_json: START", flush=True)
    def report_kg_characteristics(kg_name, kg, node_set, edge_set):
        print("\n\n\nget_knowledge_graph_from_text_json: "+kg_name+" relationships -- ", flush=True)
        for i,rel in enumerate(list(kg)):
            print("get_knowledge_graph_from_text_json: \t "+kg_name+" rel i==", i, " :: ", rel)
        print("\n\n\nget_knowledge_graph_from_text_json: "+kg_name+" nodes -- ", flush=True)
        for i,node in enumerate(list(node_set)):
            print("get_knowledge_graph_from_text_json: \t "+kg_name+" node i==", i, " :: ", node)
        print("\n\n\nget_knowledge_graph_from_text_json: "+kg_name+" edges -- ", flush=True)
        for i,edge in enumerate(list(edge_set)):
            print("get_knowledge_graph_from_text_json: \t "+kg_name+" edge i==", i, " :: ", edge)
        pass
    # currate various knowledge graphs from json
    event_id = None
    json_kg = extract_relationships_from_json(usgs_text_data[root_event], parent=root_event)
    event_root_kg = set()
    historical_kg = set()
    onepager_kg = set()
    general_vicinity_of_event_onepager_kg = set()
    event_specific_onepager_kg = set()
    moment_tensor_kg = set()
    origin_details_kg = set()
    nearby_places_kg = set()
    places_near_event_kg = set()
    place_to_info_map = dict()
    for i,rel in enumerate(json_kg):
        if ' -- metadata -- id' in rel[0]:
            event_id = "Event "+rel[2]
        elif ' -- Regional_Information -- Administrative_Region -- ISO' in rel[0]:
            event_root_kg.add((str(event_id).strip(), 'has ISO value of'.strip(), str(rel[2]).strip()))
        elif ' -- Regional_Information -- Administrative_Region -- Region' in rel[0]:
            event_root_kg.add((str(event_id).strip(), 'occured in the region referred to as the'.strip(), str(rel[2]).strip()+' Region'))
        elif ' -- Regional_Information -- Administrative_Region -- Country' in rel[0]:
            event_root_kg.add((str(event_id).strip(), 'occured in the country named'.strip(), str(rel[2]).strip()))
        elif '-- Regional_Information -- Tectonic_Summary' in rel[0]:
            historical_kg = extract_kg_relationships_from_tectonic_summary(rel[2])
        elif '-- onepager -- page-1 -- text' in rel[0]: # extract kg relations from onepage summary of specific event
            onepager_kg, event_kg_node_name = extract_kg_relationships_from_event_onepager_summary(rel[2], event_id, debugging_json_kg=json_kg)
        elif ' -- Moment-tensor -- W-phase_Moment_Tensor -- ' in rel[0]: # extract kg relations from Moment Tensor associated with a specific event
            tensor_rels = extract_kg_relationships_from_event_moment_tensor_description(rel[2], rel[0], event_id)
            moment_tensor_kg = moment_tensor_kg | tensor_rels
        elif ' -- Origin -- Details -- ' in rel[0]:
            origin_rels = extract_kg_relationships_from_event_origin_details(rel[2], rel[0], event_id)
            origin_details_kg = origin_details_kg | origin_rels
        elif ' -- Regional_Information -- Nearby_Places_' in rel[0]:
            split_meta_key = rel[0].strip().split(' -- ')
            if len(split_meta_key) > 3:
                if split_meta_key[-2] not in place_to_info_map.keys():
                    place_to_info_map[split_meta_key[-2]] = [None, None, None]
                place_to_info_map[split_meta_key[-2]][NEARBY_PLACES_KEY_TO_INDEX_MAP[split_meta_key[-1]]] = rel[2]
    event_root_kg.add((str(event_kg_node_name).strip(), 'has Event ID'.strip(), str(event_id).strip()))
    event_root_kg.add((str(event_id).strip(), 'is referred to as the'.strip(), str(event_kg_node_name).strip()))
    general_vicinity_of_event_onepager_kg = set([rel for rel in list(onepager_kg) if (" Area" in rel[0] and event_kg_node_name.split(' ')[0] not in rel[2]) or (event_kg_node_name.split(' ')[0] not in rel[0] and " Area" in rel[2])])
    event_specific_onepager_kg = onepager_kg.difference(general_vicinity_of_event_onepager_kg)
    temp_nearby_places_and_events_kg = extract_kg_relationships_from_nearby_places_info(place_to_info_map, event_id)
    nearby_places_kg = set([rel for rel in list(temp_nearby_places_and_events_kg) if event_id not in rel[0] and event_id not in rel[2]])
    places_near_event_kg = temp_nearby_places_and_events_kg.difference(nearby_places_kg)
    # define main knowledge graphs
    orig_json_kg = set(json_kg)
    general_info_kg = historical_kg | general_vicinity_of_event_onepager_kg | nearby_places_kg
    UNIVERSAL_GENERAL_INFO_KNOWLEDGE_GRAPH = UNIVERSAL_GENERAL_INFO_KNOWLEDGE_GRAPH | general_info_kg
    temp_event_kg = event_root_kg | event_specific_onepager_kg | moment_tensor_kg | origin_details_kg | places_near_event_kg
    event_kg = set()
    for rel in list(temp_event_kg):
        assert len(list(rel)) == 3 # sanity check
        parent = None
        edge = rel[1]
        child = None
        if event_id in rel[0]:
            parent = event_kg_node_name
        else:
            parent = rel[0]
        if event_id in rel[2]:
            child = event_kg_node_name
        else:
            child = rel[2]
        if parent != child:
            event_kg.add((parent, edge, child))
    # inspect the resulting knowledge graphs' nodes and edges
    #print("get_knowledge_graph_from_text_json: calling get_nodes_and_edges_from_knowledge_graph", flush=True)
    orig_json_kg_node_set, orig_json_kg_edge_set = get_nodes_and_edges_from_knowledge_graph(orig_json_kg)
    general_info_kg_node_set, general_info_kg_edge_set = get_nodes_and_edges_from_knowledge_graph(general_info_kg)
    event_kg_node_set, event_kg_edge_set = get_nodes_and_edges_from_knowledge_graph(event_kg)
    # print a report of the contents of each knowledge graph
    #report_kg_characteristics("orig_json_kg", orig_json_kg, orig_json_kg_node_set, orig_json_kg_edge_set)
    #report_kg_characteristics("general_info_kg", general_info_kg, general_info_kg_node_set, general_info_kg_edge_set)
    #report_kg_characteristics("event_kg", event_kg, event_kg_node_set, event_kg_edge_set)
    #print("get_knowledge_graph_from_text_json: STOP", flush=True)
    return UNIVERSAL_GENERAL_INFO_KNOWLEDGE_GRAPH, orig_json_kg, general_info_kg, event_kg, [orig_json_kg_node_set, orig_json_kg_edge_set, general_info_kg_node_set, general_info_kg_edge_set, event_kg_node_set, event_kg_edge_set]

def load_usgs_event_text_description(eid, json_file_path, knowledege_graph_save_dir, UNIVERSAL_GENERAL_INFO_KNOWLEDGE_GRAPH, plot_save_path=None):
    usgs_json_knowledge_graph, general_info_knowledge_graph, event_knowledge_graph = None, None, None
    usgs_json_kg_node_set, usgs_json_kg_edge_set = None, None
    general_info_kg_node_set, general_info_kg_edge_set = None, None
    event_kg_node_set, event_kg_edge_set = None, None
    with open(json_file_path, 'r') as infile:
        usgs_text_data = json.load(infile)
        assert len(list(usgs_text_data.keys())) == 1
        root_event = list(usgs_text_data.keys())[0]
        UNIVERSAL_GENERAL_INFO_KNOWLEDGE_GRAPH, usgs_json_knowledge_graph, general_info_knowledge_graph, event_knowledge_graph, \
            [usgs_json_kg_node_set, usgs_json_kg_edge_set, general_info_kg_node_set, \
             general_info_kg_edge_set, event_kg_node_set, event_kg_edge_set] = get_knowledge_graph_from_text_json(usgs_text_data, root_event, UNIVERSAL_GENERAL_INFO_KNOWLEDGE_GRAPH)
        if len(usgs_json_knowledge_graph) == 0 or len(general_info_knowledge_graph) == 0 or len(event_knowledge_graph) == 0:
            print("load_usgs_event_text_description: WARNING - UNABLE TO COMPLETE TEXT DESCRIPTION OF EVENT WITH eid == ", eid, " -- WILL OVER-WRITE WITH UNIVERSAL_GENERAL_INFO_KNOWLEDGE_GRAPH!", flush=True)
    # save knowledge graphs
    with open(knowledege_graph_save_dir+os.sep+str(eid)+"_usgs_json_knowledge_graph.pkl", 'wb') as outfile:
        pkl.dump(usgs_json_knowledge_graph, outfile)
    with open(knowledege_graph_save_dir+os.sep+str(eid)+"_general_info_knowledge_graph.pkl", 'wb') as outfile:
        pkl.dump(general_info_knowledge_graph, outfile)
    with open(knowledege_graph_save_dir+os.sep+str(eid)+"_event_knowledge_graph.pkl", 'wb') as outfile:
        pkl.dump(event_knowledge_graph, outfile)
    with open(knowledege_graph_save_dir+os.sep+str(eid)+"_usgs_json_kg_node_set.pkl", 'wb') as outfile:
        pkl.dump(usgs_json_kg_node_set, outfile)
    with open(knowledege_graph_save_dir+os.sep+str(eid)+"_usgs_json_kg_edge_set.pkl", 'wb') as outfile:
        pkl.dump(usgs_json_kg_edge_set, outfile)
    with open(knowledege_graph_save_dir+os.sep+str(eid)+"_general_info_kg_node_set.pkl", 'wb') as outfile:
        pkl.dump(general_info_kg_node_set, outfile)
    with open(knowledege_graph_save_dir+os.sep+str(eid)+"_general_info_kg_edge_set.pkl", 'wb') as outfile:
        pkl.dump(general_info_kg_edge_set, outfile)
    with open(knowledege_graph_save_dir+os.sep+str(eid)+"_event_kg_node_set.pkl", 'wb') as outfile:
        pkl.dump(event_kg_node_set, outfile)
    with open(knowledege_graph_save_dir+os.sep+str(eid)+"_event_kg_edge_set.pkl", 'wb') as outfile:
        pkl.dump(event_kg_edge_set, outfile)
    # plot graphs if requested
    if plot_save_path is not None:
        visualize_knowledge_graph(plot_save_path+os.sep+str(eid)+"_usgs_json_knowledge_graph_vis.png", usgs_json_knowledge_graph)
        visualize_knowledge_graph(plot_save_path+os.sep+str(eid)+"_general_info_knowledge_graph.png", general_info_knowledge_graph)
        visualize_knowledge_graph(plot_save_path+os.sep+str(eid)+"_event_knowledge_graph.png", event_knowledge_graph)
    general_info_knowledge_graph = sorted(list(general_info_knowledge_graph))
    random.Random(SHUFFLE_SEED_1).shuffle(general_info_knowledge_graph)
    event_knowledge_graph = sorted(list(event_knowledge_graph))
    random.Random(SHUFFLE_SEED_2).shuffle(event_knowledge_graph)
    return { 'usgs_json': sorted(list(usgs_json_knowledge_graph)), 'general_knowledge': general_info_knowledge_graph, 'event': event_knowledge_graph, }, UNIVERSAL_GENERAL_INFO_KNOWLEDGE_GRAPH
#""" # FOR DEBUGGING PURPOSES
# METHODS FOR USGS TEXT DATA -------------------------------------------------------------------------------------------------------------------------

# METHODS FOR USGS IMAGE DATA ------------------------------------------------------------------------------------------------------------------------
def load_usgs_event_image_to_array(img_file_path, plot_save_path=None):
    """
    Load an image in a .png or .jpg format into a NumPy array.
    """
    image_array = None
    if os.path.exists(img_file_path):
        with Image.open(img_file_path) as img: # Open the image file using Pillow
            img = img.convert("RGB")
            image_array = np.array(img)
            if plot_save_path is not None:
                plot_image_from_array(plot_save_path, image_array, title=img_file_path, xlabel="", ylabel="")
    return image_array
# METHODS FOR USGS IMAGE DATA ------------------------------------------------------------------------------------------------------------------------

# GENERAL METHODS FOR USGS DATA ----------------------------------------------------------------------------------------------------------------------
def load_usgs_event_text_and_img_data(events_to_modal_file_paths_map, save_dir, UNIVERSAL_GENERAL_INFO_KNOWLEDGE_GRAPH):
    print("load_usgs_event_text_and_img_data: START", flush=True)
    usgs_data_by_event_id = dict()
    for e, eid in enumerate(events_to_modal_file_paths_map.keys()):
        event_data_by_modality = dict()
        complete_text_description_available = True
        for modality_key in events_to_modal_file_paths_map[eid].keys():
            if modality_key == "time series": # time series modality
                if events_to_modal_file_paths_map[eid][modality_key] is None:# or eid not in events_to_focus_on:
                    event_data_by_modality[modality_key] = None
                else:
                    event_data_by_modality[modality_key] = {
                        "file_name": events_to_modal_file_paths_map[eid][modality_key]["file_name"], 
                        "rec_len": events_to_modal_file_paths_map[eid][modality_key]["rec_len"], 
                    }
                #""" # FOR DEBUGGING PURPOSES
            elif modality_key == "text description": # text modality
                event_data_by_modality[modality_key], UNIVERSAL_GENERAL_INFO_KNOWLEDGE_GRAPH = load_usgs_event_text_description(
                    eid, 
                    events_to_modal_file_paths_map[eid][modality_key]["file_name"], 
                    save_dir, 
                    UNIVERSAL_GENERAL_INFO_KNOWLEDGE_GRAPH, 
                    plot_save_path=save_dir if e==0 else None
                )
                if event_data_by_modality[modality_key] is None:
                    complete_text_description_available = False
                    break
            else: # case where modality is some form of image
                if events_to_modal_file_paths_map[eid][modality_key] is None or events_to_modal_file_paths_map[eid][modality_key]["file_name"] is None:
                    event_data_by_modality[modality_key] = None
                else:
                    assert ".png" in events_to_modal_file_paths_map[eid][modality_key]["file_name"] or ".jpg" in events_to_modal_file_paths_map[eid][modality_key]["file_name"]
                    event_data_by_modality[modality_key] = load_usgs_event_image_to_array(
                        events_to_modal_file_paths_map[eid][modality_key]["file_name"], 
                        plot_save_path=save_dir+os.sep+"visualizing_"+str(eid)+"_"+str(modality_key)+".png" if e==0 else None
                    )
                #""" # FOR DEBUGGING PURPOSES
        if complete_text_description_available: # commented out FOR DEBUGGING PURPOSES
            usgs_data_by_event_id[eid] = event_data_by_modality # auto-triggered FOR DEBUGGING PURPOSES
        # if e == 2: # FOR DEBUGGING PURPOSES
        #     print("load_usgs_event_text_and_img_data: WARNING - BREAKING FOR DEBUGGING PURPOSES", flush=True) # FOR DEBUGGING PURPOSES
        #     break # FOR DEBUGGING PURPOSES
    print("load_usgs_event_text_and_img_data: STOP", flush=True)
    return usgs_data_by_event_id, UNIVERSAL_GENERAL_INFO_KNOWLEDGE_GRAPH # set() FOR DEBUGGING PURPOSES

#""" # FOR DEBUGGING PURPOSES
def get_view_of_knowledge_graph(kg, args=None):
    if args is None:
        return kg
    elif "max_num_randomly_sampled_relations" in args.keys():
        if len(kg) <= args["max_num_randomly_sampled_relations"]:
            return kg
        return random.sample(kg, args["max_num_randomly_sampled_relations"]) # sample without replacement
    else:
        raise ValueError("get_view_of_knowledge_graph: Unrecognized args for obtaining view of knowledge graph")
    pass
#""" # FOR DEBUGGING PURPOSES


def currate_h5py_data_file(
    hdf5_path, regions_to_event_names_lists_map, num_samples_per_region_map, #event_names, num_samps, num_multimodal_samples_per_event,
    usgs_data_by_event_id, max_temporal_window_size, max_num_stations_per_sample, kg_context_args=None
): 
    # sanity check
    for r1, r2 in zip(sorted(list(regions_to_event_names_lists_map.keys())), sorted(list(num_samples_per_region_map.keys()))):
        assert r1 == r2

    # determine how many samples will be stored in the current data file
    num_samps = 0
    for reg_key in num_samples_per_region_map.keys(): 
        num_samps += num_samples_per_region_map[reg_key]

    # currate train samples
    with h5py.File(hdf5_path, 'w') as h5f:
        # Create group for metadata
        events_to_omit = set()
        h5f.attrs['num_samples'] = num_samps
        samp_inds_for_split = [index for index in range(num_samps)]
        random.Random(SHUFFLE_SEED_3).shuffle(samp_inds_for_split)

        # generate samples from each region
        sorted_region_keys = sorted(list(regions_to_event_names_lists_map.keys()))
        for r,region_key in enumerate(sorted_region_keys):
            num_samps_for_curr_region = num_samples_per_region_map[region_key]
            curr_reg_event_names = sorted(regions_to_event_names_lists_map[region_key])
            random.Random(SHUFFLE_SEED_4).shuffle(curr_reg_event_names)
            assert num_samps_for_curr_region >= len(curr_reg_event_names)
            num_samps_per_event = {ev_id: (num_samps_for_curr_region//len(curr_reg_event_names))+(1*(e_ind < num_samps_for_curr_region % len(curr_reg_event_names))) for e_ind, ev_id in enumerate(curr_reg_event_names)}
            queried_relations_by_event = {event_id:[] for event_id in curr_reg_event_names} # this will be used to track which knowledge graph relations have been used in a query for each event, to avoid duplicate samples
            
            # draw samples
            for s in range(num_samps_for_curr_region): 
                
                # determine which event the current sample will be associated with
                e = s % len(curr_reg_event_names)
                eid = curr_reg_event_names[e]
                
                # determine the index of the current sample within the h5f data file
                i = None
                try:
                    i = samp_inds_for_split[sum([num_samples_per_region_map[rk] for rk in sorted_region_keys[:r]]) + s]#[e*num_multimodal_samples_per_event + s] # index of current sample
                except:
                    print("currate_h5py_data_file: ERROR THROWN WHEN TRYING TO DEFINE SAMPLE INDEX", flush=True)
                    print("currate_h5py_data_file: num_samps == ", num_samps)
                    print("currate_h5py_data_file: len(samp_inds_for_split) == ", len(samp_inds_for_split))
                    print("currate_h5py_data_file: len(event_names) == ", len(event_names))
                    print("currate_h5py_data_file: region_key == ", region_key)
                    print("currate_h5py_data_file: e == ", e)
                    print("currate_h5py_data_file: num_samps_for_curr_region == ", num_samps_for_curr_region)
                    print("currate_h5py_data_file: s == ", s)
                    print("currate_h5py_data_file: hdf5_path == ", hdf5_path, flush=True)
                    raise ValueError()

                # Create sample group
                grp = h5f.create_group(f'sample_{i}')
                grp.attrs['event_id'] = str(eid)

                # Process sample fields, especially text fields (with space replacement)
                for modal_key in usgs_data_by_event_id[eid].keys():
                    
                    if modal_key == "text description":
                        #""" # FOR DEBUGGING PURPOSES
                        # grab different knowledge graphs (kgs)
                        g_kg = get_view_of_knowledge_graph(usgs_data_by_event_id[eid][modal_key]["general_knowledge"], args=kg_context_args)
                        e_kg = get_view_of_knowledge_graph(usgs_data_by_event_id[eid][modal_key]["event"], args=kg_context_args)
                        # determine which knowledge graph relationship will form the basis of the sample query and answer
                        query_rel = None
                        if len(e_kg) > 0 and len(e_kg) <= num_samps_per_event[eid]:#num_multimodal_samples_per_event: # case where the full (non-empty) event knowledge graph must be sampled from, possibly with replacement
                            # s indexes samples within a region; use per-event sample index based on queries already drawn
                            e_s = len(queried_relations_by_event[eid])
                            rel_ind = e_s % len(e_kg)
                            query_rel = e_kg[rel_ind]
                            e_kg = e_kg[:rel_ind] + e_kg[rel_ind + 1:] # remove query_relation from e_kg for sample
                            queried_relations_by_event[eid].append(query_rel)
                        else: # case where the general knowledge graph may need to be used in case randomly sampling from e_kg does not work
                            e_kg_inds = [r_ind for r_ind in range(len(e_kg))]
                            q_ind = None
                            for attempt_ind in range(2*num_samps_per_event[eid]):#num_multimodal_samples_per_event): # attempt to sample from e_kg
                                selected_rel_ind = random.choice(e_kg_inds)#e_kg)
                                selected_rel = e_kg[selected_rel_ind]
                                if selected_rel not in queried_relations_by_event[eid]:
                                    q_ind = selected_rel_ind
                                    query_rel = selected_rel
                                    queried_relations_by_event[eid].append(selected_rel)
                                    break
                            if query_rel is None: # if e_kg sampling fails, randomly draw from g_kg
                                g_kg_inds = [r_ind for r_ind in range(len(g_kg))]
                                q_ind = random.choice(g_kg_inds)#g_kg)
                                query_rel = g_kg[q_ind]
                                assert query_rel not in queried_relations_by_event[eid]
                                queried_relations_by_event[eid].append(query_rel)
                                g_kg = g_kg[:q_ind] + g_kg[q_ind+1:] # remove query_relation from g_kg for sample
                            else:
                                assert q_ind is not None
                                e_kg = e_kg[:q_ind] + e_kg[q_ind+1:] # remove query_relation from e_kg for sample
                        # grab prompt and answer to prompt
                        kg_prompt = str(query_rel[0])+" "+str(query_rel[1])+" " 
                        kg_answer = str(query_rel[2])
                        # store kgs/prompt/ans under grp
                        kg_grp = grp.create_group("text_based_knowedge_graph_sample")
                        kg_grp.create_dataset("general_knowledge", data=np.array([str(g_kg).encode('utf-8')]), compression="gzip")
                        kg_grp.create_dataset("event_knowledge", data=np.array([str(e_kg).encode('utf-8')]), compression="gzip")
                        kg_grp.create_dataset("query", data=np.array([str(kg_prompt).encode('utf-8')]), compression="gzip")
                        kg_grp.create_dataset("answer", data=np.array([str(kg_answer).encode('utf-8')]), compression="gzip")
                        #""" # FOR DEBUGGING PURPOSES
                        #pass # FOR DEBUGGING PURPOSES
                    elif modal_key == "time series":
                        usgs_ts_data = h5py.File(usgs_data_by_event_id[eid][modal_key]["file_name"], 'r')
                        curr_e_rec_len = usgs_data_by_event_id[eid][modal_key]["rec_len"]
                        # determine the start of the temporal window for the current sample's time series -- UPDATED 02/06/2026: WE ALWAYS SAMPLE TIME SERIES FROM START OF P-WAVE ONSET
                        # ts_samp_start_ind = 0
                        # if max_temporal_window_size < curr_e_rec_len:
                        #     ts_samp_start_ind = random.choice([i for i in range((curr_e_rec_len-max_temporal_window_size)+1)])
                        ts_samp_start_ind = TIME_STEP_FROM_WHICH_TO_SAMPLE_WINDOWS_IN_ORIG_RECORDINGS # p-wave onset is at 30 seconds, which for the original 100Hz signal is at time step 3000
                        # determine which stations will be included in the current sample's time series -- keep only the stations that have at least one time series with no nan values
                        sample_stations = []
                        for station in usgs_ts_data[eid].keys(): #candidate_sample_stations: 
                            keys_with_finite_recordings = []
                            for sek in usgs_ts_data[eid][station].keys(): 
                                if len(usgs_ts_data[eid][station][sek][:]) == curr_e_rec_len and np.isfinite(np.sum(usgs_ts_data[eid][station][sek][:])): 
                                    keys_with_finite_recordings.append(sek)
                            if len(keys_with_finite_recordings) > 0:
                                sample_stations.append(station)

                        if len(sample_stations) > max_num_stations_per_sample: 
                            sample_stations = random.sample(sample_stations, max_num_stations_per_sample)
                        # create the time series group for the current sample
                        ts_grp = None
                        for station_key in sorted(sample_stations): #usgs_ts_data[eid].keys(): 
                            station_sensor_keys = sorted(list(usgs_ts_data[eid][station_key].keys()))
                            for sensor_key in station_sensor_keys: 
                                if len(usgs_ts_data[eid][station_key][sensor_key][:]) == curr_e_rec_len:
                                    if not np.isfinite(np.sum(usgs_ts_data[eid][station_key][sensor_key][:])):
                                        print("currate_h5py_data_file: NON-FINITE TS SUM ENCOUNTERED IN event ", eid, " station ", station_key, " sensor ", sensor_key, " sample ", s, " -- EXCLUDING SENSOR FROM SAMPLE")
                                        print("np.sum(usgs_ts_data[eid][station_key][sensor_key][:]) == ", np.sum(usgs_ts_data[eid][station_key][sensor_key][:]))
                                        print("np.sum([1*np.isfinite(usgs_ts_data[eid][station_key][sensor_key][i]) for i in range(len(usgs_ts_data[eid][station_key][sensor_key][:]))]) == ", np.sum([1*np.isfinite(usgs_ts_data[eid][station_key][sensor_key][i]) for i in range(len(usgs_ts_data[eid][station_key][sensor_key][:]))]))
                                        print("usgs_ts_data[eid][station_key][sensor_key][:].shape == ", usgs_ts_data[eid][station_key][sensor_key][:].shape)
                                    else:
                                        curr_ts_rec = filtfilt(BUTTER_FILTER_B, BUTTER_FILTER_A, usgs_ts_data[eid][station_key][sensor_key][:])
                                        if not np.isfinite(np.sum(curr_ts_rec)):
                                            print("currate_h5py_data_file: POST-BUTTER_FILTER NON-FINITE TS SUM ENCOUNTERED!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                                            print("FULL np.sum(curr_ts_rec) == ", np.sum(curr_ts_rec))
                                            print("np.sum(usgs_ts_data[eid][station_key][sensor_key][:]) == ", np.sum(usgs_ts_data[eid][station_key][sensor_key][:]))
                                            print("eid == ", eid)
                                            print("station_key == ", station_key)
                                            print("sensor_key == ", sensor_key)
                                            print("s == ", s)
                                            raise ValueError()
                                        else:
                                            if len(curr_ts_rec) == curr_e_rec_len: 
                                                # sample a window from full time series
                                                curr_ts_rec = curr_ts_rec[ts_samp_start_ind::DOWN_SAMPLE_STEP_SIZE][:max_temporal_window_size]
                                                # normalize window / grab associated text description
                                                ts_normalized, descriptor, [_, _, _, _, _, _,] = get_normalized_usgs_time_series_with_descriptor( # RETURNS ts_normalized, descriptor, [mean, std, time_series_type, temporal_duration, sample_rt_in_hz, ]
                                                    curr_ts_rec, station_key, sensor_key, sample_rt_in_hz=LOWPASS_CUTOFF_FREQ, temporal_units_for_description="minutes", 
                                                )
                                                if not np.isfinite(np.sum(ts_normalized)):
                                                    print("currate_h5py_data_file: POST-NORMALIZATION NON-FINITE TS SUM ENCOUNTERED!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                                                    print("np.sum(ts_normalized) == ", np.sum(ts_normalized))
                                                    print("np.sum(curr_ts_rec) == ", np.sum(curr_ts_rec))
                                                    print("np.sum(usgs_ts_data[eid][station_key][sensor_key][:]) == ", np.sum(usgs_ts_data[eid][station_key][sensor_key][:]))
                                                    print("eid == ", eid)
                                                    print("station_key == ", station_key)
                                                    print("sensor_key == ", sensor_key)
                                                    print("s == ", s)
                                                    raise ValueError()
                                                else:
                                                    # store window/description under grp
                                                    if ts_grp is None:
                                                        ts_grp = grp.create_group("time_series_sample")
                                                        ts_grp.create_dataset("time_series_start_time_step_in_original_NEIC_format", data=ts_samp_start_ind)#, compression="gzip")
                                                    ts_grp.create_dataset("time_series_recording_from_station_"+station_key+"_sensor_"+sensor_key, data=ts_normalized, compression="gzip")
                                                    ts_grp.create_dataset("time_series_description_for_station_"+station_key+"_sensor_"+sensor_key, data=np.array([str(descriptor).encode('utf-8')]), compression="gzip")
                        
                        if ts_grp is None:
                            # raise ValueError("NO TIME SERIES COULD BE RECORDED FOR eid == "+str(eid)+" -- OMIT THIS EVENT FROM DATASET BEFORE PROCEEDING")
                            events_to_omit.add(eid)
                    
                    else:
                        pass # NotImplementedError() # we do not currently worry about the image modalities associated with the event

                # if s == 1: # FOR DEBUGGING PURPOSES
                #     print("currate_h5py_data_file: (sample loop) BREAKING FOR DEBUGGING PURPOSES", flush=True) # FOR DEBUGGING PURPOSES
                #     break # FOR DEBUGGING PURPOSES
            
            total_rels_queried = 0
            for event_ID in queried_relations_by_event.keys():
                total_rels_queried += len(queried_relations_by_event[event_ID])
            assert total_rels_queried == num_samps_for_curr_region#len(queried_relations) == num_multimodal_samples_per_event # COMMENTED OUT FOR DEBUGGING PURPOSES
            # if r == 2: #if e == 2: # FOR DEBUGGING PURPOSES
            #     print("currate_h5py_data_file: (region loop) BREAKING FOR DEBUGGING PURPOSES", flush=True) #(event loop) BREAKING FOR DEBUGGING PURPOSES", flush=True) # FOR DEBUGGING PURPOSES
            #     break # FOR DEBUGGING PURPOSES
        print("events_to_omit == ", events_to_omit, flush=True)
        assert len(list(events_to_omit)) == 0
    pass


def construct_kfolds_for_event_set( # note that 'folds' here are 'repeats' in actuality, but we use 'folds' as an appeal to intuition
    task_ID, event_names, usgs_data_by_event_id, num_cv_folds, root_save_dir, #num_train_events_per_fold, num_val_events_per_fold, num_multimodal_samples_per_event, 
    kg_context_args=None, max_temporal_window_size=None, max_num_stations_per_sample=4, max_events_per_region=10, num_train_samples=5000, num_test_samples=1250, 
):
    EVENT_ID_TO_REGION_NAME_MAP = None
    with open(EVENT_ID_TO_REGION_NAME_MAP_FILE_NAME, 'rb') as infile:
        EVENT_ID_TO_REGION_NAME_MAP = pkl.load(infile)

    if max_temporal_window_size is not None:
        assert max_temporal_window_size <= STANDARD_USGS_REC_NUM_TIME_STEPS
    else:
        max_temporal_window_size = STANDARD_USGS_REC_NUM_TIME_STEPS
    
    # determine which regions will be represented across all datasets, and how many events are associated with each region
    REGION_NAME_TO_EVENT_ID_MAP = dict()
    for eid in event_names:
        region_name = EVENT_ID_TO_REGION_NAME_MAP[eid]
        if region_name not in REGION_NAME_TO_EVENT_ID_MAP.keys():
            REGION_NAME_TO_EVENT_ID_MAP[region_name] = []
        REGION_NAME_TO_EVENT_ID_MAP[region_name].append(eid)
    regions_represented_across_folds = sorted(list(REGION_NAME_TO_EVENT_ID_MAP.keys()))
    random.Random(SHUFFLE_SEED_6).shuffle(regions_represented_across_folds)
    assert len(regions_represented_across_folds)*max_events_per_region <= num_train_samples+num_test_samples # ensure we can sample from every event while also balancing samples by region
    print("construct_kfolds_for_event_set: number of regions across data splits == ", len(regions_represented_across_folds))
    num_regions_per_fold_test_set = [(len(regions_represented_across_folds)//num_cv_folds)+(1*(i<len(regions_represented_across_folds)%num_cv_folds)) for i in range(num_cv_folds)]
    print("construct_kfolds_for_event_set: num_regions_per_fold_test_set == ", num_regions_per_fold_test_set)
    initial_region_name_to_num_events_map = {r:len(REGION_NAME_TO_EVENT_ID_MAP[r]) for r in REGION_NAME_TO_EVENT_ID_MAP.keys()}
    print("construct_kfolds_for_event_set: initial_region_name_to_num_events_map == ", initial_region_name_to_num_events_map)
    for region in REGION_NAME_TO_EVENT_ID_MAP:
        if len(REGION_NAME_TO_EVENT_ID_MAP[region]) > max_events_per_region:
            random.Random(SHUFFLE_SEED_5).shuffle(sorted(REGION_NAME_TO_EVENT_ID_MAP[region]))
            REGION_NAME_TO_EVENT_ID_MAP[region] = REGION_NAME_TO_EVENT_ID_MAP[region][:max_events_per_region]
    final_region_name_to_num_events_map = {r:len(REGION_NAME_TO_EVENT_ID_MAP[r]) for r in REGION_NAME_TO_EVENT_ID_MAP.keys()}
    print("construct_kfolds_for_event_set: final_region_name_to_num_events_map == ", final_region_name_to_num_events_map)
    print("construct_kfolds_for_event_set: REGION_NAME_TO_EVENT_ID_MAP == ", REGION_NAME_TO_EVENT_ID_MAP)
    # regions_by_increasing_num_associated_events = sorted(region_name_to_num_events_map.items(), key=operator.itemgetter(1, 0)) # we sort regions by event representation so that 'common' regions can be shared across folds
    # print("construct_kfolds_for_event_set: regions_by_increasing_num_associated_events == ", regions_by_increasing_num_associated_events)
    # num_regions_per_fold_test_set = [(len(regions_by_increasing_num_associated_events)//num_cv_folds)+(1*(i<len(regions_by_increasing_num_associated_events)%num_cv_folds)) for i in range(num_cv_folds)]

    # assign each region (and its associated events) to a fold
    fold_id = task_ID-1

    fold_to_region_and_event_list_map = dict()
    fold_to_num_train_samples_per_region_map = dict()
    fold_to_num_test_samples_per_region_map = dict()
    #for fold_id in range(num_cv_folds):
    fold_to_region_and_event_list_map[fold_id] = {"train": dict(), "test": dict(),}
    region_names_to_assign_to_curr_fold_test_set = regions_represented_across_folds[fold_id::num_cv_folds]#[x[0] for x in regions_by_increasing_num_associated_events[i::num_cv_folds]]
    assert len(region_names_to_assign_to_curr_fold_test_set) == num_regions_per_fold_test_set[fold_id]
    region_names_to_assign_to_curr_fold_train_set = [x for x in regions_represented_across_folds if x not in region_names_to_assign_to_curr_fold_test_set]#[x[0] for x in regions_by_increasing_num_associated_events if x[0] not in region_names_to_assign_to_curr_fold_test_set]
    # we balance sampling across regions, so the events from regions that are more frequently represented will be sampled from less than events that belong to a 'rarer' region
    fold_to_num_train_samples_per_region_map[fold_id] = {
        region_names_to_assign_to_curr_fold_train_set[j]: (num_train_samples//len(region_names_to_assign_to_curr_fold_train_set))+(1*(j<num_train_samples%len(region_names_to_assign_to_curr_fold_train_set))) 
        for j in range(len(region_names_to_assign_to_curr_fold_train_set))
    }
    fold_to_num_test_samples_per_region_map[fold_id] = {
        region_names_to_assign_to_curr_fold_test_set[j]: (num_test_samples//len(region_names_to_assign_to_curr_fold_test_set))+(1*(j<num_test_samples%len(region_names_to_assign_to_curr_fold_test_set))) 
        for j in range(len(region_names_to_assign_to_curr_fold_test_set))
    }
    for region_name in region_names_to_assign_to_curr_fold_train_set:
        fold_to_region_and_event_list_map[fold_id]['train'][region_name] = REGION_NAME_TO_EVENT_ID_MAP[region_name]
    for region_name in region_names_to_assign_to_curr_fold_test_set:
        fold_to_region_and_event_list_map[fold_id]['test'][region_name] = REGION_NAME_TO_EVENT_ID_MAP[region_name]
    
    # curate the samples for a fold
    # for fold_id in range(num_cv_folds):
    # make folder for saving fold data
    print("\nconstruct_kfolds_for_event_set: currating fold == ", fold_id, flush=True)
    curr_fold_dir = root_save_dir + os.sep + "fold_"+str(fold_id)
    if not os.path.exists(curr_fold_dir):
        os.mkdir(curr_fold_dir)
    # currate multimodal training/validation samples
    fold_train_hdf5_path = curr_fold_dir+os.sep+"train_data.h5"
    fold_val_hdf5_path = curr_fold_dir+os.sep+"val_data.h5"
    currate_h5py_data_file(
        fold_train_hdf5_path, 
        fold_to_region_and_event_list_map[fold_id]["train"], fold_to_num_train_samples_per_region_map[fold_id], # train_events, # num_train_samps, # num_multimodal_samples_per_event, 
        usgs_data_by_event_id, 
        max_temporal_window_size, 
        max_num_stations_per_sample, 
        kg_context_args=kg_context_args
    )
    currate_h5py_data_file(
        fold_val_hdf5_path, 
        fold_to_region_and_event_list_map[fold_id]["test"], fold_to_num_test_samples_per_region_map[fold_id], # val_events, # num_val_samps, # num_multimodal_samples_per_event, 
        usgs_data_by_event_id, 
        max_temporal_window_size, 
        max_num_stations_per_sample, 
        kg_context_args=kg_context_args
    )
    # save
    with open(curr_fold_dir+os.sep+'fold_meta_data.pkl', 'wb') as outfile:
        pkl.dump(fold_to_region_and_event_list_map[fold_id], outfile)
    pass


def preprocess_usgs_data(
    orig_ts_data_path, orig_TxtImg_data_path, cached_data_save_path, num_cv_folds=6, #num_train_events=945, num_val_events=189, num_multimodal_samples_per_event=1, 
    kg_context_args=None, max_temporal_window_size=None, max_num_stations_per_sample=4, max_events_per_region=10, num_train_samples=5000, num_test_samples=1250, 
):
    print("preprocess_usgs_data: START", flush=True)
    print("preprocess_usgs_data: -- orig_ts_data_path == ", orig_ts_data_path)
    print("preprocess_usgs_data: -- orig_TxtImg_data_path == ", orig_TxtImg_data_path)
    print("preprocess_usgs_data: -- cached_data_save_path == ", cached_data_save_path)
    print("preprocess_usgs_data: -- num_cv_folds == ", num_cv_folds)
    # print("preprocess_usgs_data: -- num_train_events == ", num_train_events)
    # print("preprocess_usgs_data: -- num_val_events == ", num_val_events)
    # print("preprocess_usgs_data: -- num_multimodal_samples_per_event == ", num_multimodal_samples_per_event)
    print("preprocess_usgs_data: -- kg_context_args == ", kg_context_args)
    print("preprocess_usgs_data: -- max_temporal_window_size == ", max_temporal_window_size)
    print("preprocess_usgs_data: -- max_num_stations_per_sample == ", max_num_stations_per_sample, flush=True)

    all_events_meta_data_save_dir = cached_data_save_path + os.sep + "all_events_meta_data"
    task_ID = int(os.getenv('SLURM_ARRAY_TASK_ID'))

    # Curate Knowledge Graph
    standardized_event_ids_to_focus_on = None
    all_usgs_data_by_event_id = None
    UNIVERSAL_GENERAL_INFO_KNOWLEDGE_GRAPH = None
    if task_ID > 1: 
        meta_data_loaded = False
        num_load_attempts = 0
        while not meta_data_loaded: 
            if os.path.isfile(all_events_meta_data_save_dir+os.sep+"UNIVERSAL_GENERAL_INFO_KNOWLEDGE_GRAPH.pkl"): 
                assert os.path.isfile(all_events_meta_data_save_dir+os.sep+"standardized_event_ids_to_focus_on.pkl")
                assert os.path.isfile(all_events_meta_data_save_dir+os.sep+"all_usgs_data_by_event_id.pkl")
                with open(all_events_meta_data_save_dir+os.sep+"standardized_event_ids_to_focus_on.pkl", 'rb') as infile:
                    standardized_event_ids_to_focus_on = pkl.load(infile)
                with open(all_events_meta_data_save_dir+os.sep+"all_usgs_data_by_event_id.pkl", 'rb') as infile: 
                    all_usgs_data_by_event_id = pkl.load(infile)
                with open(all_events_meta_data_save_dir+os.sep+"UNIVERSAL_GENERAL_INFO_KNOWLEDGE_GRAPH.pkl", 'rb') as infile:
                    UNIVERSAL_GENERAL_INFO_KNOWLEDGE_GRAPH = pkl.load(infile)
                meta_data_loaded = True
            else: 
                num_load_attempts += 1
                print("preprocess_usgs_data: task_ID == ", task_ID, " IS WAITING TO LOAD META DATA AFTER ", num_load_attempts, " LOAD ATTEMPTS", flush=True)
                time.sleep(META_DATA_LOADING_WAIT_TIME)
    else: 
        assert not os.path.isfile(all_events_meta_data_save_dir+os.sep+"UNIVERSAL_GENERAL_INFO_KNOWLEDGE_GRAPH.pkl")
        assert not os.path.isfile(all_events_meta_data_save_dir+os.sep+"standardized_event_ids_to_focus_on.pkl")
        assert not os.path.isfile(all_events_meta_data_save_dir+os.sep+"all_usgs_data_by_event_id.pkl")
        # determine which events we will be focused on -- those with available time series recordings
        available_usgs_ts_datafiles = sorted([x for x in os.listdir(orig_ts_data_path) if ".h5" in x])
        events_of_interest_to_neic_file_map = dict()
        for fn in available_usgs_ts_datafiles:
            usgs_data = h5py.File(orig_ts_data_path+os.sep+fn, 'r')
            curr_sorted_event_keys = sorted(list(usgs_data.keys()))
            print("preprocess_usgs_data: len(list(curr_sorted_event_keys)) == ", len(list(curr_sorted_event_keys)))
            for event_id in curr_sorted_event_keys:
                if event_id not in UNUSABLE_EVENTS:
                    events_of_interest_to_neic_file_map[event_id] = orig_ts_data_path+os.sep+fn
        
        # for each event, stitch together relevant files from different modalities
        events_of_interest_id_to_modal_file_paths_map = get_event_id_to_modal_files_map(events_of_interest_to_neic_file_map, orig_TxtImg_data_path)
        print("preprocess_usgs_data: events_of_interest_id_to_modal_file_paths_map == ", events_of_interest_id_to_modal_file_paths_map, flush=True)
        events_of_interest = sorted(list(events_of_interest_id_to_modal_file_paths_map.keys()))
        print("preprocess_usgs_data: TOTAL NUM events_of_interest == ", len(events_of_interest))
        print("preprocess_usgs_data: events_of_interest == ", events_of_interest, flush=True)

        # explore time series files
        if not os.path.exists(all_events_meta_data_save_dir):
            os.mkdir(all_events_meta_data_save_dir)
        usgs_station_channels_shared_across_all_events, _, standardized_event_ids_to_focus_on, standardized_event_ids_to_rec_len_map = explore_usgs_event_time_series_files(
            events_of_interest_id_to_modal_file_paths_map, # see https://www.geeksforgeeks.org/python/python-merging-two-dictionaries/
            events_of_interest, 
            all_events_meta_data_save_dir
        )
        
        UNIVERSAL_GENERAL_INFO_KNOWLEDGE_GRAPH = set()
        print("__MAIN__: LOADING DATA FOR ALL (SENSOR-RECORDED) EVENT STATISTICS ##################", flush=True)
        all_events_visualizations_save_dir = cached_data_save_path + os.sep + "all_events_visualizations"
        if not os.path.exists(all_events_visualizations_save_dir):
            os.mkdir(all_events_visualizations_save_dir)
        all_usgs_data_by_event_id, UNIVERSAL_GENERAL_INFO_KNOWLEDGE_GRAPH = load_usgs_event_text_and_img_data(
            {
                e:{
                    m:{
                        "file_name": events_of_interest_id_to_modal_file_paths_map[e][m], 
                        "rec_len": standardized_event_ids_to_rec_len_map[e] if m == "time series" else None, 
                    } for m in events_of_interest_id_to_modal_file_paths_map[e].keys()
                } for e in events_of_interest_id_to_modal_file_paths_map.keys() if e in standardized_event_ids_to_focus_on
            }, 
            all_events_visualizations_save_dir, 
            UNIVERSAL_GENERAL_INFO_KNOWLEDGE_GRAPH
        )
        with open(all_events_meta_data_save_dir+os.sep+"standardized_event_ids_to_focus_on.pkl", 'wb') as outfile:
            pkl.dump(standardized_event_ids_to_focus_on, outfile)
        with open(all_events_meta_data_save_dir+os.sep+"all_usgs_data_by_event_id.pkl", 'wb') as outfile: 
            pkl.dump(all_usgs_data_by_event_id, outfile)
        with open(all_events_meta_data_save_dir+os.sep+"UNIVERSAL_GENERAL_INFO_KNOWLEDGE_GRAPH.pkl", 'wb') as outfile:
            pkl.dump(UNIVERSAL_GENERAL_INFO_KNOWLEDGE_GRAPH, outfile)

    # construct cross-validation folds for down-stream training
    print("__MAIN__: CONSTRUCTING K-FOLDS ##################################")
    all_usgs_data_dir = cached_data_save_path + os.sep + "all_sensor_recorded_events"
    if not os.path.exists(all_usgs_data_dir):
        os.mkdir(all_usgs_data_dir)
        
    construct_kfolds_for_event_set(
        task_ID, 
        standardized_event_ids_to_focus_on, #events_of_interest, 
        all_usgs_data_by_event_id, 
        num_cv_folds, 
        all_usgs_data_dir, # num_train_events, # num_val_events, # num_multimodal_samples_per_event, 
        kg_context_args=kg_context_args, 
        max_temporal_window_size=max_temporal_window_size, 
        max_num_stations_per_sample=max_num_stations_per_sample, 
        max_events_per_region=max_events_per_region, 
        num_train_samples=num_train_samples, 
        num_test_samples=num_test_samples, 
    )
    
    print("preprocess_usgs_data: STOP", flush=True)
    pass
# GENERAL METHODS FOR USGS DATA ----------------------------------------------------------------------------------------------------------------------



if __name__ == "__main__":
    # read in arguments ##########################################################################################
    parser = argparse.ArgumentParser(description="Preprocess and/or Load a USGS Multi-Modal Seismic Dataset.")
    #parser.add_argument("--repo_path", type=str, required=True, help="Path to the ICE-MAP repository")
    parser.add_argument("--cached_data_save_path", type=str, required=True, help="Path to save cached pre-processed data")
    parser.add_argument("--orig_ts_data_path", type=str, required=True, help="Path to folder with original usgs time series data")
    parser.add_argument("--orig_TxtImg_data_path", type=str, required=True, help="Path to folder with original usgs text+image data")
    parser.add_argument("--num_cv_folds", type=int, required=True, help="Number of cross-validation folds to currate")
    # parser.add_argument("--num_train_events", type=int, required=True, help="Number of events to include in training data")
    # parser.add_argument("--num_val_events", type=int, required=True, help="Number of events to include in validation data")
    # parser.add_argument("--num_multimodal_samples_per_event", type=int, required=True, help="Number of samples to draw per event")
    parser.add_argument("--max_num_randomly_sampled_kg_relations", type=int, required=True, help="Number of kg relations to sample per kg per sample")
    parser.add_argument("--max_temporal_window_size", type=int, required=True, help="Number of time steps to include in final sample")
    parser.add_argument("--max_num_stations_per_sample", type=int, required=True, help="Number of recording stations to include per sample")
    parser.add_argument("--max_events_per_region", type=int, required=True, help="Maximum number of events a region can be associated with in the dataset")
    parser.add_argument("--num_train_samples", type=int, required=True, help="Number of samples to draw for the training split - recommended >= 5000 when max_events_per_region == 10")
    parser.add_argument("--num_test_samples", type=int, required=True, help="Number of samples to draw for the training split - recommended >= 1250 when max_events_per_region == 10")
    args = parser.parse_args()

    print("__MAIN__: START", flush=True)
    # repo_path = args.repo_path
    cached_data_save_path = args.cached_data_save_path
    orig_ts_data_path = args.orig_ts_data_path
    orig_TxtImg_data_path = args.orig_TxtImg_data_path

    # print("__MAIN__: repo_path == ", repo_path)
    print("__MAIN__: cached_data_save_path == ", cached_data_save_path)
    print("__MAIN__: orig_ts_data_path == ", orig_ts_data_path)
    print("__MAIN__: orig_TxtImg_data_path == ", orig_TxtImg_data_path, flush=True)

    # set up environment #########################################################################################
    print("__MAIN__: setting up environment", flush=True)
    main_script_file_path = Path(__file__)
    absolute_main_script_file_path = main_script_file_path.resolve()
    repo_directory = absolute_main_script_file_path.parent.parent
    sys.path.append(str(repo_directory))
    opentslm_src_directory = repo_directory / "OpenTSLM" / "src"
    sys.path.append(str(opentslm_src_directory))

    # import local repo dependencies
    from usgs_data.data_utils import reverse_cardinal_direction, get_nodes_and_edges_from_knowledge_graph, convert_knowledge_graph_relationship_to_text_factoid_sample, normalize_string, get_normalized_usgs_time_series_with_descriptor
    from utils.plotting import plot_time_series, plot_image_from_array, visualize_knowledge_graph, plot_sensor_event_histogram
    from utils.general_utils import set_all_seeds
    set_all_seeds(seed=INITIAL_RAND_SEED)

    # preprocess the original dataset if necessary ###############################################################
    print("__MAIN__: Preprocessing data", flush=True)
    preprocess_usgs_data(
        orig_ts_data_path, 
        orig_TxtImg_data_path, 
        cached_data_save_path, 
        num_cv_folds=args.num_cv_folds, #6, # num_train_events=args.num_train_events, #945, # num_val_events=args.num_val_events, #189, # num_multimodal_samples_per_event=args.num_multimodal_samples_per_event, #100, 
        kg_context_args={"max_num_randomly_sampled_relations": args.max_num_randomly_sampled_kg_relations,}, #10,}, 
        max_temporal_window_size=args.max_temporal_window_size, #4096, 
        max_num_stations_per_sample=args.max_num_stations_per_sample, #4, 
        max_events_per_region=args.max_events_per_region, #10, 
        num_train_samples=args.num_train_samples, #5000, 
        num_test_samples=args.num_test_samples, #1250, 
    )
    
    print("__MAIN__: DONE!!!", flush=True)
    pass
