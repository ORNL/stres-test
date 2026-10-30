# import re
import unicodedata
import numpy as np
from utils.time_series_utils import pad_time_series



def parse_descriptor(descriptor):
    """
    Generated with help from GPT-4o
    """
    # Extract values from the matching groups
    station_name = descriptor.split(' sensor ')[0][len("The station "):]#match.group("station_name")
    sensor_name = descriptor.split(' sensor ')[1].split(" data was recorded over ")[0]#match.group("sensor_name")
    sample_rt_in_hz = descriptor.split(' sampled at ')[1].split(" Hz with mean=")[0]#float(match.group("sample_rt_in_hz"))
    mean = round(float(descriptor.split(' Hz with mean=')[1].split(" and std=")[0]), 1)#match.group("mean")), 1)  # Round mean to one significant figure
    if not np.isfinite(mean):
        print("data_utils.parse_descriptor: non-finite mean == "+str(mean)+" comes from descriptor == "+str(descriptor), flush=True)
        raise ValueError("data_utils.parse_descriptor: non-finite mean == "+str(mean)+" comes from descriptor == "+str(descriptor))
    std = round(float(descriptor.split(' and std=')[1][:-1]), 1)#match.group("std")), 1)    # Round std to one significant figure
    # if not np.isfinite(std): # DECRIMENTED 12/23/2025 -- AS LONG AS THE MEAN IS FINITE, WE ALLOW INF STD TO PASS (AS THIS RESULTS IN ZERO-MAGNITUDE NORMALIZED SIGNALS)
    #     print("data_utils.parse_descriptor: non-finite std == "+str(std)+" comes from descriptor == "+str(descriptor), flush=True)
    #     raise ValueError("data_utils.parse_descriptor: non-finite std == "+str(std)+" comes from descriptor == "+str(descriptor))
    # Return the extracted values
    return {"station_name": station_name, "sensor_name": sensor_name, "sample_rt_in_hz": sample_rt_in_hz, "mean": mean, "std": std, }


def get_normalized_usgs_time_series_with_descriptor(time_series, station_name, sensor_name, sample_rt_in_hz=None, temporal_units_for_description="minutes"):
    """
    Returns zero-mean, unit-variable time series with a descriptor resembling: 'This is heart-rate data over 24 hours sampled at 50 Hz with mean=61 and std=12.'
    Implementation references: https://github.com/StanfordBDHG/OpenTSLM/blob/main/src/time_series_datasets/util.py
    """
    ASSUMED_SAMPLING_RATE = 100. # Hz - NEIC data has been standardized to 100 hz as of 12/08/2025
    ts_normalized = None
    # determine the temporal duration of the signal
    temporal_duration = None
    if sample_rt_in_hz is None:
        sample_rt_in_hz = ASSUMED_SAMPLING_RATE
    if temporal_units_for_description == "minutes":
        temporal_duration = len(time_series)/(sample_rt_in_hz*60.)
    else:
        raise ValueError("get_normalized_usgs_time_series_with_descriptor: Unsupported temporal duration unit")
    # normalize the signal
    mean = np.mean(time_series)
    std = np.std(time_series)
    if std > 1e-8:  # Avoid division by zero
        ts_normalized = (time_series - mean) / std
    else:
        ts_normalized = time_series - mean
    # describe the normalized signal
    descriptor = "The station "+str(station_name)+" sensor "+str(sensor_name)+" data was recorded over "+str(temporal_duration)+" "+str(temporal_units_for_description)+" sampled at "+str(sample_rt_in_hz)+" Hz with mean="+str(mean)+" and std="+str(std)+"."
    return ts_normalized, descriptor, [mean, std, station_name, sensor_name, temporal_duration, sample_rt_in_hz, ]
# def get_normalized_usgs_time_series_with_descriptor(time_series, station_name, sensor_name, sample_rt_in_hz=None, temporal_units_for_description="minutes"):
#     """
#     Returns zero-mean, unit-variable time series with a descriptor resembling: 
#     'This is heart-rate data over 24 hours sampled at 50 Hz with mean=61 and std=12.'
#     Implementation references: https://github.com/StanfordBDHG/OpenTSLM/blob/main/src/time_series_datasets/util.py
#     """
#     # references https://www.researchgate.net/publication/395975374_OpenTSLM_Time-Series_Language_Models_for_Reasoning_over_Multivariate_Medical_Text-_and_Time-Series_Data
#     # NCEDC_NCSN_SENSOR_CATEGORIES = ["HHZ", "HHN", "HHE", "HNZ", "HNN", "HNE", "EHZ", "EHN", "EHE", "HH1", "HH2", "HN1", "HN2", ] # assumed to conform with typical sampling rate as described in https://ncedc.org/ncsn/ # WARNING: HNZ, HNN, HNE may be 100.0 or 200.0 hz sensors, as per https://ncedc.org/ncsn/, but we assume they are 100.0 if sample_rt_in_hz is None
#     # BROADBAND_ACCELEROMETER_CATS = ["BNZ", ] # https://pubs.geoscienceworld.org/ssa/srl/article/86/6/1717/315549/A-Quick-SEED-Tutorial
#     # Ayiti_Séismes_System_SENSORS = ["ENE", "ENN", "ENZ", ] # https://pubs.geoscienceworld.org/ssa/srl/article/94/6/2725/627977/Monitoring-of-Local-Earthquakes-in-Haiti-Using-Low
#     ASSUMED_SAMPLING_RATE = 100. # Hz - NEIC data has been standardized to 100 hz as of 12/08/2025
#     ts_normalized = None

#     # determine the temporal duration of the signal
#     temporal_duration = None
#     if sample_rt_in_hz is None:
#         sample_rt_in_hz = ASSUMED_SAMPLING_RATE
#         # is_100_hz_sensor = False
#         # for sensor_category in NCEDC_NCSN_SENSOR_CATEGORIES:
#         #     if sensor_category in time_series_type: # assumed to conform with typical sampling rate as described in https://ncedc.org/ncsn/
#         #         is_100_hz_sensor = True
#         #         break
#         # is_ayiti_sensor = False
#         # for sensor_category in Ayiti_Séismes_System_SENSORS:
#         #     if sensor_category in time_series_type: # assumed to conform with typical sampling rate as described in https://pubs.geoscienceworld.org/ssa/srl/article/94/6/2725/627977/Monitoring-of-Local-Earthquakes-in-Haiti-Using-Low
#         #         is_ayiti_sensor = True
#         #         break
#         # if is_100_hz_sensor or is_ayiti_sensor:
#         #     sample_rt_in_hz = 100.
#         # elif "BNZ" in time_series_type: 
#         #     sample_rt_in_hz = 45. # middle of range given in https://pubs.geoscienceworld.org/ssa/srl/article/86/6/1717/315549/A-Quick-SEED-Tutorial
#         # else:
#         #     raise ValueError("get_normalized_usgs_time_series_with_descriptor: Could not determine sample_rt_in_hz from time_series_type")
#     if temporal_units_for_description == "minutes":
#         temporal_duration = len(time_series)/(sample_rt_in_hz*60.)
#     else:
#         raise ValueError("get_normalized_usgs_time_series_with_descriptor: Unsupported temporal duration unit")
    
#     # normalize the signal
#     mean = np.mean(time_series)
#     std = np.std(time_series)
#     if std > 1e-8:  # Avoid division by zero
#         ts_normalized = (ts - mean) / std
#     else:
#         ts_normalized = ts - mean
    
#     # describe the normalized signal
#     descriptor = "This is station "+station_name+" sensor "+sensor_name+" data over "+temporal_duration+" "+temporal_units_for_description+" sampled at "+sample_rt_in_hz+" Hz with mean="+str(mean)+" and std="+str(std)+"."
#     return ts_normalized, descriptor, [mean, std, time_series_type, temporal_duration, sample_rt_in_hz, ]


def prepare_text_timeseries_sample_batch_for_generative_eval(batch, tokenizer, device, log_file_path, universal_prompt_pretext=None,):
    """ References https://github.com/StanfordBDHG/OpenTSLM/blob/main/src/model/llm/OpenTSLMFlamingo.py """
    media_token_id = tokenizer("<image>", add_special_tokens=False)["input_ids"][-1]
    endofchunk_token_id = tokenizer("<|endofchunk|>", add_special_tokens=False)["input_ids"][-1]
    # Process time series data
    images = pad_time_series(batch).to(device)
    images = images.unsqueeze(1)  # Add time dimension
    # Process text elements of batch
    text_inputs = []
    target_text_outputs = []
    for item in batch: 
        prompt_text = item["pre_prompt"]
        if universal_prompt_pretext is not None:
            prompt_text = universal_prompt_pretext + prompt_text
        for ts_text in item["time_series_text"]:
            prompt_text += f" {tokenizer.decode([media_token_id])} {ts_text} {tokenizer.decode([endofchunk_token_id])}"
        if item["post_prompt"]:
            prompt_text += f" {item['post_prompt']}"
        text_inputs.append(prompt_text)
        target_text_outputs.append(item['answer'])
    # Tokenize prompts and answers
    tokenized_inputs = tokenizer(text_inputs, padding="longest", return_tensors="pt")
    input_ids = tokenized_inputs.input_ids.to(device)
    attention_mask = tokenized_inputs.attention_mask.to(device)
    target_ids = tokenizer(target_text_outputs, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
    return input_ids, images, attention_mask, target_ids


def normalize_string(text):
    return unicodedata.normalize('NFKC', text)

def reverse_cardinal_direction(dir):
    rev_dir = ''
    if dir == 'north':
        rev_dir = 'south'
    elif dir == 'south': 
        rev_dir = 'north'
    elif dir == 'east': 
        rev_dir = 'west'
    elif dir == 'west': 
        rev_dir = 'east'
    elif dir == 'coast':
        rev_dir = 'inland'
    elif dir == 'inland':
        rev_dir = 'off the coast'
    elif dir == 'southeast':
        rev_dir = 'northwest'
    elif dir == 'northwest':
        rev_dir = 'southeast'
    elif dir == 'southwest':
        rev_dir = 'northeast'
    elif dir == 'northeast':
        rev_dir = 'southwest'
    elif len(dir) <= 3:
        for c in dir:
            if c == 'N':
                rev_dir = rev_dir + 'S'
            elif c == 'S': 
                rev_dir = rev_dir + 'N'
            elif c == 'E': 
                rev_dir = rev_dir + 'W'
            elif c == 'W': 
                rev_dir = rev_dir + 'E'
            else:
                raise ValueError('Unrecognized cardinal direction character == '+str(c))
    else:
        raise NotImplementedError('Unrecognized dir input requested to be reversed -- dir == '+str(dir))
    return rev_dir

def get_nodes_and_edges_from_knowledge_graph(full_kg):
    kg_node_set, kg_edge_set = set(), set()
    for rel in list(full_kg):
        kg_node_set.add(rel[0])
        kg_node_set.add(rel[2])
        kg_edge_set.add(rel[1])
    return kg_node_set, kg_edge_set

def convert_knowledge_graph_relationship_to_text_factoid_sample(relation, kg_id, rel_id):
    assert len(list(relation)) == 3
    sample = {
        "id": '_'.join(['kg-'+str(kg_id), 'rel-'+str(rel_id),]), 
        "type": "factoid", 
        "prompt": str(relation[0])+" "+str(relation[1])+" ", 
        "answer": str(relation[2]), 
    }
    return sample

# def combine_samples_from_modalities_into_multimodalsample(text_samples, time_series_samples, image_samples, text_id, time_series_id, image_id):
#     mm_sample = {
#         "id": '~'.join(['txt>'+str(text_id), 'ts>'+str(time_series_id), 'img>'+str(image_id),]), 
#         "text domain": text_samples, 
#         "time series domain": time_series_samples, 
#         "image domain": image_samples, 
#     }
#     return mm_sample