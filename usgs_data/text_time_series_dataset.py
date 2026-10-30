import h5py
import torch
import numpy as np

import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from usgs_data.data_utils import parse_descriptor
from utils.time_series_utils import extend_time_series_to_match_patch_size_and_aggregate

torch.set_default_dtype(torch.float32)


class TextTimeSeriesPrompt():
    """
    Adapted from https://github.com/StanfordBDHG/OpenTSLM/blob/main/src/prompt/text_time_series_prompt.py
    """
    def __init__(self, text, time_series):
        assert time_series.ndim == 1, "Time series must be one-dimensional! You can input multiple time series, but each time series must be a one-dimensional array."
        assert time_series.size > 0, "Time series must not be empty!"
        super().__init__()
        self.__text = text
        self.__time_series = time_series
        pass

    def get_text(self):
        return self.__text

    def get_time_series(self):
        return self.__time_series


class PromptWithAnswer:
    """
    Adapted from https://github.com/StanfordBDHG/OpenTSLM/blob/main/src/prompt/prompt_with_answer.py
    """
    def __init__(self, pre_prompt, text_time_series_prompt_list, post_prompt, answer,):
        self.pre_prompt = pre_prompt
        self.text_time_series_prompt_texts = list(map(lambda x: x.get_text(), text_time_series_prompt_list))
        self.text_time_series_prompt_time_series = list(map(lambda x: x.get_time_series(), text_time_series_prompt_list))
        self.post_prompt = post_prompt
        self.answer = answer
        pass

    def to_dict(self):
        return {
            "answer": self.answer,
            "post_prompt": self.post_prompt, #.get_text(),
            "pre_prompt": self.pre_prompt, #.get_text(),
            "time_series": self.text_time_series_prompt_time_series,
            "time_series_text": self.text_time_series_prompt_texts,
        }


class TextTimeSeriesHDF5Dataset(Dataset):#QADataset):
    """
    Adapted from : 
        - https://github.com/StanfordBDHG/OpenTSLM/blob/main/src/time_series_datasets/sleep/SleepEDFQADataset.py
        - https://github.com/StanfordBDHG/OpenTSLM/blob/main/src/time_series_datasets/QADataset.py
    """
    def __init__(self, hdf5_path, log_file_path, EOS_TOKEN, time_series_sample_rt=100, include_gen_knowledge=False, condense_ts_descriptions=True):
        super().__init__()
        open(log_file_path, "a").write("HDF5Dataset.__init__: START \n")#print("HDF5Dataset.__init__: START", flush=True)
        self.hdf5_path = hdf5_path
        self.log_file_path = log_file_path
        self.EOS_TOKEN = EOS_TOKEN
        self.time_series_sample_rt = time_series_sample_rt
        self.include_gen_knowledge = include_gen_knowledge
        self.condense_ts_descriptions = condense_ts_descriptions
        self.file = None  # Will open on first access
        # Get length without loading data
        with h5py.File(hdf5_path, 'r') as f:
            self._num_samples = f.attrs['num_samples']
        open(log_file_path, "a").write("HDF5Dataset.__init__: self._num_samples == "+str(self._num_samples)+" \n")#print("HDF5Dataset.__init__: self._num_samples samples")
        self.POST_PROMPT_START = str(time_series_sample_rt)+" Hz seismic sensor recordings and the following knowledge graph triples - formatted as (node1, relationship, node2) - have been made publicly available by the USGS: " # "" FOR DEBUGGING PURPOSES
        open(log_file_path, "a").write("HDF5Dataset.__init__: self.POST_PROMPT_START == "+str(self.POST_PROMPT_START)+" \n")
        self.POST_PROMPT_END = "Given this information, the following statement is true: "#"As an expert in seismology, use the data from the USGS to complete the following statement: "
        open(log_file_path, "a").write("HDF5Dataset.__init__: self.POST_PROMPT_END == "+str(self.POST_PROMPT_END)+" \n")
        open(log_file_path, "a").write("HDF5Dataset.__init__: STOP \n")#print("HDF5Dataset.__init__: STOP", flush=True)
        pass

    def __len__(self):
        return self._num_samples
    
    def _get_post_prompt_and_answer(self, sample_kg_txt_subgroup):
        """ sample_kg_txt_subgroup == grp['text_based_knowedge_graph_sample'] """
        gen_kg_text = ""
        if self.include_gen_knowledge:
            gen_kg_text = ''.join([elem.decode('utf-8') for elem in sample_kg_txt_subgroup['general_knowledge'][()]])
        event_kg_text = ''.join([elem.decode('utf-8') for elem in sample_kg_txt_subgroup['event_knowledge'][()]]) # "" FOR DEBUGGING PURPOSES
        event_kg_query = ''.join([elem.decode('utf-8') for elem in sample_kg_txt_subgroup['query'][()]])
        event_kg_answer = ''.join([elem.decode('utf-8') for elem in sample_kg_txt_subgroup['answer'][()]])
        if not event_kg_answer.endswith(self.EOS_TOKEN):
            event_kg_answer += self.EOS_TOKEN
        post_prompt = self.POST_PROMPT_START+"\n * General Knowledge - "+gen_kg_text+"\n * Knowledge Specific to the Recorded Event - "+event_kg_text+"\n\n"+self.POST_PROMPT_END+event_kg_query
        return post_prompt, event_kg_answer

    def _get_text_time_series_prompt_list(self, sample_ts_subgroup):#row): -> List[TextTimeSeriesPrompt]:
        """ sample_ts_subgroup == sample_grp['time_series_sample'] """
        ts_rec_keys = []
        ts_descrip_keys = []
        for key in sample_ts_subgroup.keys():
            if "time_series_recording_from_station_" in key:
                ts_rec_keys.append(key)
            elif "time_series_description_for_station_" in key:
                ts_descrip_keys.append(key)
        assert len(ts_rec_keys) > 0
        assert len(ts_rec_keys) == len(ts_descrip_keys)
        # ts_recordings = [sample_ts_subgroup[k][:] for k in sorted(ts_rec_keys)]
        ts_recordings = []
        for k in sorted(ts_rec_keys): 
            if not np.isfinite(np.sum(sample_ts_subgroup[k][:])):
                raise ValueError("TextTimeSeriesHDF5Dataset._get_text_time_series_prompt_list: non-finite time series for k == "+str(k)+" -- np.sum(sample_ts_subgroup[k][:])")
            ts_recordings.append(sample_ts_subgroup[k][:])
        # ts_descriptions = [
        #     ""#''.join([elem.decode('utf-8') for elem in sample_ts_subgroup[k][()]])  # "" FOR DEBUGGING PURPOSES
        #     for k in sorted(ts_descrip_keys)
        # ]
        full_ts_descriptions = [''.join([elem.decode('utf-8') for elem in sample_ts_subgroup[k][()]]) for k in sorted(ts_descrip_keys)]
        condensed_ts_descriptions = []
        if self.condense_ts_descriptions:
            for descrip in full_ts_descriptions:
                # open(self.log_file_path, "a").write("HDF5Dataset._get_text_time_series_prompt_list: descrip == "+str(descrip)+" \n")
                key_stats = parse_descriptor(descrip)
                # open(self.log_file_path, "a").write("HDF5Dataset._get_text_time_series_prompt_list: key stats sample_rt_in_hz == "+str(key_stats["sample_rt_in_hz"])+" \n")
                # open(self.log_file_path, "a").write("HDF5Dataset._get_text_time_series_prompt_list: self.time_series_sample_rt == "+str(self.time_series_sample_rt)+" \n")
                #assert float(key_stats["sample_rt_in_hz"]) == float(self.time_series_sample_rt)
                # open(self.log_file_path, "a").write("HDF5Dataset._get_text_time_series_prompt_list: NOW MAKING condensed_descrip \n")
                if not np.isfinite(key_stats["mean"]):
                    raise ValueError("TextTimeSeriesHDF5Dataset._get_text_time_series_prompt_list: non-finite key_stats mean == "+str(key_stats['mean']))
                # if not np.isfinite(key_stats["std"]): # DECRIMENTED 12/23/2025 -- AS LONG AS THE MEAN IS FINITE, WE ALLOW INF STD TO PASS (AS THIS RESULTS IN ZERO-MAGNITUDE NORMALIZED SIGNALS)
                #     raise ValueError("TextTimeSeriesHDF5Dataset._get_text_time_series_prompt_list: non-finite key_stats std == "+str(key_stats['std']))
                condensed_descrip = str(key_stats["station_name"])+", "+str(key_stats["sensor_name"])+": avg "+str(key_stats["mean"])+", std "+str(key_stats["std"])+"." # "Station "+str(key_stats["station_name"])+", Sensor "+str(key_stats["sensor_name"])+": Mean "+str(key_stats["mean"])+", Std "+str(key_stats["std"])+"." # "" FOR DEBUGGING PURPOSES
                # open(self.log_file_path, "a").write("HDF5Dataset._get_text_time_series_prompt_list: descrip == "+str(descrip)+" \n")
                condensed_ts_descriptions.append(condensed_descrip)
        else:
            condensed_ts_descriptions = full_ts_descriptions
        return [
            TextTimeSeriesPrompt(text_description, time_series) 
            for text_description, time_series in zip(condensed_ts_descriptions, ts_recordings)#ts_descriptions, ts_recordings)
        ]

    def __getitem__(self, idx):
        # Lazy-open HDF5 file (once per worker process)
        if self.file is None:
            self.file = h5py.File(self.hdf5_path, 'r')
        # Directly access sample group
        sample_grp = self.file[f'sample_{idx}']
        # Get text-based prompt and answer from knowledge graph(s)
        post_prompt, answer = self._get_post_prompt_and_answer(sample_grp['text_based_knowedge_graph_sample'])
        # get time series and corresponding text-based descriptions from sample_grp
        ts_based_textTimeSeriesPrompt_list = self._get_text_time_series_prompt_list(sample_grp['time_series_sample'])
        return PromptWithAnswer(
            "", #self._get_pre_prompt(row).strip()),
            ts_based_textTimeSeriesPrompt_list, #self._get_text_time_series_prompt_list(row),
            post_prompt, #self._get_post_prompt(row).strip()),
            answer.strip(),
        ).to_dict()
    
    # def _get_query_kg_edge_for_item(self, idx):
    #     # Lazy-open HDF5 file (once per worker process)
    #     if self.file is None:
    #         self.file = h5py.File(self.hdf5_path, 'r')
    #     # Directly access sample group
    #     sample_grp = self.file[f'sample_{idx}']
    #     raise NotImplementedError()
    #     KG_FOO = None
    #     return KG_FOO

    def __del__(self):
        """Safely close file when dataset is destroyed"""
        if self.file:
            self.file.close()
        pass


def get_text_time_series_hdf5_data_loader(
    hdf5_path, log_file_path, EOS_TOKEN, batch_size=32, shuffle=True, num_workers=0, patch_size=4, normalize=False, 
    time_series_sample_rt=100, include_gen_knowledge=False, condense_ts_descriptions=True, 
):
    # Setup Dataset
    dataset = TextTimeSeriesHDF5Dataset(
        hdf5_path, log_file_path, EOS_TOKEN, time_series_sample_rt=time_series_sample_rt, 
        include_gen_knowledge=include_gen_knowledge, condense_ts_descriptions=condense_ts_descriptions
    )
    # Setup DataLoader
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=lambda batch: extend_time_series_to_match_patch_size_and_aggregate(
            batch, patch_size=patch_size, normalize=normalize
        )
    )
    return data_loader, dataset
