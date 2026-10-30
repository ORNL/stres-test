# import torch

TOKEN_DIMENSIONALITY_BY_LLM = {
    "Llama3.2-1B": 2048, # Llama3.2-1B EMBEDDER COLS
}
TOTAL_NUM_TOKENS_RECOGNIZED_BY_LLM = {
    "Llama3.2-1B": 128259, # Llama3.2-1B EMBEDDER ROWS
}
SPECIAL_TOKENS_BY_LLM = {
    "Llama3.2-1B '<|begin_of_text|>'": 128000, #  token ID
    "Llama3.2-1B '<|end_of_text|>'": 128001, #  token ID
}
SEISMIC_CHALLENGE_CATEGORIES = [
    "population queries", "general earthquake queries", "w-phase moment tensor queries", "earthquake origin queries", 
]
PROMPT_FILTER_STR_TO_SEISMIC_CHALLENGE_CATEGORY_MAP = {
    # population related prompts 
    "was the closest major population to": "population queries", 
    " experienced shaking level of ": "population queries", 
    "occured in an area with estimated population density (per square km) of": "population queries", 
    "triggered a shaking related fatalities alert level of": "population queries", 
    "exposed the population to shaking with a rated level of": "population queries", 
    # general earthquake specifics 
    "caused shaking level of": "general earthquake queries", 
    "occured in the region referred to as the": "general earthquake queries", 
    "Earthquake was located": "general earthquake queries", 
    "originated from GPS coordinates": "general earthquake queries", 
    "occured on the date of": "general earthquake queries", 
    "occured at UTC time": "general earthquake queries", 
    # w-phase moment tensor calcs
    "has a W-phase Moment Tensor with a Magnitude element value of": "w-phase moment tensor queries", 
    "has a W-phase Moment Tensor with a Moment element value of": "w-phase moment tensor queries", 
    "has a W-phase Moment Tensor with a Depth element value of": "w-phase moment tensor queries", 
    "has a W-phase Moment Tensor with a Contributor element value of": "w-phase moment tensor queries", 
    # origin specifics
    "has an origin characterized by a Origin Time with a value of": "earthquake origin queries", 
    "has an origin characterized by a Travel Time Residual with a value of": "earthquake origin queries", 
    "has an origin characterized by a Number of Phases with a value of": "earthquake origin queries", 
    "has an origin characterized by a Azimuthal Gap with a value of": "earthquake origin queries", 
    "has an origin characterized by a Minimum Distance with a value of": "earthquake origin queries", 
    "has an origin characterized by a FERegion with a value of": "earthquake origin queries", 
    "has an origin characterized by a Magnitude Source with a value of": "earthquake origin queries", 
    "has an origin characterized by a Location Source with a value of": "earthquake origin queries", 
    "has an origin characterized by a Magnitude with a value of": "earthquake origin queries", 
    "has an origin characterized by a Depth with a value of": "earthquake origin queries", 
}

NUM_BLOCKS_BY_FOUNDATION_MODEL_NAME = {"meta-llama/Llama-3.2-3B": 28, }