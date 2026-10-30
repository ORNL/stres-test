import os

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

import cv2
from PIL import Image
# import pandas as pd
import xarray as xr
import datashader as ds
import datashader.transfer_functions as dstf



def plot_sensor_event_histogram(log_file_path, save_path, usgs_station_sensors_to_events_map):
    """
    Plots a histogram displaying the frequency distribution of sensors across events.
    Args:
        usgs_station_sensors_to_events_map (dict): A dictionary where keys are sensor names (str) and values are lists of event names (list of str).
    Returns:
        None
    # Example usage:
    # usgs_station_sensors_to_events_map = {
    #     'Sensor A': ['Event 1', 'Event 2', 'Event 3'],
    #     'Sensor B': ['Event 2'],
    #     'Sensor C': ['Event 4', 'Event 5']
    # }
    # plot_sensor_event_histogram(usgs_station_sensors_to_events_map)
    """
    # Extract sensor names and the count of events
    sensors = list(usgs_station_sensors_to_events_map.keys())
    event_counts = [len(events) for events in usgs_station_sensors_to_events_map.values()]
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.bar(sensors, event_counts, color='skyblue')
    plt.xlabel('Sensor Names', fontsize=12)
    plt.ylabel('Number of Events', fontsize=12)
    plt.title('Sensor Event Frequency Histogram', fontsize=14)
    plt.xticks(rotation=45, ha='right')  # Rotate sensor names for better visibility
    plt.tight_layout()  # Adjust layout to prevent clipping
    plt.savefig(save_path)#plt.show() # see https://stackoverflow.com/questions/9622163/save-plot-to-image-file-instead-of-displaying-it
    plt.close()
    pass



def visualize_large_matrix_using_datashader(
    log_file_path, save_dir, m1, m1_name, canvas_width=800, canvas_height=400, cmap=['black', 'red', 'orange', 'yellow']
):
    """ Created with help from Google AI """
    open(log_file_path, "a").write("visualize_large_matrix_using_datashader: START \n")
    m1_max = np.max(m1)
    m1_min = np.min(m1)
    # Create DataArrays
    #open(log_file_path, "a").write("visualize_large_matrix_using_datashader: creating xarray \n")
    m1_xr = xr.DataArray(
        data=m1, dims=["rows", "cols"],  # Dimensions names
        coords={"rows": range(m1.shape[0]), "cols": range(m1.shape[1])}  # Coordinates
    )
    # df1 = pd.DataFrame(m1)
    # # df2 = pd.DataFrame(m2)
    # Datashader canvas configuration for comparison
    #open(log_file_path, "a").write("visualize_large_matrix_using_datashader: creating canvas \n")
    canvas = ds.Canvas(plot_width=canvas_width, plot_height=canvas_height)
    # Rasterize the matrices
    #open(log_file_path, "a").write("visualize_large_matrix_using_datashader: creating agg1 \n")
    agg1 = canvas.raster(m1_xr)#df1)
    # agg2 = canvas.raster(df2)
    # Apply the same colormap to both matrices
    #open(log_file_path, "a").write("visualize_large_matrix_using_datashader: creating heatmap1 \n")
    heatmap1 = dstf.shade(agg1, cmap=cmap, how="linear")
    # heatmap2 = dstf.shade(agg2, cmap=cmap, how="linear")
    # Save the heatmaps
    #open(log_file_path, "a").write("visualize_large_matrix_using_datashader: saving heatmap1 \n")
    heatmap1.to_pil().save(save_dir+os.sep+m1_name+"_max"+str(m1_max)+"_min"+str(m1_min)+"_heatmap.png")
    # heatmap2.to_pil().save(save_dir+os.sep+m2_name+"_max"+str(m2_max)+"_min"+str(m2_min)+"_heatmap.png")
    open(log_file_path, "a").write("visualize_large_matrix_using_datashader: STOP \n")
    pass

def animate_row_zoom_for_large_matrix(
    log_file_path, save_dir, m1, m1_name, row_index, num_frames=30, max_zoom_factor=10000, canvas_width=800, canvas_height=400, 
    cmap=['black', 'red', 'orange', 'yellow'], fps=10 # Frames per second
):
    open(log_file_path, "a").write("animate_row_zoom_for_large_matrix: START \n")
    open(log_file_path, "a").write("animate_row_zoom_for_large_matrix: m1.shape == "+str(m1.shape)+" \n")
    m1_max = np.max(m1)
    m1_min = np.min(m1)
    # Generate frames for the zoom animation
    zoom_frames = []
    full_width = m1.shape[1]  # Total number of columns
    full_height = m1.shape[0]  # Total number of rows
    frame_canvas_width = canvas_width  # Canvas width for each frame
    frame_canvas_height = canvas_height  # Canvas height for each frame
    open(log_file_path, "a").write("animate_row_zoom_for_large_matrix: full_width == "+str(full_width)+" \n")
    open(log_file_path, "a").write("animate_row_zoom_for_large_matrix: full_height == "+str(full_height)+" \n")
    open(log_file_path, "a").write("animate_row_zoom_for_large_matrix: frame_canvas_width == "+str(frame_canvas_width)+" \n")
    open(log_file_path, "a").write("animate_row_zoom_for_large_matrix: frame_canvas_height == "+str(frame_canvas_height)+" \n")

    # Gradually zoom into the matrix, focusing on the column of interest
    for zoom_factor in np.linspace(1, max_zoom_factor, num_frames):  # Zoom levels from 1x to 10x
        open(log_file_path, "a").write("animate_row_zoom_for_large_matrix: \t zoom_factor == "+str(zoom_factor)+" \n")
        # Calculate the visible range around the column of interest
        # crop_half_width = int(full_width / (2 * zoom_factor))  # How many columns on either side
        # open(log_file_path, "a").write("animate_row_zoom_for_large_matrix: \t\t crop_half_width == "+str(crop_half_width)+" \n")
        crop_half_height = int(full_height / (2 * zoom_factor))  # Rows visible on either side
        open(log_file_path, "a").write("animate_row_zoom_for_large_matrix: \t\t crop_half_height == "+str(crop_half_height)+" \n")
        # Define the cropping range around the column
        # left_bound = max(col_index - crop_half_width, 0)
        # open(log_file_path, "a").write("animate_row_zoom_for_large_matrix: \t\t left_bound == "+str(left_bound)+" \n")
        # right_bound = min(col_index + crop_half_width, full_width)
        # open(log_file_path, "a").write("animate_row_zoom_for_large_matrix: \t\t right_bound == "+str(right_bound)+" \n")
        top_bound = max(row_index - crop_half_height, 0)#max(0, full_height // 2 - crop_half_height)  # Center rows initially
        open(log_file_path, "a").write("animate_row_zoom_for_large_matrix: \t\t top_bound == "+str(top_bound)+" \n")
        bottom_bound = min(row_index + crop_half_height, full_height)#min(full_height // 2 + crop_half_height, full_height)
        open(log_file_path, "a").write("animate_row_zoom_for_large_matrix: \t\t bottom_bound == "+str(bottom_bound)+" \n")
        # Crop the submatrix
        cropped_matrix = m1[top_bound:bottom_bound, :]#[:, left_bound:right_bound]#[top_bound:bottom_bound, left_bound:right_bound]
        open(log_file_path, "a").write("animate_row_zoom_for_large_matrix: \t\t cropped_matrix.shape == "+str(cropped_matrix.shape)+" \n")
        # Create a sub-DataFrame for Datashader
        cropped_xr = xr.DataArray(data=cropped_matrix, dims=["rows", "cols"], coords={"rows": range(cropped_matrix.shape[0]), "cols": range(cropped_matrix.shape[1])})#cropped_df = pd.DataFrame(cropped_matrix)
        # Define a new Datashader canvas for the zoomed view
        canvas_zoom = ds.Canvas(plot_width=frame_canvas_width, plot_height=frame_canvas_height)
        agg_zoom = canvas_zoom.raster(cropped_xr)#cropped_df)
        # Apply the same colormap to the zoomed view
        heatmap_zoomed = dstf.shade(agg_zoom, cmap=cmap, how="linear")
        # Convert Datashader output to a PIL image
        zoom_frames.append(heatmap_zoomed.to_pil())

    # Save the frames as an animated video
    open(log_file_path, "a").write("animate_row_zoom_for_large_matrix: Saving Animation \n")
    output_video_path = save_dir+os.sep+m1_name+"_max"+str(m1_max)+"_min"+str(m1_min)+"_zoom_into_row"+str(row_index)+"_heatmap.mp4"
    frame_width, frame_height = zoom_frames[0].size
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    # Add frames to the video
    for frame in zoom_frames:
        video_writer.write(cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))
    video_writer.release()
    # print(f"Zoom animation saved as {output_video_path}")
    open(log_file_path, "a").write("animate_row_zoom_for_large_matrix: STOP \n")
    pass



def visualize_knowledge_graph(save_path, knowledge_graph):
    """
    Visualizes a knowledge graph using matplotlib and networkx.
    Args:
        knowledge_graph (list of tuples): A list of tuples in the format
            (str(node1), str(edge), str(node2)).
    """
    # Create a directed graph object
    G = nx.DiGraph()
    # Add edges and nodes from the knowledge graph
    for node1, edge, node2 in knowledge_graph:
        G.add_edge(node1, node2, label=edge)
    # Draw the graph
    pos = nx.circular_layout(G)  # Define layout for nodes
    plt.figure(figsize=(40, 32))  # Set the figure size
    # Draw nodes and edges
    nx.draw_networkx_nodes(G, pos, node_size=1000, node_color="lightblue")
    nx.draw_networkx_edges(G, pos, edgelist=G.edges(), edge_color="gray", arrows=True)
    nx.draw_networkx_labels(G, pos, font_size=10, font_color="black")
    # Draw edge labels
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color="red", font_size=8)
    # Remove axis
    plt.axis('off')  
    plt.title('Knowledge Graph', fontsize=14)
    plt.savefig(save_path)#plt.show() # see https://stackoverflow.com/questions/9622163/save-plot-to-image-file-instead-of-displaying-it
    plt.close()
    pass

def plot_image_from_array(save_path, image_array, title="Image", xlabel="", ylabel=""):
    """
    Plot an image from a NumPy array.
    """
    plt.imshow(image_array)
    plt.title(title)
    if xlabel == "" and ylabel == "": 
        plt.axis('off')  # Turn off axes for better visualization
    else:
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
    plt.savefig(save_path)#plt.show() # see https://stackoverflow.com/questions/9622163/save-plot-to-image-file-instead-of-displaying-it
    plt.close()
    pass

def plot_time_series(save_path, data, title="Time Series Plot", xlabel="Time Step", ylabel="Value", use_legend=False, vertical_channel_offset=0.):
    """
    Plots a time series using Matplotlib.
    Parameters:
    - save_path (str): path to save image to
    - data (numpy.ndarray): Input time series data to plot.
    - title (str): Title of the plot (default: "Time Series Plot").
    - xlabel (str): Label for the x-axis (default: "Time").
    - ylabel (str): Label for the y-axis (default: "Value").
    """
    CHANNEL_DIM = 0
    TEMPORAL_DIM = 1
    assert len(data.shape) == 2
    # Create the x-axis as the time steps (ranges from 0 to len(data) - 1)
    time_steps = np.arange(data.shape[TEMPORAL_DIM])
    # Create the plot
    plt.figure(figsize=(10, 5))
    for i in range(data.shape[CHANNEL_DIM]):
        plt.plot(time_steps, vertical_channel_offset*i + data[i,:], alpha=0.3, label="Channel"+str(i))#, marker='.', linestyle='-', alpha=0.3, label="Channel"+str(i))
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if use_legend:
        plt.legend(loc='center left', bbox_to_anchor=(1,0.5))
    plt.grid(alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path)#plt.show() # see https://stackoverflow.com/questions/9622163/save-plot-to-image-file-instead-of-displaying-it
    plt.close()
    pass


def categorical_scatter_plot(save_path, categories, values, title=None, xlabel=None, ylabel=None, sublabels=None, sublabel_colormap=None, zoom_bounds=None):
    """
    (CREATED WITH HELP OF ORNL ASSISTANT)
    Generate a categorical scatter plot with categories along the x-axis and continuous
    numerical values on the y-axis. Works with lists or NumPy arrays for inputs.

    Parameters:
        categories (list or np.ndarray): A sequence indicating the category of each data point.
        values (list or np.ndarray): A sequence of corresponding numerical values.
        title (str, optional): The title of the plot. Default is None.
        xlabel (str, optional): The label for the x-axis. Default is None.
        ylabel (str, optional): The label for the y-axis. Default is None.
        sublabels (list or np.ndarray, optional): data point-level labels
        sublabel_colormap (dict, optional): maps sublabels to matplotlib color(s)
        zoom_bounds (list of int or float, optional): the lower and upper (y) boundary for the plot

    Returns:
        None
    """
    point_color = 'orangered'
    if sublabels is not None:
        if sublabel_colormap is not None:
            point_color = [sublabel_colormap[l] for l in sublabels]
        else:
            point_color = ['orangered' for _ in values]

    # Convert inputs to NumPy arrays (to make them easier to process)
    categories = np.array(categories)
    values = np.array(values)
    
    # Get unique categories and create a mapping for them along the x-axis
    unique_categories = np.unique(categories)
    category_mapping = {category: idx for idx, category in enumerate(unique_categories)}

    # Map each category value to the corresponding x-coordinate
    x_coords = np.array([category_mapping[cat] for cat in categories])

    # Add some jitter to x-coordinates to improve clarity
    jitter = np.random.uniform(-0.2, 0.2, size=x_coords.shape)  # Random jitter
    x_coords_jittered = x_coords + jitter

    # Create the plot
    plt.figure(figsize=(15, 9))
    # plt.scatter(x_coords_jittered, values, alpha=0.6, s=40, c='orangered', edgecolor='white', linewidth=0.5)
    plt.scatter(x_coords_jittered, values, alpha=0.5, s=40, c=point_color, edgecolor='white', linewidth=0.5)

    if sublabel_colormap is not None:
        legend_handles = []
        for label_key in sublabel_colormap.keys():
            legend_handles.append(
                mlines.Line2D(
                    [],[],color=sublabel_colormap[label_key], alpha=0.5, marker='o', markersize=5, linestyle=None, label=label_key
                )
            )
        plt.legend(handles=legend_handles, loc='center left', bbox_to_anchor=(1,0.5)) # see https://stackoverflow.com/questions/4700614/how-to-put-the-legend-outside-the-plot

    # Customize the plot
    plt.xticks(ticks=range(len(unique_categories)), labels=unique_categories, fontsize=12, rotation=90)
    if title:
        plt.title(title, fontsize=16)
    if xlabel:
        plt.xlabel(xlabel, fontsize=14)
    else:
        plt.xlabel("Categories", fontsize=14)
    if ylabel:
        plt.ylabel(ylabel, fontsize=14)
    else:
        plt.ylabel("Values", fontsize=14)
    if zoom_bounds is not None:
        plt.ylim(zoom_bounds[0], zoom_bounds[1])
    
    # Enhance y-axis gridlines for readability
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    # Save the plot
    plt.tight_layout()
    plt.savefig(save_path)#plt.show() # see https://stackoverflow.com/questions/9622163/save-plot-to-image-file-instead-of-displaying-it
    plt.close()
    pass
