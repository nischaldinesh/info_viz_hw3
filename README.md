## Overview
This project is a web-based visualization tool for comparing various outlier detection algorithms using heatmaps. The application generates heatmaps for datasets such as Blobs, Moons, and Circles using multiple algorithms. The web interface allows users to select a dataset and a colormap via a drop-down menu and radio buttons, respectively. A common colorbar is displayed alongside the heatmaps.

## Tech Stack
- Python 3
- Flask: Web framework used to build the web application.
- Matplotlib: Plotting library used to generate heatmaps and standalone colorbars.
- scikit-learn: Provides functions to generate datasets (Blobs, Moons, Circles) and perform various statistical operations.

## Code Explanation
- data_preparation.py: Contains functions get_blobs(), get_moons(), and get_circles() that generate synthetic datasets using scikit-learn. These datasets are scaled using StandardScaler and serve as input data for the heatmap visualizations.

- algorithms.py: Implements functions that create and fit various outlier detection models from PyOD and scikit-learn on a given dataset.

- app.py: The core of the web application.
  - Imports functions from data_preparation.py and algorithms.py
  - Contains the generate_heatmap() function that uses Matplotlib to create heatmap visualizations for a given model and dataset.
  - Contains generate_colorbar_image() to create a common colorbar image.
  - Organizes models into groups by algorithm type using get_models_grouped().
  - Reads user selections dataset and colormap via query parameters and generates the corresponding heatmaps.

- templates/index.html
  - A form for dataset selection (drop-down) and colormap selection (radio buttons).
  - The heatmaps arranged horizontally and grouped by algorithm type with a common colorbar displayed on the right.
  - Custom CSS is used to style the interface.

- requirements.txt: Lists all required Python packages to set up the project environment.
