<div align="center">
  <h1>Tornado Plots for Silicon Module Offset Detection</h1>
  <p>Welcome to the Tornado Plots for Silicon Module Offset Detection repository! This project is designed to calculate module offsets and visualize the shifts of individual holes using tornado plots. These visualizations provide valuable insights to ensure precise wirebonding and avoid wire setting offsets.</p>
</div>

---


## Getting Started

### Prerequisites
To run these scripts, setting up a new Conda environment is required. The environment configuration is provided in the **environment.yml** file. Please ensure you are using an **Ubuntu Linux system**, as running on other operating systems may result in numerous PackageNotFound errors.

### Installation
#### 1. Clone the repository:
     git clone https://github.com/abhinavrajgupta/Tornado-Plots
#### 2. Navigate to the project directory:
     cd Tornado-Plots
#### 3. Create and activate the Conda environment:
     conda env create -f environment.yml
     conda activate ArrowPlots

### Adding Modules and Imgaes
The code in this repository is structured to specify paths for uploading images. Ensure that you place your upload module in the appropriate directory to maintain compatibility with the codebase. The path format follows this structure: 
######  current_directory/Modules/(Upload_Module_Here)
        mkdir Modules
#### Upload your Images here
##### [For Example you will upload M15 in the Modules file such that the path is structured as Modules/M15/(images and text file here)]

## How It Works
### 1. Upload Images:
  - Place your OGP-captured images in the specified directory.
### 2. Run Center Detection:
  - Execute the detection script to mark the intersection of three lines appearing as Mercedes Benz Logo.
### 3. Data Analysis:
  - Run the python script to mark actual center of modules, find offsets, angles, and create the tornado plot.

## Usage:
### 1. Run Detection Script:
    python run_detect.py
### 2. Generate Plots:
###### Before running this script, please open **ArrowPlotScript.py** and update your current directory as base directory in the first function
    python ArrowPlotScript.py

## Results:
### 1. Detection Results: Images with labels for mercedes benz intersection saved at:
    ../Modules/Module_Name/ReusltsArrowPlots
### 2. Tornado Plots: Visual representation of offset of individual holes in a silicon module:
    ../Modules/Module_Name/ReusltsArrowPlots/ArrowPlot.jpg


## Contributing
#### Contributions are welcome! Feel free to fork the repository and submit pull requests.

