# Image-level-micro-gesture-classification
Machine Vision and Digital Image Analysis

## How to run
### Google Drive + Colab
In this practical assignment, the repository was loaded into a personal Google Drive folder and the experiments were conducted on Google Colab to allow parallelization using GPU.

1. Open one notebook in the notebooks/ directory using Google Colab.
2. Mount repository from Drive using the integrated notebook cell.
3. In the '_File_' section, insert the data to be analyzed. This is done to avoid huge computational time due to loading images from Drive for every iteration and to account for limited space. File can be unzipped from the notebook by adding the following command on a notebook cell.

```console
!unzip -q /content/train.zip -d /content/train
```

4. After loading and unzipping data, the notebook can be run without any major issue.


### Visual Studio Code
To avoid loading the whole repository on Google Drive, a Visual Studio approach can be used. An example run is shown in vscode.ipynb, and it can be used for testing data easily.

1. Download repository and open root directory
2. Load train/ dataset inside the directory
3. Open vscode.ipynb and run each cell

Two approaches for data can be used:

- Loading of a zip data file in the root directory and unzip using the command line.
- Loading of data directory directly into the root directory.


## Data handling

Data structure contains 32 folders, and each of them representing an individual gesture class. Within each folder, the images are representative samples for the particular gesture class the folder belongs to.

The dataset was stored in the following structure:

```
data/
├── train/
│   ├── class_1/
│   ├── class_2/
│   |...
|   ├── class_n/ 
```

The dataset consists of multiple JPEG files in the various class folders, and each of these corresponds to representations of gestures in the image.