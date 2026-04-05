# Biomedical-waste-detection
This repository provides code, a representative dataset subset, and full reproducibility resources (including configuration files and environment setup) to enable replication of the proposed biomedical waste classification framework.

## Reproducibility Instructions

### Environment Setup
Install dependencies:
pip install -r requirements.txt

### Dataset
The dataset used in this study is primarily self-collected and supplemented with publicly available datasets, including the Kaggle biomedical waste dataset:
https://www.kaggle.com/datasets/engineeringubu/pharmaceutical-and-biomedical-waste

A fully anonymized representative subset of the dataset is provided in the `/data` folder of this repository, including sample images and corresponding annotations in JSON format.  
The annotations contain object class labels (sharps, infectious, pharmaceutical, pathological) and bounding box coordinates [x, y, width, height].

This subset preserves the structure of the original dataset and is intended to support reproducibility and demonstration of the proposed model pipeline- (https://www.kaggle.com/datasets/engineeringubu/pharmaceutical-and-biomedical-waste).

### Training
Run:
python Training_testing.py

### Inference
Run:
python Algorithm.py

### Preprocessing
Run:
python preprocessing.py

### Reproducibility
- Fixed random seed: 42
- Same training/testing split as described in the paper

### Expected Results
Running the above pipeline should reproduce results comparable to those reported in the manuscript.

### Full Dataset Access
The complete dataset is not publicly available due to privacy and regulatory constraints.  
Access can be requested from the corresponding author under a Data Use Agreement (DUA).

### Notes
Running the provided scripts with the included dataset subset enables verification of the implementation and reproduces results consistent with the methodology described in the manuscript.