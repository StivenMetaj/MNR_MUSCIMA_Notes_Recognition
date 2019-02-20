# MUSCIMA Notes Recognition

The aim of this project is to use Faster-RCNN to recognize notes from the CVC-MUSCIMA dataset, with MUSCIMA++ annotations.

## Steps to reproduce the experiments
+ open a terminal
### 1) Download the source code of this project
+ `git clone https://github.com/StivenMetaj/DDM_MUSCIMA_Project.git`
### 2) Prepare the dataset
+ enter in the project directory:
`cd DDM_MUSCIMA_Project` (we will refer to project directory with $PROJECT_ROOT)
+ create some dir:
`mkdir -p data/CVCMUSCIMA/MUSCIMA++`
+ download the CVCMUSCIMA_WI dataset:
`wget http://www.cvc.uab.es/cvcmuscima/CVCMUSCIMA_WI.zip`
+ extract it:
`unzip CVCMUSCIMA_WI.zip -d data/CVCMUSCIMA`
+ download the CVCMUSCIMA_SR dataset:
`wget http://www.cvc.uab.es/cvcmuscima/CVCMUSCIMA_SR.zip`
+ extract it:
`unzip CVCMUSCIMA_SR.zip -d data/CVCMUSCIMA`
+ download MUSCIMA++ annotations:
`curl --remote-name-all https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11372/LRT-2372{/MUSCIMA-pp_v1.0.zip}`
+ extract them:
`unzip MUSCIMA-pp_v1.0.zip -d data/CVCMUSCIMA/MUSCIMA++`
### 3) Convert dataset into COCO format
+ convert dataset into COCO format:
`python3 conversion/convert_to_coco.py`
+ at this time, you should have the following folder structure:

`.`  
`├── conversion`  
`│   ├── convert_to_coco.py`  
`│   ├── convert_to_voc.py`  
`│   ├── split.py`  
`│   └── utils.py`  
`├── data`  
`│   ├── CVCMUSCIMA`   
`│   │   ├── CvcMuscima-Distortions`  
`│   │   ├── CVCMUSCIMA_WI`  
`│   │   └── MUSCIMA++`  
`│   └── mnr`  
`│       ├── annotations`  
`│       ├── test2019`  
`│       ├── train2019`  
`│       └── val2019`  
`├── maskrcnn-benchmark`  
`└── README.md`

### 4) Prepare facebookreasearch/maskrcnn-benchmark library
+ follow the instructions to install the library in $PROJECT_ROOT/maskrcnn-benchmark/INSTALL.md
+ return to $PROJECT_ROOT
+ create a symbolic link to MUSCIMA dataset:
`ln -s data/mnr maskrcnn-benchmark/datasets/mnr`
### 5) Do the experiments
+ have fun (work in progress) TODO finire


## Notes
Source codes of library maskrcnn-benchmark was downloaded (on 5 February 2019) and modified when necessary.
If you want to check or download the latest version of this library, you can find it here: https://github.com/facebookresearch/maskrcnn-benchmark.
If you have to use this project with a newer version of maskrcnn-benchmark, you can download it, replace the included version inside this project, install, and then redo the changes we made.

The main changes we made were to add the following files inside $PROJECT_ROOT/maskrcnn-benchmark:
+ configs/muscima/e2e_faster_rcnn_R_50_FPN_1x_muscima.yaml
+ configs/muscima/e2e_faster_rcnn_R_50_FPN_1x_muscima_pretrained_imagenet.yaml
+ demo/muscima_predictor_demo.py
+ maskrcnn_benchmark/config/paths_catalog.py
+ tools/evaluate_and_plot.py

Other minor changes we made were mainly to support the VOC format, but we recommend using the COCO format.