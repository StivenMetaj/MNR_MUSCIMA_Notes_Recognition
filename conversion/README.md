# MUSCIMA Notes Recognition

The aim of this project is to use Faster-RCNN to recognize notes from the CVC-MUSCIMA dataset, with MUSCIMA++ annotations.

## Steps to reproduce the experiments
+ open a terminal
### 1) Download the source code of this project
+ `git clone https://github.com/StivenMetaj/DDM_MUSCIMA_Project.git`
### 2) Prepare the dataset
+ enter in the project directory:
`cd DDM_MUSCIMA_Project`
+ create some dir:
`mkdir -p CVCMUSCIMA/MUSCIMA++`
+ download the CVCMUSCIMA_WI dataset:
`wget http://www.cvc.uab.es/cvcmuscima/CVCMUSCIMA_WI.zip`
+ extract it:
`unzip CVCMUSCIMA_WI.zip -d CVCMUSCIMA`
+ download the CVCMUSCIMA_SR dataset:
`wget http://www.cvc.uab.es/cvcmuscima/CVCMUSCIMA_SR.zip`
+ extract it:
`unzip CVCMUSCIMA_SR.zip -d CVCMUSCIMA`
+ download MUSCIMA++ annotations:
`curl --remote-name-all https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11372/LRT-2372{/MUSCIMA-pp_v1.0.zip}`
+ extract them:
`unzip MUSCIMA-pp_v1.0.zip -d CVCMUSCIMA/MUSCIMA++`
+ convert dataset into PASCAL VOC 2007 format:
`python3 convert.py`
### 3) Prepare facebookreasearch/maskrcnn-benchmark library
+ download library source code:
`git clone https://github.com/facebookresearch/maskrcnn-benchmark.git`
+ follow instructions to install the library on maskrcnn-benchmark/INSTALL.md
+ return to the root of this project
+ overwrite some library file in order to support MUSCIMA dataset:
`cp -r overwrite/maskrcnn-benchmark ./`
+ create a symbolic link to MUSCIMA dataset:
`ln -s $MNR2019 maskrcnn-benchmark/datasets/MNR2019`
### 4) Do the experiments
+ have fun (work in progress)
