#MUSCIMA Notes Recognition

The aim of this project is to use Faster-RCNN to recognize notes from the CVC-MUSCIMA dataset, with MUSCIMA++ annotations.

## Steps to reproduce the experiments
+ open a terminal
### 1) Download the source code of this project
+ git clone https://github.com/StivenMetaj/DDM_MUSCIMA_Project.git
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
### 3) Do the experiments
+ have fun (work in progress)
