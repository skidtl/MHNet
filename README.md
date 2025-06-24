# MHNet: Multi-scale Hierarchical Extraction Network for Small Object Detection in UAV Images

**Submission ID:** 9496 

**Authors:**

- Ziyang Xing
- Xuebin Xu
- Meiling Sun
- Kuihe Yang

**Affiliation:**

Ziyang Xing, Xuebin Xu, Kuihe Yang are with Hebei University of Science and Technology

M. Sun is with Shijiazhuang Preschool Teachers College

------

**Description of MHNet**

Unmanned aerial vehicle (UAV) image have the characteristics of small object sizes, dense distributions, and complex backgrounds. Existing detection methods perform well under normal circumstances, but perform poorly when processing UAV images. In this paper, we propose MHNet, a small object detection framework for UAV images, which solves the above problems through multi-scale feature processing and more efficient feature fusion. First, we design a multi-scale hierarchical convolution (MHC) module that extracts features at different scales, layer by layer, providing finer-grained feature information and a larger receptive field. Second, we designed the SPPFC module to capture the multi-scale features extracted by the backbone. We introduce contextual anchor attention (CAA) in the SPPFC module to bolster contextual dependency and fortify feature information across various scales, thereby augmenting the semantic information of high-level features. At the same time, this paper uses an auxiliary detection head, combined with a new feature fusion architecture to improve the prediction ability of small objects. The CAA module downsample the input features of the auxiliary detection head to enhance the feature information of the other two detection heads. This design effectively promotes the fusion of high-level and low-level information. Multiple experiments on VisDrone2019 and UAVDT have demonstrated the effectiveness of MHNet. On VisDrone2019, with mAP and mAP50 reaching 45.8% and 28.2%, respectively. Compared with the benchmark, our MHNet improves mAP and mAP50 by 6.7% and 4.8%, respectively.

------

**Dataset Processing**

Import the dataset file into the main file. The dataset file and the ` ultralytics` file are at the same level.

The dataset file format is as follows:

    dateset    
        |-images
                |--train
                |--val
                |--test
        |-labels
                |--train
                |--val
                |--test

The label format is txt. Add a dataset configuration file (yaml) under the ultralytics/cfg/datasets folder.

------

**Environment Configuration**

Use pip to install the `ultralytics` package in a [`Python>=3.8`] environment that includes [`PyTorch>=1.8`].

```
pip install ultralytics
```

------

**Important Script**

|              Script               |                         Description                          |
| :-------------------------------: | :----------------------------------------------------------: |
|             MHNet.pt              |        MHNet model weights trained using VisDrone2019        |
|             train.py              |                     Model training files                     |
|              val.py               |                       Model test file                        |
|     ultralytics/cfg/datasets      |      Configuration files for storing different datasets      |
| ultralytics/cfg/models/MHNet.yaml |            Overall architecture of the MH network            |
|      ultralytics/nn/modules/      | This folder contains various feature extraction modules used in the MHNet network. |

------

**Image Detection**

Can be used directly in the command line to detect images.

```bash
yolo predict model=MHNet.pt source='yourimage.png'
```

------

**Model Training**

```
python train.py
```

