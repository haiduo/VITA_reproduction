# VITA_reproduction
A code reproduction of VITA.

## Installation

1. see  [Mask2Former](https://github.com/facebookresearch/Mask2Former/edit/main/INSTALL.md)
2. see  [IFC](https://github.com/sukjunhwang/IFC#steps)
3. The Datasets need to be placed up the root of detertrons as follows ：[catalogue_desc](https://github.com/haiduo/VITA_reproduction/blob/main/catalogue.png)
4. You need to add  the [YouTubeVIS 2019 model weight of Mask2Former about Video Instance Segmentation](https://dl.fbaipublicfiles.com/maskformer/video_mask2former/ytvis_2019/video_maskformer2_R50_bs16_8ep/model_final_34112b.pkl) to the weights folder as for a pre-training model.

# Note
- Based on **detectron2**.
- The codes are under **projects**/ folder, which follows the convention of detectron2.
- You can easily import our project to the latest detectron2 by following below.
- inserting [VITA_1 folder](https://github.com/haiduo/VITA_reproduction/tree/main/detectron2/projects/VITA_1)

