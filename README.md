This repo is the unofficial pytorch implementation of paper "[Contour-oriented Cropland Extraction from High Resolution Remote Sensing Imagery Using Richer Convolution Features Network](https://ieeexplore.ieee.org/document/8820430)".And I add some tricks on this method.

![image-20210624203834465](20210624203834465.png)


# 1. 数据集准备

## 1.1. 数据制作

- 利用arcgis制作耕地面矢量；
- 面矢量转线矢量；
- 面矢量转栅格，线矢量转栅格；
- 耕地线膨胀两个像素后，叠加到耕地面上，其中背景的像素值为0，耕地面的像素值为1，耕地线为2。

## 1.2.  数据目录 

> 
> data
> 
>     train_images
>
>   		  ***.tif
>
> 			  ***.tif
>
> 			  ***.tif
>
>              ...........
>
> 	  train_labels
> 	
>   		  ***.tif
>
> 			  ***.tif
>
> 			  ***.tif
>
>              ...........



# 2. 模型训练

准备好数据集后， 根据需要进行配置文件的修改，依次训练分割和边缘检测模型，最后进行预测。



# 3. 项目文件说明

分割模型训练

```python
python train_seg.py
```

边缘检测模型训练

```
python train_edge.py
```

模型推理（分割+边缘+后处理）

```
python predict_large.py
```



