# PyTorch U-Net on Cityscapes Dataset


This repository contains my first try to get a [U-Net](https://arxiv.org/abs/1505.04597) network training from the [Cityscapes dataset](https://www.cityscapes-dataset.com/).
This ended up being a bit more challenging then I expected as the data processing tools in python are not as straight forward as I expected.
Within the PyTorch framework, they provide a "dataloader" for the Cityscapes dataset, but it is not really suitable for any segmentation task.
I built off of there initial code to allow for the same random transforms to be applied to both the rgb image and labels.
Additionally, the number of classes used for training have been trimmed down and can be easily changed by updating the *mapping* data type within the dataset.py loader.


The network outputs a [N, classes, W, H] size tensor which needs to then be converted into a prediction.
To find the classification for a given pixel, the argmax of the classes responses is calculated for each and correspond to the class.
Before saving to disk, I convert this classid back into a rgb color to allow for visual comparison to the groundtruth labels.
I found that the network prediction gave ok visual results after four epochs.





## Training the Model

Please look into the `train.py` file for details on the possible arguments.
You will first need to download the cityscapes dataset and extract it.
One would normally use the loss type of "segment" if you want to do pixel-wise segmentation.
The "reconstruction" will try to just reconstruct the rgb label as the output (which is not super usefull in most cases).

```
python3 script_train.py --datadir <path_to_data> --batch_size 4 --num_gpu 1 --losstype segment
```





## Credits


* Dataloader is based off the official PyTorch vision version - [link](https://github.com/pytorch/vision/blob/ee5b4e82fe25bd4a0f0ab22ccdbcfc3de1b3b265/torchvision/datasets/cityscapes.py)
* U-Net model and original training script is from GuhoChoi Kind PyTorch tutorial - [link](https://github.com/GunhoChoi/Kind-PyTorch-Tutorial/tree/master/12_Semantic_Segmentation)



