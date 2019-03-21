
import time
import os.path
import argparse
import numpy as np
import matplotlib.pyplot as plt
from model_unet import *
from PIL import Image
import torchvision


# our CLI parser
parser = argparse.ArgumentParser()
parser.add_argument("--datadir", type=str, default="/home/patrick/datasets/CampusLoopDataset/", help="directory of images")
parser.add_argument("--batch_size", type=int, default=1, help="batch size")
parser.add_argument("--num_gpu", type=int, default=1, help="number of gpus")
parser.add_argument("--losstype", type=str, default="segment", help="choose between segment & reconstruction")
args = parser.parse_args()


# our transform that is applied to all incoming images
transform_image = torchvision.transforms.Compose([
    torchvision.transforms.Resize(size=(128, 256), interpolation=Image.BILINEAR),
    torchvision.transforms.ToTensor()
])


# load the images in the folder
img_data = torchvision.datasets.ImageFolder(root=args.datadir, transform=transform_image)
img_batch = torch.utils.data.DataLoader(img_data, batch_size=args.batch_size, shuffle=False, num_workers=4)
print(img_data)


# load pretrained model if it is there
print("loading unet model...")
file_model = './unet.pkl'
if os.path.isfile(file_model):
    generator = torch.load(file_model)
    print("    - model restored from file....")
    print("    - filename = %s" % file_model)
else:
    print("unable to load unet.pkl model file")
    exit()


# make the result directory
if not os.path.exists('./predict/'):
    os.makedirs('./predict/')


# Loop through the dataset and evaluate how well the network predicts
print("\nevaluating network (will take a while)...")
for idx_batch, (imagergb, labelrgb) in enumerate(img_batch):

    # send to the GPU and do a forward pass
    x = Variable(imagergb).cuda(0)
    y = generator.forward(x)

    # enforce that we are only doing segmentation network type
    if args.losstype != "segment":
        print("this test script only works for \"segment\" unet classification...")
        exit()

    # max over the classes should be the prediction
    # our prediction is [N, classes, W, H]
    # so we max over the second dimension and take the max response
    pred_class = torch.zeros((y.size()[0], y.size()[2], y.size()[3]))
    for idx in range(0, y.size()[0]):
        pred_class[idx] = torch.argmax(y[idx], dim=0).cpu().int()

    # unsqueese so we have [N, 1, W, H] size
    # this allows for debug saving of the images to file...
    pred_class = pred_class.unsqueeze(1).float()

    # debug saving generated classes to file
    v_utils.save_image(pred_class.float()/y.size()[1], "./predict/gen_image_{}_{}.png".format(0, idx_batch))
    v_utils.save_image(x.cpu().data, "./predict/original_image_{}_{}.png".format(0, idx_batch))
