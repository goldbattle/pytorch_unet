
import time
import os.path
import argparse
import numpy as np
import matplotlib.pyplot as plt
from model_unet import *
from datasets import CityscapesDataset


# our CLI parser
parser = argparse.ArgumentParser()
parser.add_argument("--datadir", type=str, default="/home/patrick/datasets/cityscapes/", help="directory the Cityscapes dataset is in")
parser.add_argument("--batch_size", type=int, default=1, help="batch size")
parser.add_argument("--num_gpu", type=int, default=1, help="number of gpus")
parser.add_argument("--losstype", type=str, default="segment", help="choose between segment & reconstruction")
args = parser.parse_args()


# cityscapes dataset loading
img_data = CityscapesDataset(args.datadir, split='val', mode='fine', augment=False)
img_batch = torch.utils.data.DataLoader(img_data, batch_size=args.batch_size, shuffle=False, num_workers=4)
print(img_data)


# initiate generator and optimizer
print("creating unet model...")
generator = nn.DataParallel(UnetGenerator(3, img_data.num_classes, 64), device_ids=[i for i in range(args.num_gpu)]).cuda()


# load pretrained model if it is there
file_model = './unet.pkl'
if os.path.isfile(file_model):
    generator = torch.load(file_model)
    print("    - model restored from file....")
    print("    - filename = %s" % file_model)


# Loop through the dataset and evaluate how well the network predicts
print("\nevaluating network (will take a while)...")
history_accuracy = []
history_time = []
for idx_batch, (imagergb, label_class, labelrgb) in enumerate(img_batch):

    # send to the GPU and do a forward pass
    start_time = time.time()
    x = Variable(imagergb).cuda(0)
    y_ = Variable(label_class).cuda(0)
    y = generator.forward(x)
    end_time = time.time()

    # we "squeeze" the groundtruth if we are using cross-entropy loss
    # this is because it expects to have a [N, W, H] image where the values
    # in the 2D image correspond to the class that that pixel should be 0 < pix[u,v] < classes
    if args.losstype == "segment":
        y_ = torch.squeeze(y_)

    # max over the classes should be the prediction
    # our prediction is [N, classes, W, H]
    # so we max over the second dimension and take the max response
    # if we are doing rgb reconstruction, then just directly save it to file
    pred_class = torch.zeros((y.size()[0], y.size()[2], y.size()[3]))
    if args.losstype == "segment":
        for idx in range(0, y.size()[0]):
            pred_class[idx] = torch.argmax(y[idx], dim=0).cpu().int()
            #pred_rgb[idx] = img_data.class_to_rgb(maxindex)
    else:
        print("this test script only works for \"segment\" unet classification...")
        exit()

    # unsqueese so we have [N, 1, W, H] size
    # this allows for debug saving of the images to file...
    pred_class = pred_class.unsqueeze(1).float()
    label_class = label_class.unsqueeze(1).float()

    # now compare the groundtruth to the predicted
    # we should record the accuracy for the class
    acc_sum = (pred_class == label_class).sum()
    acc = float(acc_sum) / (label_class.size()[0]*label_class.size()[2]*label_class.size()[3])
    history_accuracy.append(acc)
    history_time.append((end_time-start_time))

    # debug saving generated classes to file
    #v_utils.save_image(pred_class.float()/img_data.num_classes, "./result/gen_image_{}_{}.png".format(0, idx_batch))
    #v_utils.save_image(label_class.float()/img_data.num_classes, "./result/label_image_{}_{}.png".format(0, idx_batch))
    #v_utils.save_image(x.cpu().data, "./result/original_image_{}_{}.png".format(0, idx_batch))


# finally output the accuracy
print("\nNETWORK RESULTS")
print("    - avg timing = %.4f (sec)" % (sum(history_time)/len(history_time)))
print("    - avg accuracy = %.4f" % (sum(history_accuracy)/len(history_accuracy)))
