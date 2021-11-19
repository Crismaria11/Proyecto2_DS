import io
from torch import load
import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms
from torchvision.io import read_image
import numpy as np
from PIL import Image

import matplotlib.pyplot as plt

Tensor = torch.cuda.FloatTensor

IMG_SIZE = 256


def create_corner_rect(bb, color='red'):
    bb = np.array(bb)
    return plt.Rectangle((bb[1], bb[0]), bb[3] - bb[1], bb[2] - bb[0], color=color,
                         fill=False, lw=3)


def show_corner_bb(im, bb):
    plt.imshow(im)
    plt.gca().add_patch(create_corner_rect(bb))


class FasterRCNNDetector(torch.nn.Module):
    def __init__(self, pretrained=False, **kwargs):
        super(FasterRCNNDetector, self).__init__()
        # load pre-trained model incl. head
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=pretrained,
                                                                          pretrained_backbone=pretrained)

        # get number of input features for the classifier custom head
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features

        # replace the pre-trained head with a new one
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 4)

    def forward(self, images, targets=None):
        return self.model(images, targets)


class TorchEngine():
    def __init__(self):
        self.model = None
        self.load_model()
        self.cpu_device = None

    def load_model(self):
        print('loading torch model')
        model = FasterRCNNDetector()
        model.load_state_dict(load('./model/model-wts.pth'))
        print('model loading done')
        model.eval()
        self.model = model
        self.cpu_device = torch.device("cpu")
        print('model is now in eval mode')

    def process_img_for_prediction(self, img):
        # convert image to Tensor
        return transforms.ToTensor()(img).unsqueeze_(0)

    def predict(self, img, imgpath):
        img_tensor = self.process_img_for_prediction(img)
        if not self.model:
            raise Exception('Model is not loaded, please run setup function before attempting to predict')

        img_t = read_image(imgpath)

        with torch.no_grad():
            out_bb = self.model(img_tensor)
        bb_hat = out_bb[0]['boxes'].detach().cpu().numpy()
        bb_hat = bb_hat.astype(int)
        print(bb_hat)
        for box in bb_hat:
            show_corner_bb(img_t.squeeze(0), box)
        plt.show()

        return img, False
