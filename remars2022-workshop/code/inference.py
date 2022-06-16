import json
import logging
import sys
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import boto3

import io
import pickle

s3_client = boto3.client('s3')

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

from torchvision.models import vgg16

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_modified_vgg16_unet(in_channels=3):
    """ Get a modified VGG16-Unet model with customized input channel numbers.
    """
    class Modified_VGG16Unet(VGG16Unet):
        def __init__(self):
            super().__init__(in_channels=in_channels)
    return Modified_VGG16Unet


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ConvRelu(in_channels, middle_channels),
            ConvRelu(middle_channels, out_channels),
        )

    def forward(self, x):
        return self.block(x)

class ConvRelu(nn.Module):
    def __init__(self, in_, out):
        super().__init__()
        self.conv = nn.Conv2d(in_, out, 3, padding=1)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x

class VGG16Unet(nn.Module):
    def __init__(self, in_channels=3, num_filters=32, pretrained=False):
        super().__init__()
        # Get VGG16 net as encoder
        self.encoder = vgg16(pretrained=pretrained).features
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU(inplace=True)

        # Modify encoder architecture
        self.encoder[0] = nn.Conv2d(
            in_channels, 64, kernel_size=3, stride=1, padding=1)
        self.conv1 = nn.Sequential(
            self.encoder[0], self.relu, self.encoder[2], self.relu)
        self.conv2 = nn.Sequential(
            self.encoder[5], self.relu, self.encoder[7], self.relu)
        self.conv3 = nn.Sequential(
            self.encoder[10], self.relu, self.encoder[12], self.relu,
            self.encoder[14], self.relu)
        self.conv4 = nn.Sequential(
            self.encoder[17], self.relu, self.encoder[19], self.relu,
            self.encoder[21], self.relu)
        self.conv5 = nn.Sequential(
            self.encoder[24], self.relu, self.encoder[26], self.relu,
            self.encoder[28], self.relu)

        # Build decoder
        self.center = DecoderBlock(
            512, num_filters*8*2, num_filters*8)
        self.dec5 = DecoderBlock(
            512 + num_filters*8, num_filters*8*2, num_filters*8)
        self.dec4 = DecoderBlock(
            512 + num_filters*8, num_filters*8*2, num_filters*8)
        self.dec3 = DecoderBlock(
            256 + num_filters*8, num_filters*4*2, num_filters*2)
        self.dec2 = DecoderBlock(
            128 + num_filters*2, num_filters*2*2, num_filters)
        self.dec1 = ConvRelu(64 + num_filters, num_filters)

        # Final output layer outputs logits, not probability
        self.final = nn.Conv2d(num_filters, 1, kernel_size=1)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(self.pool(conv1))
        conv3 = self.conv3(self.pool(conv2))
        conv4 = self.conv4(self.pool(conv3))
        conv5 = self.conv5(self.pool(conv4))
        center = self.center(self.pool(conv5))
        dec5 = self.dec5(torch.cat([center, conv5], 1))
        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(torch.cat([dec2, conv1], 1))
        x_out = self.final(dec1)
        return x_out
    
    
# defining model and loading weights to it.
def model_fn(model_dir):
    model = get_modified_vgg16_unet(in_channels=4)()
    with open(os.path.join(model_dir, "model.pth"), "rb") as f:
        model.load_state_dict(torch.load(f, map_location='cpu'))
    model.to(device).eval()
    return model

    
    
# data preprocessing
def input_fn(request_body, request_content_type):
    assert request_content_type == "application/json"
    
    # Get bucket name and file from the input path
    s3_path = json.loads(request_body)["inputs"]
    global path_parts
    path_parts=s3_path.replace("s3://","").split("/")
    global BUCKET_NAME
    BUCKET_NAME=path_parts.pop(0)
    BUCKET_FILE_NAME="/".join(path_parts)
    
    # Extract data from the s3 bucket
    my_array_data2 = io.BytesIO()
    s3_client.download_fileobj(BUCKET_NAME, BUCKET_FILE_NAME, my_array_data2)
    my_array_data2.seek(0)
    global idx_refs, src_im_height, src_im_width
    data, idx_refs, (src_im_height, src_im_width) = pickle.load(my_array_data2)

    data = torch.tensor(data, dtype=torch.float32, device=device)
    return data


# inference
def predict_fn(data, model):
    with torch.no_grad():
        model.eval()
        
    subarr_preds_list = []
    batch_size=1
    for batch_i in range(0, data.shape[0], batch_size):
        if batch_i + batch_size <= data.shape[0]:
            subarr_pred = model(data[
                batch_i:batch_i+batch_size, ...
            ])
        else:
            subarr_pred = model(data[
                batch_i:, ...
            ])
        subarr_preds_list.append(subarr_pred.cpu().data.numpy())
    subarr_preds = np.concatenate(subarr_preds_list, axis=0)

    return subarr_preds


# postprocess
def output_fn(subarr_preds, content_type):
    assert content_type == "application/json"

    # upload without using disk
    my_array_data = io.BytesIO()
    pickle.dump([subarr_preds,idx_refs, src_im_height, src_im_width] , my_array_data)
    my_array_data.seek(0)
    OUTPUT_FILE = "/".join(path_parts[:-1]) + '/output_pred.pkl'
    s3_client.upload_fileobj(my_array_data, BUCKET_NAME, OUTPUT_FILE)
    file_name = 's3://' + BUCKET_NAME + '/' + OUTPUT_FILE
    return json.dumps(file_name)
