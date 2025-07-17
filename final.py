#!/usr/bin/python3

import jetson_inference
import jetson_utils
import argparse

# Set up argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("filename", type=str, help="Path to the image file to classify")
parser.add_argument("--model", type=str, default="/home/nvidia/jetson-inference/python/training/classification/models/skin-disease/resnet18.onnx", help="Path to your ONNX model")
parser.add_argument("--labels", type=str, default="/home/nvidia/jetson-inference/python/training/classification/data/skin-disease/labels.txt", help="Path to class labels file")
parser.add_argument("--input_blob", type=str, default="input_0", help="Name of the input layer")
parser.add_argument("--output_blob", type=str, default="output_0", help="Name of the output layer")
opt = parser.parse_args()

# Load the image
img = jetson_utils.loadImage(opt.filename)

# Load your custom model
net = jetson_inference.imageNet(
    argv=[
        "--model=" + opt.model,
        "--labels=" + opt.labels,
        "--input_blob=" + opt.input_blob,
        "--output_blob=" + opt.output_blob
    ]
)

# Classify the image
class_idx, confidence = net.Classify(img)
class_desc = net.GetClassDesc(class_idx)

# Output results
print(f"Image is recognized as '{class_desc}' (class #{class_idx}) with {confidence * 100:.2f}% confidence")