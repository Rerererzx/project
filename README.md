# project
Skin Disease Classification on Jetson
This project uses a deep learning model to identify different skin diseases from images. It runs on NVIDIA Jetson devices using the Jetson Inference library.

The Algorithm
The code loads an image, runs it through a trained ONNX model, and prints out the predicted skin disease along with a confidence score.

Steps:

Load the image.

Load the ONNX model and label file.

Run the image through the model.

Print the prediction and confidence.

This works entirely on the Jetson device without needing the internet.

Running this project
Make sure your Jetson device has jetson-inference installed.

Place your model and labels in this folder:
models/skin_disease_model/
├── model.onnx
└── labels.txt

Run the script like this:
python3 final.py path/to/image.jpg

If your files are in a different location:
python3 final.py path/to/image.jpg \
  --model=your_model.onnx \
  --labels=your_labels.txt \
  --input_blob=input_0 \
  --output_blob=output_0
Required Libraries
Python 3

jetson-inference

jetson-utils

Example Output
Image is recognized as 'psoriasis' (class #2) with 92.47% confidence
Video link: https://www.youtube.com/watch?v=PoyLSNTM9BY