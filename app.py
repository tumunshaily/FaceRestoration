
from flask import Flask,render_template,request
import argparse
import cv2
import glob
import numpy as np
import os
from basicsr.utils import imwrite

from gfpgan import GFPGANer

app = Flask(__name__)

@app.route('/',methods=['GET'])
def hello_world():
    return render_template('index.html')
@app.route('/',methods=['POST'])   
def restore():
    imageFile = request.files['imageFile']
    image_path = "./static/" + "orignal.jpg"
    imageFile.save(image_path)
    print(image_path)

    restorer = GFPGANer(
        model_path="C:/Users/tumun/Downloads/GFPGANCleanv1-NoCE-C2.pth",
        upscale=2,
        arch='clean',
        channel_multiplier=2,
        bg_upsampler=None)
    input_img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    cropped_faces, restored_faces, restored_img = restorer.enhance(
        input_img)
    output = cv2.imwrite("./static/RestoredImage.jpg",restored_img)
    return render_template('index.html')


if __name__ == '__main__':
    app.run(port=3000,debug=True)