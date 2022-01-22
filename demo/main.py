from flask import Flask, render_template, request, redirect, jsonify, send_from_directory

import io
from PIL import Image
import base64
import numpy as np
import cv2

import glob
import time

import tensorflow as tf
from model import Pix2Pix

app = Flask(__name__)
model_style_1 = None
model_style_2 = None

@app.route('/history')
def history():
    images = glob.glob('outputs/*.png')
    images.sort(reverse=True)
    return render_template('history.html', images=images)

@app.route('/outputs/<path:path>')
def static_file(path):
    return send_from_directory('outputs', path)

@app.route('/', methods=['GET'])
def main():
    images = glob.glob('outputs/*.png')
    images.sort(reverse=True)
    return render_template('main.html', images=images[:12])

@app.route('/', methods=['POST'])
def parse_request():
    data = request.form.to_dict()
    image = data['image']
    checkpoint = data['checkpoint']
    ts = time.time()

    # Decode base64 image
    input_image = image.split(',')[1]
    input_image = Image.open(io.BytesIO(base64.b64decode(input_image)))

    # Load image as a numpy array with the correct format:
    # RGB and values normalized between -1 and 1.
    input_image = np.array(input_image)[:, :, -1] # Keep only alpha channel
    input_image = np.where(input_image > 127, 0, 255)
    input_image = np.repeat(np.expand_dims(input_image, axis=-1), 3, axis=-1)

    # Predict image
    input_image_ = (input_image.astype(np.float32) / 127.5) - 1
    input_image_ = np.expand_dims(input_image_, axis=0)
    if checkpoint == 'Style 1':
        output_image = model_style_1(input_image_, training=True)[0].numpy()
    else: # checkpoint == 'Style 2':
        output_image = model_style_2(input_image_, training=True)[0].numpy()
    output_image = (output_image * 0.5 + 0.5) * 255.0

    export_image = np.concatenate((input_image, output_image), axis=1).astype(np.uint8)
    Image.fromarray(export_image).save('outputs/{}.png'.format(str(ts)))

    # Encode image to base64
    output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
    _, image_bytes = cv2.imencode('.png', output_image)
    output_image = b'data:image/png;base64,' + base64.b64encode(image_bytes)

    return output_image

if __name__ == '__main__':
    model_style_1 = Pix2Pix()
    model_style_1.load_weights(tf.train.latest_checkpoint('./checkpoints/style_1_hed_l1'))

    model_style_2 = Pix2Pix()
    model_style_2.load_weights(tf.train.latest_checkpoint('./checkpoints/style_2_hed_l1'))

    app.run(host='0.0.0.0')