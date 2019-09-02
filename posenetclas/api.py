"""
API for speech to text

Date: September 2018
Author: Lara Lloret Iglesias
Email: lloret@ifca.unican.es
Github: laramaktub


Descriptions:
The API will use the model files inside ../models/api. If not found it will use the model files of the last trained model.
If several checkpoints are found inside ../models/api/ckpts we will use the last checkpoint.

Warnings:
There is an issue of using Flask with Keras: https://github.com/jrosebr1/simple-keras-rest-api/issues/1
The fix done (using tf.get_default_graph()) will probably not be valid for standalone wsgi container e.g. gunicorn,
gevent, uwsgi.
"""

import json
import os
import tempfile
import warnings
from datetime import datetime
import pkg_resources
import builtins
import re
import urllib.request
import flask

import numpy as np
import requests
from werkzeug.exceptions import BadRequest
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K

from posenetclas import paths, utils, config, label_wav
from posenetclas.data_utils import load_class_names, load_class_info, mount_nextcloud
from posenetclas import image_demo

CONF = config.conf_dict()


# Mount NextCloud folders (if NextCloud is available)
try:
    mount_nextcloud('ncplants:/data/output', paths.get_base_dir())
    #mount_nextcloud('ncplants:/models', paths.get_models_dir())
except Exception as e:
    print(e)

# Empty model variables for inference (will be loaded the first time we perform inference)
loaded = False
graph, model, conf, class_names, class_info = None, None, None, None, None

# Additional parameters
allowed_extensions = set(['wav']) # allow only certain file extensions
top_K = 5  # number of top classes predictions to return




def catch_error(f):
    def wrap(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            raise e
    return wrap


def catch_url_error(url_list):

    url_list=url_list['urls']
    # Error catch: Empty query
    if not url_list:
        raise BadRequest('Empty query')

    for i in url_list:
        # Error catch: Inexistent url
        try:
            url_type = requests.head(i).headers.get('content-type')
        except:
            raise BadRequest("""Failed url connection:
            Check you wrote the url address correctly.""")
        print("tipo de imagen ------------------------------------------>    ", url_type.split('/')[0])
        # Error catch: Wrong formatted urls
        if url_type.split('/')[0] != 'image':
            raise BadRequest("""Url image format error:
            Some urls were not in image format.""")


def catch_localfile_error(file_list):

    # Error catch: Empty query
    if not file_list[0].filename:
        raise BadRequest('Empty query')

    # Error catch: Image format error
    for f in file_list:
        extension = f.split('.')[-1]
        if extension not in allowed_extensions:
            raise BadRequest("""Local image format error:
            At least one file is not in a standard image format (jpg|jpeg|png).""")


@catch_error
def predict_url(urls, merge=True):
    """
    Function to predict an url
    """

    timestamp = datetime.now().strftime('%Y-%m-%d_%H%M%S')
    timestamp_folder="/tmp/"+timestamp+"/"

    try:
        os.stat(timestamp_folder)
    except:
        os.mkdir(timestamp_folder)

    catch_url_error(urls)
    imagename=os.path.basename(urls['urls'][0])
    urllib.request.urlretrieve(urls['urls'][0], '/tmp/'+timestamp+"/"+imagename)
    
    return format_prediction(image_demo.posenet_image(timestamp))

    



@catch_error
def predict_file(filenames, merge=True):
    """
    Function to predict a local image

    """

    timestamp = datetime.now().strftime('%Y-%m-%d_%H%M%S')
    timestamp_folder="/tmp/"+timestamp+"/"

    catch_localfile_error(filenames)

    return format_prediction(image_demo.posenet_image(timestamp))



@catch_error
def predict_data(images, merge=True):
    """
    Function to predict an image file
    """
    timestamp = datetime.now().strftime('%Y-%m-%d_%H%M%S')
    timestamp_folder="/tmp/"+timestamp+"/"

    try:
        os.stat(timestamp_folder)
    except:
        os.mkdir(timestamp_folder)

    if not isinstance(images, list):
        images = [images]
    filenames = []
    for image in images:

        thename=image['files'].filename
        thefile=timestamp_folder+thename
        image['files'].save(thefile)

    # Stream the file back

    return format_prediction(image_demo.posenet_image(timestamp)), flask.send_file(filename_or_fp=thefile, as_attachment=True,  attachment_filename=os.path.basename(output_path))



def format_prediction(labels):


    for label in labels:
        d = {
            "status": "ok",
            "output": labels[0]["output"],
            "predictions": [],
        }

        for thekey in label:
            if thekey!="output":
                pred={
                thekey:label[thekey]
                }
                d["predictions"].append(pred)


    return d


def image_link(pred_lab):
    """
    Return link to Google images
    """
    base_url = 'https://www.google.es/search?'
    params = {'tbm':'isch','q':pred_lab}
    link = base_url + requests.compat.urlencode(params)
    return link


def wikipedia_link(pred_lab):
    """
    Return link to wikipedia webpage
    """
    base_url = 'https://en.wikipedia.org/wiki/'
    link = base_url + pred_lab.replace(' ', '_')
    return link


def metadata():
    d = {
        "author": None,
        "description": None,
        "url": None,
        "license": None,
        "version": None,
    }
    return d




@catch_error
def get_metadata():
    """
    Function to read metadata
    """

    module = __name__.split('.', 1)

    pkg = pkg_resources.get_distribution(module[0])
    meta = {
        'Name': None,
        'Version': None,
        'Summary': None,
        'Home-page': None,
        'Author': None,
        'Author-email': None,
        'License': None,
    }

    for line in pkg.get_metadata_lines("PKG-INFO"):
        for par in meta:
            if line.startswith(par):
                _, value = line.split(": ", 1)
                meta[par] = value

    # Update information with Docker info (provided as 'CONTAINER_*' env variables)
    r = re.compile("^CONTAINER_(.*?)$")
    container_vars = list(filter(r.match, list(os.environ)))
    for var in container_vars:
        meta[var.capitalize()] = os.getenv(var)

    return meta
