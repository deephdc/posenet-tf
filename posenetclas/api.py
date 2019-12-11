"""
API for speech to text

Date: September 2018
Author: Lara Lloret Iglesias
Email: lloret@ifca.unican.es
Github: laramaktub
"""

from collections import OrderedDict
import os
import pkg_resources
import re
import shutil

from aiohttp.web import HTTPBadRequest
import requests
import urllib.request
from webargs import fields, validate

from posenetclas import config, test_utils, utils


CONF = config.conf_dict()

# Additional parameters
allowed_extensions = set(['png', 'jpg', 'jpeg', 'PNG', 'JPG', 'JPEG'])  # allow only certain file extensions


def catch_error(f):
    def wrap(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            raise HTTPBadRequest(reason=e)
    return wrap


def catch_url_error(url_list):

    # Error catch: Empty query
    if not url_list:
        raise ValueError('Empty query')

    for i in url_list:
        if not i.startswith('data:image'):  # don't do the checks for base64 encoded images

            # Error catch: Inexistent url
            try:
                url_type = requests.head(i).headers.get('content-type')
            except Exception:
                raise ValueError("Failed url connection: "
                                 "Check you wrote the url address correctly.")

            # Error catch: Wrong formatted urls
            if url_type.split('/')[0] != 'image':
                raise ValueError("Url image format error: Some urls were not in image format. "
                                 "Check you didn't uploaded a preview of the image rather than the image itself.")


def catch_localfile_error(file_list):

    # Error catch: Empty query
    if not file_list:
        raise ValueError('Empty query')

    # Error catch: Image format error
    for f in file_list:
        extension = os.path.basename(f.content_type).split('/')[-1]
        # extension = mimetypes.guess_extension(f.content_type)
        if extension not in allowed_extensions:
            raise ValueError("Local image format error: "
                             "At least one file is not in a standard image format ({}).".format(allowed_extensions))


def warm():
    test_utils.load_predict_model()


@catch_error
def predict(**args):

    if (not any([args['urls'], args['files']]) or
            all([args['urls'], args['files']])):
        raise Exception("You must provide either 'url' or 'data' in the payload")

    if args['files']:
        args['files'] = [args['files']]  # patch until list is available
        return predict_data(args)
    elif args['urls']:
        args['urls'] = [args['urls']]  # patch until list is available
        return predict_url(args)


def predict_url(args):
    """
    Function to predict an url
    """
    catch_url_error(args['urls'])

    # Download images
    dir_path = utils.create_tmp_folder()
    filepaths = []
    for url in args['urls']:
        tmp_path = os.path.join(dir_path, os.path.basename(url))
        urllib.request.urlretrieve(url, tmp_path)
        filepaths.append(tmp_path)

    output_dir = None
    if args['accept'] in ['image/png', 'application/zip']:
        output_dir = utils.create_tmp_folder()

    try:
        outputs = test_utils.predict_images(filepaths=filepaths, output_dir=output_dir)
    finally:
        for f in filepaths:
            os.remove(f)

    return format_prediction(outputs, output_dir, content_type=args['accept'])


def predict_data(args):
    """
    Function to predict an image file
    """
    catch_localfile_error(args['files'])
    filepaths = [f.filename for f in args['files']]

    output_dir = None
    if args['accept'] in ['image/png', 'application/zip']:
        output_dir = utils.create_tmp_folder()

    try:
        outputs = test_utils.predict_images(filepaths=filepaths, output_dir=output_dir)
    finally:
        for f in filepaths:
            os.remove(f)

    return format_prediction(outputs, output_dir, content_type=args['accept'])


def format_prediction(outputs, output_dir, content_type):

    if content_type == 'image/png':
        return open(outputs[0]['img_path'], 'rb')

    elif content_type == 'application/json':
        return outputs

    elif content_type == 'application/zip':
        f = shutil.make_archive(base_name=output_dir,
                                format='zip',
                                root_dir=output_dir)
        return open(f, 'rb')


def get_predict_args():

    parser = OrderedDict()

    # Add data and url fields
    parser['files'] = fields.Field(required=False,
                                   missing=None,
                                   type="file",
                                   data_key="data",
                                   location="form",
                                   description="Select the image you want to classify.")

    # Use field.String instead of field.Url because I also want to allow uploading of base 64 encoded data strings
    parser['urls'] = fields.String(required=False,
                                   missing=None,
                                   description="Select an URL of the image you want to classify.")
    # missing action="append" --> append more than one url

    # Add format type of the response
    parser['accept'] = fields.Str(description="Media type(s) that is/are acceptable for the response.",
                                  missing='application/zip',
                                  validate=validate.OneOf(['application/zip', 'image/png', 'application/json']))

    return parser


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
