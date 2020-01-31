"""
Gather all module's test

Date: December 2019
Author: Ignacio Heredia
Email: iheredia@ifca.unican.es
Github: ignacioheredia
"""
import os
import subprocess
import time
import json
from shutil import copyfile

import magic

from posenetclas import paths


module_name = 'posenetclas'
test_url = 'https://file-examples.com/wp-content/uploads/2017/10/file_example_JPG_100kB.jpg'

# ===========
# Local Tests
# ===========


def test_load():
    print('Testing local: module load ...')
    import posenetclas.api


def test_metadata():
    print('Testing local: metadata ...')
    from posenetclas.api import get_metadata

    get_metadata()


def test_predict_url():
    print('Testing local: predict url ...')
    from posenetclas.api import predict_url

    for content_type in ['image/png', 'application/json', 'application/zip']:
        print('    Testing {}'.format(content_type))

        args = {'urls': [test_url], 'accept': content_type}
        r = predict_url(args)


def test_predict_data():
    print('Testing local: predict data ...')
    from deepaas.model.v2.wrapper import UploadedFile
    from posenetclas.api import predict_data

    for content_type in ['image/png', 'application/json', 'application/zip']:
        print('    Testing {}'.format(content_type))

        fpath = os.path.join(paths.get_base_dir(), 'data', 'samples', 'runner.jpg')
        tmp_fpath = os.path.join(paths.get_base_dir(), 'data', 'samples', 'tmp_file.jpg')
        copyfile(fpath, tmp_fpath)  # copy to tmp because we are deleting the file after prediction
        file = UploadedFile(name='data', filename=tmp_fpath, content_type='image/jpg')

        args = {'files': [file], 'accept': content_type}
        r = predict_data(args)


# ==========
# CURL Tests
# ==========


def test_curl_load():
    print('Testing curl: module load ...')

    r = subprocess.run('curl -X GET "http://0.0.0.0:5000/v2/models/" -H "accept: application/json"',
                       shell=True, check=True, stdout=subprocess.PIPE).stdout
    r = json.loads(r)
    models = [m['name'] for m in r['models']]
    if module_name not in models:
        raise Exception('Model is not correctly loaded.')


def test_curl_metadata():
    print('Testing curl: metadata ...')

    r = subprocess.run('curl -X GET "http://0.0.0.0:5000/v2/models/{}/" -H "accept: application/json"'.format(module_name),
                       shell=True, check=True, stdout=subprocess.PIPE).stdout
    if r == b'404: Not Found':
        raise Exception('Model is not correctly loaded.')
    r = json.loads(r)


def test_curl_predict_url():
    print('Testing curl: predict url ...')
    from urllib.parse import quote_plus

    for content_type in ['image/png', 'application/json', 'application/zip']:
        print('    Testing {}'.format(content_type))
        r = subprocess.run('curl -X POST "http://0.0.0.0:5000/v2/models/{}/predict/?urls={}" -H "accept: {}"'.format(module_name,
                                                                                                                     quote_plus(test_url),
                                                                                                                     content_type),
                           shell=True, check=True, stdout=subprocess.PIPE).stdout
        if r == b'404: Not Found':
            raise Exception('Model is not correctly loaded.')

    output_type = magic.from_buffer(r, mime=True)
    if output_type != content_type:
        raise Exception('Output type {} is different from expected {}'.format(output_type, content_type))


if __name__ == '__main__':
    print('Testing locally ...')
    test_load()
    test_metadata()
    test_predict_url()
    test_predict_data()

    print('Testing through CURL ...')
    r = subprocess.run('deepaas-run --listen-ip 0.0.0.0 --nowarm &', shell=True)  # launch deepaas
    time.sleep(20)  # wait for deepaas to be ready
    test_curl_load()
    test_curl_metadata()
    test_curl_predict_url()
    r = subprocess.run("kill $(ps aux | grep 'deepaas-run' | awk '{print $2}')", shell=True)   # kill deepaas
