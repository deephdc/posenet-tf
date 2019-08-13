#!/usr/bin/env bash
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# 
# modified by v.kozlov @2018-07-18 
# to include JupyterCONF environment check
#
######

jOPTS=""

# Check if JupyterCONF environment is specified (can be passed to docker via "--env JupyterCONF=value")
# If so, check for:
#    jupyter_config_user.py - Jupyter config file defined by user, e.g. with pre-configured password
#    jupyterSSL.key   - private key file for usage with SSL/TLS
#    jupyterSSL.pem   - the full path to an SSL/TLS certificate file
#
# Idea: local directory at host machine with those files is mounted to docker container, e.g.
#     --volume=host_dir:dir_in_container --env JupyterCONF=dir_in_container
# such that SSL connection is established and user-defined jupyter config is used

if [ -v JupyterCONF ]; then
    [[ -f $JupyterCONF/jupyter_config_user.py ]] && jConfig="$JupyterCONF/jupyter_config_user.py" && jOPTS=$jOPTS" --config=u'$jConfig'"
    [[ -f $JupyterCONF/jupyterSSL.key ]] && jKeyfile="$JupyterCONF/jupyterSSL.key" && jOPTS=$jOPTS" --keyfile=u'$jKeyfile'"
    [[ -f $JupyterCONF/jupyterSSL.pem ]] && jCertfile="$JupyterCONF/jupyterSSL.pem" && jOPTS=$jOPTS" --certfile=u'$jCertfile'"
fi

# mainly for debugging:
echo "opts: $jOPTS"

jupyter notebook $jOPTS "$@"
