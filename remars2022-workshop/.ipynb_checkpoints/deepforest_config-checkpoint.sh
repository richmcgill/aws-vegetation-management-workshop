#!/bin/bash

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

# This script setup a conda environment that contains dependencies of
# the 'solaris' module.

# Exit when error occurs
set -e

# Open file
cp ./dataset.py ~/anaconda3/envs/tutorial_env/lib/python3.8/site-packages/deepforest/dataset.py