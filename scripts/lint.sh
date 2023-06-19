#!/bin/bash

python3 -m black idisc
python3 -m black scripts
python3 -m black splits
python3 -m isort idisc
python3 -m isort scripts
python3 -m isort splits
