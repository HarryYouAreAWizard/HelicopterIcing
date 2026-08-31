#!/bin/sh

# Install additional dependencies.
# TODO: Solve by using a custom Docker image.
# curl https://bootstrap.pypa.io/get-pip.py | python3
# pip3 install --no-cache matplotlib

# echo I am here:
# pwd

# python3 -m venv .venv
source .venv/bin/activate
pip install matplotlib

# echo following files exist
# ls -a

echo Running the project
# Run the first experiment...
python3 k8helloworld