#!/bin/sh

# Install additional dependencies.
# TODO: Solve by using a custom Docker image.
# curl https://bootstrap.pypa.io/get-pip.py | python3
# pip3 install --no-cache matplotlib


# install dependencies
echo Installing dependencies
pip install --quiet numpy matplotlib opencv-python-headless scikit-learn

# echo following files exist
# ls -a

echo 
echo 
echo 
echo Running the project
python3 main.py