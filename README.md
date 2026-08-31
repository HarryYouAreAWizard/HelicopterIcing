
<!-- move videos -->
example:

rsync -avzP data/Video_Test_F2787_125743_01_VIDCKPT_sec.mpg springfield:~/data



<!-- make the Dockerfile -->
Make a file and name it just Dockerfile

<!-- setup the Dockerfile -->
Config the packages needed and specify the workspace. In my example:

# use python
FROM python:3.12

# install the needed packages
RUN pip install --no-cache-dir opencv-python-headless numpy matplotlib

WORKDIR /home/noah/HelicopterIcing


<!-- build it -->
docker build -t my-image .


<!-- Fail -->
We cannot unpack the docker image at the remote, since we cannot use docker...
