

# # copy the code to springfield
scp -r hi/*.py springfield:~/
scp -r hi/*.sh springfield:~/

# # copy dependencies into the project folder
# scp -r src*/ springfield:~/test_get_masked_image/

# copy the .yaml to springfield
# scp test_get_masked_image.yaml springfield:/home/


frink run -f on_springfield.yaml 


# copy the figures back to my local machine

scp -r springfield:~/case_videos /home/noah/HelicopterIcing/hi/
scp -r springfield:~/figures /home/noah/HelicopterIcing/hi
# scp -r springfield:~/scan_output /home/noah/HelicopterIcing/hi

# scp -r springfield:~/segnet_output /home/noah/HelicopterIcing/hi/

