

# copy the code to springfield
scp -r test_get_masked_image/ springfield:~/

# copy dependencies into the project folder
scp -r src*/ springfield:~/test_get_masked_image/

# copy the .yaml to springfield
# scp test_get_masked_image.yaml springfield:/home/


frink run -f test_get_masked_image.yaml 


# copy the figures back to my local machine
scp -r springfield:~/figures /home/noah/HelicopterIcing
