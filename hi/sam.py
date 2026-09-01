



"""
Segment Anything Model from facebook
"""




def load_sam():
    from transformers import SamModel, SamProcessor

    model = SamModel.from_pretrained("facebook/sam-vit-huge", device_map="auto")
    processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")



def load_segnet():
    # from
    # https://huggingface.co/commaai/comma10k-segnet#load-trained-model
    import segmentation_models_pytorch as smp
    import albumentations as A

    hub_repo = "commaai/comma10k-segnet"
    model = smp.from_pretrained(hub_repo)
    transform = A.Compose.from_pretrained(hub_repo)


    model_init_params = {
        "encoder_name": "tu-efficientnet_b2",
        "encoder_depth": 5,
        "encoder_weights": None,
        "decoder_use_norm": "batchnorm",
        "decoder_channels": (256, 128, 64, 32, 16),
        "decoder_attention_type": None,
        "decoder_interpolation": "nearest",
        "in_channels": 3,
        "classes": 5,
        "activation": None,
        "aux_params": None
    }


    return model