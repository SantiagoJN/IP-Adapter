import torch
path_root = "/media/raid/santiagojn/IPAdapter/results/test_train_serrano_custom_encoder4/checkpoint-130000/"
ckpt = path_root + "pytorch_model.bin"
sd = torch.load(ckpt, map_location="cpu")
image_proj_sd = {}
ip_sd = {}
for k in sd:
    if k.startswith("unet"):
        pass
    elif k.startswith("image_proj_model"):
        image_proj_sd[k.replace("image_proj_model.", "")] = sd[k]
    elif k.startswith("adapter_modules"):
        ip_sd[k.replace("adapter_modules.", "")] = sd[k]

torch.save({"image_proj": image_proj_sd, "ip_adapter": ip_sd}, path_root + "ip_adapter.bin")