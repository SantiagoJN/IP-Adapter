import os
import random
import argparse
from pathlib import Path
import json
import itertools
import time

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from transformers import CLIPImageProcessor
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel, DDIMScheduler
# from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel
from diffusers import ControlNetModel
from custom_pipeline_controlnet_inpaint import StableDiffusionControlNetInpaintPipeline # Modified pipeline!
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection


from ip_adapter.ip_adapter import ImageProjModel, IPAdapterTRAIN
from ip_adapter.utils import is_torch2_available
if is_torch2_available():
    from ip_adapter.attention_processor import IPAttnProcessor2_0 as IPAttnProcessor, AttnProcessor2_0 as AttnProcessor
else:
    from ip_adapter.attention_processor import IPAttnProcessor, AttnProcessor

import datetime
import configparser
import numpy as np
import logging
import matplotlib.pyplot as plt

from torch.autograd import Variable
from torchviz import make_dot, make_dot_from_trace

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

# import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
custom_encoder = True

# From https://stackoverflow.com/questions/52988876/how-can-i-visualize-what-happens-during-loss-backward
def getBack(var_grad_fn):
    print(var_grad_fn)
    for n in var_grad_fn.next_functions:
        if n[0]:
            try:
                tensor = getattr(n[0], 'variable')
                print(n[0])
                print('Tensor with grad found:', tensor)
                print(' - gradient:', tensor.grad)
                print()
            except AttributeError as e:
                getBack(n[0])


# Dataset
class MyDataset(torch.utils.data.Dataset):

    def __init__(self, json_file, tokenizer, size=512, t_drop_rate=0.05, i_drop_rate=0.05, ti_drop_rate=0.05, image_root_path=""):
        super().__init__()

        self.tokenizer = tokenizer
        self.size = size
        self.i_drop_rate = i_drop_rate
        self.t_drop_rate = t_drop_rate
        self.ti_drop_rate = ti_drop_rate
        self.image_root_path = image_root_path

        self.data = json.load(open(json_file)) # list of dict: [{"image_file": "1.png", "text": "A dog"}]

        self.transform = transforms.Compose([
            transforms.Resize(self.size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(self.size),
            transforms.ToTensor(),
            transforms.Normalize([0], [1]), #! Make sure to Properly normalize the images
        ])
        self.clip_image_processor = CLIPImageProcessor()
        
    # TODO: Habrá que devolver también la imagen de "estilo"
    def __getitem__(self, idx):
        #* Lo único que queremos usar para condicionar nuestro adapter es la imagen renderizada. Luego esa imagen
        #*  se le debería pasar a nuestro FVAE encoder, y el IP-Adapter se encarga de aprender la relación
        item = self.data[idx] 
        text = item["text"]
        
        pil_image_path = item["pil_image"]
        pil_image = self.transform(Image.open(os.path.join(self.image_root_path, pil_image_path)).convert("RGB"))
        image_path = item["image"]
        image = self.transform(Image.open(os.path.join(self.image_root_path, image_path)).convert("RGB"))
        control_image_path = item["control_image"]
        control_image = self.transform(Image.open(os.path.join(self.image_root_path, control_image_path)).convert("RGB"))
        mask_image_path = item["mask_image"]
        mask_image = self.transform(Image.open(os.path.join(self.image_root_path, mask_image_path)).convert("RGB"))
        
        pil_mod = (pil_image * 0.5) + 0.5 # ! Keep an eye on this
        clip_image = self.clip_image_processor(images=pil_mod, return_tensors="pt", do_rescale=False).pixel_values

        # read mask
        # print("[TODO] Once everything goes fine, modify which mask is loaded for each getitem (data, json...)")
        # raw_masked = Image.open("datasets/test_custom_pipeline/data")
        # masked_image = self.transform(raw_masked.convert("RGB"))
        # raw_mask = Image.open("datasets/test_custom_pipeline/data")
        # mask = self.transform(raw_mask.convert("RGB"))
        # raw_depth = Image.open("datasets/test_custom_pipeline/data")
        # depth = self.transform(raw_depth.convert("RGB"))

        # drop
        drop_image_embed = 0
        rand_num = random.random()
        if rand_num < self.i_drop_rate:
            drop_image_embed = 1
        elif rand_num < (self.i_drop_rate + self.t_drop_rate):
            text = ""
        elif rand_num < (self.i_drop_rate + self.t_drop_rate + self.ti_drop_rate):
            text = ""
            drop_image_embed = 1
        # get text and tokenize
        text_input_ids = self.tokenizer(
            text,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids
        
        return {
            "pil_image": pil_image,
            "image": image,
            "control_image": control_image,
            "mask_image": mask_image,
            "text_input_ids": text_input_ids,
            "clip_image": clip_image,
            "drop_image_embed": drop_image_embed, 
            # "masked_image": masked_image,
            # "mask": mask,
            # "depth": depth
        }

    def __len__(self):
        return len(self.data)
    

# Arrange samples in batches
def collate_fn(data):
    pil_images = torch.stack([example["pil_image"] for example in data])
    images = torch.stack([example["image"] for example in data])
    control_images = torch.stack([example["control_image"] for example in data])
    mask_images = torch.stack([example["mask_image"] for example in data])
    text_input_ids = torch.cat([example["text_input_ids"] for example in data], dim=0)
    clip_images = torch.cat([example["clip_image"] for example in data], dim=0)
    drop_image_embeds = [example["drop_image_embed"] for example in data]
    # masked_images = torch.stack([example["masked_image"] for example in data])
    # masks = torch.stack([example["mask"] for example in data])
    # depths = torch.stack([example["depth"] for example in data])
    
    # print(f'¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬clip_images:{clip_images.shape}')
    return {
        "pil_images": pil_images,
        "images": images,
        "control_images": control_images,
        "mask_images": mask_images,
        "text_input_ids": text_input_ids,
        "clip_images": clip_images,
        "drop_image_embeds": drop_image_embeds,
        # "masked_images": masked_images,
        # "masks": masks,
        # "depths": depths
    }
    

class IPAdapter(torch.nn.Module):
    """IP-Adapter"""
    def __init__(self, unet, image_proj_model, adapter_modules, ckpt_path=None):
        super().__init__()
        self.unet = unet
        self.image_proj_model = image_proj_model
        self.adapter_modules = adapter_modules

        if ckpt_path is not None:
            self.load_from_checkpoint(ckpt_path)
    

    #* encoder_hidden_states -> text embedding
    #* image_embeds -> image embedding
    def forward(self, noisy_latents, timesteps, encoder_hidden_states, image_embeds):
        ip_tokens = self.image_proj_model(image_embeds) # ! FVAE Encoder here
        print(f'encoder_hidden_states before: {encoder_hidden_states.shape}')
        encoder_hidden_states = torch.cat([encoder_hidden_states, ip_tokens], dim=1)
        print(f'IP_tokens: {ip_tokens.shape}')
        print(f'encoder_hidden_states after: {encoder_hidden_states.shape}')
        exit()
        #* Parece que aquí lo que hace es coger las (dos) fuentes de info externas que tiene el U-NET, y las concatena
        # Predict the noise residual
        noise_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample
        # print(f'Ip tokens: {ip_tokens.shape}')
        # print(f'Noisy latents: {noisy_latents.shape}')
        # print(f'Noise pred shape after UNET: {noise_pred.shape}')
        
        return noise_pred

    def load_from_checkpoint(self, ckpt_path: str):
        # Calculate original checksums
        orig_ip_proj_sum = torch.sum(torch.stack([torch.sum(p) for p in self.image_proj_model.parameters()]))
        orig_adapter_sum = torch.sum(torch.stack([torch.sum(p) for p in self.adapter_modules.parameters()]))

        state_dict = torch.load(ckpt_path, map_location="cpu")

        # Load state dict for image_proj_model and adapter_modules
        self.image_proj_model.load_state_dict(state_dict["image_proj"], strict=True)
        self.adapter_modules.load_state_dict(state_dict["ip_adapter"], strict=True)

        # Calculate new checksums
        new_ip_proj_sum = torch.sum(torch.stack([torch.sum(p) for p in self.image_proj_model.parameters()]))
        new_adapter_sum = torch.sum(torch.stack([torch.sum(p) for p in self.adapter_modules.parameters()]))

        # Verify if the weights have changed
        assert orig_ip_proj_sum != new_ip_proj_sum, "Weights of image_proj_model did not change!"
        assert orig_adapter_sum != new_adapter_sum, "Weights of adapter_modules did not change!"

        print(f"Successfully loaded weights from checkpoint {ckpt_path}")

    
    
def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_ip_adapter_path",
        type=str,
        default=None,
        help="Path to pretrained ip adapter model. If not specified weights are initialized randomly.",
    )
    parser.add_argument(
        "--data_json_file",
        type=str,
        default=None,
        required=True,
        help="Training data",
    )
    parser.add_argument(
        "--data_root_path",
        type=str,
        default="",
        required=True,
        help="Training data root path",
    )
    parser.add_argument(
        "--image_encoder_path",
        type=str,
        default=None,
        required=True,              # ! <<<<<<<<
        help="Path to CLIP image encoder",  # ! This should be changed ! 
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-ip_adapter",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images"
        ),
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--train_batch_size", type=int, default=8, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=2000,
        help=(
            "Save a checkpoint of the training state every X updates"
        ),
    )
    parser.add_argument(
        "--save_epochs",
        type=int,
        default=50,
        help=(
            "Save a checkpoint of the training state every X epochs"
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args
    

def main():
    args = parse_args()

    logging_dir = Path(args.output_dir, args.logging_dir)
    # Get the relevant dimensions from the config file
    config = configparser.RawConfigParser()
    config.read('/home/santiagojn/IP-Adapter/config.txt')
    configs_dict = dict(config.items('Configs'))
    relevants_text = configs_dict['relevant'] # Raw text that contains the relevant dimensions' identifiers
    relevant_dimensions = np.array([int(num.strip()) for num in relevants_text.split(',')]) # Text to array

    now = datetime.now() 
    timestamp = now.strftime("%Y-%d-%m, %H.%M.%S")
    writer_dir = f"runs/{timestamp}"
    writer = SummaryWriter(writer_dir)

    # relevant_dimensions = [3, 6, 7, 11, 13, 17, 19]
    # print(f'[WARNING] Selecting the following dimensions as relevant.\nMake sure this is correct before training\n{relevant_dimensions}')

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )
    
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    
    # Save args into an external json file
    path_to_metadata = f"{args.output_dir}/config.json"
    with open(path_to_metadata, 'w') as f:
        json.dump(vars(args), f, indent=4, sort_keys=True) # Save hyperparameters in case we want to check them later
    
    logging.basicConfig(filename=f'{args.output_dir}/run.log', filemode='w', format='%(asctime)s - %(message)s', level=logging.INFO)
    logging.info('Start training...')
    logging.info(f'Using image encoder in path {args.image_encoder_path}')

    # Load scheduler, tokenizer and models.
    """ # In our custom implementation, we don't need to have the components separated
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
    """
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler") # load this for sampling the noise inside ip_adapter
    # load controlnet
    # controlnet_model_path = "/media/raid/santiagojn/IPAdapter/control_v11f1p_sd15_depth"
    controlnet_model_path = "/media/raid/santiagojn/IPAdapter/controlnet-depth-sdxl-1.0"
    controlnet = ControlNetModel.from_pretrained(controlnet_model_path, torch_dtype=torch.float16)

    # noise scheduler
    noise_scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
        steps_offset=1,
    )

    # vae
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae").to(dtype=torch.float16)

    # load SD pipeline
    pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        controlnet=controlnet,
        torch_dtype=torch.float16,
        scheduler=noise_scheduler,
        vae=vae,
        feature_extractor=None,
        safety_checker=None
    )

    if custom_encoder:
        image_encoder = torch.hub.load('/media/raid/santiagojn/IPAdapter/disentangling-vae-master', # hub config location
                        'latent_extractor', 
                        ckpt_path=args.image_encoder_path, # encoder checkpoint
                        source='local')
        emb_dim = len(relevant_dimensions)
    else:
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(args.image_encoder_path) # ! FVAE Encoder
        emb_dim = image_encoder.config.projection_dim

    # image_encoder.eval().cuda()
    # freeze parameters of models to save more memory
    
    # ! Very careful with what we set to require grad or not
    pipe.unet.requires_grad_(False)
    vae.requires_grad_(False)
    # text_encoder.requires_grad_(False) #?
    
    image_encoder.requires_grad_(False)
    
    #ip-adapter
    image_proj_model = ImageProjModel(
        cross_attention_dim=pipe.unet.config.cross_attention_dim,
        # clip_embeddings_dim=image_encoder.config.projection_dim, # ! <<<<<<<<<<<<<< 20D?
        clip_embeddings_dim=emb_dim,
        clip_extra_context_tokens=4,
    )
    # init adapter modules 
    """ --->done in set_ip_adapter
    attn_procs = {}
    unet_sd = pipe.unet.state_dict()
    for name in pipe.unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else pipe.unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = pipe.unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(pipe.unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = pipe.unet.config.block_out_channels[block_id]
        if cross_attention_dim is None:
            attn_procs[name] = AttnProcessor()
        else:
            layer_name = name.split(".processor")[0]
            weights = {
                "to_k_ip.weight": unet_sd[layer_name + ".to_k.weight"],
                "to_v_ip.weight": unet_sd[layer_name + ".to_v.weight"],
            }
            attn_procs[name] = IPAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim)
            attn_procs[name].load_state_dict(weights)
    pipe.unet.set_attn_processor(attn_procs)
    """
    # adapter_modules = torch.nn.ModuleList(pipe.unet.attn_processors.values())

    # from ZeST: ip_model = IPAdapterXL(pipe, image_encoder_path, ip_ckpt, device)
    # unet -> pipe (luego ya nos encargamos de sacar el noise)
    # image_proj_model -> alternative implementation with image_encoder_path
    # adapter_modules lo he metido nuevo en la clase padre IPAdapter, porque lo quiere luego el optimizer -> lo he quitado para que pueda pillar el unet bien
    #ip_adapter = IPAdapter(unet, image_proj_model, adapter_modules, args.pretrained_ip_adapter_path)
    ip_adapter = IPAdapterTRAIN(pipe, args.image_encoder_path, args.pretrained_ip_adapter_path, 
                                accelerator.device, custom_FVAE=custom_encoder)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    #unet.to(accelerator.device, dtype=weight_dtype)
    
    # vae.to(accelerator.device, dtype=weight_dtype)
    # text_encoder.to(accelerator.device, dtype=weight_dtype)
    image_encoder.to(accelerator.device, dtype=weight_dtype)
    
    adapter_modules = torch.nn.ModuleList(ip_adapter.pipe.unet.attn_processors.values())
    # pipe_controlnet = ip_adapter.pipe.controlnet # ? We should also account for the controlnet module, no? --> handled in the attention processors? --> no, its condition is already handled in the input of the unet :)
    
    # optimizer
    params_to_opt = itertools.chain(ip_adapter.image_proj_model.parameters(),  adapter_modules.parameters())#, pipe_controlnet.parameters()) # ! This should be properly set; otherwise, backpropagation doesn't work
    optimizer = torch.optim.AdamW(params_to_opt, lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # dataloader
    train_dataset = MyDataset(args.data_json_file, tokenizer=pipe.tokenizer, size=args.resolution, image_root_path=args.data_root_path)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )
    
    # Prepare everything with our `accelerator`.
    ip_adapter, optimizer, train_dataloader = accelerator.prepare(ip_adapter, optimizer, train_dataloader)
    print(f'[INFO] Training during {len(train_dataloader)} batches of {args.train_batch_size} size; around {len(train_dataloader)*args.train_batch_size} samples.')
    
    global_step = 0
    for epoch in range(1, args.num_train_epochs+1):
        begin = time.perf_counter()
        for step, batch in enumerate(train_dataloader):
            load_data_time = time.perf_counter() - begin
            with accelerator.accumulate(ip_adapter):
                

                """ Se supone que todo esto ya lo hace el pipeline solo...
                    # Convert images to latent space
                    with torch.no_grad():
                        latents = vae.encode(batch["images"].to(accelerator.device, dtype=weight_dtype)).latent_dist.sample()
                        latents = latents * vae.config.scaling_factor

                    # Sample noise that we'll add to the latents
                    noise = torch.randn_like(latents)
                    bsz = latents.shape[0] # Batch size
                    # Sample a random timestep for each image
                    timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device)
                    timesteps = timesteps.long()

                    # Add noise to the latents according to the noise magnitude at each timestep
                    # (this is the forward diffusion process)
                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                
                    with torch.no_grad():
                        if custom_encoder:
                            image_embeds, _ = image_encoder(batch["images"].to(accelerator.device, dtype=weight_dtype)) # get the returned *mean*
                            image_embeds = image_embeds[:,relevant_dimensions] # Removing _irrelevant_ dimensions
                        else:
                            image_embeds = image_encoder(batch["clip_images"].to(accelerator.device, dtype=weight_dtype)).image_embeds # ! Ver qué saca aquí
                        # print(f'~~~~~~~~~~~~~~~~~~~~~~~~~~~{(batch["clip_images"]).shape}')
                        
                        # print(f'IMAGE EMBEDS({image_embeds.shape}): {image_embeds}')
                        # exit()
                    image_embeds_ = []
                    for image_embed, drop_image_embed in zip(image_embeds, batch["drop_image_embeds"]):
                        if drop_image_embed == 1:
                            image_embeds_.append(torch.zeros_like(image_embed))
                        else:
                            image_embeds_.append(image_embed)
                    image_embeds = torch.stack(image_embeds_)
                
                    with torch.no_grad():
                        encoder_hidden_states = text_encoder(batch["text_input_ids"].to(accelerator.device))[0]
                    
                    # print(f'BS: {args.train_batch_size}')
                    # print(f'Noise: {noise}')
                    noise_pred = ip_adapter(noisy_latents, timesteps, encoder_hidden_states, image_embeds)
                """
                
                bsz = batch["images"].shape[0]
                # print(f'@@@@@@BATCH SIZE: {bsz}')
                # Sample a random timestep for each image
                # timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=accelerator.device)
                # timesteps = timesteps.long()

                
                
                #! Make sure that images look like they should before training a model
                # plt.imshow((((batch["control_images"][0]).permute(1,2,0)).detach()).cpu())
                # plt.show()
                # plt.imshow((((batch["pil_images"][0]).permute(1,2,0)).detach()).cpu())
                # plt.show()
                # plt.imshow((((batch["images"][0]).permute(1,2,0)).detach()).cpu())
                # plt.show()
                # plt.imshow((((batch["mask_images"][0]).permute(1,2,0)).detach()).cpu())
                # plt.show()
                # exit()

                # Here we call the forward, hoping that it returns a noise and its estimation
                noise, noise_pred, timestep = ip_adapter.compute_noises(
                        pil_image=batch["pil_images"], 
                        image=batch["images"], # " image batch to be used as the starting point "
                        control_image=batch["control_images"],  # " The ControlNet input condition to provide guidance to the `unet` for generation "
                        mask_image=batch["mask_images"],  # " image batch to mask `image` "
                        seed=42, 
                        num_samples=1, # we send just 1 batch of n samples (otherwise it replicates internal variables..) # ! Change when it works
                        num_inference_steps=30, 
                        controlnet_conditioning_scale=0.99, 
                        # timesteps=timesteps,
                        noise_scheduler=noise_scheduler,
                        weight_dtype=weight_dtype,
                        vae=vae)

                # print(f'Returned from compute_noises with requires grads: noise - {noise.requires_grad} and noise_pred - {noise_pred.requires_grad}')
                # print(f'Noise shape: {noise.shape}') # [16, 4, 32, 32] => [bs, unet_shape, (latent_shape)]
                # print(f'################## Calling make_dot() function')
                # make_dot(noise_pred, params=dict(ip_adapter.pipe.unet.named_parameters()), show_attrs=True, show_saved=True)

                #? "computes the average squared difference between pixel values in the generated image and the ground truth image"
                # Computes the noise reconstructed at certain "step" (defined at the beginning of the loop)
                loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
                # print(f'Loss with type {loss.type}, requires grad: {loss.requires_grad}')
                loss = Variable(loss, requires_grad = True) # otherwise we get nan's after the first iteration (?)
                # print(f'Loss changed type to {loss.type}')
                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean().item()
                
                # Backpropagate
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()


                if accelerator.is_main_process:
                    ct = datetime.now()
                    log_msg = "[{}] Epoch {}, step {}, timestep {}, data_time: {:.4f}, time: {:.4f}, step_loss: {}".format(
                        ct, epoch, step, timestep, load_data_time, time.perf_counter() - begin, avg_loss)
                    logging.info(log_msg)
                    print(log_msg)
            
            global_step += 1
            
            begin = time.perf_counter()
        
        # Profiling
        writer.add_scalar('loss', avg_loss, epoch)

        if epoch % args.save_epochs == 0 and epoch != 0:
            save_path = os.path.join(args.output_dir, f"checkpoint-{epoch}")
            print(f'Saving in path {save_path}')
            accelerator.save_state(save_path, safe_serialization=False) # ! https://github.com/tencent-ailab/IP-Adapter/issues/263
    
    writer.close() # Make sure it is closed properly
            
                
if __name__ == "__main__":
    main()    
