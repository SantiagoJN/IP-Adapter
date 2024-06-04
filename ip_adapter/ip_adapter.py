import os
from typing import List

import torch
from diffusers import StableDiffusionPipeline
from diffusers.pipelines.controlnet import MultiControlNetModel
from PIL import Image
from safetensors import safe_open
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
import torchvision.transforms as transforms 
import matplotlib.pyplot as plt
import random

from .utils import is_torch2_available, get_generator

if is_torch2_available():
    from .attention_processor import (
        AttnProcessor2_0 as AttnProcessor,
    )
    from .attention_processor import (
        CNAttnProcessor2_0 as CNAttnProcessor,
    )
    from .attention_processor import (
        IPAttnProcessor2_0 as IPAttnProcessor,
    )
else:
    from .attention_processor import AttnProcessor, CNAttnProcessor, IPAttnProcessor
from .resampler import Resampler

import datetime
import configparser
import numpy as np


class ImageProjModel(torch.nn.Module):
    """Projection Model"""

    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024, clip_extra_context_tokens=4):
        super().__init__()

        # print(f'INITIALIZING ImageProjModel with cross_attention_dim:{cross_attention_dim}, \
        #         clip_embeddings_dim:{clip_embeddings_dim} and clip_extra_context_tokens:{clip_extra_context_tokens}')
        self.generator = None
        self.cross_attention_dim = cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.proj = torch.nn.Linear(clip_embeddings_dim, self.clip_extra_context_tokens * cross_attention_dim)
        self.norm = torch.nn.LayerNorm(cross_attention_dim)

    def forward(self, image_embeds):
        # print(f'---FORWARD---')
        # print(f'Image_embeds {image_embeds.shape}: {image_embeds}')
        embeds = image_embeds
        with torch.no_grad(): # https://stackoverflow.com/questions/75517324/runtimeerror-inference-tensors-cannot-be-saved-for-backward-to-work-around-you
            clip_extra_context_tokens = self.proj(embeds).reshape(
                -1, self.clip_extra_context_tokens, self.cross_attention_dim
            )
            clip_extra_context_tokens = self.norm(clip_extra_context_tokens)
        # print(f'clip_extra_context_tokens shape {clip_extra_context_tokens.shape}')
        # exit()
        return clip_extra_context_tokens


class MLPProjModel(torch.nn.Module):
    """SD model with image prompt"""
    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024):
        super().__init__()
        
        self.proj = torch.nn.Sequential(
            torch.nn.Linear(clip_embeddings_dim, clip_embeddings_dim),
            torch.nn.GELU(),
            torch.nn.Linear(clip_embeddings_dim, cross_attention_dim),
            torch.nn.LayerNorm(cross_attention_dim)
        )
        
    def forward(self, image_embeds):
        clip_extra_context_tokens = self.proj(image_embeds)
        return clip_extra_context_tokens


class IPAdapter(torch.nn.Module): # ? Previously it didn't inherit from torch.nn.Module (and thus didn't have the self.training parameter...)
    def __init__(self, sd_pipe, image_encoder_path, ip_ckpt, device, num_tokens=4, custom_FVAE=False):
        super().__init__() # ? It didn't have this either (maybe since this was intended to work in inference?)
        self.device = device
        self.image_encoder_path = image_encoder_path
        self.ip_ckpt = ip_ckpt
        self.num_tokens = num_tokens

        self.pipe = sd_pipe.to(self.device)
        self.set_ip_adapter()

        self.custom_FVAE = custom_FVAE
        # self.adapter_modules = adapter_modules

        # Get the relevant dimensions from the config file
        config = configparser.RawConfigParser()
        config.read('/home/santiagojn/IP-Adapter/config.txt')
        configs_dict = dict(config.items('Configs'))
        relevants_text = configs_dict['relevant']
        self.relevant_dimensions = np.array([int(num.strip()) for num in relevants_text.split(',')]) # Text to array
        if self.custom_FVAE:
            print(f'[WARNING] Selecting the following dimensions as relevant.\nMake sure this is correct before using them\n{self.relevant_dimensions}')


        # load image encoder
        if not self.custom_FVAE: # The default initialization
            self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(self.image_encoder_path).to(
                self.device, dtype=torch.float16
            )
        else: # Using our custom trained FactorVAE encoder to obtain the image embeddings
            self.image_encoder = torch.hub.load('/media/raid/santiagojn/IPAdapter/disentangling-vae-master', # hub config location
                        'latent_extractor', 
                        ckpt_path=self.image_encoder_path, # encoder checkpoint
                        source='local')
            self.image_encoder.to(self.device, dtype=torch.float16)

        self.clip_image_processor = CLIPImageProcessor()
        # image proj model
        self.image_proj_model = self.init_proj()

        if self.ip_ckpt is not None:
            self.load_ip_adapter()

    def init_proj(self):
        if not self.custom_FVAE: # Default implementation
            image_proj_model = ImageProjModel(
                cross_attention_dim=self.pipe.unet.config.cross_attention_dim,
                clip_embeddings_dim=self.image_encoder.config.projection_dim,
                clip_extra_context_tokens=self.num_tokens,
            ).to(self.device, dtype=torch.float16)
        else: # CUSTOM VAE
            image_proj_model = ImageProjModel(
                cross_attention_dim=self.pipe.unet.config.cross_attention_dim,
                clip_embeddings_dim=len(self.relevant_dimensions),
                clip_extra_context_tokens=4,
            ).to(self.device, dtype=torch.float16)
        return image_proj_model

    def set_ip_adapter(self):
        unet = self.pipe.unet
        attn_procs = {}
        for name in unet.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]
            if cross_attention_dim is None:
                attn_procs[name] = AttnProcessor()
            else:
                attn_procs[name] = IPAttnProcessor(
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim,
                    scale=1.0,
                    num_tokens=self.num_tokens,
                ).to(self.device, dtype=torch.float16)
        unet.set_attn_processor(attn_procs)
        if hasattr(self.pipe, "controlnet"):
            if isinstance(self.pipe.controlnet, MultiControlNetModel):
                for controlnet in self.pipe.controlnet.nets:
                    controlnet.set_attn_processor(CNAttnProcessor(num_tokens=self.num_tokens))
            else:
                self.pipe.controlnet.set_attn_processor(CNAttnProcessor(num_tokens=self.num_tokens))
        self.adapter_modules = torch.nn.ModuleList(unet.attn_processors.values())

    def load_ip_adapter(self):
        if os.path.splitext(self.ip_ckpt)[-1] == ".safetensors":
            state_dict = {"image_proj": {}, "ip_adapter": {}}
            with safe_open(self.ip_ckpt, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if key.startswith("image_proj."):
                        state_dict["image_proj"][key.replace("image_proj.", "")] = f.get_tensor(key)
                    elif key.startswith("ip_adapter."):
                        state_dict["ip_adapter"][key.replace("ip_adapter.", "")] = f.get_tensor(key)
        else:
            state_dict = torch.load(self.ip_ckpt, map_location="cpu")
        self.image_proj_model.load_state_dict(state_dict["image_proj"])
        ip_layers = torch.nn.ModuleList(self.pipe.unet.attn_processors.values())
        ip_layers.load_state_dict(state_dict["ip_adapter"])

    #@torch.inference_mode()
    def get_image_embeds(self, pil_image=None, clip_image_embeds=None):
        if not self.custom_FVAE: # Default implementation~~
            if pil_image is not None:
                if isinstance(pil_image, Image.Image):
                    pil_image = [pil_image]
                clip_image = self.clip_image_processor(images=pil_image, return_tensors="pt").pixel_values
                clip_image_embeds = self.image_encoder(clip_image.to(self.device, dtype=torch.float16)).image_embeds
            else:
                clip_image_embeds = clip_image_embeds.to(self.device, dtype=torch.float16)
            image_embeds = clip_image_embeds # To keep it consistent with our custom implementation
            print(f'Image embeds with CLIP: \n{image_embeds}')

        else: # When we are using our custom encoder from FactorVAE
            with torch.no_grad(): # ? Don't compute gradients here (see image_encoder() in tutorial_train.py)
                # print(f'Applying custom image embedding with torch.no_grad()')
                if not torch.is_tensor(pil_image):
                    transform = transforms.Compose([transforms.ToTensor()])
                    pil_image = transform(pil_image)
                    # Adding a "null" dimension in position 0, since the encoder expects a shape (batch, channels, width, height)
                    # TODO: Aquí igual hay que comprobar una condición de si hay 4 dimensiones o no (por si le metemos un batch directamente)
                    pil_image = pil_image.unsqueeze(0) 
                image_embeds, _ = self.image_encoder(pil_image.to(self.device, dtype=torch.float16))
                
                # print(f'Image embeds: {image_embeds.shape}')
                image_embeds = image_embeds[:,self.relevant_dimensions]
                # print(f'After filtering the relevant dimensions: {image_embeds.shape}')
        
        # Applying image proj model
        image_prompt_embeds = self.image_proj_model(image_embeds)
        # print(f'\tProjected image embeds to shape {image_prompt_embeds.shape}')
        # print(f'Calling proj model with zeros')
        uncond_image_prompt_embeds = self.image_proj_model(torch.zeros_like(image_embeds))
        return image_prompt_embeds, uncond_image_prompt_embeds

    def set_scale(self, scale):
        for attn_processor in self.pipe.unet.attn_processors.values():
            if isinstance(attn_processor, IPAttnProcessor):
                attn_processor.scale = scale

    def generate(
        self,
        pil_image=None,
        clip_image_embeds=None,
        prompt=None,
        negative_prompt=None,
        scale=1.0,
        num_samples=4,
        seed=None,
        guidance_scale=7.5,
        num_inference_steps=30,
        **kwargs,
    ):
        self.set_scale(scale)

        if pil_image is not None:
            num_prompts = 1 if isinstance(pil_image, Image.Image) else len(pil_image)
        else:
            num_prompts = clip_image_embeds.size(0)

        if prompt is None:
            prompt = "best quality, high quality"
        if negative_prompt is None:
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

        if not isinstance(prompt, List):
            prompt = [prompt] * num_prompts
        if not isinstance(negative_prompt, List):
            negative_prompt = [negative_prompt] * num_prompts

        image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds(
            pil_image=pil_image, clip_image_embeds=clip_image_embeds
        )
        print(f'Default IPAdapter generate, image_prompt_embeds: {image_prompt_embeds.shape}')
        bs_embed, seq_len, _ = image_prompt_embeds.shape
        image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
        image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, num_samples, 1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)

        with torch.inference_mode():
            prompt_embeds_, negative_prompt_embeds_ = self.pipe.encode_prompt(
                prompt,
                device=self.device,
                num_images_per_prompt=num_samples,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )
            print(f'Concatenating prompt_embeds_ with shape {prompt_embeds_.shape}, and image_prompt_embeds with shape {image_prompt_embeds.shape}')
            prompt_embeds = torch.cat([prompt_embeds_, image_prompt_embeds], dim=1)
            negative_prompt_embeds = torch.cat([negative_prompt_embeds_, uncond_image_prompt_embeds], dim=1)

        generator = get_generator(seed, self.device)

        images = self.pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
            **kwargs,
        ).images

        return images


class IPAdapterTRAIN(IPAdapter):
    """Class used to handle training with multiple sources of conditioning"""

    """ 
        A function that is called at the end of each denoising steps during the inference.
        We use it to recover the intermediate noise_pred value, necessary to compute our training loss. 
    """
    """
    def callback_function(self, pipe, step, timestep, callback_kwargs):
        # print(f"callback function [{step} =? {self.stopping_step}]")
        # print(f'Timesteps: {self.timesteps}')
        # print(f'Step {step}, ts {timestep}\n\t-noise = {noise}\n\t-noise_pred = {noise_pred}\n------------------------------')
        # print('[TODO]: Once this works and prints its respective noise, save it to compute the loss.')
        # https://huggingface.co/docs/diffusers/v0.27.2/en/using-diffusers/callback#pipeline-callbacks
        #! timestep or step?
        if step == self.stopping_step: # If it is time to stop, return the current noises (all images in the batch end at the same time)
            # print(f"saving @ step {step}")
            pipe._interrupt = True
            # self.noise = callback_kwargs["noise"]
            # self.noise_pred = callback_kwargs["noise_pred"]
            # TODO: Se podría guardar el proceso de difusión (igual que al finald de https://huggingface.co/docs/diffusers/v0.27.2/en/using-diffusers/callback#pipeline-callbacks)
        
        return callback_kwargs
    """

    """
        CUSTOM Forward function which should be a mix between the other two functions
    """
    def compute_noises(self, pil_image, prompt=None, negative_prompt=None, scale=1.0, weight_dtype=torch.float32,
                num_samples=4, seed=None, num_inference_steps=30, timesteps=-1, noise_scheduler=None, vae=None, **kwargs):
        self.set_scale(scale)

        #! Which noise pair do we get?
        #*      - One at a random point in the #inference_steps
        # self.stopping_step = random.randint(0,num_inference_steps-1) # When the diffusion process reaches this point, it stops and returns its respective noises
        # selfs.timesteps = timesteps # To have access in the callback_function
        # plt.imshow((((pil_image[0]).permute(1,2,0)).detach()).cpu())
        # plt.show()

        #* It handles batches :D
        num_prompts = 1 if isinstance(pil_image, Image.Image) else len(pil_image)
        
        if prompt is None:
            prompt = "best quality, high quality"
        if negative_prompt is None:
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

        if not isinstance(prompt, List):
            prompt = [prompt] * num_prompts
        if not isinstance(negative_prompt, List):
            negative_prompt = [negative_prompt] * num_prompts


        #* 1) Get ip_tokens, encoding the _style_ images (pil_image) with our FVAE encoder (it already is applying the image_proj_model)
        image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds(pil_image)
        # ip_tokens = self.image_proj_model(image_prompt_embeds)
        # bs_embed, seq_len, _ = image_prompt_embeds.shape
        # image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
        # image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
        # uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, num_samples, 1)
        # uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)

        """ #!This code with 4 outputs from encode_prompt is used for XL models only!
        with torch.inference_mode():
            (
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            ) = self.pipe.encode_prompt( #! Ver bien la documentación de esto, y por qué parece que no lo puedo usar cuando se usa en todos los envs que he usado so far
                prompt,
                num_images_per_prompt=num_samples,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
                device=self.device
            )
            # Concat of the image prompt embeds (and their respective negative prompt)
            prompt_embeds = torch.cat([prompt_embeds, image_prompt_embeds], dim=1)
            negative_prompt_embeds = torch.cat([negative_prompt_embeds, uncond_image_prompt_embeds], dim=1)

            self.generator = get_generator(seed, self.device)
        

            #* 3) Instead of returning an image, it should return a prediction of the noise
            images = self.pipe(
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                num_inference_steps=num_inference_steps,
                generator=self.generator,
                callback_on_step_end=self.callback_function,
                callback_on_step_end_tensor_inputs=["noise_pred", "noise"], # get the inner noise_pred!
                **kwargs,
            ).images
        """
        
        """ This is now done inside the pipeline
        with torch.inference_mode(): # It should not compute the graph here (see the text_encoder call of tutorial_train.py)
            prompt_embeds_, negative_prompt_embeds_ = self.pipe.encode_prompt(
                prompt,
                device=self.device,
                num_images_per_prompt=num_samples,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt
            )
            # print(f'prompt_embeds_: {prompt_embeds_.shape}')
            # print(f'negative_prompt_embeds_: {negative_prompt_embeds_.shape}')
            # print(f'image_prompt_embeds: {image_prompt_embeds.shape}')
            
            # prompt_embeds = torch.cat([prompt_embeds_, image_prompt_embeds], dim=1)
            # negative_prompt_embeds = torch.cat([negative_prompt_embeds_, uncond_image_prompt_embeds], dim=1)
            prompt_embeds = prompt_embeds_
            negative_prompt_embeds = negative_prompt_embeds_
        """

        generator = get_generator(seed, self.device)

        #* 2) Manually generate the latents that will be used as input to the unet later. (Then, the pipe does not need the generator (?))
        # Inside the function, it will get the noisy latents as input to the unet, and here we have the ground truth noise, which we will be able
        #   to compare against the noise_pred.
        vae.to(self.device, dtype=torch.float16) # To make the type of network weights coincide with the input samples
        with torch.no_grad():
            # latents = self.pipe._encode_vae_image(image=kwargs["image"].to(self.device, dtype=weight_dtype), generator=generator, weight_dtype=weight_dtype)#.latent_dist.sample()
            latents = vae.encode(kwargs["image"].to(self.device, dtype=torch.float16)).latent_dist.sample()
            # latents = self.pipe.vae.encode(kwargs["image"].to(self.device, dtype=torch.float32)).latent_dist.sample()
            latents = latents * vae.config.scaling_factor

        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents)
        bsz = latents.shape[0] # Batch size
        # Sample a random timestep for each image
        timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device)
        timesteps = timesteps.long()

        # ! FOR DEBUGGING PURPOSES. REMOVE IT FOR ACTUAL TRAINING
        # ! FOR DEBUGGING PURPOSES. REMOVE IT FOR ACTUAL TRAINING
        # ! FOR DEBUGGING PURPOSES. REMOVE IT FOR ACTUAL TRAINING
        # ! FOR DEBUGGING PURPOSES. REMOVE IT FOR ACTUAL TRAINING
        timesteps = torch.randint(699, 700, (bsz,), device=latents.device)
        timesteps = timesteps.long()

        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
        # noisy_latents.to(dtype=torch.float16)

        
        # TODO: Pasarle a la pipeline el pil_image, y que se calcule dentro los image embeds? (no sé si esto puede tener alguna influencia para que se pueda calcular el gradiente)
        #* 3) Instead of returning an image, it should return a prediction of the noise
        noise_pred = self.pipe( # ! Pay special attention to the meaning of each parameter to this function. Make sure I'm using them all properly.
            # prompt_embeds=prompt_embeds,
            # negative_prompt_embeds=negative_prompt_embeds,
            prompt = prompt,
            negative_prompt = negative_prompt,
            ip_adapter_image_embeds = [image_prompt_embeds], # ? Pipe does not expect prompt_embeds to have both prompt and image embeds
            guidance_scale=0, # * Don't perform classifier-free guidance
            num_inference_steps=num_inference_steps,
            generator=generator,
            inference_timesteps=timesteps, # * Timestep at which we want to compute the noise_pred
            latents=noisy_latents, # * Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image generation
            strength=0.2, # ! Very careful with this value; if == 1, latents will be initialized to pure noise
            # callback_on_step_end=self.callback_function,
            # callback_on_step_end_tensor_inputs=["noise_pred", "noise"], # get the inner noise_pred!
            **kwargs,
        )

        return noise, noise_pred, timesteps[0]


class IPAdapterXL(IPAdapter):
    """SDXL"""

    def generate(
        self,
        pil_image,
        prompt=None,
        negative_prompt=None,
        scale=1.0,
        num_samples=4,
        seed=None,
        num_inference_steps=30,
        **kwargs,
    ):
        self.set_scale(scale)

        num_prompts = 1 if isinstance(pil_image, Image.Image) else len(pil_image)

        if prompt is None:
            prompt = "best quality, high quality"
        if negative_prompt is None:
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

        if not isinstance(prompt, List):
            prompt = [prompt] * num_prompts
        if not isinstance(negative_prompt, List):
            negative_prompt = [negative_prompt] * num_prompts

        image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds(pil_image)
        bs_embed, seq_len, _ = image_prompt_embeds.shape
        image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
        image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, num_samples, 1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)

        with torch.inference_mode():
            (
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            ) = self.pipe.encode_prompt(
                prompt,
                num_images_per_prompt=num_samples,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )
            prompt_embeds = torch.cat([prompt_embeds, image_prompt_embeds], dim=1)
            negative_prompt_embeds = torch.cat([negative_prompt_embeds, uncond_image_prompt_embeds], dim=1)

        self.generator = get_generator(seed, self.device)
        
        images = self.pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            num_inference_steps=num_inference_steps,
            generator=self.generator,
            **kwargs,
        ).images

        return images


class IPAdapterPlus(IPAdapter):
    """IP-Adapter with fine-grained features"""

    def init_proj(self):
        image_proj_model = Resampler(
            dim=self.pipe.unet.config.cross_attention_dim,
            depth=4,
            dim_head=64,
            heads=12,
            num_queries=self.num_tokens,
            embedding_dim=self.image_encoder.config.hidden_size,
            output_dim=self.pipe.unet.config.cross_attention_dim,
            ff_mult=4,
        ).to(self.device, dtype=torch.float16)
        return image_proj_model

    @torch.inference_mode()
    def get_image_embeds(self, pil_image=None, clip_image_embeds=None):
        if isinstance(pil_image, Image.Image):
            pil_image = [pil_image]
        clip_image = self.clip_image_processor(images=pil_image, return_tensors="pt").pixel_values
        clip_image = clip_image.to(self.device, dtype=torch.float16)
        clip_image_embeds = self.image_encoder(clip_image, output_hidden_states=True).hidden_states[-2]
        image_prompt_embeds = self.image_proj_model(clip_image_embeds)
        uncond_clip_image_embeds = self.image_encoder(
            torch.zeros_like(clip_image), output_hidden_states=True
        ).hidden_states[-2]
        uncond_image_prompt_embeds = self.image_proj_model(uncond_clip_image_embeds)
        return image_prompt_embeds, uncond_image_prompt_embeds


class IPAdapterFull(IPAdapterPlus):
    """IP-Adapter with full features"""

    def init_proj(self):
        image_proj_model = MLPProjModel(
            cross_attention_dim=self.pipe.unet.config.cross_attention_dim,
            clip_embeddings_dim=self.image_encoder.config.hidden_size,
        ).to(self.device, dtype=torch.float16)
        return image_proj_model


class IPAdapterPlusXL(IPAdapter):
    """SDXL"""

    def init_proj(self):
        image_proj_model = Resampler(
            dim=1280,
            depth=4,
            dim_head=64,
            heads=20,
            num_queries=self.num_tokens,
            embedding_dim=self.image_encoder.config.hidden_size,
            output_dim=self.pipe.unet.config.cross_attention_dim,
            ff_mult=4,
        ).to(self.device, dtype=torch.float16)
        return image_proj_model

    @torch.inference_mode()
    def get_image_embeds(self, pil_image):
        if isinstance(pil_image, Image.Image):
            pil_image = [pil_image]
        clip_image = self.clip_image_processor(images=pil_image, return_tensors="pt").pixel_values
        clip_image = clip_image.to(self.device, dtype=torch.float16)
        clip_image_embeds = self.image_encoder(clip_image, output_hidden_states=True).hidden_states[-2]
        image_prompt_embeds = self.image_proj_model(clip_image_embeds)
        uncond_clip_image_embeds = self.image_encoder(
            torch.zeros_like(clip_image), output_hidden_states=True
        ).hidden_states[-2]
        uncond_image_prompt_embeds = self.image_proj_model(uncond_clip_image_embeds)
        return image_prompt_embeds, uncond_image_prompt_embeds

    def generate(
        self,
        pil_image,
        prompt=None,
        negative_prompt=None,
        scale=1.0,
        num_samples=4,
        seed=None,
        num_inference_steps=30,
        **kwargs,
    ):
        self.set_scale(scale)

        num_prompts = 1 if isinstance(pil_image, Image.Image) else len(pil_image)

        if prompt is None:
            prompt = "best quality, high quality"
        if negative_prompt is None:
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

        if not isinstance(prompt, List):
            prompt = [prompt] * num_prompts
        if not isinstance(negative_prompt, List):
            negative_prompt = [negative_prompt] * num_prompts

        image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds(pil_image)
        bs_embed, seq_len, _ = image_prompt_embeds.shape
        image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
        image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, num_samples, 1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)

        with torch.inference_mode():
            (
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            ) = self.pipe.encode_prompt(
                prompt,
                num_images_per_prompt=num_samples,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )
            prompt_embeds = torch.cat([prompt_embeds, image_prompt_embeds], dim=1)
            negative_prompt_embeds = torch.cat([negative_prompt_embeds, uncond_image_prompt_embeds], dim=1)

        generator = get_generator(seed, self.device)

        images = self.pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            num_inference_steps=num_inference_steps,
            generator=generator,
            **kwargs,
        ).images

        return images
