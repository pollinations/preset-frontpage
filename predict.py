# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path
import sys
import cog
# sys.path.append(".")
# sys.path.append('./taming-transformers')
from taming.models import vqgan 
import torch
from omegaconf import OmegaConf

from ldm.util import instantiate_from_config

import argparse, os, sys, glob
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from einops import rearrange
from torchvision.utils import make_grid
import transformers
import gc
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from open_clip import tokenizer
import open_clip
import argparse
import tempfile
from pathlib import Path
import argparse
import shutil
import os
import cv2
import glob
import torch
from collections import OrderedDict
import numpy as np
from main_test_swinir import define_model, setup, get_image_pair


from predict import Predictor # swin

import os


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        clip_model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
        def load_model_from_config(config, ckpt, verbose=False):
            print(f"Loading model from {ckpt}")
            pl_sd = torch.load(ckpt, map_location="cuda:0")
            sd = pl_sd["state_dict"]
            model = instantiate_from_config(config.model)
            m, u = model.load_state_dict(sd, strict=False)
            if len(m) > 0 and verbose:
                print("missing keys:")
                print(m)
            if len(u) > 0 and verbose:
                print("unexpected keys:")
                print(u)

            model = model.half().cuda()
            model.eval()
            return model

        config = OmegaConf.load("/latent-diffusion/configs/latent-diffusion/txt2img-1p4B-eval.yaml") 
        model = load_model_from_config(config, "/content/models/ldm-model.ckpt") 

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model = model.to(device)

        # SWIN

        model_dir = 'experiments/pretrained_models'

        self.model_zoo = {
            'real_sr': {
                4: os.path.join(model_dir, '003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth')
            },
            'gray_dn': {
                15: os.path.join(model_dir, '004_grayDN_DFWB_s128w8_SwinIR-M_noise15.pth'),
                25: os.path.join(model_dir, '004_grayDN_DFWB_s128w8_SwinIR-M_noise25.pth'),
                50: os.path.join(model_dir, '004_grayDN_DFWB_s128w8_SwinIR-M_noise50.pth')
            },
            'color_dn': {
                15: os.path.join(model_dir, '005_colorDN_DFWB_s128w8_SwinIR-M_noise15.pth'),
                25: os.path.join(model_dir, '005_colorDN_DFWB_s128w8_SwinIR-M_noise25.pth'),
                50: os.path.join(model_dir, '005_colorDN_DFWB_s128w8_SwinIR-M_noise50.pth')
            },
            'jpeg_car': {
                10: os.path.join(model_dir, '006_CAR_DFWB_s126w7_SwinIR-M_jpeg10.pth'),
                20: os.path.join(model_dir, '006_CAR_DFWB_s126w7_SwinIR-M_jpeg20.pth'),
                30: os.path.join(model_dir, '006_CAR_DFWB_s126w7_SwinIR-M_jpeg30.pth'),
                40: os.path.join(model_dir, '006_CAR_DFWB_s126w7_SwinIR-M_jpeg40.pth')
            }
        }

        parser = argparse.ArgumentParser()
        parser.add_argument('--task', type=str, default='real_sr', help='classical_sr, lightweight_sr, real_sr, '
                                                                        'gray_dn, color_dn, jpeg_car')
        parser.add_argument('--scale', type=int, default=1, help='scale factor: 1, 2, 3, 4, 8')  # 1 for dn and jpeg car
        parser.add_argument('--noise', type=int, default=15, help='noise level: 15, 25, 50')
        parser.add_argument('--jpeg', type=int, default=40, help='scale factor: 10, 20, 30, 40')
        parser.add_argument('--training_patch_size', type=int, default=128, help='patch size used in training SwinIR. '
                                                                                 'Just used to differentiate two different settings in Table 2 of the paper. '
                                                                                 'Images are NOT tested patch by patch.')
        parser.add_argument('--large_model', action='store_true',
                            help='use large model, only provided for real image sr')
        parser.add_argument('--model_path', type=str,
                            default=self.model_zoo['real_sr'][4])
        parser.add_argument('--folder_lq', type=str, default=None, help='input low-quality test image folder')
        parser.add_argument('--folder_gt', type=str, default=None, help='input ground-truth test image folder')

        self.args = parser.parse_args('')

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.tasks = {
            'Real-World Image Super-Resolution': 'real_sr',
            'Grayscale Image Denoising': 'gray_dn',
            'Color Image Denoising': 'color_dn',
            'JPEG Compression Artifact Reduction': 'jpeg_car'
        }

    def upscale(self, input_dir, output_dir, task_type='Real-World Image Super-Resolution', jpeg=40, noise=15):

        self.args.task = self.tasks[task_type]
        self.args.noise = noise
        self.args.jpeg = jpeg

        # set model path
        if self.args.task == 'real_sr':
            self.args.scale = 4
            self.args.model_path = self.model_zoo[self.args.task][4]
        elif self.args.task in ['gray_dn', 'color_dn']:
            self.args.model_path = self.model_zoo[self.args.task][noise]
        else:
            self.args.model_path = self.model_zoo[self.args.task][jpeg]

        if self.args.task == 'real_sr':
            self.args.folder_lq = input_dir
        else:
            self.args.folder_gt = input_dir

        model = define_model(self.args)
        model.eval()
        model = model.to(self.device)

        # setup folder and path
        folder, save_dir, border, window_size = setup(self.args)
        os.makedirs(save_dir, exist_ok=True)

        for input_file in os.listdir(input_dir):
            out_path = input_file.replace(input_dir, output_dir)
            test_results = OrderedDict()
            test_results['psnr'] = []
            test_results['ssim'] = []
            test_results['psnr_y'] = []
            test_results['ssim_y'] = []
            test_results['psnr_b'] = []
            # psnr, ssim, psnr_y, ssim_y, psnr_b = 0, 0, 0, 0, 0

            for idx, path in enumerate(sorted(glob.glob(os.path.join(folder, '*')))):
                # read image
                imgname, img_lq, img_gt = get_image_pair(self.args, path)  # image to HWC-BGR, float32
                img_lq = np.transpose(img_lq if img_lq.shape[2] == 1 else img_lq[:, :, [2, 1, 0]],
                                      (2, 0, 1))  # HCW-BGR to CHW-RGB
                img_lq = torch.from_numpy(img_lq).float().unsqueeze(0).to(self.device)  # CHW-RGB to NCHW-RGB

                # inference
                with torch.no_grad():
                    # pad input image to be a multiple of window_size
                    _, _, h_old, w_old = img_lq.size()
                    h_pad = (h_old // window_size + 1) * window_size - h_old
                    w_pad = (w_old // window_size + 1) * window_size - w_old
                    img_lq = torch.cat([img_lq, torch.flip(img_lq, [2])], 2)[:, :, :h_old + h_pad, :]
                    img_lq = torch.cat([img_lq, torch.flip(img_lq, [3])], 3)[:, :, :, :w_old + w_pad]
                    output = model(img_lq)
                    output = output[..., :h_old * self.args.scale, :w_old * self.args.scale]

                # save image
                output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
                if output.ndim == 3:
                    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))  # CHW-RGB to HCW-BGR
                output = (output * 255.0).round().astype(np.uint8)  # float32 to uint8
                cv2.imwrite(str(out_path), output)

    def predict(
        self,
        Prompt: str = cog.Input(description="Your text prompt.", default=""),
        Steps: int = cog.Input(description="Number of steps to run the model", default=100),
        ETA: int = cog.Input(description="Can be 0 or 1", default=1),
        Samples_in_parallel: int = cog.Input(description="Batch size", default=4),
        Diversity_scale: float = cog.Input(description="As a rule of thumb, higher values of scale produce better samples at the cost of a reduced output diversity.", default=10.),
        Width: int = cog.Input(description="Width", default=256),
        Height: int = cog.Input(description="Height", default=256)
    ) -> None:
        """Run a single prediction on the model"""
        Prompts = Prompt

        Iterations = 1
        output_path = "/outputs"
        PLMS_sampling=True

        
        os.system(f"rm -rf /content/steps")
        os.makedirs("/content/steps", exist_ok=True)

        frames = []
        def save_img_callback(pred_x0, i):
            # print(pred_x0)
            frame_id = len(frames)
            x_samples_ddim = self.model.decode_first_stage(pred_x0)
            imgs = torch.clamp((x_samples_ddim+1.0)/2.0, min=0.0, max=1.0)
            grid = imgs
            #grid = rearrange(grid, 'n b c h w -> (n b) c h w')
            rows = len(imgs)
            # check if rows is quadratic and if yes take the square root
            height = int(rows**0.5)
            grid = make_grid(imgs, nrow=height)
            # to image
            grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
            step_out = os.path.join("/content/steps", f'aaa_{frame_id:04}.png')
            Image.fromarray(grid.astype(np.uint8)).save(step_out)

            if frame_id % 10 == 0:
                progress_out = os.path.join(output_path, "aaa_progress.png") 
                Image.fromarray(grid.astype(np.uint8)).save(progress_out)
            frames.append(frame_id)

        def run(opt):
            torch.cuda.empty_cache()
            gc.collect()
            if opt.plms:
                opt.ddim_eta = 0
                sampler = PLMSSampler(self.model)
            else:
                sampler = DDIMSampler(self.model)
            
            os.makedirs(opt.outdir, exist_ok=True)
            outpath = opt.outdir

            sample_path = os.path.join(outpath, "samples")
            os.makedirs(sample_path, exist_ok=True)
            base_count = len(os.listdir(sample_path))

            all_samples=list()
            samples_ddim, x_samples_ddim = None, None
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    with self.model.ema_scope():
                        uc = None
                        if opt.scale > 0:
                            uc = self.model.get_learned_conditioning(opt.n_samples * [""])
                        for prompt in opt.prompts:
                            print(prompt)
                            for n in range(opt.n_iter):
                                c = self.model.get_learned_conditioning(opt.n_samples * [prompt])
                                shape = [4, opt.H//8, opt.W//8]
                                samples_ddim, _ = sampler.sample(S=opt.ddim_steps,
                                                                conditioning=c,
                                                                batch_size=opt.n_samples,
                                                                shape=shape,
                                                                verbose=False,
                                                                img_callback=save_img_callback,
                                                                unconditional_guidance_scale=opt.scale,
                                                                unconditional_conditioning=uc,
                                                                eta=opt.ddim_eta,
                                                                x_T=samples_ddim)

                                x_samples_ddim = self.model.decode_first_stage(samples_ddim)
                                x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, min=0.0, max=1.0)
                                all_samples.append(x_samples_ddim)
                # additionally, save as grid
                grid = torch.stack(all_samples, 0)
                grid = rearrange(grid, 'n b c h w -> (n b) c h w')
                rows = opt.n_samples
                # check if rows is quadratic and if yes take the square root
                height = int(rows**0.5)
                grid = make_grid(grid, nrow=height)
                # to image
                grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
                #Image.fromarray(grid.astype(np.uint8)).save(os.path.join(outpath, f'zzz_{prompt.replace(" ", "-")}.png'))
                # save individual images
                for n,x_sample in enumerate(all_samples[0]):
                    x_sample = x_sample.squeeze()
                    x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                    prompt_filename = prompt.replace(" ", "-")
                    Image.fromarray(x_sample.astype(np.uint8)).save(os.path.join(output_path, f"{output_path}/yyy_{prompt_filename}_{n}.png"))

        os.system(f"rm {output_path}/aaa_*.png")

        args = argparse.Namespace(
            prompts = Prompts.split("->"), 
            outdir=output_path,
            ddim_steps = Steps,
            ddim_eta = ETA,
            n_iter = Iterations,
            W=Width,
            H=Height,
            n_samples=Samples_in_parallel,
            scale=Diversity_scale,
            plms=PLMS_sampling
        )
        run(args)

        # last_frame=!ls -w1 -t /content/steps/*.png | head -1
        # last_frame = last_frame[0]
        # !cp -v $last_frame /content/steps/aaa_0000.png
        # !cp -v $last_frame /content/steps/aaa_0001.png
        self.upscale("/content/steps", "/content/steps-upscaled")
        encoding_options = "-c:v libx264 -crf 20 -preset slow -vf format=yuv420p -c:a aac -movflags +faststart"
        os.system(f"ffmpeg -y -r 10 -i /content/steps-upscaled/aaa_%04d.png {encoding_options} {output_path}/zzz_output.mp4")
