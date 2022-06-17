# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import argparse
import gc
import glob
import os
import shutil
import sys
import tempfile
from collections import OrderedDict
from pathlib import Path

import cog
import cv2
import numpy as np
import open_clip
import torch
import transformers
from cog import BasePredictor, Input, Path
from einops import rearrange
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.util import instantiate_from_config
from omegaconf import OmegaConf
from open_clip import tokenizer
from PIL import Image

# sys.path.append(".")
# sys.path.append('./taming-transformers')
from taming.models import vqgan
from torchvision.utils import make_grid
from tqdm import tqdm, trange

from main_test_swinir import define_model, get_image_pair, setup


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        clip_model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai"
        )

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

        config = OmegaConf.load(
            "/latent-diffusion/configs/latent-diffusion/txt2img-1p4B-eval.yaml"
        )
        model = load_model_from_config(config, "/content/models/ldm-model.ckpt")

        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.model = model.to(device)

        # SWIN

        model_dir = "experiments/pretrained_models"

        self.model_zoo = {
            "real_sr": {
                4: os.path.join(
                    model_dir, "003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth"
                )
            },
            "gray_dn": {
                15: os.path.join(
                    model_dir, "004_grayDN_DFWB_s128w8_SwinIR-M_noise15.pth"
                ),
                25: os.path.join(
                    model_dir, "004_grayDN_DFWB_s128w8_SwinIR-M_noise25.pth"
                ),
                50: os.path.join(
                    model_dir, "004_grayDN_DFWB_s128w8_SwinIR-M_noise50.pth"
                ),
            },
            "color_dn": {
                15: os.path.join(
                    model_dir, "005_colorDN_DFWB_s128w8_SwinIR-M_noise15.pth"
                ),
                25: os.path.join(
                    model_dir, "005_colorDN_DFWB_s128w8_SwinIR-M_noise25.pth"
                ),
                50: os.path.join(
                    model_dir, "005_colorDN_DFWB_s128w8_SwinIR-M_noise50.pth"
                ),
            },
            "jpeg_car": {
                10: os.path.join(model_dir, "006_CAR_DFWB_s126w7_SwinIR-M_jpeg10.pth"),
                20: os.path.join(model_dir, "006_CAR_DFWB_s126w7_SwinIR-M_jpeg20.pth"),
                30: os.path.join(model_dir, "006_CAR_DFWB_s126w7_SwinIR-M_jpeg30.pth"),
                40: os.path.join(model_dir, "006_CAR_DFWB_s126w7_SwinIR-M_jpeg40.pth"),
            },
        }

        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--task",
            type=str,
            default="real_sr",
            help="classical_sr, lightweight_sr, real_sr, "
            "gray_dn, color_dn, jpeg_car",
        )
        parser.add_argument(
            "--scale", type=int, default=1, help="scale factor: 1, 2, 3, 4, 8"
        )  # 1 for dn and jpeg car
        parser.add_argument(
            "--noise", type=int, default=15, help="noise level: 15, 25, 50"
        )
        parser.add_argument(
            "--jpeg", type=int, default=40, help="scale factor: 10, 20, 30, 40"
        )
        parser.add_argument(
            "--training_patch_size",
            type=int,
            default=128,
            help="patch size used in training SwinIR. "
            "Just used to differentiate two different settings in Table 2 of the paper. "
            "Images are NOT tested patch by patch.",
        )
        parser.add_argument(
            "--large_model",
            action="store_true",
            help="use large model, only provided for real image sr",
        )
        parser.add_argument(
            "--model_path", type=str, default=self.model_zoo["real_sr"][4]
        )
        parser.add_argument(
            "--folder_lq",
            type=str,
            default=None,
            help="input low-quality test image folder",
        )
        parser.add_argument(
            "--folder_gt",
            type=str,
            default=None,
            help="input ground-truth test image folder",
        )

        self.args = parser.parse_args("")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tasks = {
            "Real-World Image Super-Resolution": "real_sr",
            "Grayscale Image Denoising": "gray_dn",
            "Color Image Denoising": "color_dn",
            "JPEG Compression Artifact Reduction": "jpeg_car",
        }

    def upscale(
        self,
        input_dir,
        output_dir,
        task_type="Real-World Image Super-Resolution",
        jpeg=40,
        noise=15,
    ):
        os.makedirs(output_dir, exist_ok=True)
        self.args.task = self.tasks[task_type]
        self.args.noise = noise
        self.args.jpeg = jpeg

        # set model path
        if self.args.task == "real_sr":
            self.args.scale = 4
            self.args.model_path = self.model_zoo[self.args.task][4]
        elif self.args.task in ["gray_dn", "color_dn"]:
            self.args.model_path = self.model_zoo[self.args.task][noise]
        else:
            self.args.model_path = self.model_zoo[self.args.task][jpeg]

        if self.args.task == "real_sr":
            self.args.folder_lq = input_dir
        else:
            self.args.folder_gt = input_dir

        model = define_model(self.args)
        model.eval()
        model = model.to(self.device)

        # setup folder and path
        folder, save_dir, border, window_size = setup(self.args)
        os.makedirs(save_dir, exist_ok=True)

        test_results = OrderedDict()
        test_results["psnr"] = []
        test_results["ssim"] = []
        test_results["psnr_y"] = []
        test_results["ssim_y"] = []
        test_results["psnr_b"] = []
        # psnr, ssim, psnr_y, ssim_y, psnr_b = 0, 0, 0, 0, 0

        for idx, path in enumerate(sorted(glob.glob(os.path.join(folder, "*")))):
            print("upscaling", path)
            out_path = path.replace(input_dir, output_dir)
            # read image
            imgname, img_lq, img_gt = get_image_pair(
                self.args, path
            )  # image to HWC-BGR, float32
            img_lq = np.transpose(
                img_lq if img_lq.shape[2] == 1 else img_lq[:, :, [2, 1, 0]], (2, 0, 1)
            )  # HCW-BGR to CHW-RGB
            img_lq = (
                torch.from_numpy(img_lq).float().unsqueeze(0).to(self.device)
            )  # CHW-RGB to NCHW-RGB

            # inference
            with torch.no_grad():
                # pad input image to be a multiple of window_size
                _, _, h_old, w_old = img_lq.size()
                h_pad = (h_old // window_size + 1) * window_size - h_old
                w_pad = (w_old // window_size + 1) * window_size - w_old
                img_lq = torch.cat([img_lq, torch.flip(img_lq, [2])], 2)[
                    :, :, : h_old + h_pad, :
                ]
                img_lq = torch.cat([img_lq, torch.flip(img_lq, [3])], 3)[
                    :, :, :, : w_old + w_pad
                ]
                output = model(img_lq)
                output = output[
                    ..., : h_old * self.args.scale, : w_old * self.args.scale
                ]

            # save image
            output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
            if output.ndim == 3:
                output = np.transpose(
                    output[[2, 1, 0], :, :], (1, 2, 0)
                )  # CHW-RGB to HCW-BGR
            output = (output * 255.0).round().astype(np.uint8)  # float32 to uint8
            cv2.imwrite(str(out_path), output)

    def predict(
        self,
        Prompt: str = cog.Input(description="Your text prompt.", default=""),
        Steps: int = cog.Input(
            description="Number of steps to run the model", default=100
        ),
        ETA: int = cog.Input(description="Can be 0 or 1", default=1),
        Samples_in_parallel: int = cog.Input(description="Batch size", default=4),
        Diversity_scale: float = cog.Input(
            description="As a rule of thumb, higher values of scale produce better samples at the cost of a reduced output diversity.",
            default=10.0,
        ),
        Width: int = cog.Input(description="Width", default=256),
        Height: int = cog.Input(description="Height", default=256),
    ) -> None:
        """Run a single prediction on the model"""

        Iterations = 1
        output_path = "/outputs"
        PLMS_sampling = True

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

            all_samples = list()
            samples_ddim, x_samples_ddim = None, None
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    with self.model.ema_scope():
                        uc = None
                        if opt.scale > 0:
                            uc = self.model.get_learned_conditioning(
                                opt.n_samples * [""]
                            )
                        for prompt in opt.prompts:
                            print(prompt)
                            for n in range(opt.n_iter):
                                c = self.model.get_learned_conditioning(
                                    opt.n_samples * [prompt]
                                )
                                shape = [4, opt.H // 8, opt.W // 8]
                                samples_ddim, _ = sampler.sample(
                                    S=opt.ddim_steps,
                                    conditioning=c,
                                    batch_size=opt.n_samples,
                                    shape=shape,
                                    verbose=False,
                                    unconditional_guidance_scale=opt.scale,
                                    unconditional_conditioning=uc,
                                    eta=opt.ddim_eta,
                                    x_T=samples_ddim,
                                )

                                x_samples_ddim = self.model.decode_first_stage(
                                    samples_ddim
                                )
                                x_samples_ddim = torch.clamp(
                                    (x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0
                                )
                                all_samples.append(x_samples_ddim)
                # additionally, save as grid
                grid = torch.stack(all_samples, 0)
                grid = rearrange(grid, "n b c h w -> (n b) c h w")
                rows = opt.n_samples
                # check if rows is quadratic and if yes take the square root
                height = int(rows**0.5)
                grid = make_grid(grid, nrow=height)
                # to image
                grid = 255.0 * rearrange(grid, "c h w -> h w c").cpu().numpy()
                # Image.fromarray(grid.astype(np.uint8)).save(os.path.join(outpath, f'zzz_{prompt.replace(" ", "-")}.png'))
                # save individual images
                os.makedirs("/content/tmp", exist_ok=True)
                clean_folder("/content/tmp")
                for n, x_sample in enumerate(all_samples[0]):
                    x_sample = x_sample.squeeze()
                    x_sample = 255.0 * rearrange(
                        x_sample.cpu().numpy(), "c h w -> h w c"
                    )
                    prompt_filename = prompt.replace(" ", "-")
                    Image.fromarray(x_sample.astype(np.uint8)).save(
                        os.path.join(
                            output_path, f"/content/tmp/{prompt_filename}_{n}.png"
                        )
                    )

        Modifiers = ["cyber", "cgsociety", "pixar"]
        for Modifier in Modifiers:
            Prompts = modify(Prompt, Modifier)
            args = argparse.Namespace(
                prompts=Prompts.split("->"),
                outdir=output_path,
                ddim_steps=Steps,
                ddim_eta=ETA,
                n_iter=Iterations,
                W=Width,
                H=Height,
                n_samples=Samples_in_parallel,
                scale=Diversity_scale,
                plms=PLMS_sampling,
            )
            run(args)
        self.upscale("/content/tmp", output_path)


def clean_folder(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print("Failed to delete %s. Reason: %s" % (file_path, e))


def modify(Prompt, Modifiers):
    if Modifiers == "cyber":
        return f"cyber cyber {Prompt} {Prompt} {Prompt} digital art by michael whelan"
    if Modifiers == "cgsociety":
        return f"{Prompt} {Prompt} {Prompt} digital art by michael whelan by cgsociety , cyberpunk"
    if Modifiers == "pixar":
        return f"{Prompt} {Prompt} {Prompt} by pixar 3d render"
    print("Unknown modifier:", Modifiers)
    return Prompt
