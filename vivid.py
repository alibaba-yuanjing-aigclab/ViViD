import argparse
from datetime import datetime
from pathlib import Path

import torch
from diffusers import AutoencoderKL, DDIMScheduler
from omegaconf import OmegaConf
from PIL import Image
from torchvision import transforms
from transformers import CLIPVisionModelWithProjection

from src.models.pose_guider import PoseGuider
from src.models.unet_2d_condition import UNet2DConditionModel
from src.models.unet_3d import UNet3DConditionModel
from src.pipelines.pipeline_pose2vid_long import Pose2VideoPipeline
from src.utils.util import get_fps, read_frames, save_videos_grid

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",type=str,default="./configs/prompts/test.yaml")
    parser.add_argument("-W", type=int, default=384)
    parser.add_argument("-H", type=int, default=512)
    parser.add_argument("-L", type=int, default=24)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cfg", type=float, default=3.5)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--fps", type=int)
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    config = OmegaConf.load(args.config)

    if config.weight_dtype == "fp16":
        weight_dtype = torch.float16
    else:
        weight_dtype = torch.float32

    vae = AutoencoderKL.from_pretrained(
        config.pretrained_vae_path,
    ).to("cuda", dtype=weight_dtype)

    reference_unet = UNet2DConditionModel.from_pretrained_2d(
        config.pretrained_base_model_path,
        subfolder="unet",
        unet_additional_kwargs={
            "in_channels": 5,
        }
    ).to(dtype=weight_dtype, device="cuda")

    inference_config_path = config.inference_config
    infer_config = OmegaConf.load(inference_config_path)
    denoising_unet = UNet3DConditionModel.from_pretrained_2d(
        config.pretrained_base_model_path,
        config.motion_module_path,
        subfolder="unet",
        unet_additional_kwargs=infer_config.unet_additional_kwargs,
    ).to(dtype=weight_dtype, device="cuda")

    pose_guider = PoseGuider(320, block_out_channels=(16, 32, 96, 256)).to(
        dtype=weight_dtype, device="cuda"
    )


    image_enc = CLIPVisionModelWithProjection.from_pretrained(
        config.image_encoder_path
    ).to(dtype=weight_dtype, device="cuda")

    sched_kwargs = OmegaConf.to_container(infer_config.noise_scheduler_kwargs)
    scheduler = DDIMScheduler(**sched_kwargs)

    seed = config.get("seed",args.seed)
    generator = torch.manual_seed(seed)

    width, height = args.W, args.H
    clip_length = config.get("L",args.L)  
    steps = args.steps
    guidance_scale = args.cfg

    # load pretrained weights
    denoising_unet.load_state_dict(
        torch.load(config.denoising_unet_path, map_location="cpu"),
        strict=False,
    )
    reference_unet.load_state_dict(
        torch.load(config.reference_unet_path, map_location="cpu"),
    )

    pose_guider.load_state_dict(
        torch.load(config.pose_guider_path, map_location="cpu"),
    )

    pipe = Pose2VideoPipeline(
        vae=vae,
        image_encoder=image_enc,
        reference_unet=reference_unet,
        denoising_unet=denoising_unet,
        pose_guider=pose_guider,
        scheduler=scheduler,
    )
    pipe = pipe.to("cuda", dtype=weight_dtype)

    date_str = datetime.now().strftime("%Y%m%d")
    time_str = datetime.now().strftime("%H%M")
    save_dir_name = f"{time_str}--seed_{seed}-{args.W}x{args.H}"

    save_dir = Path(f"output/{date_str}/{save_dir_name}")
    save_dir.mkdir(exist_ok=True, parents=True)

    model_video_paths = config.model_video_paths
    cloth_image_paths = config.cloth_image_paths

    transform = transforms.Compose(
        [transforms.Resize((height, width)), transforms.ToTensor()]
    )


    for model_image_path in model_video_paths:
        src_fps = get_fps(model_image_path)

        model_name = Path(model_image_path).stem
        agnostic_path=model_image_path.replace("videos","agnostic")
        agn_mask_path=model_image_path.replace("videos","agnostic_mask")
        densepose_path=model_image_path.replace("videos","densepose")

        video_tensor_list=[]
        video_images=read_frames(model_image_path)

        for vid_image_pil in video_images[:clip_length]:
            video_tensor_list.append(transform(vid_image_pil))

        video_tensor = torch.stack(video_tensor_list, dim=0)  # (f, c, h, w)
        video_tensor = video_tensor.transpose(0, 1)

        agnostic_list=[]
        agnostic_images=read_frames(agnostic_path)
        for agnostic_image_pil in agnostic_images[:clip_length]:
            agnostic_list.append(agnostic_image_pil)

        agn_mask_list=[]
        agn_mask_images=read_frames(agn_mask_path)
        for agn_mask_image_pil in agn_mask_images[:clip_length]:
            agn_mask_list.append(agn_mask_image_pil)

        pose_list=[]
        pose_images=read_frames(densepose_path)
        for pose_image_pil in pose_images[:clip_length]:
            pose_list.append(pose_image_pil)

        video_tensor = video_tensor.unsqueeze(0)


        for cloth_image_path in cloth_image_paths:
            cloth_name =  Path(cloth_image_path).stem
            cloth_image_pil = Image.open(cloth_image_path).convert("RGB")

            cloth_mask_path=cloth_image_path.replace("cloth","cloth_mask")
            cloth_mask_pil = Image.open(cloth_mask_path).convert("RGB")

            pipeline_output = pipe(
                agnostic_list,
                agn_mask_list,
                cloth_image_pil,
                cloth_mask_pil,
                pose_list,
                width,
                height,
                clip_length,
                steps,
                guidance_scale,
                generator=generator,
            )
            video = pipeline_output.videos

            video = torch.cat([video_tensor,video], dim=0)
            save_videos_grid(
                video,
                f"{save_dir}/{model_name}_{cloth_name}_{args.H}x{args.W}_{int(guidance_scale)}_{time_str}.mp4",
                n_rows=2,
                fps=src_fps if args.fps is None else args.fps,
            )


if __name__ == "__main__":
    main()
