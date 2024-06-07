from PIL import ImageOps, ImageFilter, Image
import numpy as np
import os
from torchvision import transforms
import torch
import av
from pathlib import Path
from einops import rearrange
import torchvision
def get_fps(video_path):
    container = av.open(video_path)
    video_stream = next(s for s in container.streams if s.type == "video")
    fps = video_stream.average_rate
    container.close()
    return fps

def save_videos_from_pil(pil_images, path, fps=8):
    import av

    save_fmt = Path(path).suffix
    os.makedirs(os.path.dirname(path), exist_ok=True)
    width, height = pil_images[0].size

    if save_fmt == ".mp4":
        codec = "libx264"
        container = av.open(path, "w")
        stream = container.add_stream(codec, rate=fps)

        stream.width = width
        stream.height = height

        for pil_image in pil_images:
            # pil_image = Image.fromarray(image_arr).convert("RGB")
            av_frame = av.VideoFrame.from_image(pil_image)
            container.mux(stream.encode(av_frame))
        container.mux(stream.encode())
        container.close()

    elif save_fmt == ".gif":
        pil_images[0].save(
            fp=path,
            format="GIF",
            append_images=pil_images[1:],
            save_all=True,
            duration=(1 / fps * 1000),
            loop=0,
        )
    else:
        raise ValueError("Unsupported file type. Use .mp4 or .gif.")

def save_videos_grid(videos: torch.Tensor, path: str, rescale=False, n_rows=6, fps=8):
    videos = rearrange(videos, "b t c h w -> t b c h w")
    height, width = videos.shape[-2:]
    outputs = []

    for x in videos:
        x = torchvision.utils.make_grid(x, nrow=n_rows)  # (c h w)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)  # (h w c)
        if rescale:
            x = (x + 1.0) / 2.0  # -1,1 -> 0,1
        x = (x * 255).numpy().astype(np.uint8)
        x = Image.fromarray(x)

        outputs.append(x)

    os.makedirs(os.path.dirname(path), exist_ok=True)

    save_videos_from_pil(outputs, path, fps)

def sam_based_agnostic(mask_folder,frame_folder):
    frames_list=sorted(os.listdir(frame_folder))
    mask_list=sorted(os.listdir(mask_folder))
    agn=[]
    agn_mask=[]
    for idx, frame in enumerate(frames_list[:len(mask_list)]):

        image1=Image.open(os.path.join(frame_folder,frame))
        width, height = image1.size

        image2 = Image.new('RGB', (width, height),(126,126,126))

        mask=Image.open(os.path.join(mask_folder,mask_list[idx]))

        blurred_mask_image = mask.filter(ImageFilter.GaussianBlur(radius=4))  # adjustment

        blurred_mask_array = np.array(blurred_mask_image)

        smoothed_mask_array = np.where(blurred_mask_array > 50, 255, 0)  # adjustment

        smoothed_mask = Image.fromarray(smoothed_mask_array.astype(np.uint8))
        mask=ImageOps.invert(smoothed_mask)
        
        agn_mask.append(smoothed_mask)
        result = Image.composite(image1, image2, mask)
        
        agn.append(result)

    return agn,agn_mask

agn,agn_mask=sam_based_agnostic("/path/to/mask_folder","/path/to/frame_folder")

transform = transforms.Compose([
    transforms.ToTensor()  
])

agn=[transform(n) for n in agn]
agn=torch.stack(agn)
agn=agn.unsqueeze(0)

agn_mask=[transform(n) for n in agn_mask]
agn_mask=torch.stack(agn_mask)
agn_mask=agn_mask.unsqueeze(0)

fps=24
save_videos_grid(agn,"/path/to/saved_agnostic",fps=fps)
save_videos_grid(agn_mask,"/path/to/saved_agnostic_mask",fps=fps)