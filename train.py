from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import ProjectConfiguration, set_seed
from omegaconf import DictConfig, OmegaConf
from torchvision import transforms
import shutil
import os
import numpy as np
import accelerate
import logging
import torch
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler
from diffusers.models.embeddings import get_2d_rotary_pos_embed_lumina
from transformers import AutoModel, AutoTokenizer, GemmaModel
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from safetensors.torch import save_file
from torch.optim.lr_scheduler import LambdaLR
import random
import torch.nn.functional as F
import json
from PIL import Image
from dataset import TrainDataset

from src.lumina import LuminaNextDiT2DModel
#from src.model import LuminaT2IAdapter
from src.utils import count_parameters, pil_to_numpy, pt_to_pil, make_image_grid
#from ..data import CirclesFillDataset
#from src.pipeline import LuminaAdapterPipeline

def get_xt(x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor):
    bs = x0.shape[0]
    t = t.view(bs, *[1 for _ in range(x0.dim() - 1)])
    xt = t * x1 + (1 - t) * x0

    return xt

def load_lumina_models(cfg: DictConfig, device="cuda"):
    pretrained_model_name_or_path = cfg.model.pretrained_model_name_or_path

    # load text encoder
    text_encoder = GemmaModel.from_pretrained(
        pretrained_model_name_or_path, subfolder="text_encoder"
    ).to(device)

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path, subfolder="tokenizer"
    )

    # load vae
    vae = AutoencoderKL.from_pretrained(
        pretrained_model_name_or_path, subfolder="vae", torch_dtype=torch.float32
    ).to(device)

    # load dit
    dit = LuminaNextDiT2DModel.from_pretrained(
        pretrained_model_name_or_path, subfolder="transformer"
    ).to(device)

    return {
        "dit": dit,
        "vae": vae,
        "text_encoder": text_encoder,
        "tokenizer": tokenizer,
    }



def from_loader(loader: DataLoader):
    while True:
        for batch in loader:
            yield batch

def construct_dataset(base_path):
    transform = transforms.Compose([
        transforms.Resize((512, 512)),  # Resize to 512x512
        transforms.ToTensor(),          # Convert to Tensor
    ])
    dataset = TrainDataset(base_dir=base_path, transform=transform)
    return dataset

def create_dataloader(cfg: DictConfig, dataset: Dataset):
    def collate_fn(batch):
        bs = len(batch)
        imgs = []
        condition_imgs = []
        prompts = []

        for data in batch:
            img, condition_img, prompt = (
                data["img"],
                data["condition_img"],
                data["prompt"],
            )
            if img is None:
                continue
            imgs.append(img)
            condition_imgs.append(condition_img)
            prompts.append(prompt)

        # all images in batch failed to load
        if len(imgs) == 0:
            return {
                "imgs": None,
                "condition_imgs": None,
                "prompts": None,
            }

        # handle failed image loads — repeat until batch size with first image
        if len(imgs) < bs:
            for i in range(bs - len(imgs)):
                imgs.append(imgs[0])
                condition_imgs.append(condition_imgs[0])
                prompts.append(prompts[0])

        return {
            "imgs": torch.stack(imgs),
            "condition_imgs": torch.stack(condition_imgs),
            "prompts": prompts,
        }

    return DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        collate_fn=collate_fn,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
        shuffle=True,
        drop_last=True,
    )


def load_adapter(cfg: DictConfig, dit: LuminaNextDiT2DModel, device):
    adapter_config = {
        **dit.config,
        "rank": cfg.model.adapter.rank,
        "num_layers": cfg.model.adapter.num_layers,
    }

    adapter = LuminaT2IAdapter(**adapter_config).to(device)
    return adapter


def get_weight_dtype(precision: str):
    weight_dtype = torch.float32
    if precision == "fp16":
        weight_dtype = torch.float16
    elif precision == "bf16":
        weight_dtype = torch.bfloat16
    return weight_dtype


def save_checkpoint(
    cfg: DictConfig,
    accelerator: Accelerator,
    checkpoint_dir: str,
    global_step: int,
):
    accelerator.print("Saving Checkpoint")
    if cfg.train.max_checkpoints is not None:
        checkpoints = os.listdir(checkpoint_dir)
        checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

        # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
        if len(checkpoints) >= cfg.train.max_checkpoints:
            num_to_remove = len(checkpoints) - cfg.train.max_checkpoints + 1
            removing_checkpoints = checkpoints[0:num_to_remove]

            accelerator.print(
                f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
            )
            accelerator.print(
                f"removing checkpoints: {', '.join(removing_checkpoints)}"
            )

            for removing_checkpoint in removing_checkpoints:
                removing_checkpoint = os.path.join(checkpoint_dir, removing_checkpoint)
                shutil.rmtree(removing_checkpoint)

        save_path = os.path.join(
            checkpoint_dir,
            f"checkpoint-{global_step}",
        )
        accelerator.save_state(save_path)
        model = accelerator.unwrap_model(adapter)
        model_save_path = os.path.join(save_path, "adapter.safetensors")
        save_file(model.state_dict(), model_save_path)


def log_validation_image(
    accelerator: Accelerator, name: str, image: Image.Image, global_step: int
):
    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            np_images = pil_to_numpy(image)[0]
            tracker.writer.add_images(name, np_images, global_step, dataformats="HWC")
        elif tracker.name == "wandb":
            import wandb

            tracker.log({name: wandb.Image(image)}, step=global_step)


def generate_validation_grid(self, cfg, pipe, imgs, prompts, condition_imgs):
    validation_imgs = []
    for img, prompt, condition_img in zip(imgs, prompts, condition_imgs):
        gen_img = pipe(
            prompt=prompt,
            image=condition_img,
            height=cfg.dataset.params.resolution[0],
            width=cfg.dataset.params.resolution[1],
            # num_inference_steps=self.cfg.train.validation_inference_steps,
        )

        # compare with ground truth
        validation_img = make_image_grid(
            [condition_img, img, gen_img],
            rows=3,
            cols=1,
        )
        validation_imgs.append(validation_img)

    validation_grid = make_image_grid(
        validation_imgs, rows=1, cols=len(validation_imgs)
    )

    return validation_grid


def validation_step(
    cfg,
    dataset: Dataset,
    transformer,
    tokenizer,
    text_encoder,
    vae,
):
    euler_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        cfg.model.pretrained_model_name_or_path, subfolder="scheduler"
    )
    pipe = LuminaAdapterPipeline(
        transformer=transformer,
        scheduler=euler_scheduler,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        adapter=adapter,
    )

    batch = [dataset[i] for i in range(cfg.train.num_train_validation)]
    prompts = [data["prompt"] for data in batch]
    imgs = [pt_to_pil(data["img"].unsqueeze(0)) for data in batch]
    condition_imgs = [pt_to_pil(data["condition_img"].unsqueeze(0)) for data in batch]

    print("validation step")
    print(imgs, condition_imgs)

    train_validation_grid = generate_validation_grid(
        cfg, pipe, imgs, prompts, condition_imgs
    )
    # self.log_validation_image(train_validation_grid, name="train_validation")


def load_optimizer(cfg: DictConfig):
    optimizer = torch.optim.AdamW(
        [param for param in adapter.parameters() if param.requires_grad],
        lr=cfg.train.lr,
        fused=True,
    )
    return optimizer


def load_scheduler(cfg: DictConfig, optimizer):
    num_warmup_steps = cfg.train.num_warmup_steps

    def warmup_schedule(step):
        if step < num_warmup_steps:
            return float(step) / float(max(1, num_warmup_steps))  # Linear warmup
        return 1.0  # Keep LR constant after warmup

    # Create a scheduler with warmup
    scheduler = LambdaLR(optimizer, lr_lambda=warmup_schedule)
    return scheduler


def encode_prompt(
    prompt_batch,
    text_encoder,
    tokenizer,
    proportion_empty_prompts,
    is_train=True,
    device="cpu",
):
    captions = []
    for caption in prompt_batch:
        if random.random() < proportion_empty_prompts:
            captions.append("")
        elif isinstance(caption, str):
            captions.append(caption)
        elif isinstance(caption, (list, np.ndarray)):
            # take a random caption if there are multiple
            captions.append(random.choice(caption) if is_train else caption[0])

    with torch.no_grad():
        text_inputs = tokenizer(
            captions,
            padding=True,
            pad_to_multiple_of=8,
            max_length=256,
            truncation=True,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids.to(device)
        prompt_masks = text_inputs.attention_mask.to(device)

        prompt_embeds = text_encoder(
            input_ids=text_input_ids,
            attention_mask=prompt_masks,
            output_hidden_states=True,
        ).hidden_states[-2]

    return prompt_embeds, prompt_masks


def train_lumina(cfg: DictConfig):
    
    weight_dtype = get_weight_dtype(cfg.train.mixed_precision)

    checkpoint_dir = os.path.join(cfg.output_dir, "checkpoints")

    set_seed(cfg.train.seed)
    
    # load models
    models = load_lumina_models(cfg)
    
    dit, vae, text_encoder, tokenizer = (
        models["dit"],
        models["vae"],
        models["text_encoder"],
        models["tokenizer"],
    )
    

    # freeze weights of dit, vae, text encoder
    dit.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    dit.eval()
    vae.eval()
    text_encoder.eval()
    
    # load adapter

    # load dataset and dataloader
    dataset = construct_dataset(cfg.dataset.name)
    dataloader = create_dataloader(cfg, dataset)
    
    # init project
    if accelerator.is_main_process:
        os.makedirs(cfg.output_dir, exist_ok=True)
        os.makedirs(logging_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)

        # save model config
        with open(os.path.join(cfg.output_dir, "adapter.json"), "w") as f:
            json.dump({"rank": adapter.rank, "num_layers": adapter.num_layers}, f)

        accelerator.init_trackers(
            cfg.name,
            {
                "lr": cfg.train.lr,
                "batch_size": cfg.train.batch_size,
                "mixed_precision": cfg.train.mixed_precision,
                "gradient_accumulation_steps": cfg.train.gradient_accumulation_steps,
                "max_grad_norm": cfg.train.max_grad_norm,
                "num_layers": cfg.model.adapter.num_layers,
                "rank": cfg.model.adapter.rank,
                "num_warmup_steps": cfg.train.num_warmup_steps,
            },
            {
                "wandb": {"dir": "../../wandb"},
            },
        )

        # save cfg to project dir
        OmegaConf.save(cfg, os.path.join(cfg.output_dir, "config.yaml"))

    # resume from last checkpoint if exists
    global_step = 0
    initial_global_step = 0

    if cfg.resume_from_checkpoint:
        # get the latest checkpoint
        dirs = os.listdir(checkpoint_dir)
        dirs = [d for d in dirs if d.startswith("checkpoint")]
        dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
        path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint does not exist. Starting a new training run."
            )
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(checkpoint_dir, path))
            global_step = int(path.split("-")[1])
            initial_global_step = global_step
            global_step = global_step
    else:
        initial_global_step = 0

    pbar = tqdm(
        range(0, cfg.train.train_steps),
        initial=initial_global_step,
        desc="Train Steps",
        disable=not accelerator.is_local_main_process,
    )
    dataloader = accelerate.skip_first_batches(dataloader, initial_global_step)
    optimizer = load_optimizer(cfg, adapter)
    scheduler = load_scheduler(cfg, optimizer)

    # prepare with accelerator
    adapter, optimizer, scheduler, dataloader = accelerator.prepare(
        adapter,
        optimizer,
        scheduler,
        dataloader,
    )

    loader = from_loader(dataloader)

    num_steps = (
        cfg.train.train_steps - initial_global_step
    ) * cfg.train.gradient_accumulation_steps
    train_loss = 0.0

    for i in range(num_steps):
        with accelerator.accumulate(adapter):
            batch = next(loader)
            imgs, condition_imgs, prompts = (
                batch["imgs"],
                batch["condition_imgs"],
                batch["prompts"],
            )

            if imgs is None:
                continue

            bs = imgs.shape[0]

            # sample timesteps
            timesteps = torch.rand(bs, device=imgs.device)

            # embed prompts
            prompt_embeds, prompt_masks = encode_prompt(
                prompts,
                text_encoder,
                tokenizer,
                proportion_empty_prompts=0,
                is_train=True,
                device=imgs.device,
            )

            # encode imgs x1 with vae
            with torch.no_grad():
                x1 = vae.encode(imgs).latent_dist.sample() * vae.config.scaling_factor

            # sample random noise x0
            x0 = torch.randn_like(x1, device=x1.device)

            # get xt
            xt = get_xt(x0, x1, timesteps)

            # get features from condition img
            image_rotary_emb = get_2d_rotary_pos_embed_lumina(
                dit.head_dim, 384, 384, linear_factor=1, ntk_factor=1
            )

            with accelerator.autocast():
                block_hidden_states = adapter(condition_imgs, image_rotary_emb)

            # pred velocity
            pred_v = dit(
                hidden_states=xt,
                timestep=timesteps,
                encoder_hidden_states=prompt_embeds,
                encoder_mask=prompt_masks,
                image_rotary_emb=image_rotary_emb,
                block_hidden_states=block_hidden_states,
                cross_attention_kwargs={},
                return_dict=False,
            )[0]
            pred_v = pred_v.chunk(2, dim=1)[0]

            # compute loss
            loss = F.mse_loss(pred_v.float(), (x1 - x0).float(), reduction="mean")

            avg_loss = accelerator.gather(loss.repeat(cfg.train.batch_size)).mean()
            train_loss += avg_loss.item() / cfg.train.gradient_accumulation_steps

            # backprop
            optimizer.zero_grad()
            accelerator.backward(loss)

            if accelerator.sync_gradients:
                # clip grad norm
                accelerator.clip_grad_norm_(
                    adapter.parameters(), cfg.train.max_grad_norm
                )

            optimizer.step()
            scheduler.step()

            if accelerator.sync_gradients:
                pbar.update(1)
                logs = {
                    "train_loss": train_loss,
                    "lr": scheduler.get_last_lr()[0],
                }
                pbar.set_postfix(**logs)
                global_step += 1
                train_loss = 0.0

                if (
                    global_step % cfg.train.log_every == 0
                    and accelerator.is_main_process
                ):
                    accelerator.log(logs, step=global_step)

                if (
                    global_step
                ) % cfg.train.checkpoint_every == 0 and accelerator.is_main_process:
                    save_checkpoint(
                        cfg, accelerator, adapter, checkpoint_dir, global_step
                    )

                if (
                    global_step
                ) % cfg.train.validate_every == 0 and accelerator.is_main_process:
                    validation_step(
                        cfg,
                        dataset=dataset,
                        transformer=dit,
                        tokenizer=tokenizer,
                        text_encoder=text_encoder,
                        vae=vae,
                        adapter=adapter,
                    )

