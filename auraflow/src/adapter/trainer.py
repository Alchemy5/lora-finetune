import pdb;pdb.set_trace()
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import ProjectConfiguration, set_seed
from omegaconf import DictConfig
import torch
import os
import logging
from transformers import BitsAndBytesConfig
from ..auraflow.auraflow import AuraFlowTransformer2DModel
from transformers import AutoTokenizer, UMT5EncoderModel
from diffusers import AutoencoderKL
from model import LoRAAttention, LoRA
from ..data.dataset import StyleDataset
from torch.utils.data import DataLoader, Dataset
import accelerate
import tqdm
from torch.optim.lr_scheduler import LambdaLR
from typing import Union, Optional, List
import torch.nn.functional as F

def get_weight_dtype(precision: str):
    weight_dtype = torch.float32
    if precision == "fp16":
        weight_dtype = torch.float16
    elif precision == "bf16":
        weight_dtype = torch.bfloat16
    return weight_dtype

def load_models(cfg: DictConfig, weight_dtype=torch.float32, device="cpu"):
    pretrained_model_name_or_path = cfg.model.pretrained_model_name_or_path

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=weight_dtype,
    )

    # load transformer
    transformer = AuraFlowTransformer2DModel.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="transformer",
    ).to(dtype=weight_dtype, device=device)

    # load text encoder
    text_encoder = UMT5EncoderModel.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        quantization_config=quantization_config,
    )
    # ).to(dtype=weight_dtype, device=device)

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path, subfolder="tokenizer"
    )

    # load vae
    vae = AutoencoderKL.from_pretrained(
        pretrained_model_name_or_path, subfolder="vae"
    ).to(device)

    return {
        "transformer": transformer,
        "text_encoder": text_encoder,
        "tokenizer": tokenizer,
        "vae": vae,
    }

def replace_with_lora_attention(blocks, lora_rank=4):
    for block in blocks:
        block.attn = LoRAAttention(
            query_dim=block.attn.query_dim,
            cross_attention_dim=block.attn.cross_attention_dim,
            dim_head=block.attn.dim_head,
            heads=block.attn.heads,
            qk_norm=block.attn.qk_norm,
            out_dim=block.attn.out_dim,
            bias=block.attn.bias,
            out_bias=block.attn.out_bias,
            processor=block.attn.processor,
            lora_rank=lora_rank,
        )

def freeze_transformer_weights(model):
    # Freeze all parameters in the model
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze only LoRA parameters
    for name, module in model.named_modules():
        if isinstance(module, LoRA):
            for param in module.parameters():
                param.requires_grad = True

def create_dataloader(cfg: DictConfig, dataset: Dataset):
    def collate_fn(batch):
        imgs = []

        for data in batch:
            img = data
            if img is None:
                continue
            imgs.append(img)

        return {
            "imgs": torch.stack(imgs),
        }

    return DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        collate_fn=collate_fn,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
        shuffle=True,
    )

def load_scheduler(cfg: DictConfig, optimizer):
    num_warmup_steps = cfg.train.num_warmup_steps

    def warmup_schedule(step):
        if step < num_warmup_steps:
            return float(step) / float(max(1, num_warmup_steps))  # Linear warmup
        return 1.0  # Keep LR constant after warmup

    # Create a scheduler with warmup
    scheduler = LambdaLR(optimizer, lr_lambda=warmup_schedule)
    return scheduler

def from_loader(loader: DataLoader):
    while True:
        for batch in loader:
            yield batch

def encode_prompt(
    text_encoder,
    tokenizer,
    prompt: Union[str, List[str]],
    negative_prompt: Union[str, List[str]] = None,
    do_classifier_free_guidance: bool = True,
    num_images_per_prompt: int = 1,
    device: Optional[torch.device] = None,
    prompt_embeds: Optional[torch.Tensor] = None,
    negative_prompt_embeds: Optional[torch.Tensor] = None,
    prompt_attention_mask: Optional[torch.Tensor] = None,
    negative_prompt_attention_mask: Optional[torch.Tensor] = None,
    max_sequence_length: int = 256,
):
    if device is None:
        device = "cpu"

    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    max_length = max_sequence_length
    if prompt_embeds is None:
        text_inputs = tokenizer(
            prompt,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt",
        )
        text_input_ids = text_inputs["input_ids"]
        untruncated_ids = tokenizer(
            prompt, padding="longest", return_tensors="pt"
        ).input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
            text_input_ids, untruncated_ids
        ):
            removed_text = tokenizer.batch_decode(
                untruncated_ids[:, max_length - 1 : -1]
            )
            logging.warning(
                "The following part of your input was truncated because T5 can only handle sequences up to"
                f" {max_length} tokens: {removed_text}"
            )

        text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
        prompt_embeds = text_encoder(**text_inputs)[0]
        prompt_attention_mask = (
            text_inputs["attention_mask"].unsqueeze(-1).expand(prompt_embeds.shape)
        )
        prompt_embeds = prompt_embeds * prompt_attention_mask

    dtype = text_encoder.dtype

    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    bs_embed, seq_len, _ = prompt_embeds.shape
    # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)
    prompt_attention_mask = prompt_attention_mask.reshape(bs_embed, -1)
    prompt_attention_mask = prompt_attention_mask.repeat(num_images_per_prompt, 1)

    # get unconditional embeddings for classifier free guidance
    if do_classifier_free_guidance and negative_prompt_embeds is None:
        negative_prompt = negative_prompt or ""
        uncond_tokens = (
            [negative_prompt] * batch_size
            if isinstance(negative_prompt, str)
            else negative_prompt
        )
        max_length = prompt_embeds.shape[1]
        uncond_input = tokenizer(
            uncond_tokens,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt",
        )
        uncond_input = {k: v.to(device) for k, v in uncond_input.items()}
        negative_prompt_embeds = text_encoder(**uncond_input)[0]
        negative_prompt_attention_mask = (
            uncond_input["attention_mask"]
            .unsqueeze(-1)
            .expand(negative_prompt_embeds.shape)
        )
        negative_prompt_embeds = negative_prompt_embeds * negative_prompt_attention_mask

    if do_classifier_free_guidance:
        # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
        seq_len = negative_prompt_embeds.shape[1]

        negative_prompt_embeds = negative_prompt_embeds.to(dtype=dtype, device=device)

        negative_prompt_embeds = negative_prompt_embeds.repeat(
            1, num_images_per_prompt, 1
        )
        negative_prompt_embeds = negative_prompt_embeds.view(
            batch_size * num_images_per_prompt, seq_len, -1
        )

        negative_prompt_attention_mask = negative_prompt_attention_mask.reshape(
            bs_embed, -1
        )
        negative_prompt_attention_mask = negative_prompt_attention_mask.repeat(
            num_images_per_prompt, 1
        )
    else:
        negative_prompt_embeds = None
        negative_prompt_attention_mask = None

    return (
        prompt_embeds,
        prompt_attention_mask,
        negative_prompt_embeds,
        negative_prompt_attention_mask,
    )

def sample_timesteps(batch_size: int, sampling: str = "uniform"):
    if sampling == "uniform":
        return torch.rand(batch_size)
    elif sampling == "cubic":
        t = torch.rand(batch_size)
        return 1 - t**3
    elif sampling == "sigmoid":
        return torch.sigmoid(torch.randn(batch_size))
    else:
        raise ValueError(f"Timestep sampling method {sampling} not supported")

def get_xt(x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor):
    bs = x0.shape[0]
    t = t.view(bs, *[1 for _ in range(x0.dim() - 1)])
    xt = (1 - t) * x1 + t * x0

    return xt

def train_adapter(cfg: DictConfig):
    weight_dtype = get_weight_dtype(cfg.train.mixed_precision)
    logging_dir = os.path.join(cfg.output_dir, "logs")
    checkpoint_dir = os.path.join(cfg.output_dir, "checkpoints")
    project_config = ProjectConfiguration(
        project_dir=cfg.output_dir, logging_dir=logging_dir
    )
    accelerator = Accelerator(
        mixed_precision=cfg.train.mixed_precision,
        gradient_accumulation_steps=cfg.train.gradient_accumulation_steps,
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
        project_config=project_config,
        log_with=cfg.log_with,
    )

    set_seed(cfg.train.seed)
    torch.set_float32_matmul_precision("high")

    logging.info("Loading Models")
    models = load_models(cfg, weight_dtype, accelerator.device)

    transformer, text_encoder, tokenizer, vae = (
        models["transformer"],
        models["text_encoder"],
        models["tokenizer"],
        models["vae"],
    )

    logging.info("Loaded Transformer, Text Encoder, Tokenizer, and VAE")

    logging.info("Loaded DiT, VAE, Text Encoder, and Tokenizer")

    replace_with_lora_attention(transformer.joint_transformer_blocks, lora_rank=4)
    replace_with_lora_attention(transformer.single_transformer_blocks, lora_rank=4)
    logging.info("Injected in LoRA weights!")

    # freeze weights of dit, vae, text encoder
    freeze_transformer_weights(transformer) # freeze all weights in transformer except LoRA weights
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    transformer.train()
    vae.eval()
    text_encoder.eval()

    # enable gradient checkpointing
    if cfg.train.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    # load dataset and dataloader
    dataset = StyleDataset(cfg.images_dir)
    dataloader = create_dataloader(cfg, dataset)

     # init project
    if accelerator.is_main_process:
        os.makedirs(cfg.output_dir, exist_ok=True)
        os.makedirs(logging_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)

    
    logging.info("*** Tuning AuraFlow ***")
    logging.info(f"*** Per Device Batch Size: {cfg.train.batch_size} ***")
    logging.info(f"*** Total Train Steps: {cfg.train.train_steps} ***")
    logging.info(f"*** Mixed Precision: {cfg.train.mixed_precision} ***")
    logging.info(
        f"*** Gradient Accumulation Steps: {cfg.train.gradient_accumulation_steps} ***"
    )
    logging.info(f"*** Device: {accelerator.device} ***")

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
    #dataloader = accelerate.skip_first_batches(dataloader, initial_global_step)

    # Define an optimizer that only updates LoRA parameters
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, transformer.parameters()), 
        lr=cfg.train.learning_rate
    )

    scheduler = load_scheduler(cfg, optimizer)

    # prepare with accelerator
    transformer, optimizer, scheduler, dataloader = accelerator.prepare(
        transformer,
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
        with accelerator.accumulate(transformer):
            batch = next(loader)
            imgs = (
                batch["imgs"]
            )

            if imgs is None:
                continue

            bs = imgs.shape[0]

            with torch.no_grad():
                # embed prompts
                (
                    prompt_embeds,
                    prompt_attention_mask,
                    negative_prompt_embeds,
                    negative_prompt_attention_mask,
                ) = encode_prompt(
                    text_encoder=text_encoder,
                    tokenizer=tokenizer,
                    prompt=["generate an inspirational image"],
                    do_classifier_free_guidance=False,
                    device=accelerator.device,
                )

                # encode imgs x1 with vae
                x1 = vae.encode(imgs).latent_dist.sample() * vae.config.scaling_factor
            
            # sample timesteps
            timesteps = torch.rand(bs, device=accelerator.device)
            timesteps = sample_timesteps(bs, cfg.train.timestep_sampling).to(
                accelerator.device
            )

            # sample random noise x0
            x0 = torch.randn_like(x1, device=accelerator.device)

            # get xt
            xt = get_xt(x0, x1, timesteps)

            # pred velocity
            pred_v = transformer(
                xt.to(weight_dtype),
                encoder_hidden_states=prompt_embeds,
                timestep=timesteps,
                return_dict=False,
            )[0]
            
            # compute loss
            loss = F.mse_loss(pred_v.float(), (x0 - x1).float(), reduction="mean")

            avg_loss = accelerator.gather(loss.repeat(cfg.train.batch_size)).mean()
            train_loss += avg_loss.item() / cfg.train.gradient_accumulation_steps

            # backprop
            optimizer.zero_grad()
            accelerator.backward(loss)

            if accelerator.sync_gradients:
                # clip grad norm
                accelerator.clip_grad_norm_(
                    transformer.parameters(), cfg.train.max_grad_norm
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
                    global_step % cfg.train.log_every == 0
                    and accelerator.is_main_process
                ):
                    accelerator.log(logs, step=global_step)
    
    accelerator.wait_for_everyone()
    accelerator.end_training()
    accelerator.print("Done Training")
    