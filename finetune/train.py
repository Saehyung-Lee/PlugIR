import datetime
import sys
import math
import os
import time
import warnings
import copy
import pickle

import torch
import torch.utils.data
import torchvision
import utils
from sampler import RASampler
from torch import nn
from torch.utils.data.dataloader import default_collate
from torchvision.transforms.functional import InterpolationMode
from collections import OrderedDict
from typing import Dict, Callable, Optional, Any, Tuple, Union
import numpy as np
import statistics
import logging
from visdial import VisDial
import torch
import torch.utils.data
from torch.nn.functional import normalize
import losses

from transformers import AutoProcessor, BlipForImageTextRetrieval

class BlipForRetrieval(BlipForImageTextRetrieval):
    def get_text_features(self,
                          input_ids: torch.LongTensor,
                          attention_mask: Optional[torch.LongTensor] = None,
                          return_dict: Optional[bool] = None,
                          ) -> torch.FloatTensor:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        question_embeds = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=return_dict,
        )
        question_embeds = question_embeds[0] if not return_dict else question_embeds.last_hidden_state

        text_feat = normalize(self.text_proj(question_embeds[:, 0, :]), dim=-1)

        return text_feat

    def get_image_features(
            self,
            pixel_values: torch.FloatTensor,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> torch.FloatTensor:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        with torch.no_grad():
            vision_outputs = self.vision_model(
                pixel_values=pixel_values,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            image_embeds = vision_outputs[0]

            image_feat = normalize(self.vision_proj(image_embeds[:, 0, :]), dim=-1)

        return image_feat

def criterion(img_features, txt_features, temp, device, args):
    if args.loss == 'contrastive':
        return losses.Contrastive(img_features, txt_features, temp, device)
    elif args.loss == 'recall':
        criterion = losses.RecallatK()
        return criterion(img_features, txt_features)

def train_one_epoch(model, optimizer, data_loader, device, epoch, args, model_ema=None, scaler=None, processor=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value}"))
    metric_logger.add_meter("img/s", utils.SmoothedValue(window_size=10, fmt="{value}"))

    header = f"Epoch: [{epoch}]"
    for i, (imgs, txts) in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
        start_time = time.time()
        txts = processor(text=txts, padding=True, return_tensors='pt')
        imgs = imgs.to(device)
        txts = txts.to(device)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            img_features = model.module.get_image_features(imgs)
            txt_features = model.module.get_text_features(**txts)
            loss = criterion(img_features, txt_features, args.temp, device, args)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            if args.clip_grad_norm is not None:
                # we should unscale the gradients of optimizer's assigned params if do gradient clipping
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if args.clip_grad_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)

            optimizer.step()

        batch_size = imgs.shape[0]
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters["img/s"].update(batch_size / (time.time() - start_time))

def evaluate(model, data_loader, device, args, log_suffix="", processor=None):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = f"Test: {log_suffix}"

    num_processed_samples = 0
    with torch.inference_mode():
        for i in range(11):
            data_loader.dataset.dialog_len = i
            for imgs, txts in metric_logger.log_every(data_loader, args.print_freq, header):
                start_time = time.time()
                txts = processor(text=txts, padding=True, return_tensors='pt')
                imgs = imgs.to(device)
                txts = txts.to(device)
                img_features = model.module.get_image_features(imgs)
                txt_features = model.module.get_text_features(**txts)
                loss = criterion(img_features, txt_features, args.temp, device, args)

                # FIXME need to take into account that the datasets
                # could have been padded in distributed setup
                batch_size = imgs.shape[0]
                metric_logger.update(loss=loss.item())
                metric_logger.meters[f"loss{i}"].update(loss.item())
                metric_logger.meters["img/s"].update(batch_size / (time.time() - start_time))
                num_processed_samples += batch_size
            logging.info("Test loss at dialog_len %d : %f"%(i, metric_logger.meters[f"loss{i}"].global_avg))
    # gather the stats from all processes

    num_processed_samples = utils.reduce_across_processes(num_processed_samples)
    if (
        hasattr(data_loader.dataset, "__len__")
        and len(data_loader.dataset) != num_processed_samples
        and torch.distributed.get_rank() == 0
    ):
        # See FIXME above
        warnings.warn(
            f"It looks like the dataset has {len(data_loader.dataset)} samples, but {num_processed_samples} "
            "samples were used for the validation, which might bias the results. "
            "Try adjusting the batch size and / or the world size. "
            "Setting the world size to 1 is always a safe bet."
        )

    metric_logger.synchronize_between_processes()

    return metric_logger.loss.global_avg

def load_data(args, processor):
    # Data loading code
    logging.info("Loading data")
    val_resize_size, val_crop_size, train_crop_size = args.val_resize_size, args.val_crop_size, args.train_crop_size
    interpolation = InterpolationMode(args.interpolation)

    logging.info("Loading training data")
    st = time.time()
    name = args.data_path.split('/')[-1]
    dataset_train = VisDial(args.data_path, split='train', transform=processor)
    dataset_test = VisDial(args.data_path, split='val', dialog_len=0, transform=processor)
    logging.info(f"Took {time.time() - st}")

    logging.info("Creating data loaders")
    logging.info('distributed:')
    logging.info(args.distributed)
    if args.distributed:
        if hasattr(args, "ra_sampler") and args.ra_sampler:
            train_sampler = RASampler(dataset_train, shuffle=True, repetitions=args.ra_reps)
        else:
            train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test, shuffle=False)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset_train)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    return dataset_train, dataset_test, train_sampler, test_sampler

def main(args):
    checkpoint = None
    if args.output_dir:
        utils.mkdir(args.output_dir)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(args.output_dir, 'training.log')),
            logging.StreamHandler()
        ])
    logger = logging.getLogger()
    

    utils.init_distributed_mode(args)

    device = torch.device(args.device)

    if args.use_deterministic_algorithms:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True

    logging.info("Creating model")
    if args.torch_seed is not None:
        torch.manual_seed(args.torch_seed)
    model = BlipForRetrieval.from_pretrained("Salesforce/blip-itm-large-coco").to(device)
    processor = AutoProcessor.from_pretrained("Salesforce/blip-itm-large-coco")

    model.to(device)

    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    dataset, dataset_test, train_sampler, test_sampler = load_data(args, processor)
    logging.info(args)

    collate_fn = None
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    data_loader_test = torch.utils.data.DataLoader(
            dataset_test, batch_size=args.test_batch_size, sampler=test_sampler, num_workers=args.workers, pin_memory=True
    )

    custom_keys_weight_decay = []
    if args.bias_weight_decay is not None:
        custom_keys_weight_decay.append(("bias", args.bias_weight_decay))
    if args.transformer_embedding_decay is not None:
        for key in ["class_token", "position_embedding", "relative_position_bias_table"]:
            custom_keys_weight_decay.append((key, args.transformer_embedding_decay))

    '''
    TODO: train only text encoder
    '''
    for name, p in model.named_parameters():
        if 'text' in name:
            print(f'update {name}')
            p.requires_grad = True
        else:
            print(f'not update {name}')
            p.requires_grad = False
    parameters = utils.set_weight_decay(
        model,
        args.weight_decay,
        norm_weight_decay=args.norm_weight_decay,
        custom_keys_weight_decay=custom_keys_weight_decay if len(custom_keys_weight_decay) > 0 else None,
    )

    opt_name = args.opt.lower()
    if opt_name.startswith("sgd"):
        optimizer = torch.optim.SGD(
            parameters,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov="nesterov" in opt_name,
        )
    elif opt_name == "rmsprop":
        optimizer = torch.optim.RMSprop(
            parameters, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, eps=0.0316, alpha=0.9
        )
    elif opt_name == "adamw":
        optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise RuntimeError(f"Invalid optimizer {args.opt}. Only SGD, RMSprop and AdamW are supported.")

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    args.lr_scheduler = args.lr_scheduler.lower()
    if args.lr_scheduler == "steplr":
        main_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    elif args.lr_scheduler == "cosineannealinglr":
        main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs - args.lr_warmup_epochs, eta_min=args.lr_min
        )
    elif args.lr_scheduler == "exponentiallr":
        main_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_gamma)
    else:
        raise RuntimeError(
            f"Invalid lr scheduler '{args.lr_scheduler}'. Only StepLR, CosineAnnealingLR and ExponentialLR "
            "are supported."
        )

    if args.lr_warmup_epochs > 0:
        if args.lr_warmup_method == "linear":
            warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=args.lr_warmup_decay, total_iters=args.lr_warmup_epochs
            )
        elif args.lr_warmup_method == "constant":
            warmup_lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
                optimizer, factor=args.lr_warmup_decay, total_iters=args.lr_warmup_epochs
            )
        else:
            raise RuntimeError(
                f"Invalid warmup lr method '{args.lr_warmup_method}'. Only linear and constant are supported."
            )
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_lr_scheduler, main_lr_scheduler], milestones=[args.lr_warmup_epochs]
        )
    else:
        lr_scheduler = main_lr_scheduler

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    model_ema = None
    if args.model_ema:
        # Decay adjustment that aims to keep the decay independent from other hyper-parameters originally proposed at:
        # https://github.com/facebookresearch/pycls/blob/f8cd9627/pycls/core/net.py#L123
        #
        # total_ema_updates = (Dataset_size / n_GPUs) * epochs / (batch_size_per_gpu * EMA_steps)
        # We consider constant = Dataset_size for a given dataset/setup and ommit it. Thus:
        # adjust = 1 / total_ema_updates ~= n_GPUs * batch_size_per_gpu * EMA_steps / epochs
        adjust = args.world_size * args.batch_size * args.model_ema_steps / args.epochs
        alpha = 1.0 - args.model_ema_decay
        alpha = min(1.0, alpha * adjust)
        model_ema = utils.ExponentialMovingAverage(model_without_ddp, device=device, decay=1.0 - alpha)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model_without_ddp.load_state_dict(checkpoint["model"])
        if not args.test_only:
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        args.start_epoch = checkpoint["epoch"] + 1
        if model_ema:
            model_ema.load_state_dict(checkpoint["model_ema"])
        if scaler:
            scaler.load_state_dict(checkpoint["scaler"])

    '''
    TODO: loading test
    '''
    if args.pretrained:
        checkpoint = torch.load(args.pretrained, map_location="cpu")
        checkpoint = checkpoint["model_ema"]
        state_dict = {}
        for k, v in checkpoint.items():
            if 'heads' not in k:
                state_dict[k] = v
        msg = model.load_state_dict(state_dict, strict=False)
        logging.info("Load pretrained model with msg: {}".format(msg))
            
    if args.test_only:
        # We disable the cudnn benchmarking because it can noticeably affect the accuracy
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        if model_ema:
            evaluate(model_ema, data_loader_test, device=device, log_suffix="EMA", args=args)
        else:
            evaluate(model, data_loader_test, device=device, args=args)
        return

    logging.info("Start training")
    start_time = time.time()
    best = 99999
    for epoch in range(args.start_epoch, args.epochs):
        logging.info('epoch %d'%epoch)
        if args.distributed:
            train_sampler.set_epoch(epoch)
        train_one_epoch(model, optimizer, data_loader, device, epoch, args, model_ema, scaler, processor=processor)
        lr_scheduler.step()
        test_loss = evaluate(model, data_loader_test, device=device, processor=processor, args=args)
        if model_ema:
            ema_test_loss = evaluate(model_ema, data_loader_test, device=device, log_suffix="EMA", processor=processor, args=args)
        else:
            ema_test_loss = 99999
        if args.output_dir and best > min(test_loss, ema_test_loss):
            logging.info(f"Save the checkpoint at epoch {epoch}")
            best = min(test_loss, ema_test_loss)
            checkpoint = {
                "model": model_without_ddp.state_dict(),
                "epoch": epoch,
                "args": args,
            }
            utils.save_on_master(checkpoint, os.path.join(args.output_dir, f"epoch{epoch}.pth"))
            del checkpoint
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logging.info(f"Training time {total_time_str}")


def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Classification Training", add_help=add_help)

    ''' Add '''
    parser.add_argument("--torch-seed", type=int)
    parser.add_argument("--num-exp", type=int)
    parser.add_argument("--save-after-epoch", type=int, default=100)
    parser.add_argument("--pretrained", type=str)
    parser.add_argument("--loss", type=str, default="recall", choices=["contrastive", "recall"])
    parser.add_argument("--temp", type=float, default=1)

    parser.add_argument("--data-path", default="VisDial", type=str, help="dataset path")
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument(
        "-b", "--batch-size", default=64, type=int, help="images per gpu, the total batch size is $NGPU x batch_size"
    )
    parser.add_argument("--test-batch-size", default=100, type=int)
    parser.add_argument("--epochs", default=36, type=int, metavar="N", help="number of total epochs to run")
    parser.add_argument(
        "-j", "--workers", default=8, type=int, metavar="N", help="number of data loading workers (default: 16)"
    )
    parser.add_argument("--opt", default="adamw", type=str, help="optimizer")
    parser.add_argument("--lr", default=0.0000125, type=float, help="initial learning rate")
    parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=1e-4,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
        dest="weight_decay",
    )
    parser.add_argument(
        "--norm-weight-decay",
        default=None,
        type=float,
        help="weight decay for Normalization layers (default: None, same value as --wd)",
    )
    parser.add_argument(
        "--bias-weight-decay",
        default=None,
        type=float,
        help="weight decay for bias parameters of all layers (default: None, same value as --wd)",
    )
    parser.add_argument(
        "--transformer-embedding-decay",
        default=None,
        type=float,
        help="weight decay for embedding parameters for vision transformer models (default: None, same value as --wd)",
    )
    parser.add_argument(
        "--label-smoothing", default=0.0, type=float, help="label smoothing (default: 0.0)", dest="label_smoothing"
    )
    parser.add_argument("--lr-scheduler", default="exponentiallr", type=str, help="the lr scheduler")
    parser.add_argument("--lr-warmup-epochs", default=0, type=int, help="the number of epochs to warmup (default: 0)")
    parser.add_argument(
        "--lr-warmup-method", default="constant", type=str, help="the warmup method (default: constant)"
    )
    parser.add_argument("--lr-warmup-decay", default=0.01, type=float, help="the decay for lr")
    parser.add_argument("--lr-step-size", default=30, type=int, help="decrease lr every step-size epochs")
    parser.add_argument("--lr-gamma", default=0.93, type=float, help="decrease lr by a factor of lr-gamma")
    parser.add_argument("--lr-min", default=0.0, type=float, help="minimum lr of lr schedule (default: 0.0)")
    parser.add_argument("--print-freq", default=10, type=int, help="print frequency")
    parser.add_argument("--output-dir", default=".", type=str, help="path to save outputs")
    parser.add_argument("--resume", default="", type=str, help="path of checkpoint")
    parser.add_argument("--start-epoch", default=0, type=int, metavar="N", help="start epoch")
    parser.add_argument(
        "--sync-bn",
        dest="sync_bn",
        help="Use sync batch norm",
        action="store_true",
    )
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )
    parser.add_argument("--auto-augment", default=None, type=str, help="auto augment policy (default: None)")
    parser.add_argument("--random-erase", default=0.0, type=float, help="random erasing probability (default: 0.0)")

    # Mixed precision training parameters
    parser.add_argument("--amp", action="store_true", help="Use torch.cuda.amp for mixed precision training")

    # distributed training parameters
    parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--dist-url", default="env://", type=str, help="url used to set up distributed training")
    parser.add_argument(
        "--model-ema", action="store_true", help="enable tracking Exponential Moving Average of model parameters"
    )
    parser.add_argument(
        "--model-ema-steps",
        type=int,
        default=32,
        help="the number of iterations that controls how often to update the EMA model (default: 32)",
    )
    parser.add_argument(
        "--model-ema-decay",
        type=float,
        default=0.99998,
        help="decay factor for Exponential Moving Average of model parameters (default: 0.99998)",
    )
    parser.add_argument(
        "--use-deterministic-algorithms", action="store_true", help="Forces the use of deterministic algorithms only."
    )
    parser.add_argument(
        "--interpolation", default="bilinear", type=str, help="the interpolation method (default: bilinear)"
    )
    parser.add_argument(
        "--val-resize-size", default=256, type=int, help="the resize size used for validation (default: 256)"
    )
    parser.add_argument(
        "--val-crop-size", default=224, type=int, help="the central crop size used for validation (default: 224)"
    )
    parser.add_argument(
        "--train-crop-size", default=224, type=int, help="the random crop size used for training (default: 224)"
    )
    parser.add_argument("--clip-grad-norm", default=None, type=float, help="the maximum gradient norm (default None)")
    parser.add_argument("--ra-sampler", action="store_true", help="whether to use Repeated Augmentation in training")
    parser.add_argument(
        "--ra-reps", default=3, type=int, help="number of repetitions for Repeated Augmentation (default: 3)"
    )
    parser.add_argument("--weights", default=None, type=str, help="the weights enum name to load")

    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
