from pickletools import optimize
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from models import Uni_Sign
import utils as utils
from datasets import LIS_Dataset

import os
import time
import argparse, json, datetime
from pathlib import Path
import math
import sys
from timm.optim import create_optimizer
from models import get_requires_grad_dict
from transformers import get_scheduler
from SLRT_metrics import translation_performance
from config import *
from typing import Iterable, Optional

def main(args):
    utils.init_distributed_mode_ds(args)
    print("\nargs : \n", args)
    utils.set_seed(args.seed)

    print(f"Creating dataset:")
    train_data = LIS_Dataset(path=train_label_paths[args.dataset], args=args, phase='train')
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=train_data.collate_fn,
        sampler=train_sampler,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    dev_data = LIS_Dataset(path=dev_label_paths[args.dataset], args=args, phase='dev')
    dev_sampler = SequentialSampler(dev_data)
    dev_dataloader = DataLoader(
        dev_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=dev_data.collate_fn,
        sampler=dev_sampler,
        pin_memory=args.pin_mem,
    )

    print(f"Creating model:")
    model = Uni_Sign(args=args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model.train()
    for param in model.parameters():
        if param.requires_grad:
            param.data = param.data.to(torch.float32)


    if args.finetune != '':
        print('***********************************')
        print('Load Checkpoint...')
        print('***********************************')

        print(f"Recupero Checkpoint da: {args.finetune}")


        state_dict = torch.load(args.finetune, map_location='cpu')['model']

        ret = model.load_state_dict(state_dict, strict=False)
        print('Missing keys: \n', '\n'.join(ret.missing_keys))
        print('Unexpected keys: \n', '\n'.join(ret.unexpected_keys))

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True
        )
        model_without_ddp = model.module

    n_parameters = utils.count_parameters_in_MB(model_without_ddp)
    print(f'number of params: {n_parameters}M')

    optimizer = create_optimizer(args, model_without_ddp)

    if args.quick_break <= 0:
        args.quick_break = len(train_dataloader)

    lr_scheduler = get_scheduler(
        name='cosine',
        optimizer=optimizer,
        num_warmup_steps=int(args.warmup_epochs * len(train_dataloader) / args.gradient_accumulation_steps),
        num_training_steps=int(args.epochs * len(train_dataloader) / args.gradient_accumulation_steps),
    )

    output_dir = Path(args.output_dir)
    start_epoch = 0

    # RESUME support
    if args.resume:
        print(f"***************************************************\nRipristino da checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location='cpu')
        try:
            model_without_ddp.load_state_dict(checkpoint['model'])
            print("Modello caricato correttamente.")
        except Exception as e:
            print(f"Errore caricamento modello : {e}")


        # OPTIMIZER
        if 'optimizer' in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint['optimizer'])
                print("Stato dell'optimizer caricato correttamente.")
            
                for i, g in enumerate(optimizer.param_groups):
                    print(f"   - LR gruppo {i}: {g['lr']}")
            except Exception as e:
                print("Errore nel caricamento dell'optimizer:", e)
        else:
            print("Nessuno stato dell'optimizer trovato nel checkpoint.")
    
        # LR SCHEDULER
        if 'lr_scheduler' in checkpoint:
            try:
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
                print("Stato del lr_scheduler caricato correttamente.")
            except Exception as e:
                print("Errore nel caricamento del lr_scheduler:", e)
        else:
            print("Nessuno stato del lr_scheduler trovato nel checkpoint.")
    
        # EPOCH
        start_epoch = checkpoint.get('epoch', 0)
        print(f"Riprendo da epoca : {start_epoch}\n***************************************************")

    if args.eval:
        if utils.is_main_process():
            print("📄 test result")
            test_stats = evaluate(args, dev_dataloader, model, model_without_ddp)
        return

    print(f"Start training for {args.epochs} epochs")

    start_time = time.time()

    for epoch in range(start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        train_stats = train_one_epoch(args, model, train_dataloader, optimizer, epoch, model_without_ddp=model_without_ddp)

        if args.output_dir:
            checkpoint_paths = [output_dir / f'checkpoint_{epoch}.pth']
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': get_requires_grad_dict(model_without_ddp),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch + 1,
                }, checkpoint_path)

        test_stats = evaluate(args, dev_dataloader, model, model_without_ddp)
        print(f"BLEU-4 of the network on the {len(dev_dataloader)} dev videos: {test_stats['bleu4']:.2f}")

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def train_one_epoch(args, model, data_loader, optimizer, epoch, model_without_ddp):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = f'Epoch: [{epoch}/{args.epochs}]'
    print_freq = 10
    optimizer.zero_grad()

    for step, (src_input, tgt_input) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        if (step + 1) % args.quick_break == 0:
            if args.output_dir:
                output_dir = Path(args.output_dir)
                checkpoint_paths = [output_dir / f'checkpoint.pth']
                for checkpoint_path in checkpoint_paths:
                    utils.save_on_master({
                        'model': get_requires_grad_dict(model_without_ddp),
                        'optimizer': optimizer.state_dict(),
                        'epoch': epoch + 1,
                    }, checkpoint_path)

        for key in src_input.keys():
            if isinstance(src_input[key], torch.Tensor):
                src_input[key] = src_input[key].to(device)

        stack_out = model(src_input, tgt_input)
        total_loss = stack_out['loss']

        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss_value = total_loss.item()
        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            sys.exit(1)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def evaluate(args, data_loader, model, model_without_ddp):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    with torch.no_grad():
        tgt_pres = []
        tgt_refs = []

        for step, (src_input, tgt_input) in enumerate(metric_logger.log_every(data_loader, 10, header)):
            for key in src_input.keys():
                if isinstance(src_input[key], torch.Tensor):
                    src_input[key] = src_input[key].to(device)

            stack_out = model(src_input, tgt_input)
            total_loss = stack_out['loss']
            metric_logger.update(loss=total_loss.item())

            output = model_without_ddp.generate(stack_out, max_new_tokens=100, num_beams=4)

            for i in range(len(output)):
                tgt_pres.append(output[i])
                tgt_refs.append(tgt_input['gt_sentence'][i])

    tokenizer = model_without_ddp.mt5_tokenizer
    padding_value = tokenizer.eos_token_id

    pad_tensor = torch.ones(150 - len(tgt_pres[0])).to(device) * padding_value
    tgt_pres[0] = torch.cat((tgt_pres[0], pad_tensor.long()), dim=0)
    tgt_pres = pad_sequence(tgt_pres, batch_first=True, padding_value=padding_value)
    tgt_pres = tokenizer.batch_decode(tgt_pres, skip_special_tokens=True)

    if args.dataset in {'LIS', 'LIS_TEST'}:
        tgt_pres = [' '.join(list(r.replace(" ", '').replace("\n", ''))) for r in tgt_pres]
        tgt_refs = [' '.join(list(r.replace("，", ',').replace("？", "?").replace(" ", ''))) for r in tgt_refs]

    if all(hyp.strip() == "" for hyp in tgt_pres):
        print("[WARNING] Tutte le ipotesi sono vuote, salto il calcolo delle metriche.")
        return {"bleu4": 0.0, "rouge": 0.0, "loss": 0.0}

    bleu_dict, rouge_score = translation_performance(tgt_refs, tgt_pres)
    for k, v in bleu_dict.items():
        metric_logger.meters[k].update(v)
    metric_logger.meters['rouge'].update(rouge_score)

    metric_logger.synchronize_between_processes()
    print('* BLEU-4 {top1.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.bleu4, losses=metric_logger.loss))

    if utils.is_main_process() and utils.get_world_size() == 1 and args.eval:
        with open(args.output_dir + '/tmp_pres.txt', 'w') as f:
            for line in tgt_pres:
                f.write(line + '\n')
        with open(args.output_dir + '/tmp_refs.txt', 'w') as f:
            for line in tgt_refs:
                f.write(line + '\n')

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


if __name__ == '__main__':
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    parser = argparse.ArgumentParser('Uni-Sign scripts', parents=[utils.get_args_parser()])
    parser.add_argument('--resume', default='', help='path al checkpoint da cui riprendere il pre-training')
    args = parser.parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args)


"""
!python3.9 /content/Uni-Sign-LIS/pre_training_v2.py \
  --batch-size 4 \
  --gradient-accumulation-steps 1 \
  --epochs 10 \                      # continua da 5 a 10
  --opt AdamW \
  --lr 3e-4 \
  --quick_break 2048 \
  --output_dir ./out/stage1_pretraining \
  --dataset LIS \
  --resume ./out/stage1_pretraining/checkpoint_4.pth

"""