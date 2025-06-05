
from pickletools import optimize
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
##############
from torch.utils.data import RandomSampler, SequentialSampler
##############

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

    print("\nargs : \n",args)
    utils.set_seed(args.seed)

    print(f"Creating dataset:")
    train_data = LIS_Dataset(path=train_label_paths[args.dataset], 
                                  args=args, phase='train')
    print(train_data)
    
    
    train_sampler = RandomSampler(train_data)
    
    train_dataloader = DataLoader(train_data,
                                 batch_size=args.batch_size, 
                                 num_workers=args.num_workers, 
                                 collate_fn=train_data.collate_fn,
                                 sampler=train_sampler, 
                                 pin_memory=args.pin_mem,
                                 drop_last=True)


    dev_data = LIS_Dataset(path=dev_label_paths[args.dataset], 
                                args=args, phase='dev')
    print(dev_data)
   
    dev_sampler = SequentialSampler(dev_data)
    
    dev_dataloader = DataLoader(dev_data,
                                 batch_size=args.batch_size,
                                 num_workers=args.num_workers, 
                                 collate_fn=dev_data.collate_fn,
                                 sampler=dev_sampler, 
                                 pin_memory=args.pin_mem)

    print(f"Creating model:")
    model = Uni_Sign(
                    args=args,
                    )
  
    
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
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    n_parameters = utils.count_parameters_in_MB(model_without_ddp)
    print(f'number of params: {n_parameters}M')

    optimizer = create_optimizer(args, model_without_ddp)
    
    if args.quick_break <= 0:
        args.quick_break = len(train_dataloader)

    lr_scheduler = get_scheduler(
                name='cosine',
                optimizer=optimizer,
                num_warmup_steps=int(args.warmup_epochs * len(train_dataloader)/args.gradient_accumulation_steps),
                num_training_steps=int(args.epochs * len(train_dataloader)/args.gradient_accumulation_steps),
            )
    
    """
    model, optimizer, lr_scheduler = utils.init_deepspeed(args, model, optimizer, lr_scheduler)
    model_without_ddp = model.module.module
    """
    ##################################
    model_without_ddp = model
    ##################################
    # print(model_without_ddp)
    print(optimizer)

    output_dir = Path(args.output_dir)

    start_time = time.time()
    max_accuracy = 0

    if args.eval:
        if utils.is_main_process():
            print("📄 test result")
            test_stats = evaluate(args, dev_dataloader, model, model_without_ddp)

        return 
    print(f"Start training for {args.epochs} epochs")

    for epoch in range(0, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        
        train_stats = train_one_epoch(args, model, train_dataloader, optimizer, epoch, model_without_ddp=model_without_ddp)

        if args.output_dir:
            checkpoint_paths = [output_dir / f'checkpoint_{epoch}.pth']
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': get_requires_grad_dict(model_without_ddp),
                }, checkpoint_path)
        test_stats = evaluate(args, dev_dataloader, model, model_without_ddp)
        print(f"BLEU-4 of the network on the {len(dev_dataloader)} dev videos: {test_stats['bleu4']:.2f}")

        if max_accuracy < test_stats["bleu4"]:
            max_accuracy = test_stats["bleu4"]
            if args.output_dir and utils.is_main_process():
                checkpoint_paths = [output_dir / 'best_checkpoint.pth']
                for checkpoint_path in checkpoint_paths:
                    utils.save_on_master({
                        'model': get_requires_grad_dict(model_without_ddp),
                    }, checkpoint_path)
            
        print(f'Max BLEU-4: {max_accuracy:.2f}%')
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
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
    print_freq = 10
    optimizer.zero_grad()

    """
    target_dtype = None
    if model.bfloat16_enabled():
        target_dtype = torch.bfloat16
    """
    for step, (src_input, tgt_input) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        if (step + 1) % args.quick_break == 0:
            if args.output_dir:
                output_dir = Path(args.output_dir)
                checkpoint_paths = [output_dir / f'checkpoint.pth']
                for checkpoint_path in checkpoint_paths:
                    utils.save_on_master({
                        'model': get_requires_grad_dict(model_without_ddp),
                    }, checkpoint_path)

        """if target_dtype != None:"""
        for key in src_input.keys():
            if isinstance(src_input[key], torch.Tensor):
                src_input[key] = src_input[key].to(device)
                

        stack_out = model(src_input, tgt_input)
        
        total_loss = stack_out['loss']
        """
        model.backward(total_loss)
        model.step()
        """

        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss_value = total_loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)
            
        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    return  {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def evaluate(args, data_loader, model, model_without_ddp):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    """
    target_dtype = None
    if model.bfloat16_enabled():
        target_dtype = torch.bfloat16
    """
        
    with torch.no_grad():
        tgt_pres = []
        tgt_refs = []
 
        for step, (src_input, tgt_input) in enumerate(metric_logger.log_every(data_loader, 10, header)):
            """if target_dtype != None:"""


            print(f"[DEBUG] STEP {step} - Input al modello:")
            for key, value in src_input.items():
                if isinstance(value, torch.Tensor):
                    print(f"{key}: shape {value.shape}, dtype {value.dtype}")

            for key in src_input.keys():
                if isinstance(src_input[key], torch.Tensor):
                    src_input[key] = src_input[key].to(device)
                    
            
            stack_out = model(src_input, tgt_input)
            total_loss = stack_out['loss']
            metric_logger.update(loss=total_loss.item())
        
            output = model_without_ddp.generate(stack_out, 
                                                max_new_tokens=100, 
                                                num_beams = 4,
                        )
            
            print(f"\n[DEBUG] Output grezzo (token IDs):")
            for i, out in enumerate(output):
                print(f"Output[{i}]: {out.tolist()}")

            for i in range(len(output)):
                tgt_pres.append(output[i])
                tgt_refs.append(tgt_input['gt_sentence'][i])

    tokenizer = model_without_ddp.mt5_tokenizer
    padding_value = tokenizer.eos_token_id

    # DEBUG 3 - Token speciali
    print("\n[DEBUG] Token speciali del tokenizer:")
    print(f"[DEBUG] pad_token: {tokenizer.pad_token} ({tokenizer.pad_token_id})")
    print(f"[DEBUG] eos_token: {tokenizer.eos_token} ({tokenizer.eos_token_id})")

    # DEBUG 4 - Controllo contenuto sequenze
    print("\n[DEBUG] Verifica lunghezza sequenze e token speciali:")
    for i, seq in enumerate(tgt_pres):
        token_list = seq.tolist()
        print(f"Seq {i}: len={len(token_list)}, tokens={token_list}")
        if all(t in [tokenizer.pad_token_id, tokenizer.eos_token_id] for t in token_list):
            print(f"Seq {i} contiene solo token speciali.")
    
    pad_tensor = torch.ones(150 - len(tgt_pres[0])).to(device) * padding_value

    tgt_pres[0] = torch.cat((tgt_pres[0],pad_tensor.long()),dim = 0)

    tgt_pres = pad_sequence(tgt_pres,batch_first=True,padding_value=padding_value)

    # DEBUG 5 - Decodifica output
    print("\n[DEBUG] Decodifica delle predizioni:")
    for i, seq in enumerate(tgt_pres):
        decoded = tokenizer.decode(seq, skip_special_tokens=True)
        print(f"{i}: '{decoded}'")

    tgt_pres = tokenizer.batch_decode(tgt_pres, skip_special_tokens=True)


    # DEBUG 6 - Controllo riferimenti
    print("\n[DEBUG] Riferimenti (target):")
    for i, ref in enumerate(tgt_refs):
        print(f"{i}: '{ref}'")

            
    if args.dataset == 'CSL_News' or args.dataset == 'LIS':
        tgt_pres = [' '.join(list(r.replace(" ",'').replace("\n",''))) for r in tgt_pres]
        tgt_refs = [' '.join(list(r.replace("，", ',').replace("？","?").replace(" ",''))) for r in tgt_refs]

    # DEBUG 7 Fallback se tutte le predizioni sono vuote 
    if all(hyp.strip() == "" for hyp in tgt_pres):
        print("[WARNING] Tutte le ipotesi sono vuote, salto il calcolo delle metriche.")
        return {"bleu4": 0.0, "rouge": 0.0, "loss": 0.0}
    

    bleu_dict, rouge_score = translation_performance(tgt_refs, tgt_pres)
    for k,v in bleu_dict.items():
        metric_logger.meters[k].update(v)
    metric_logger.meters['rouge'].update(rouge_score)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* BLEU-4 {top1.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.bleu4, losses=metric_logger.loss))
    
    if utils.is_main_process() and utils.get_world_size() == 1 and args.eval:
        with open(args.output_dir+'/tmp_pres.txt','w') as f:
            for i in range(len(tgt_pres)):
                f.write(tgt_pres[i]+'\n')
        with open(args.output_dir+'/tmp_refs.txt','w') as f:
            for i in range(len(tgt_refs)):
                f.write(tgt_refs[i]+'\n')

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


if __name__ == '__main__':
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    parser = argparse.ArgumentParser('Uni-Sign scripts', parents=[utils.get_args_parser()])
    args = parser.parse_args()
    
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args)

