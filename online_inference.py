import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from models import Uni_Sign
import utils as utils
from datasets import LIS_Dataset_online
from pathlib import Path
from config import *
import argparse
import os
import cv2
import numpy as np
import pickle
from tqdm import tqdm


def main(args):
    print(args)
    utils.set_seed(args.seed)

    with open(args.pose_pkl, 'rb') as f:
        pose_data = pickle.load(f)

    print(f"Creating dataset:")
    online_data = LIS_Dataset_online(args=args)
    online_data.rgb_data = args.rgb_video
    online_data.pose_data = pose_data

    online_sampler = torch.utils.data.SequentialSampler(online_data)
    online_dataloader = DataLoader(
        online_data,
        batch_size=1,
        collate_fn=online_data.collate_fn,
        sampler=online_sampler,
    )

    print(f"Creating model:")
    model = Uni_Sign(args=args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    for name, param in model.named_parameters():
        if param.requires_grad:
            param.data = param.data.to(torch.float32)

    if args.finetune != '':
        print('***********************************')
        print('Load Checkpoint...')
        print('***********************************')
        state_dict = torch.load(args.finetune, map_location='cpu')['model']
        ret = model.load_state_dict(state_dict, strict=True)
        print('Missing keys: \n', '\n'.join(ret.missing_keys))
        print('Unexpected keys: \n', '\n'.join(ret.unexpected_keys))
    else:
        raise NotImplementedError

    model.eval()
    if torch.cuda.is_available():
        model.to(torch.bfloat16)
        target_dtype = torch.bfloat16
    else:
        model.to(torch.float32)
        target_dtype = torch.float32

    inference(online_dataloader, model, target_dtype, device)


def inference(data_loader, model, target_dtype, device):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'


    with torch.no_grad():
        tgt_pres = []

        for step, (src_input, tgt_input) in enumerate(metric_logger.log_every(data_loader, 10, header)):
            if target_dtype is not None:
                for key in src_input.keys():
                    if isinstance(src_input[key], torch.Tensor):
                        src_input[key] = src_input[key].to(target_dtype).to(device)

            stack_out = model(src_input, tgt_input)

            output = model.generate(
                stack_out,
                max_new_tokens=100,
                num_beams=4,
            )

            for i in range(len(output)):
                tgt_pres.append(output[i])

    tokenizer = model.mt5_tokenizer
    padding_value = tokenizer.eos_token_id

    pad_tensor = torch.ones(150 - len(tgt_pres[0])).to(device) * padding_value
    tgt_pres[0] = torch.cat((tgt_pres[0], pad_tensor.long()), dim=0)

    tgt_pres = pad_sequence(tgt_pres, batch_first=True, padding_value=padding_value)
    tgt_pres = tokenizer.batch_decode(tgt_pres, skip_special_tokens=True)

    print(f"Prediction result is: {tgt_pres[0]}")


if __name__ == '__main__':
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    parser = argparse.ArgumentParser('Uni-Sign inference script', parents=[utils.get_args_parser()])
    parser.add_argument("--rgb_video", required=True, help="Path to rgb video prepared .mp4 file")
    parser.add_argument("--pose_pkl", required=True, help="Path to precomputed pose .pkl file")
    

    args = parser.parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args)

"""
rgb_format_path = "/content/drive/MyDrive/LIS3_TEST/rgb_format_resized/08_11_2023_20.mp4"
pose_format_path = "/content/drive/MyDrive/LIS3_TEST/pose_format/08_11_2023_20.pkl"
stage3_best_checkpoint_path = "/content/drive/MyDrive/Uni-Sign-Output/stage3_finetuning_4/checkpoint_0.pth"
output_dir = "./out/results"

!python3.9 /content/Uni-Sign-LIS/online_inference.py\
  --rgb_video {rgb_format_path} \
  --pose_pkl {pose_format_path} \
  --finetune {stage3_best_checkpoint_path} \
  --output_dir {output_dir} \
  --rgb_support 
"""

