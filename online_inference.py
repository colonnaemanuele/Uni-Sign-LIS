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
    online_data.rgb_data = args.online_video
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
    model.cuda()
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
    model.to(torch.bfloat16)

    inference(online_dataloader, model)


def inference(data_loader, model):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    target_dtype = torch.bfloat16

    with torch.no_grad():
        tgt_pres = []

        for step, (src_input, tgt_input) in enumerate(metric_logger.log_every(data_loader, 10, header)):
            if target_dtype is not None:
                for key in src_input.keys():
                    if isinstance(src_input[key], torch.Tensor):
                        src_input[key] = src_input[key].to(target_dtype).cuda()

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

    pad_tensor = torch.ones(150 - len(tgt_pres[0])).cuda() * padding_value
    tgt_pres[0] = torch.cat((tgt_pres[0], pad_tensor.long()), dim=0)

    tgt_pres = pad_sequence(tgt_pres, batch_first=True, padding_value=padding_value)
    tgt_pres = tokenizer.batch_decode(tgt_pres, skip_special_tokens=True)

    print(f"Prediction result is: {tgt_pres[0]}")


if __name__ == '__main__':
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    parser = argparse.ArgumentParser('Uni-Sign inference script', parents=[utils.get_args_parser()])
    parser.add_argument("--pose_pkl", required=True, help="Path to precomputed pose .pkl file")

    args = parser.parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args)

"""
python run_inference.py \
  --online_video /path/to/video_01.mp4 \
  --pose_pkl /path/to/video_01.pkl \
  --finetune /path/to/best_checkpoint.pth \
  --output_dir ./results \
  --dataset LIS \
  --task SLT
"""