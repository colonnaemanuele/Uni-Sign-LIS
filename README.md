<h3 align="center"><a href="" style="color:#9C276A">
Uni-Sign: Toward Unified Sign Language Understanding at Scale</a></h3>


[![arXiv](https://img.shields.io/badge/Arxiv-2501.15187-AD1C18.svg?logo=arXiv)](https://arxiv.org/abs/2501.15187) 
[![CSL-Dataset](https://img.shields.io/badge/HuggingFace🤗-%20CSL%20News-blue.svg)](https://huggingface.co/datasets/ZechengLi19/CSL-News)
[![CSL-Dataset](https://img.shields.io/badge/BaiDu☁-%20CSL%20News-green.svg)](https://pan.baidu.com/s/17W6kIreNMHYtD4y2llKmDg?pwd=ncvo) 

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/uni-sign-toward-unified-sign-language/sign-language-recognition-on-ms-asl)](https://paperswithcode.com/sota/sign-language-recognition-on-ms-asl?p=uni-sign-toward-unified-sign-language)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/uni-sign-toward-unified-sign-language/sign-language-recognition-on-wlasl100)](https://paperswithcode.com/sota/sign-language-recognition-on-wlasl100?p=uni-sign-toward-unified-sign-language)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/uni-sign-toward-unified-sign-language/sign-language-recognition-on-wlasl-2000)](https://paperswithcode.com/sota/sign-language-recognition-on-wlasl-2000?p=uni-sign-toward-unified-sign-language)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/uni-sign-toward-unified-sign-language/sign-language-recognition-on-csl-daily)](https://paperswithcode.com/sota/sign-language-recognition-on-csl-daily?p=uni-sign-toward-unified-sign-language)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/uni-sign-toward-unified-sign-language/gloss-free-sign-language-translation-on-csl)](https://paperswithcode.com/sota/gloss-free-sign-language-translation-on-csl?p=uni-sign-toward-unified-sign-language)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/uni-sign-toward-unified-sign-language/gloss-free-sign-language-translation-on-2)](https://paperswithcode.com/sota/gloss-free-sign-language-translation-on-2?p=uni-sign-toward-unified-sign-language)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/uni-sign-toward-unified-sign-language/gloss-free-sign-language-translation-on-3)](https://paperswithcode.com/sota/gloss-free-sign-language-translation-on-3?p=uni-sign-toward-unified-sign-language)
</h5>

![Uni-Sign](docs/framework.png)


## Installazione
E' suggerito creare un nuovo ambiente conda.
```bash
# create environment
conda create --name Uni-Sign python=3.9
conda activate Uni-Sign
# install other relevant dependencies
pip install -r requirements.txt
```

## Preparazione Dataset

A partire dai video mp4 originali e dalle annotazioni in formato txt : 
1. Ritagliare i video per escludere i bordi neri
2. Estrarre i keypoints usando il modello RTMPose-x da MMPose per estrarre i keypoints
3. Convertire la struttura del file json contenente i keypoints in modo da essere compatibile con il modello Uni-Sign, e convertire il risultante file json in pkl
4. Allineare i video con le annotazioni, creando un file json `LIS_Labels` con la seguente struttura :
   ```json
   [
     {
        "video": "{giorno}_{video}.mp4",
        "pose": "{giorno}_{video}.pkl",
        "text": "annotazione video"
    },
   ]
   ```
   e posizionarlo in `/Uni-Sign/dataset/data`
5. posizionare i video rgb in formato mp4 ritagliati nella directory `/Uni-Sign/dataset/rgb_format`, salvandoli con nome `{giorno}_{video}.mp4`, mentre i file pkl contenenti i keypoints nella directory `/Uni-Sign/dataset/pose_format`, salvandoli con nome `{giorno}_{video}.pkl`

Per far ciò è possibile usare il notebook `./pose_extraction.ipynb` (i file andranno rinominati e posizionati manualmente!)
 
La struttura finale ella directory del dataset deve essere la seguente : 
```
/Uni-Sign/dataset/
├── LIS
│   ├── data 
│   │   └── LIS_Labels.json
│   │
│   ├── rgb_format  # mp4 format
│   │   ├── 12_10_2023_0.mp4 
│   │   ├── 12_10_2023_1.mp4
│   │   └── ...
│   │ 
│   └── pose_format # pkl format
│       ├── 12_10_2023_0.pkl 
│       ├── 12_10_2023_1.pkl
│       └── ...
│   
```

## Pre-trained Weights
Scaricare i pesi del modello [mt5-base](https://huggingface.co/google/mt5-base) e posizionarli in `./pretrained_weight/mt5-base`

## Addestramento e Valutazione

Tutti gli script devono essere eseguiti nella directory Uni-Sign .

### Addestramento
**Stage 1**: pose-only pre-training.
```bat
./script/LIS_stage1.bat
```
**Stage 2**: RGB-pose pre-training.
```bat
./script/LIS_stage2.bat
```
**Stage 3**: downstream fine-tuning.
```bat
./script/LIS_stage3.bat
```

### Valutazione
Dopo aver completato lo stage 3 di fine-tuning, per valutare le performance : 
```bat
./script/LIS_eval_stage3.bat
```


## 👍 Acknowledgement
The codebase of Uni-Sign is adapted from [GFSLT-VLP](https://github.com/zhoubenjia/GFSLT-VLP), while the implementations of the pose/temporal encoders are derived from [CoSign](https://openaccess.thecvf.com/content/ICCV2023/papers/Jiao_CoSign_Exploring_Co-occurrence_Signals_in_Skeleton-based_Continuous_Sign_Language_Recognition_ICCV_2023_paper.pdf). We sincerely appreciate the authors of CoSign for personally sharing their code 🙏. \
We are also grateful for the following projects our Uni-Sign arise from:
* 🤟[SSVP-SLT](https://github.com/facebookresearch/ssvp_slt): a excellent sign language translation framework! 
* 🏃️[MMPose](https://github.com/open-mmlab/mmpose): an open-source toolbox for pose estimation.
* 🤠[FUNASR](https://github.com/modelscope/FunASR): a high-performance speech-to-text toolkit.


## 📑 Citation

```
@article{li2025uni,
  title={Uni-Sign: Toward Unified Sign Language Understanding at Scale},
  author={Li, Zecheng and Zhou, Wengang and Zhao, Weichao and Wu, Kepeng and Hu, Hezhen and Li, Houqiang},
  journal={arXiv preprint arXiv:2501.15187},
  year={2025}
}
```
