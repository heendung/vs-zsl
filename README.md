# Revisiting Document Representations for Large-Scale <br/> Zero-Shot Learning

Official implementation for the paper [Revisiting Document Representations for Large-Scale Zero-Shot Learning](https://arxiv.org/abs/2104.10355) <br/> by Jihyung Kil, Wei-Lun Chao, NAACL 2021.

<p align="center">
  <img src="./figs/zsl_app.png" width="80%" height="5%"></center>
</p>


<strong>[Update 03/20/22]:</strong>  Add environment, visual features and labels of our split, and codes for weighted average semantic represenations and DeViSE<sup>*</sup>.

## Environment
Import our conda environment:
```
conda env create -f ZSL_fv.yaml
conda activate ZSL_fv
```


## Dataset
#### Wikipedia Documents
The (non) filtered Wikipedia sentences are available on [here](https://drive.google.com/drive/u/0/folders/1oFo4CsYcU0t7EOb9DwJX26JQgT_i9BIw). Please refer to the related README for more details.

#### Semantic Representations
Extract the semantic representations from the (non) filtered sentences:

```
CUDA_VISIBLE_DEVICES=0 python3 get_sem_rep.py --wiki_set data/21k_true_wiki_sents_vis_sec_clu --pool avg_pool --flt vis_sec_clu --max_seq_len 64 --max_sent all
```

#### Visual Features
We use the [ResNet visual features](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/zero-shot-learning/zero-shot-learning-the-good-the-bad-and-the-ugly) (He et al., 2016) provided by (Xian et al., 2018a).

[Visual features and labels](https://buckeyemailosu-my.sharepoint.com/:f:/g/personal/kil_5_buckeyemail_osu_edu/Ep5hL2cYY7dNoMXGh--dXisBirfkyQ4TmV1orWPl73xW2w?e=jJa6Wp) of our 1K/2-Hop/3-Hop/ALL split

#### Visual Attributes
For AwA2 and aPY, we use [visual attributes](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/zero-shot-learning/zero-shot-learning-the-good-the-bad-and-the-ugly) provided by (Xian et al., 2018a). 

#### Data Split
Please refer to README on [here](https://drive.google.com/drive/u/0/folders/1oFo4CsYcU0t7EOb9DwJX26JQgT_i9BIw) how to split ImageNet into our settings (i.e., 2-Hop, 3-Hop, ALL).

For AwA2 and aPY, we follow the [proposed split](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/zero-shot-learning/zero-shot-learning-the-good-the-bad-and-the-ugly) provided by (Xian et al., 2018a). <br/><br/>

## Code
We leverage three Zero-Shot Learning models in our experiments:
* DeViSE (Frome et al., 2013): DeViSE and DeViSE<sup>*</sup> are based on the implementation from [here](https://github.com/TristHas/GOZ).
* EXEM (Changpinyo et al., 2020): We use its official [implementation](https://github.com/pujols/Zero-shot-learning-journal).
* HVE (Liu et al., 2020): The official implementation can be found on [here](https://github.com/ShaoTengLiu/Hyperbolic_ZSL).<br/><br/>

Weighted Average Semantic Represenations (<strong>a<sub>c</sub></strong>):

 * Train <strong>b<sub>ψ</sub></strong> by minimizing the objectives (4) and (5) in the paper (e.g., ε: 0.95, τ: 0.96, BERT<sub>p-<strong>w</strong></sub>, Vis<sub>sec-clu</sub>):
 
    - [vis_sec_clu_sem_rep_pre_trained_fv.pt](https://buckeyemailosu-my.sharepoint.com/:f:/g/personal/kil_5_buckeyemail_osu_edu/Ep5hL2cYY7dNoMXGh--dXisBirfkyQ4TmV1orWPl73xW2w?e=jJa6Wp): pre-trained sentence representations filtered by Vis<sub>sec-clu</sub>
    - [vis_sec_clu_avg_sem_rep_pre_trained_fv/bert_sem.pt](https://buckeyemailosu-my.sharepoint.com/:f:/g/personal/kil_5_buckeyemail_osu_edu/Ep5hL2cYY7dNoMXGh--dXisBirfkyQ4TmV1orWPl73xW2w?e=jJa6Wp): pre-trained average semantic represenations filtered by Vis<sub>sec-clu</sub>  
```
python3 train_b_psi_comb_fv.py --output  fine_tune_b_psi --tau 0.96 --type pre_trained --train True --discri discriminate --epochs 5 --split all_data --batch_size 512 --eps 0.95 --lr 1e-4 --sent_rep vis_sec_clu_sem_rep_pre_trained_fv.pt --avg_rep vis_sec_clu_avg_sem_rep_pre_trained_fv/bert_sem.pt
```
   


 * Obtain the weighted average semantic represenations (<strong>a<sub>c</sub></strong>) after training:
```
python3 train_b_psi_comb_fv.py --output fine_tune_b_psi --tau 0.96 --type pre_trained --train False --discri discriminate --epochs 1  --split all_data --batch_size 768 --eps 0.95 
``` 
<br/>
Train DeViSE<sup>*</sup>:

```
python3 DeVise_star.py --data_dir /local/scratch/jihyung --output devise_result --tau 0.96 --type pre_trained --eps 0.95 --bs 768 --split all_data --marg 0.2 --lr 0.0004 --num_epochs 50 --sem_type bert_p_w --sem_rep fine_tune_b_psi_eps_0.95_tau_0.96_pre_trained_fv/all_data_semantic_rep_after_train_epochs_1.pt
``` 

## Citation
If you find the code and data useful, please cite the following paper:
```
@inproceedings{kil2021revisiting,
  title={Revisiting Document Representations for Large-Scale Zero-Shot Learning},
  author={Kil, Jihyung and Chao, Wei-Lun},
  booktitle={Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies},
  pages={3117--3128},
  year={2021}
}
```
