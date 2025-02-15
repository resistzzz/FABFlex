## Fast and Accurate Blind Flexible Docking 

### Overview:  
```
|---baselines       # directory about the data of baselines
|---binddataset     # directory of the preprocessed dataset
|---ckpt            # directory to store the trained ckpts
|---fabflex         # codes about the FABFlex model
|   |---inference.py    # run entire inference of FABFlex on test sets
|   |---inference_post_optim.py     # run inference with post-optimization
|   |---inference_protein.py        # run inference to output protein
|   |---inference_without_post_optim.py     # run inference without post-optimization
|   |---main_fatwo_joint.py         # training for model
|   |---main_pro_joint.py           # pretraining for pocket docking module
|---requirements.txt    # environment file
```

### How to make inference using FABFlex:
1. download the datasets.
2. download the ckpt from <a href="https://drive.google.com/drive/folders/1WXhDX1wuYrvtwwEZyZakAy5lxpNcQ0A5?usp=sharing" class="underline" target="_blank">Google Drive</a>
3. install the environment
4. run `fabflex/inference.py` as follows:
```
python fabflex/inference.py \
    --batch_size 4 \
    --root_path ./binddataset \
    --use_iterative --total_iterative 6 \
    --pocket_radius 20 --force_fix_radius --min_pocket_radius 20 \
    --n_iter 1 --mean_layers 5 --coord_scale 5 \
    --addNoise 0 --hidden_size 512 \
    --pocket_pred_hidden_size 128 \
    --geometry_reg_step_size 0.001 --geometry_reg_step 1 \
    --use_ln_mlp --rm_layernorm --mlp_hidden_scale 1 --stage 2
    --ckpt $CKPT_PATH
```
If you want to inference with post-optimization, run `fabflex/inference_post_optim.py` as follows:
```
python fabflex/inference_post_optim.py \
    --batch_size 1 \
    --root_path ./binddataset \
    --use_iterative --total_iterative 6 \
    --pocket_radius 20 --force_fix_radius --min_pocket_radius 20 \
    --n_iter 1 --mean_layers 5 --coord_scale 5 \
    --addNoise 0 --hidden_size 512 \
    --pocket_pred_hidden_size 128 \
    --geometry_reg_step_size 0.001 --geometry_reg_step 1 \
    --use_ln_mlp --rm_layernorm --mlp_hidden_scale 1 --stage 2
    --ckpt $CKPT_PATH
    --out_path $OUT_PATH
```

### TODO  
- [ ] [Code] Clean all related codes
- [x] [Code] Release inference codes of FABFlex
- [ ] [Dataset] Upload preprocessed datasets, it is approximately 22.8 GB
- [ ] [Ckpt] Upload checkpoint
- [x] [Readme] Write a README
- [ ] [Inference] Add a inference code from SMILES and amino sequence



