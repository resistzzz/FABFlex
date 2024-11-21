import torch

TankBind_model = '/root/autodl-tmp/binddataset/ckpt/tankbind_ckpt.pt'
FABind_model = '/root/autodl-tmp/binddataset/ckpt/fabind_best_model.bin'
FABindPlus_model = '/root/autodl-tmp/binddataset/ckpt/best_layer5_ckpt.bin'
DynamicBind_model = '/home/workspace/DynamicBind/workdir/big_score_model_sanyueqi_with_time/ema_inference_epoch314_model.pt'
FABFlex_model = '/root/autodl-tmp/results/epoch_235_Ti6/pytorch_model.bin'

def return_parameters(model_path, model_name=''):
    # 假设你已经有state_dict存储的文件，加载它
    state_dict = torch.load(model_path)

    # 计算模型的参数量
    total_params = sum(param.numel() for param in state_dict.values())

    print(f"{model_name}模型的参数量: {total_params}")


return_parameters(TankBind_model, model_name='TankBind')
return_parameters(FABind_model, model_name='FABind')
return_parameters(FABindPlus_model, model_name='FABind+')
return_parameters(DynamicBind_model, model_name='DynamicBind')
return_parameters(FABFlex_model, model_name='FABFlex')