import torch

path = "./model_tiny.pth"
ckpt = torch.load(path)
orig_keys = ckpt.keys()
new_ckpt = {}

### stem
for key in orig_keys:
    sp_key = key.split(".")
    if sp_key[0] == "stem":
        new_ckpt[f"downsample_layers.0.{sp_key[1]}.{sp_key[2]}"] = ckpt[key]

### downsamples
for stage_id in range(4):
    for key in orig_keys:
        sp_key = key.split(".")
        if sp_key[0] == "stages" and sp_key[1] == f"{stage_id}" and sp_key[2] == "downsample":
            new_ckpt[f"downsample_layers.{stage_id}.{sp_key[3]}.{sp_key[4]}"] = ckpt[key]

### stages
for stage_id in range(4):
    for key in orig_keys:
        sp_key = key.split(".")
        if sp_key[0] == "stages" and sp_key[1] == f"{stage_id}" and sp_key[2] == "blocks":
            if sp_key[4] == "gamma":
                new_ckpt[f"stages.{stage_id}.{sp_key[3]}.{sp_key[4]}"] = ckpt[key]
            elif sp_key[4] == "mlp":
                n = 1 if sp_key[5] == "fc1" else 2
                new_ckpt[f"stages.{stage_id}.{sp_key[3]}.pointwise_conv{n}.{sp_key[6]}"] = ckpt[key]
            elif sp_key[4] == "norm":
                new_ckpt[f"stages.{stage_id}.{sp_key[3]}.{sp_key[4]}.{sp_key[5]}"] = ckpt[key]
            elif sp_key[4] == "conv_dw":
                st = ".".join(sp_key[5:])
                new_ckpt[f"stages.{stage_id}.{sp_key[3]}.depthwise_conv.{st}"] = ckpt[key]
            else:
                print(sp_key[4])

new_ckpt = {f"backbone.{k}": new_ckpt[k] for k in new_ckpt.keys()}
torch.save(new_ckpt, "model_tiny_mmseg.pth")

