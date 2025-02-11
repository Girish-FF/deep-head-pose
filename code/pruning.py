import torch
import time
import copy
import torch.nn as nn
import torch.nn.utils.prune as prune
from hopenet import Hopenet
import torchvision
from torchprofile import profile_macs

############ FUNCTIONS #####################################
def get_num_parameters(model: nn.Module, count_nonzero_only=False) -> int:
    """
    calculate the total number of parameters of model
    :param count_nonzero_only: only count nonzero weights
    """
    num_counted_elements = 0
    for param in model.parameters():
        if count_nonzero_only:
            num_counted_elements += param.count_nonzero()
        else:
            num_counted_elements += param.numel()
    return num_counted_elements

def get_model_size(model: nn.Module, data_width=32, count_nonzero_only=False) -> int:
    """
    calculate the model size in bits
    :param data_width: #bits per element
    :param count_nonzero_only: only count nonzero weights
    """
    return get_num_parameters(model, count_nonzero_only) * data_width

Byte = 8
KiB = 1024 * Byte
MiB = 1024 * KiB
GiB = 1024 * MiB
############################################################

# Instantiate the Hopenet model
Hnet = Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)
saved_state_dict = torch.load("hopenet_robust_alpha1.pkl")
Hnet.load_state_dict(saved_state_dict)
model = copy.deepcopy(Hnet)

# Function to calculate the number of parameters in a model
# def count_parameters(model):
#     return sum(p.numel() for p in model.parameters())


# Profile MACs and inference time for a dummy input
dummy_input = torch.randn(1, 3, 224, 224)  # Define a dummy input shape
macs = profile_macs(model, dummy_input)
print(f"MACs for the model: {macs}")

# Measure inference time for a dummy input
# with torch.autograd.profiler.profile(use_cuda=False) as prof:
start = time.time()
output = model(dummy_input)
print(f"inference time for dummy input: {(time.time()-start)*1000}")
print(f"Model size before pruning: {get_model_size(model)/MiB:.2f}")
# print(prof.key_averages().table(sort_by="self_cpu_time_total"))
# Print the number of parameters in the model before pruning
print(f"Number of parameters before pruning: {get_num_parameters(model, False)}")
print(f"Number of non-zero parameters before pruning: {get_num_parameters(model, True)}")
# for i, (name, module) in enumerate(model.named_modules()):
#     # print(name)
#     if isinstance(module, torch.nn.Conv2d):
#         print("#"*10,i+1,"#"*10)
#         print(name)
#         print(module)
#         print("#"*100)

# mod1 = model.conv1
# # print("Parameters before pruning")
# # print(list(mod1.named_parameters()))
# print(" Before Pruning ")
# for name, param in mod1.named_parameters():
#     # if name in ['bias']:
#         print(name, param.size(), param.count_nonzero())
# for name, param in mod1.named_buffers():
#     # if name in ['bias']:
#         print(name, param.size(), param.count_nonzero())

# prune.ln_structured(mod1, "weight", 0.5, 1, 0)
# # prune.l1_unstructured(mod1, "weight", 0.5)
# # prune.l1_unstructured(mod1, "bias", 0.5)

# print(" After Pruning ")        
# for name, param in mod1.named_parameters():
#     # if name in ['bias']:
#         print(name, param.size(), param.count_nonzero())
# for name, param in mod1.named_buffers():
#     # if name in ['bias']:
#         print(name, param.size(), param.count_nonzero())

# prune.remove(mod1, "weight")
# # prune.remove(mod1, "bias")

# for name, param in mod1.named_parameters():
#     # if name in ['bias']:
#         print(name, param.size(), param.count_nonzero())
# for name, param in mod1.named_buffers():
#     # if name in ['bias']:
#         print(name, param.size(), param.count_nonzero())

# # print("Parameters after pruning")
# # print(list(mod1.named_parameters()))
# for name, param in mod1.named_parameters():
#     if name=="weight":
#         # idx = torch.argwhere(param!=0)
#         idx = torch.nonzero(param.sum(dim=(1,2,3)))
#         print(idx.size())

# for name,param in mod1.named_parameters():
#     # print(param[idx].T) 
#     if name=="weight":mod1.weight = nn.Parameter(param[idx].squeeze(1))
#     elif name=="bias":mod1.bias = nn.Parameter(param[idx].squeeze(1))

# print("Parameters after pruning and slicing")
# # print(list(mod1.named_parameters()))
# for name, param in mod1.named_parameters():
#     # if name in ['bias']:
#         print(name, param.size(), param.count_nonzero())

# ###########################################
# for name, module in model.named_modules():
#     if isinstance(module, torch.nn.Conv2d):
#         prune.ln_structured(module, "weight", 0.5, 2, 0)
#         # prune.remove(module, "weight")
#     elif isinstance(module, torch.nn.BatchNorm2d):
#         prune.l1_unstructured(module, "weight", 0.5)

    
# # Define pruning parameters
# pruning_params = {
#     'amount': 0.5,  # Specify the amount of pruning (e.g., 50%)
#     'prune_method': prune.L1Unstructured,  # Choose the pruning method (e.g., L1Unstructured)
# }

# # Apply channel-wise pruning to the model
# for name, module in model.named_modules():
#     if isinstance(module, nn.Conv2d):
#         prune.ln_structured(module, name='weight', amount=pruning_params['amount'], n=1, dim=0)  # Prune along channels
#         prune.remove(module, "weight")

# # Verify the sparsity pattern of the pruned model's weights
# for name, module in model.named_modules():
#     if isinstance(module, nn.Conv2d):
#         print(f"Sparsity pattern for {name}: {torch.sum(module.weight == 0).item() / module.weight.nelement()}")

# print(f"Model size after pruning: {get_model_size(model)/MiB:.2f}")
# # Print the number of parameters in the pruned model
# print(f"Number of parameters before pruning: {get_num_parameters(model, False)}")
# print(f"Number of non-zero parameters before pruning: {get_num_parameters(model, True)}")

# # Profile MACs and inference time for a dummy input
# dummy_input = torch.randn(1, 3, 224, 224)  # Define a dummy input shape
# macs = profile_macs(model, dummy_input)
# print(f"MACs for the pruned model: {macs}")

# # Measure inference time for a dummy input
# with torch.autograd.profiler.profile(use_cuda=False) as prof:
#     output = model(dummy_input)
# print(prof.key_averages().table(sort_by="self_cpu_time_total"))

# print()