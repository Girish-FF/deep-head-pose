import torch
import copy
import torchvision
from torchvision import transforms
from hopenet import Hopenet # Assuming you have a Hopenet model implementation
from torch.ao.quantization import get_default_qconfig
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
from torch.ao.quantization import QConfigMapping, QConfigAny, QConfig, float16_static_qconfig
import datasets


# Load the pre-trained Hopenet model
pretrained_model_path = "hopenet_alpha2.pkl"
model = Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)
model.load_state_dict(torch.load(pretrained_model_path, weights_only=True))
# torch.save(model.state_dict(), "hopenet.pth")

# model = torch.load("output\\snapshots\\Pruned90HN_Retrain.pth")


new_model = copy.deepcopy(model)

new_model.to("cpu").eval()

transformations = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
data_dir = "C:\\Users\\Girish\\Desktop\\FF-Projects\\Datasets\\testFaceNonface\\Face"
filename_list = "filenames_custom.txt"
pose_dataset = datasets.custom(data_dir, filename_list, transformations)

test_loader = torch.utils.data.DataLoader(dataset=pose_dataset,
                                            batch_size=128,)


# # Specify the quantization configuration
# quantization_config = torch.quantization.get_default_qconfig('fbgemm')
# print(quantization_config)

# # Apply post-training static quantization to the model
# quantized_model = torch.quantization.quantize_dynamic(
#     new_model, qconfig_spec=quantization_config)

# # Save the quantized model
# quantized_model_path = "quantized_hopenet.pth"
# torch.jit.save(quantized_model, quantized_model_path)

qconfig = get_default_qconfig("x86")
qconfig_mapping = QConfigMapping().set_global(qconfig)
# qconfig_mapping = QConfigMapping().set_global(float16_static_qconfig)
# Modify the qconfig to use 16-bit precision
# qconfig = QConfig(
#     activation=QConfigDynamic(dtype=torch.float16),
#     weight=QConfigDynamic(dtype=torch.float16)
# )

# qconfig_mapping = {"": qconfig}
# qconfig_mapping = {"": float16_static_qconfig}

example_inputs = torch.randn(4,3,224,224)

prepared_model = prepare_fx(new_model, qconfig_mapping, example_inputs)
print(prepared_model.graph)

def calibrate(model, data_loader):
    with torch.inference_mode():
        for image, target in data_loader:
            model(image)
            break
calibrate(prepared_model, test_loader)  # run calibration on sample data

quantized_model = convert_fx(prepared_model)
print(quantized_model)
# Save the quantized model
quantized_model_path = "Hopenet_alpha2_Q8Bit.pth"
torch.jit.save(torch.jit.script(quantized_model), quantized_model_path)
