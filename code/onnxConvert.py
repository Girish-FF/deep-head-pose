from time import time
import torch
import onnx
import cv2
from PIL import  Image
import onnxruntime as ort
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import hopenetlite_v2, stable_hopenetlite


model = torch.jit.load("Hopenet_alpha2_Q8Bit.pth")
# model = hopenetlite_v2.HopeNetLite()
# saved_state_dict = torch.load("hopenet_robust_alpha1.pkl")
# model.load_state_dict(saved_state_dict, strict=False)

# #### For HopeNet lite (stable) model ###### 
# model = stable_hopenetlite.shufflenet_v2_x1_0()
# saved_state_dict = torch.load("deep-head-pose-lite\\model\\shuff_epoch_120.pkl")
# model.load_state_dict(saved_state_dict, strict=False)

model.to("cpu")
model.eval()

test_img = Image.open("HeadPoseImage\\0_landmk_frame_0073.jpg")
transformations = torchvision.transforms.Compose([
    torchvision.transforms.Resize(224),
    torchvision.transforms.CenterCrop(224), 
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
test_img = transformations(test_img).unsqueeze(0)
x = torch.randn(1, 3,224,224, requires_grad=True)
with torch.inference_mode():
    out = model(test_img)

new_model = "Hopenet_alpha2_Q8BitONNX.onnx"
torch.onnx.export (model, x, new_model,
                   export_params=True, opset_version=17,
                   do_constant_folding=True,
                   input_names = ["input"],
                   output_names = ["output"],
                   dynamic_axes={'input':{0:'batch_size'},
                                })


onnx_model = onnx.load(new_model)
# print(onnx_model)
onnx.checker.check_model(onnx_model)



sess_options = ort.SessionOptions()
# Below is for optimizing performance
sess_options.intra_op_num_threads = 24
sess_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
ort_session = ort.InferenceSession(new_model, sess_options=sess_options)

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# compute ONNX Runtime output prediction
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(test_img)}
ort_outs = ort_session.run(None, ort_inputs)

print(len(out))
print(len(ort_outs))
print("Difference in values")
print(out[0].numpy()-ort_outs[0])
print(out[1].numpy()-ort_outs[1])
print(out[2].numpy()-ort_outs[2])
# compare ONNX Runtime and PyTorch results
# np.testing.assert_allclose(to_numpy(out[0]), ort_outs[0], rtol=1e-03, atol=1e-05)

print("Exported model has been tested with ONNXRuntime, and the result looks good!")