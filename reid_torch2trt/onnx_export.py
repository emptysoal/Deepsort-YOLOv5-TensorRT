import torch
from model import Net

model_path = "./ckpt.t7"
onnx_file = "deepsort.onnx"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = Net(reid=True)
state_dict = torch.load(model_path, map_location="cpu")["net_dict"]
model.load_state_dict(state_dict)
model.to(device)
model.eval()

input_names = ['input']
output_names = ['output']

input_tensor = torch.randn(1, 3, 128, 64, device=device)


torch.onnx.export(
    model,
    input_tensor,
    onnx_file,
    input_names=input_names,
    output_names=output_names,
    do_constant_folding=True,
    verbose=True,
    keep_initializers_as_inputs=True,
    opset_version=12,
    dynamic_axes={"input": {0: "nBatchSize"}, "output": {0: "nBatchSize"}}
)
