import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.quantized as nnq
import onnx

project_root = Path(__file__).parents[1]
sys.path.append(str(project_root))

from layer_info import (
    ShapeParam,
    Conv2DShapeParam,
    LinearShapeParam,
    MaxPool2DShapeParam,
)

from lib.models.vgg import VGG
import network_parser.torch2onnx



def parse_pytorch(model: nn.Module, input_shape=(1, 3, 32, 32)) -> list[ShapeParam]:
    layers = []
    dummy_input = torch.randn(*input_shape)  # Test input
    hooks = []

    def hook_fn(module, inputs, output):
        input_shape = inputs[0].shape
        output_shape = output.shape

        if isinstance(module, (nn.Conv2d, nnq.Conv2d)):  # Parse convolution layer
            layers.append(
                Conv2DShapeParam(
                    N=input_shape[0],  # Batch size
                    H=input_shape[2],   # Input image size
                    W=input_shape[3],  
                    R=module.kernel_size[0], # Kernel size
                    S=module.kernel_size[1],  
                    E=output_shape[2], # Output image size
                    F=output_shape[3],  
                    C=input_shape[1], # Input/output channels
                    M=module.out_channels,  
                    U=module.stride[0], # Stride & Padding
                    P=module.padding[0]  
                )
            )
        elif isinstance(module, (nn.MaxPool2d, nnq.MaxPool2d)):  # Parse max pooling layer
            kernel_size = module.kernel_size if isinstance(module.kernel_size, int) else module.kernel_size[0]
            stride = module.stride if isinstance(module.stride, int) else module.stride[0]
            layers.append(
                MaxPool2DShapeParam(
                    N=input_shape[0], 
                    kernel_size=kernel_size,
                    stride=stride
                )
            )
        elif isinstance(module, (nn.Linear, nnq.Linear)):  # Parse fully connected (linear) layer
            layers.append(
                LinearShapeParam(
                    N=input_shape[0], 
                    in_features=module.in_features, 
                    out_features=module.out_features
                )
            )

    for layer in model.modules():
        if isinstance(layer, (nn.Conv2d, nn.Linear, nn.MaxPool2d, nnq.Conv2d, nnq.Linear, nnq.MaxPool2d)):
            hooks.append(layer.register_forward_hook(hook_fn))

    model(dummy_input)

    for hook in hooks:
        hook.remove()

    for i, layer in enumerate(layers):
        print(f"Layer {i}: {layer.__class__.__name__}")
        print(f"Parameters: {layer.to_dict()}")
        print()

    return layers


def parse_onnx(model: onnx.ModelProto) -> list[ShapeParam]:
    layers = []
    import onnxruntime as ort
    from onnx import numpy_helper
    from onnx import shape_inference

    def get_tensor_shape(model: onnx.ModelProto, tensor_name: str):
        inferred_model = shape_inference.infer_shapes(model)

        # Search for the tensor with the given name
        for value_info in inferred_model.graph.value_info:
            if value_info.name == tensor_name:
                return [dim.dim_value for dim in value_info.type.tensor_type.shape.dim]

        # If not found; search the model's inputs
        for input_info in inferred_model.graph.input:
            if input_info.name == tensor_name:
                return [dim.dim_value for dim in input_info.type.tensor_type.shape.dim]

        # If still not found; search the model's outputs
        for output_info in inferred_model.graph.output:
            if output_info.name == tensor_name:
                return [dim.dim_value for dim in output_info.type.tensor_type.shape.dim]

        return None

    model = onnx.shape_inference.infer_shapes(model)  # 推斷形狀
    for node in model.graph.node:
        layer_type = node.op_type
        if layer_type == "Conv":
            input_name = node.input[0]  # 第一個輸入是 feature map
            output_name = node.output[0]  # 第一個輸出是 feature map
            weight_name = node.input[1]  # 第二個輸入是 weights
            
            input_shape = get_tensor_shape(model, input_name)  # [N, C, H, W]
            output_shape = get_tensor_shape(model, output_name)  # [N, M, E, F]
            
            weight_tensor = next((w for w in model.graph.initializer if w.name == weight_name), None)
            if weight_tensor and input_shape and output_shape:
                weight_array = numpy_helper.to_array(weight_tensor)
                M, C, R, S = weight_array.shape  # (out_channels, in_channels, kernel_h, kernel_w)
                
                # 讀取 stride 和 padding
                attr = {a.name: a for a in node.attribute}
                U = attr.get("strides", None)
                U = U.ints[0] if U else 1  # 預設 stride=1
                
                P = attr.get("pads", None)
                P = P.ints[0] if P else 0  # 預設 padding=0

                _, C, H, W = input_shape
                _, M, E, F = output_shape
                
                layers.append(Conv2DShapeParam(N=1, H=H, W=W, R=R, S=S, E=E, F=F, C=C, M=M, U=U, P=P))
        
        elif layer_type == "MaxPool":
            kernel_size = None
            stride = None
            for attr in node.attribute:
                if attr.name == "kernel_shape":
                    kernel_size = attr.ints[0]
                if attr.name == "strides":
                    stride = attr.ints[0]
            layers.append(MaxPool2DShapeParam(N=1, kernel_size=kernel_size, stride=stride))
        elif layer_type == "Gemm":  # 全連接層
            in_features = None
            out_features = None
            for inp in model.graph.initializer:
                if inp.name == node.input[1]:
                    weight_array = numpy_helper.to_array(inp)
                    out_features, in_features = weight_array.shape
            layers.append(LinearShapeParam(N=1, in_features=in_features, out_features=out_features))
    return layers

def compare_layers(answer, layers):
    if len(answer) != len(layers):
        print(
            f"Layer count mismatch: answer has {len(answer)}, but ONNX has {len(layers)}"
        )

    min_len = min(len(answer), len(layers))

    for i in range(min_len):
        ans_layer = vars(answer[i])
        layer = vars(layers[i])

        diffs = {
            k: (ans_layer[k], layer[k])
            for k in ans_layer
            if k in layer and ans_layer[k] != layer[k]
        }

        if diffs:
            print(f"Difference in layer {i + 1} ({type(answer[i]).__name__}):")
            for k, (ans_val, val) in diffs.items():
                print(f"  {k}: answer = {ans_val}, onnx = {val}")

    if len(answer) > len(layers):
        print(f"Extra layers in answer: {answer[len(layers) :]}")
    elif len(layers) > len(answer):
        print(f"Extra layers in yours: {layers[len(answer) :]}")


def run_tests() -> None:
    """Run tests on the network parser functions."""
    answer = [
        Conv2DShapeParam(N=1, H=32, W=32, R=3, S=3, E=32, F=32, C=3, M=32, U=1, P=1),
        Conv2DShapeParam(N=1, H=32, W=32, R=3, S=3, E=32, F=32, C=32, M=32, U=1, P=1),
        MaxPool2DShapeParam(N=1, kernel_size=2, stride=2),
        Conv2DShapeParam(N=1, H=16, W=16, R=3, S=3, E=16, F=16, C=32, M=64, U=1, P=1),
        Conv2DShapeParam(N=1, H=16, W=16, R=3, S=3, E=16, F=16, C=64, M=64, U=1, P=1),
        MaxPool2DShapeParam(N=1, kernel_size=2, stride=2),
        Conv2DShapeParam(N=1, H=8, W=8, R=3, S=3, E=8, F=8, C=64, M=128, U=1, P=1),
        Conv2DShapeParam(N=1, H=8, W=8, R=3, S=3, E=8, F=8, C=128, M=128, U=1, P=1),
        MaxPool2DShapeParam(N=1, kernel_size=2, stride=2),
        Conv2DShapeParam(N=1, H=4, W=4, R=3, S=3, E=4, F=4, C=128, M=256, U=1, P=1),
        Conv2DShapeParam(N=1, H=4, W=4, R=3, S=3, E=4, F=4, C=256, M=256, U=1, P=1),
        MaxPool2DShapeParam(N=1, kernel_size=2, stride=2),
        LinearShapeParam(N=1, in_features=1024, out_features=256),
        LinearShapeParam(N=1, in_features=256, out_features=128),
        LinearShapeParam(N=1, in_features=128, out_features=10),
    ]

    # Test with the PyTorch model.
    model = VGG()
    layers_pth = parse_pytorch(model)

    # Define the input shape.
    dummy_input = torch.randn(1, 3, 32, 32)
    # Save the model to ONNX.
    torch2onnx.torch2onnx(model, "parser_onnx.onnx", dummy_input)
    # Load the ONNX model.
    model_onnx = onnx.load("parser_onnx.onnx")
    layers_onnx = parse_onnx(model_onnx)

    # Display results.
    print("PyTorch Network Parser:")
    if layers_pth == answer:
        print("Correct!")
    else:
        print("Wrong!")
        compare_layers(answer, layers_pth)

    print("ONNX Network Parser:")
    if layers_onnx == answer:
        print("Correct!")
    else:
        print("Wrong!")
        compare_layers(answer, layers_onnx)


if __name__ == "__main__":
    run_tests()
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.quantized as nnq
import onnx

project_root = Path(__file__).parents[1]
sys.path.append(str(project_root))

from layer_info import (
    ShapeParam,
    Conv2DShapeParam,
    LinearShapeParam,
    MaxPool2DShapeParam,
)

from lib.models.vgg import VGG
import network_parser.torch2onnx



def parse_pytorch(model: nn.Module, input_shape=(1, 3, 32, 32)) -> list[ShapeParam]:
    layers = []
    dummy_input = torch.randn(*input_shape)  # Test input
    hooks = []

    def hook_fn(module, inputs, output):
        input_shape = inputs[0].shape
        output_shape = output.shape

        if isinstance(module, (nn.Conv2d, nnq.Conv2d)):  # Parse convolution layer
            layers.append(
                Conv2DShapeParam(
                    N=input_shape[0],  # Batch size
                    H=input_shape[2],   # Input image size
                    W=input_shape[3],  
                    R=module.kernel_size[0], # Kernel size
                    S=module.kernel_size[1],  
                    E=output_shape[2], # Output image size
                    F=output_shape[3],  
                    C=input_shape[1], # Input/output channels
                    M=module.out_channels,  
                    U=module.stride[0], # Stride & Padding
                    P=module.padding[0]  
                )
            )
        elif isinstance(module, (nn.MaxPool2d, nnq.MaxPool2d)):  # Parse max pooling layer
            kernel_size = module.kernel_size if isinstance(module.kernel_size, int) else module.kernel_size[0]
            stride = module.stride if isinstance(module.stride, int) else module.stride[0]
            layers.append(
                MaxPool2DShapeParam(
                    N=input_shape[0], 
                    kernel_size=kernel_size,
                    stride=stride
                )
            )
        elif isinstance(module, (nn.Linear, nnq.Linear)):  # Parse fully connected (linear) layer
            layers.append(
                LinearShapeParam(
                    N=input_shape[0], 
                    in_features=module.in_features, 
                    out_features=module.out_features
                )
            )

    for layer in model.modules():
        if isinstance(layer, (nn.Conv2d, nn.Linear, nn.MaxPool2d, nnq.Conv2d, nnq.Linear, nnq.MaxPool2d)):
            hooks.append(layer.register_forward_hook(hook_fn))

    model(dummy_input)

    for hook in hooks:
        hook.remove()

    for i, layer in enumerate(layers):
        print(f"Layer {i}: {layer.__class__.__name__}")
        print(f"Parameters: {layer.to_dict()}")
        print()

    return layers


def parse_onnx(model: onnx.ModelProto) -> list[ShapeParam]:
    layers = []
    import onnxruntime as ort
    from onnx import numpy_helper
    from onnx import shape_inference

    def get_tensor_shape(model: onnx.ModelProto, tensor_name: str):
        inferred_model = shape_inference.infer_shapes(model)

        # Search for the tensor with the given name
        for value_info in inferred_model.graph.value_info:
            if value_info.name == tensor_name:
                return [dim.dim_value for dim in value_info.type.tensor_type.shape.dim]

        # If not found; search the model's inputs
        for input_info in inferred_model.graph.input:
            if input_info.name == tensor_name:
                return [dim.dim_value for dim in input_info.type.tensor_type.shape.dim]

        # If still not found; search the model's outputs
        for output_info in inferred_model.graph.output:
            if output_info.name == tensor_name:
                return [dim.dim_value for dim in output_info.type.tensor_type.shape.dim]

        return None

    model = onnx.shape_inference.infer_shapes(model)  # 推斷形狀
    for node in model.graph.node:
        layer_type = node.op_type
        if layer_type == "Conv":
            input_name = node.input[0]  # 第一個輸入是 feature map
            output_name = node.output[0]  # 第一個輸出是 feature map
            weight_name = node.input[1]  # 第二個輸入是 weights
            
            input_shape = get_tensor_shape(model, input_name)  # [N, C, H, W]
            output_shape = get_tensor_shape(model, output_name)  # [N, M, E, F]
            
            weight_tensor = next((w for w in model.graph.initializer if w.name == weight_name), None)
            if weight_tensor and input_shape and output_shape:
                weight_array = numpy_helper.to_array(weight_tensor)
                M, C, R, S = weight_array.shape  # (out_channels, in_channels, kernel_h, kernel_w)
                
                # 讀取 stride 和 padding
                attr = {a.name: a for a in node.attribute}
                U = attr.get("strides", None)
                U = U.ints[0] if U else 1  # 預設 stride=1
                
                P = attr.get("pads", None)
                P = P.ints[0] if P else 0  # 預設 padding=0

                _, C, H, W = input_shape
                _, M, E, F = output_shape
                
                layers.append(Conv2DShapeParam(N=1, H=H, W=W, R=R, S=S, E=E, F=F, C=C, M=M, U=U, P=P))
        
        elif layer_type == "MaxPool":
            kernel_size = None
            stride = None
            for attr in node.attribute:
                if attr.name == "kernel_shape":
                    kernel_size = attr.ints[0]
                if attr.name == "strides":
                    stride = attr.ints[0]
            layers.append(MaxPool2DShapeParam(N=1, kernel_size=kernel_size, stride=stride))
        elif layer_type == "Gemm":  # 全連接層
            in_features = None
            out_features = None
            for inp in model.graph.initializer:
                if inp.name == node.input[1]:
                    weight_array = numpy_helper.to_array(inp)
                    out_features, in_features = weight_array.shape
            layers.append(LinearShapeParam(N=1, in_features=in_features, out_features=out_features))
    return layers

def compare_layers(answer, layers):
    if len(answer) != len(layers):
        print(
            f"Layer count mismatch: answer has {len(answer)}, but ONNX has {len(layers)}"
        )

    min_len = min(len(answer), len(layers))

    for i in range(min_len):
        ans_layer = vars(answer[i])
        layer = vars(layers[i])

        diffs = {
            k: (ans_layer[k], layer[k])
            for k in ans_layer
            if k in layer and ans_layer[k] != layer[k]
        }

        if diffs:
            print(f"Difference in layer {i + 1} ({type(answer[i]).__name__}):")
            for k, (ans_val, val) in diffs.items():
                print(f"  {k}: answer = {ans_val}, onnx = {val}")

    if len(answer) > len(layers):
        print(f"Extra layers in answer: {answer[len(layers) :]}")
    elif len(layers) > len(answer):
        print(f"Extra layers in yours: {layers[len(answer) :]}")


def run_tests() -> None:
    """Run tests on the network parser functions."""
    answer = [
        Conv2DShapeParam(N=1, H=32, W=32, R=3, S=3, E=32, F=32, C=3, M=32, U=1, P=1),
        Conv2DShapeParam(N=1, H=32, W=32, R=3, S=3, E=32, F=32, C=32, M=32, U=1, P=1),
        MaxPool2DShapeParam(N=1, kernel_size=2, stride=2),
        Conv2DShapeParam(N=1, H=16, W=16, R=3, S=3, E=16, F=16, C=32, M=64, U=1, P=1),
        Conv2DShapeParam(N=1, H=16, W=16, R=3, S=3, E=16, F=16, C=64, M=64, U=1, P=1),
        MaxPool2DShapeParam(N=1, kernel_size=2, stride=2),
        Conv2DShapeParam(N=1, H=8, W=8, R=3, S=3, E=8, F=8, C=64, M=128, U=1, P=1),
        Conv2DShapeParam(N=1, H=8, W=8, R=3, S=3, E=8, F=8, C=128, M=128, U=1, P=1),
        MaxPool2DShapeParam(N=1, kernel_size=2, stride=2),
        Conv2DShapeParam(N=1, H=4, W=4, R=3, S=3, E=4, F=4, C=128, M=256, U=1, P=1),
        Conv2DShapeParam(N=1, H=4, W=4, R=3, S=3, E=4, F=4, C=256, M=256, U=1, P=1),
        MaxPool2DShapeParam(N=1, kernel_size=2, stride=2),
        LinearShapeParam(N=1, in_features=1024, out_features=256),
        LinearShapeParam(N=1, in_features=256, out_features=128),
        LinearShapeParam(N=1, in_features=128, out_features=10),
    ]

    # Test with the PyTorch model.
    model = VGG()
    layers_pth = parse_pytorch(model)

    # Define the input shape.
    dummy_input = torch.randn(1, 3, 32, 32)
    # Save the model to ONNX.
    torch2onnx.torch2onnx(model, "parser_onnx.onnx", dummy_input)
    # Load the ONNX model.
    model_onnx = onnx.load("parser_onnx.onnx")
    layers_onnx = parse_onnx(model_onnx)

    # Display results.
    print("PyTorch Network Parser:")
    if layers_pth == answer:
        print("Correct!")
    else:
        print("Wrong!")
        compare_layers(answer, layers_pth)

    print("ONNX Network Parser:")
    if layers_onnx == answer:
        print("Correct!")
    else:
        print("Wrong!")
        compare_layers(answer, layers_onnx)


if __name__ == "__main__":
    run_tests()
