import torch
import torchvision.models as models

# 1. Load ResNet-34 pretrained từ torchvision
model = models.resnet34(pretrained=True)
model.eval()  # chuyển sang chế độ đánh giá (không học)

# 2. Tạo dummy input (giả lập ảnh đầu vào)
dummy_input = torch.randn(1, 3, 224, 224)  # batch size 1, RGB, 224x224

# 3. Xuất sang định dạng ONNX
torch.onnx.export(
    model,                     # mô hình PyTorch
    dummy_input,               # đầu vào mẫu
    "resnet34.onnx",           # tên file output
    input_names=['input'],     # tên đầu vào
    output_names=['output'],   # tên đầu ra
    dynamic_axes={             # cho phép kích thước batch thay đổi
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    },
    opset_version=11           # chọn phiên bản ONNX opset
)

print("✅ Đã lưu model ONNX vào 'resnet34.onnx'")
