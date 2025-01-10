import torch

print("GPU 사용 가능 여부:", torch.cuda.is_available())
print("사용 중인 GPU 이름:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
