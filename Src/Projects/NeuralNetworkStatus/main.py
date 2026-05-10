import torch
import torch.nn as nn

# Create a simple model and visualize the structure
model1 = nn.Sequential(
    nn.Linear(1, 10),
    nn.ReLU(),   
    nn.Linear(10, 1)       
)
print(f'Model1 architecture: {model1}\n')

model2 = nn.Linear(1, 10)
print(f'Model2 Architecture: {model2}\n')

# Get the weights: model_state_dict()
print(f'=' * 50)
print(f'Model1 Parameters: {model1.state_dict()}\n')
print(f'Model2 Parameters: {model2.state_dict()}')


# Discover the parameters numbers
print(f'=' * 50)

total_params1 = sum(p.numel() for p in model1.parameters())
trainable_params1 = sum(p.numel() for p in model1.parameters() if p.requires_grad)
total_params2 = sum(p.numel() for p in model2.parameters())
trainable_params2 = sum(p.numel() for p in model2.parameters() if p.requires_grad)

print(f"Total parameters of model 1: {total_params1:,}")
print(f"Treinable parameters of model 1: {trainable_params1:,}")
print(f"Total parameters of model 1: {total_params2:,}")
print(f"Treinable parameters of model 1: {trainable_params2:,}")

# Discover layers/parameters: .parameters() or named_parameters()
print(f'=' * 50)
for name, p in model1.named_parameters():
    print(f"MODEL1: Layer: {name:15} | Parameters: {p.numel():>10}")

for name, p in model2.named_parameters():
    print(f"MODEL2: Layer: {name:15} | Parameters: {p.numel():>10}")


# See the force of each layer
print(f'=' * 50)
for name, param in model1.named_parameters():
    if 'weight' in name:
        norm = torch.norm(param).item()
        print(f"Layer strength {name}: {norm:.4f}")


# Sparsity of weights
print(f'=' * 50)
for name, p in model1.named_parameters():
    next_zero = (p.abs() < 1e-3).float().mean().item() * 100
    print(f"Layer {name} is {next_zero:.1f}% empty (negligible weights)")

# Dtypes of data
print(f'=' * 50)
for name, p in model1.named_parameters():
    print(f"Layer {name} | Type: {p.dtype} | Memory: {p.element_size() * p.numel() / 1024:.2f} KB")