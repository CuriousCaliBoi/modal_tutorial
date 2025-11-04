import modal

# Create an image with PyTorch installed
image = modal.Image.debian_slim().pip_install("torch")

app = modal.App("example-gpu", image=image)


# Request a GPU for this function
@app.function(gpu="T4")
def gpu_function():
    import torch
    
    # Check if GPU is available
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"GPU is available: {torch.cuda.get_device_name(0)}")
        
        # Create a tensor on the GPU
        x = torch.randn(1000, 1000, device=device)
        y = torch.randn(1000, 1000, device=device)
        z = torch.matmul(x, y)
        
        print(f"Matrix multiplication result shape: {z.shape}")
        print(f"Result is on GPU: {z.is_cuda}")
    else:
        print("GPU is not available")


@app.local_entrypoint()
def main():
    gpu_function.remote()


