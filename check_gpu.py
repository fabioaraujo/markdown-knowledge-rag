import torch

print("🔍 Verificando disponibilidade de GPU...\n")

if torch.cuda.is_available():
    print("✅ GPU DISPONÍVEL!")
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   CUDA Version: {torch.version.cuda}")
    print(f"   Memória Total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print(f"   Memória Livre: {torch.cuda.mem_get_info()[0] / 1024**3:.2f} GB")
else:
    print("❌ GPU NÃO DISPONÍVEL")
    print("   PyTorch está usando CPU")
    print("\n   Para instalar CUDA:")
    print("   https://pytorch.org/get-started/locally/")
