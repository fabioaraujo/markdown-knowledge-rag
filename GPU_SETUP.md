# Instruções para manter PyTorch com CUDA

Devido à forma como o `uv run` funciona, ele sempre reinstala as dependências.

## ✅ Solução: Usar o ambiente diretamente

### 1. Instalar PyTorch com CUDA uma vez:
```bash
uv pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
```

### 2. Ativar o ambiente e usar python diretamente:
```bash
# Ativar ambiente
.venv\Scripts\Activate.ps1

# Rodar scripts
python kb_rag.py
python check_gpu.py
```

### Ou use um alias:
```bash
# Adicione ao seu perfil do PowerShell
function uvpy { .\.venv\Scripts\python.exe @args }

# Depois use:
uvpy kb_rag.py
uvpy check_gpu.py
```

## 🔧 Alternativa: Criar script batch

```batch
@echo off
.venv\Scripts\python.exe %*
```

Salve como `run.bat` e use: `.\run kb_rag.py`
