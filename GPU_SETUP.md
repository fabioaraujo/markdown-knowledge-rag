# 🎮 Configuração GPU (CUDA) - Guia Completo

## ⚠️ PROBLEMA: uv run reinstala PyTorch CPU

O comando `uv run` sempre sincroniza as dependências do `pyproject.toml`, que por padrão instala PyTorch CPU. Isso sobrescreve a instalação GPU!

## ✅ SOLUÇÃO: Usar o ambiente virtual diretamente

### 1. Verificar se tem GPU NVIDIA
```bash
nvidia-smi
```

Deve mostrar sua placa de vídeo e versão CUDA.

### 2. Instalar dependências básicas
```bash
uv sync
```

### 3. Instalar PyTorch com CUDA
```bash
# Para CUDA 12.1 (mais comum)
uv pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio

# Para CUDA 11.8 (GPUs mais antigas)
# uv pip install --index-url https://download.pytorch.org/whl/cu118 torch torchvision torchaudio
```

### 4. Verificar instalação
```bash
python check_gpu.py
```

Deve mostrar:
```
✅ GPU DISPONÍVEL!
   GPU: NVIDIA GeForce RTX 4070
   CUDA Version: 12.1
   Memória Total: 11.99 GB
```

### 5. Executar o projeto

**Opção A: Ativando o ambiente (Recomendado)**
```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# Linux/Mac
# source .venv/bin/activate

# Agora pode usar python normalmente
python kb_rag.py
python check_gpu.py
```

**Opção B: Script auxiliar**
```bash
# Windows
.\run.bat kb_rag.py

# O script run.bat executa direto no ambiente sem ativar
```

**❌ NÃO FAÇA ISSO:**
```bash
# Isso vai reinstalar PyTorch CPU!
uv run kb_rag.py  # ❌ ERRADO
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
