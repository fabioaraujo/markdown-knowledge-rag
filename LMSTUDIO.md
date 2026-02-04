# 🏠 Usando LM Studio (IA Local)

Este projeto suporta **LM Studio** para rodar a IA completamente local, sem precisar de API keys ou internet!

## 🚀 Configuração do LM Studio

### 1. Instalar LM Studio
- Baixe em: https://lmstudio.ai/
- Instale e abra o programa

### 2. Baixar um Modelo
Recomendações de modelos:

**Modelos leves (4-8GB RAM):**
- `Phi-3-mini` (3.8B parâmetros)
- `TinyLlama` (1.1B parâmetros)

**Modelos médios (16GB RAM):**
- `Mistral-7B-Instruct`
- `Llama-3.2-7B-Instruct`

**Modelos poderosos (32GB+ RAM):**
- `Mixtral-8x7B-Instruct`
- `Llama-3.1-13B-Instruct`

### 3. Iniciar o Servidor
1. No LM Studio, vá em **"Local Server"**
2. Selecione o modelo baixado
3. Clique em **"Start Server"**
4. Servidor iniciará em: `http://localhost:1234`

## 💻 Configuração do Projeto

### 1. Instalar dependências
```bash
# Dependências básicas
uv sync

# PyTorch com GPU (acelera embeddings 10-50x)
uv pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
```

📖 Mais detalhes sobre GPU: [GPU_SETUP.md](GPU_SETUP.md)

### 2. Executar com LM Studio
```python
from kb_rag import KnowledgeBaseRAG

kb = KnowledgeBaseRAG(
    docs_path="./docs",
    provider="lmstudio",
    lmstudio_url="http://localhost:1234/v1",
    embedding_model="all-MiniLM-L6-v2"
)

kb.setup()
kb.query("Sua pergunta aqui")
```

### Executar o projeto:
```bash
# Ativar ambiente
.venv\Scripts\Activate.ps1  # Windows
# source .venv/bin/activate  # Linux/Mac

# Executar
python kb_rag.py
```

⚠️ **Não use `uv run kb_rag.py`** - ele reinstala PyTorch CPU e perde aceleração GPU!

O código já está configurado para usar LM Studio por padrão!

## 🎯 Vantagens

✅ **100% Local** - Sem enviar dados para cloud  
✅ **Privacidade Total** - Seus documentos não saem do computador  
✅ **Sem Custos** - Não precisa de API keys pagas  
✅ **Funciona Offline** - Após baixar os modelos  
⚡ **Aceleração GPU** - Embeddings 10-50x mais rápidos com CUDA  

## ⚙️ Embeddings Locais

O sistema usa **sentence-transformers** para gerar embeddings localmente:
- Modelo padrão: `all-MiniLM-L6-v2` (apenas 80MB!)
- Rápido e eficiente
- Qualidade comparável ao OpenAI para português

Outros modelos disponíveis:
- `paraphrase-multilingual-MiniLM-L12-v2` (melhor para português)
- `all-mpnet-base-v2` (mais preciso, mas maior)

Para trocar o modelo:
```python
kb = KnowledgeBaseRAG(
    embedding_model="paraphrase-multilingual-MiniLM-L12-v2"
)
```

## 🔄 Alternando entre OpenAI e LM Studio

```python
# LM Studio (local)
kb = KnowledgeBaseRAG(provider="lmstudio")

# OpenAI (cloud)
kb = KnowledgeBaseRAG(provider="openai")
```

## 📊 Performance

| Provider | Velocidade | Custo | Privacidade |
|----------|------------|-------|-------------|
| LM Studio | Depende do hardware | Grátis | 100% Local |
| OpenAI | Muito rápido | ~$0.002/1K tokens | Cloud |

## 🐛 Troubleshooting

**Erro de conexão:**
- Verifique se o LM Studio está rodando
- Confirme a URL: `http://localhost:1234`

**Respostas lentas:**
- Use modelos menores (Phi-3, TinyLlama)
- Verifique se tem GPU disponível

**Erro de memória:**
- Escolha modelo menor
- Reduza `chunk_size` no código
