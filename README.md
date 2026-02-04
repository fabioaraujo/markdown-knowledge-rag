# KB-RAG - Base de Conhecimento com RAG

Sistema RAG (Retrieval-Augmented Generation) para criar uma base de conhecimento inteligente usando arquivos Markdown como fonte.

**✨ Funciona 100% local com LM Studio ou na nuvem com OpenAI!**

## 🚀 Características

- 📝 Usa arquivos Markdown como fonte de conhecimento
- 🏠 **100% Local com LM Studio** (privacidade total!)
- ☁️ Ou use OpenAI para velocidade máxima
- 🔍 Busca semântica usando embeddings
- 🤖 Respostas contextualizadas com LLM
- 📚 Citação de fontes
- 💾 Persistência do banco vetorial (ChromaDB)
- 🚫 Sem custos (modo local)

## 📦 Instalação

Este projeto usa [uv](https://github.com/astral-sh/uv) para gerenciamento de dependências.

```

### Opção 1: LM Studio (Local - Recomendado) 🏠

1. **Instale o LM Studio**: https://lmstudio.ai/
2. **Baixe um modelo** (ex: Mistral-7B, Llama-3.2, Phi-3)
3. **Inicie o servidor** no LM Studio
4. **Pronto!** Rode `uv run kb_rag.py`

📖 Guia completo: [LMSTUDIO.md](LMSTUDIO.md)

### Opção 2: OpenAI (Cloud) ☁️

```bash
cp .env.example .env
# Edite .env e adicione sua OPENAI_API_KEY
```

No código, use `provider="openai"Configurar API key
cp .env.example .env
# Edite .env e adicione sua OPENAI_API_KEY
```

## 🎯 Uso

### Adicionar documentos

Coloque seus arquivos `.md` na pasta `docs/`:

```
docs/
├── conceitos.md
├── tutoriais.md
└─LM Studio (Local)
kb = KnowledgeBaseRAG(
    docs_path="./docs",
    provider="lmstudio",  # 👈 Modo local!
    lmstudio_url="http://localhost:1234/v1",
    embedding_model="all-MiniLM-L6-v2"
)

# OU OpenAI (Cloud)
kb = KnowledgeBaseRAG(
    docs_path="./docs",
    provider="openai"  # 👈 Modo cloud
)

# Configurar

```bash
uv run kb_rag.py
```

### Uso programático

```pSentence-Transformers**: Embeddings locais (não precisa de API)
- **OpenAI** (opcional): API para embeddings e LLM na cloud
from kb_rag import KnowledgeBaseRAG

# Inicializar
kb = KnowledgeBaseRAG(docs_path="./docs")

# Configurar (primeira vez ou force_rebuild=True)
kb.setup(force_rebuild=False)

# Fazer consulta
resultado = kb.query("Como funciona o sistema RAG?")
print(resultado['result'])
```

## 🛠️ Estrutura

- `kb_rag.py` - Código principal do sistema RAG
- `docs/` - Seus arquivos Markdown de conhecimento
- `chroma_db/` - Banco vetorial persistido (gerado automaticamente)
- `.env` - Configuração da API key

## 📝 Dependências

- **LangChain**: Framework para construção de aplicações com LLMs
- **ChromaDB**: Banco vetorial para armazenamento de embeddings
- **OpenAI**: API para embeddings e LLM

## 💡 Exemplo

```python
kb = KnowledgeBaseRAG()
kb.setup()

# Consulta
resultado = kb.query("O que é RAG?")
print(resultado['result'])
# Saída: RAG (Retrieval-Augmented Generation) é uma técnica...

# Ver fontes
for doc in resultado['source_documents']:
    print(f"- {doc.metadata['source']}")
```

## 🔄 Atualizar Base de Conhecimento

Após adicionar/modificar arquivos na pasta `docs/`:

```python
kb.setup(force_rebuild=True)  # Reconstrói o banco vetorial
```

## 📄 Licença

MIT
