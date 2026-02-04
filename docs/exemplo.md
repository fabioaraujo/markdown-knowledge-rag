# Base de Conhecimento - Exemplo

## Sobre este Projeto

Este é um sistema RAG (Retrieval-Augmented Generation) que permite criar uma base de conhecimento usando arquivos Markdown como fonte de dados.

## Como Funciona

O sistema funciona em várias etapas:

1. **Carregamento**: Lê todos os arquivos .md da pasta `docs/`
2. **Chunking**: Divide os documentos em pedaços menores para processamento
3. **Vetorização**: Converte os chunks em embeddings usando OpenAI
4. **Armazenamento**: Salva os vetores no ChromaDB
5. **Consulta**: Busca os chunks mais relevantes e gera resposta com LLM

## Vantagens do RAG

- Conhecimento sempre atualizado (basta atualizar os arquivos)
- Não precisa retreinar modelo
- Fontes citadas nas respostas
- Escalável para grandes volumes de documentos

## Como Adicionar Conteúdo

Simplesmente adicione novos arquivos .md na pasta `docs/` e execute o sistema com `force_rebuild=True`.
