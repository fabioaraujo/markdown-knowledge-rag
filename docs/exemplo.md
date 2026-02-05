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

## Equipe do Projeto

### Desenvolvedores

- **João Silva** - Desenvolvedor Backend
  - Responsável pela API REST
  - Implementação do sistema de autenticação
  - Email: joao@example.com

- **Juliana Santos** - Desenvolvedora Frontend
  - Design e implementação da interface
  - Responsável pelo sistema de componentes React
  - Especialista em UX/UI
  - Email: juliana@example.com

- **Carlos Oliveira** - DevOps
  - Infraestrutura e CI/CD
  - Monitoramento e logs

### Gerente de Projeto

- **Maria Fernandes** - Product Owner
  - Definição de requisitos
  - Contato: maria@example.com

## Reuniões

### Sprint Planning - 15/01/2026

Participantes: João, Juliana, Carlos e Maria

Tópicos discutidos:
- Nova feature de busca avançada
- Juliana apresentou mockups da nova interface
- Carlos configurou o ambiente de staging

### Daily Standup - 20/01/2026

- Juliana reportou progresso no componente de filtros
- João integrou a API de busca
- Carlos resolveu issues de performance

## Notas Técnicas

A arquitetura proposta por Juliana para o frontend usa React 18 com hooks customizados para gerenciar estado global. Esta abordagem foi aprovada pela equipe.
