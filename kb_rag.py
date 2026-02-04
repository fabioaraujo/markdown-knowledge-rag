"""
Sistema RAG (Retrieval-Augmented Generation) para base de conhecimento usando Markdown.
Suporta OpenAI e LM Studio (local).
"""
import os
from pathlib import Path
from typing import Literal
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import MarkdownTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_classic.chains.retrieval_qa.base import RetrievalQA


class KnowledgeBaseRAG:
    """Sistema RAG para consulta de base de conhecimento em Markdown."""
    
    def __init__(self, 
                 docs_path: str = "./docs", 
                 persist_dir: str = "./chroma_db",
                 provider: Literal["openai", "lmstudio"] = "lmstudio",
                 lmstudio_url: str = "http://localhost:1234/v1",
                 embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Inicializa o sistema RAG.
        
        Args:
            docs_path: Caminho para diretório com arquivos markdown
            persist_dir: Diretório para persistir o banco vetorial
            provider: "openai" ou "lmstudio" (padrão: lmstudio)
            lmstudio_url: URL do servidor LM Studio (padrão: http://localhost:1234/v1)
            embedding_model: Modelo de embeddings local (padrão: all-MiniLM-L6-v2)
        """
        load_dotenv()
        self.docs_path = Path(docs_path)
        self.persist_dir = Path(persist_dir)
        self.provider = provider
        self.lmstudio_url = lmstudio_url
        self.embedding_model = embedding_model
        self.vectorstore = None
        self.qa_chain = None
        
        # Valida configuração baseada no provider
        if self.provider == "openai":
            if not os.getenv("OPENAI_API_KEY"):
                raise ValueError("OPENAI_API_KEY não encontrada. Configure no arquivo .env")
        else:
            print(f"🏠 Usando LM Studio local em {self.lmstudio_url}")
            print(f"📊 Embeddings: {self.embedding_model}")
    
    def load_documents(self):
        """Carrega todos os arquivos markdown do diretório."""
        print(f"📁 Carregando documentos de {self.docs_path}...")
        from langchain_core.documents import Document
        
        documents = []
        md_files = list(self.docs_path.rglob("*.md"))
        
        for file_path in md_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                doc = Document(
                    page_content=content,
                    metadata={"source": str(file_path)}
                )
                documents.append(doc)
            except Exception as e:
                print(f"❌ Erro ao carregar {file_path}: {e}")
        
        print(f"✅ {len(documents)} documentos carregados")
        return documents
    
    def split_documents(self, documents):
        """Divide documentos em chunks menores."""
        print("✂️ Dividindo documentos em chunks...")
        splitter = MarkdownTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        chunks = splitter.split_documents(documents)
        print(f"✅ {len(chunks)} chunks criados")
        return chunks
    
    def create_vectorstore(self, chunks):
        """Cria e persiste o banco vetorial."""
        print("🔢 Criando embeddings e banco vetorial...")
        
        # Escolhe embeddings baseado no provider
        if self.provider == "openai":
            embeddings = OpenAIEmbeddings()
        else:
            # Embeddings locais com HuggingFace
            print(f"📥 Baixando modelo de embeddings: {self.embedding_model}...")
            embeddings = HuggingFaceEmbeddings(
                model_name=self.embedding_model,
                model_kwargs={'device': 'cuda'},  # Usa GPU
                encode_kwargs={'normalize_embeddings': True}
            )
        
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=str(self.persist_dir)
        )
        print(f"✅ Banco vetorial criado em {self.persist_dir}")
    
    def load_vectorstore(self):
        """Carrega banco vetorial existente."""
        print(f"📂 Carregando banco vetorial de {self.persist_dir}...")
        
        # Usa os mesmos embeddings do provider configurado
        if self.provider == "openai":
            embeddings = OpenAIEmbeddings()
        else:
            embeddings = HuggingFaceEmbeddings(
                model_name=self.embedding_model,
                model_kwargs={'device': 'cuda'},  # Usa GPU
                encode_kwargs={'normalize_embeddings': True}
            )
        
        self.vectorstore = Chroma(
            persist_directory=str(self.persist_dir),
            embedding_function=embeddings
        )
        print("✅ Banco vetorial carregado")
    
    def setup(self, force_rebuild: bool = False):
        """
        Configura o sistema RAG.
        
        Args:
            force_rebuild: Se True, reconstrói o banco vetorial do zero
        """
        # Verifica se já existe banco vetorial
        if self.persist_dir.exists() and not force_rebuild:
            self.load_vectorstore()
        else:
            # Cria novo banco vetorial
            documents = self.load_documents()
            if not documents:
                raise ValueError(f"Nenhum documento .md encontrado em {self.docs_path}")
            
            chunks = self.split_documents(documents)
            self.create_vectorstore(chunks)
        
        # Configura chain de Q&A
        print("🤖 Configurando chain de Q&A...")
        
        if self.provider == "openai":
            llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
        else:
            # Conecta ao LM Studio
            llm = ChatOpenAI(
                base_url=self.lmstudio_url,
                api_key="lm-studio",  # LM Studio não valida, mas é obrigatório
                temperature=0.7,
                streaming=True
            )
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(
                search_kwargs={"k": 3}
            ),
            return_source_documents=True
        )
        print("✅ Sistema pronto para consultas!")
    
    def query(self, question: str) -> dict:
        """
        Faz uma consulta à base de conhecimento.
        
        Args:
            question: Pergunta a ser respondida
            
        Returns:
            Dict com resposta e documentos fonte
        """
        if not self.qa_chain:
            raise RuntimeError("Sistema não configurado. Execute setup() primeiro.")
        
        print(f"\n❓ Pergunta: {question}")
        result = self.qa_chain.invoke({"query": question})
        
        print(f"\n💡 Resposta: {result['result']}")
        print(f"\n📚 Fontes ({len(result['source_documents'])} documentos):")
        for i, doc in enumerate(result['source_documents'], 1):
            source = doc.metadata.get('source', 'Desconhecido')
            print(f"  {i}. {source}")
        
        return result


def main():
    """Exemplo de uso do sistema RAG."""
    print("🚀 Sistema RAG - Base de Conhecimento\n")
    
    # Escolha o provider: "lmstudio" (local) ou "openai"
    provider = "lmstudio"  # Mude para "openai" se quiser usar OpenAI
    
    # Inicializa o sistema
    kb = KnowledgeBaseRAG(
        docs_path="./docs",
        provider=provider,
        lmstudio_url="http://localhost:1234/v1",  # URL padrão do LM Studio
        embedding_model="all-MiniLM-L6-v2"  # Modelo local de embeddings
    )
    
    # Configura (usa banco existente se disponível)
    kb.setup(force_rebuild=False)
    
    # Exemplos de consultas
    perguntas = [
        "O que é este projeto?",
        "Como funciona o sistema?",
    ]
    
    for pergunta in perguntas:
        kb.query(pergunta)
        print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
