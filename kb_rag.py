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
                 docs_path: str = None, 
                 persist_dir: str = "./chroma_db",
                 provider: Literal["openai", "lmstudio"] = "lmstudio",
                 lmstudio_url: str = "http://localhost:1234/v1",
                 embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Inicializa o sistema RAG.
        
        Args:
            docs_path: Caminho para diretório com arquivos markdown (padrão: lê do .env ou usa ./docs)
            persist_dir: Diretório para persistir o banco vetorial
            provider: "openai" ou "lmstudio" (padrão: lmstudio)
            lmstudio_url: URL do servidor LM Studio (padrão: http://localhost:1234/v1)
            embedding_model: Modelo de embeddings local (padrão: all-MiniLM-L6-v2)
        """
        load_dotenv()
        
        # Lê docs_path do .env se não for fornecido
        if docs_path is None:
            docs_path = os.getenv("DOCS_PATH", "./docs")
        
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
        
        print(f"   Encontrados {len(md_files)} arquivos .md")
        
        for file_path in md_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                doc = Document(
                    page_content=content,
                    metadata={"source": str(file_path)}
                )
                documents.append(doc)
                print(f"   ✓ {file_path.name} ({len(content)} caracteres)")
            except Exception as e:
                print(f"   ❌ Erro ao carregar {file_path}: {e}")
        
        print(f"✅ {len(documents)} documentos carregados com sucesso")
        return documents
    
    def split_documents(self, documents):
        """Divide documentos em chunks menores."""
        print("✂️ Dividindo documentos em chunks...")
        splitter = MarkdownTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        chunks = splitter.split_documents(documents)
        
        # Mostra distribuição de chunks por arquivo
        from collections import Counter
        sources = [chunk.metadata.get('source', 'Desconhecido') for chunk in chunks]
        source_counts = Counter(sources)
        
        print(f"\n   Chunks por arquivo:")
        for source, count in source_counts.items():
            filename = Path(source).name if source != 'Desconhecido' else source
            print(f"   - {filename}: {count} chunks")
        
        print(f"\n✅ Total: {len(chunks)} chunks criados")
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
        
        # Cria retriever customizado que usa busca híbrida
        from langchain.schema.retriever import BaseRetriever
        from langchain.schema import Document
        from typing import List
        
        class HybridRetriever(BaseRetriever):
            vectorstore: any
            k: int = 8
            
            def _get_relevant_documents(self, query: str) -> List[Document]:
                # Busca semântica ampla
                all_results = self.vectorstore.similarity_search(query, k=self.k*3)
                
                # Filtra por palavra-chave se encontrar
                query_lower = query.lower()
                filtered = [
                    doc for doc in all_results 
                    if any(word in doc.page_content.lower() for word in query_lower.split())
                ]
                
                return filtered[:self.k] if filtered else all_results[:self.k]
            
            async def _aget_relevant_documents(self, query: str) -> List[Document]:
                return self._get_relevant_documents(query)
        
        retriever = HybridRetriever(vectorstore=self.vectorstore, k=8)
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )
        print("✅ Sistema pronto para consultas!")
    
    def list_indexed_files(self):
        """Lista todos os arquivos indexados no banco vetorial."""
        if not self.vectorstore:
            print("❌ Banco vetorial não carregado")
            return
        
        try:
            # Pega uma amostra de todos os documentos
            all_docs = self.vectorstore.get()
            sources = set()
            
            if 'metadatas' in all_docs and all_docs['metadatas']:
                for metadata in all_docs['metadatas']:
                    if metadata and 'source' in metadata:
                        sources.add(metadata['source'])
            
            print(f"\n📚 Arquivos indexados no banco vetorial: {len(sources)}")
            for source in sorted(sources):
                filename = Path(source).name
                print(f"   ✓ {filename}")
                
            print(f"\n📊 Total de chunks no banco: {len(all_docs.get('ids', []))}")
        except Exception as e:
            print(f"❌ Erro ao listar arquivos: {e}")
    
    def search_raw(self, query: str, k: int = 10):
        """Busca direta no vectorstore sem LLM (para debug)."""
        if not self.vectorstore:
            print("❌ Banco vetorial não carregado")
            return
        
        print(f"\n🔍 Buscando por: '{query}'\n")
        results = self.vectorstore.similarity_search(query, k=k)
        
        print(f"Encontrados {len(results)} chunks:\n")
        for i, doc in enumerate(results, 1):
            source = Path(doc.metadata.get('source', 'Desconhecido')).name
            preview = doc.page_content[:200].replace('\n', ' ')
            print(f"{i}. [{source}]")
            print(f"   {preview}...\n")
    
    def search_hybrid(self, query: str, k: int = 10):
        """Busca híbrida: filtra por palavra-chave + ranking semântico."""
        if not self.vectorstore:
            print("❌ Banco vetorial não carregado")
            return []
        
        # Pega mais resultados da busca semântica
        all_results = self.vectorstore.similarity_search(query, k=k*3)
        
        # Filtra resultados que contêm a palavra-chave exata (case-insensitive)
        query_lower = query.lower()
        filtered = [
            doc for doc in all_results 
            if query_lower in doc.page_content.lower()
        ]
        
        # Se encontrou com filtro, retorna filtrados. Senão, retorna todos
        return filtered[:k] if filtered else all_results[:k]
    
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
        provider=provider,
        lmstudio_url="http://localhost:1234/v1",  # URL padrão do LM Studio
        embedding_model="all-MiniLM-L6-v2"  # Modelo local de embeddings
    )
    
    # Configura (usa banco existente se disponível)
    kb.setup(force_rebuild=True)
    
    print("\n" + "="*80)
    print("💬 Sistema pronto! Digite suas perguntas (ou 'sair' para encerrar)")
    print("\n📌 Comandos especiais:")
    print("   /listar         - Lista arquivos indexados")
    print("   /buscar <termo> - Busca direta sem LLM (debug)")
    print("   /rebuild        - Reconstrói o banco vetorial")
    print("   sair            - Encerra o programa")
    print("="*80 + "\n")
    
    # Loop interativo de perguntas
    while True:
        try:
            pergunta = input("❓ Sua pergunta: ").strip()
            
            if not pergunta:
                continue
            
            # Comandos especiais
            if pergunta.lower() == '/listar':
                kb.list_indexed_files()
                print("\n" + "="*80 + "\n")
                continue
            
            if pergunta.lower().startswith('/buscar '):
                termo = pergunta[8:].strip()
                if termo:
                    # Busca híbrida
                    results = kb.search_hybrid(termo, k=10)
                    print(f"\n🔍 Busca híbrida por: '{termo}'")
                    print(f"Encontrados {len(results)} chunks com a palavra-chave:\n")
                    for i, doc in enumerate(results, 1):
                        source = Path(doc.metadata.get('source', 'Desconhecido')).name
                        # Destaca a palavra no preview
                        preview = doc.page_content[:300].replace('\n', ' ')
                        print(f"{i}. [{source}]")
                        print(f"   {preview}...\n")
                else:
                    print("❌ Use: /buscar <termo>")
                print("\n" + "="*80 + "\n")
                continue
            
            if pergunta.lower() == '/rebuild':
                print("\n🔄 Reconstruindo banco vetorial...")
                kb.setup(force_rebuild=True)
                print("✅ Banco reconstruído!\n")
                print("="*80 + "\n")
                continue
                
            if pergunta.lower() in ['sair', 'exit', 'quit', 'q']:
                print("\n👋 Encerrando sistema. Até logo!")
                break
            
            kb.query(pergunta)
            print("\n" + "="*80 + "\n")
            
        except KeyboardInterrupt:
            print("\n\n👋 Encerrando sistema. Até logo!")
            break
        except Exception as e:
            print(f"\n❌ Erro: {e}\n")
            continue


if __name__ == "__main__":
    main()
