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
        self.parent_documents = {}  # Cache de documentos completos para Parent Document Retrieval
        
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
                
                # Melhoria 3: Adiciona nome do arquivo no início do conteúdo
                filename = file_path.stem  # Nome sem extensão
                enhanced_content = f"# Arquivo: {filename}\n\n{content}"
                
                doc = Document(
                    page_content=enhanced_content,
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
        # Melhoria 1: Aumenta overlap de 100 para 200 (20% do chunk_size)
        splitter = MarkdownTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = splitter.split_documents(documents)
        
        # Melhoria 2: Armazena documentos completos para Parent Document Retrieval
        for doc in documents:
            source = doc.metadata.get('source')
            if source:
                self.parent_documents[source] = doc.page_content
        
        # Adiciona referência ao documento pai em cada chunk
        for chunk in chunks:
            chunk.metadata['parent_source'] = chunk.metadata.get('source')
        
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
            # Embeddings locais com HuggingFace - modo offline
            print(f"📥 Carregando modelo de embeddings: {self.embedding_model}...")
            import torch
            import os
            
            # Configura modo offline
            os.environ['TRANSFORMERS_OFFLINE'] = '1'
            os.environ['HF_HUB_OFFLINE'] = '1'
            
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f"   Device: {device} (modo offline)")
            
            try:
                embeddings = HuggingFaceEmbeddings(
                    model_name=self.embedding_model,
                    model_kwargs={'device': device, 'trust_remote_code': True, 'local_files_only': True},
                    encode_kwargs={'normalize_embeddings': True, 'batch_size': 32}
                )
            except Exception as e:
                print(f"   ⚠️ Erro: {e}")
                print(f"   📡 Modelo não encontrado em cache. Tentando download online...")
                # Remove modo offline temporariamente
                del os.environ['TRANSFORMERS_OFFLINE']
                del os.environ['HF_HUB_OFFLINE']
                embeddings = HuggingFaceEmbeddings(
                    model_name=self.embedding_model,
                    model_kwargs={'device': device},
                    encode_kwargs={'normalize_embeddings': True}
                )
        
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=str(self.persist_dir)
        )
        print(f"✅ Banco vetorial criado em {self.persist_dir}")
        print(f"📦 Cache de Parent Documents: {len(self.parent_documents)} arquivos completos armazenados")
    
    def _populate_parent_cache(self):
        """Carrega documentos markdown e popula o cache de Parent Documents."""
        from langchain_core.documents import Document
        
        md_files = list(self.docs_path.rglob("*.md"))
        
        for file_path in md_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Adiciona nome do arquivo no início (como no load_documents)
                filename = file_path.stem
                enhanced_content = f"# Arquivo: {filename}\n\n{content}"
                
                # Armazena conteúdo completo com o mesmo caminho usado no banco
                source_path = str(file_path)
                self.parent_documents[source_path] = enhanced_content
                
            except Exception as e:
                # Ignora erros silenciosamente para não poluir output
                pass
    
    def load_vectorstore(self):
        """Carrega banco vetorial existente."""
        print(f"📂 Carregando banco vetorial de {self.persist_dir}...")
        
        # Usa os mesmos embeddings do provider configurado
        if self.provider == "openai":
            embeddings = OpenAIEmbeddings()
        else:
            import torch
            import os
            
            # Configura modo offline para usar cache local
            os.environ['TRANSFORMERS_OFFLINE'] = '1'
            os.environ['HF_HUB_OFFLINE'] = '1'
            
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
            try:
                embeddings = HuggingFaceEmbeddings(
                    model_name=self.embedding_model,
                    model_kwargs={'device': device, 'local_files_only': True},
                    encode_kwargs={'normalize_embeddings': True}
                )
            except Exception as e:
                print(f"   ⚠️ Erro ao carregar modelo do cache: {e}")
                print(f"   📡 Removendo modo offline e tentando novamente...")
                # Remove modo offline
                if 'TRANSFORMERS_OFFLINE' in os.environ:
                    del os.environ['TRANSFORMERS_OFFLINE']
                if 'HF_HUB_OFFLINE' in os.environ:
                    del os.environ['HF_HUB_OFFLINE']
                embeddings = HuggingFaceEmbeddings(
                    model_name=self.embedding_model,
                    model_kwargs={'device': device},
                    encode_kwargs={'normalize_embeddings': True}
                )
        
        self.vectorstore = Chroma(
            persist_directory=str(self.persist_dir),
            embedding_function=embeddings
        )
        print("✅ Banco vetorial carregado")
        
        # Carrega documentos markdown para popular cache de Parent Documents
        print("📂 Carregando documentos para cache de Parent Documents...")
        self._populate_parent_cache()
        print(f"📚 Cache populado: {len(self.parent_documents)} documentos completos")
    
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
        from langchain_core.retrievers import BaseRetriever
        from langchain_core.documents import Document
        from typing import List
        
        class HybridRetriever(BaseRetriever):
            vectorstore: any
            parent_documents: dict
            # Aumentado de 5 para 10 para melhor cobertura
            k: int = 10  # Número de documentos únicos a retornar (com contexto completo)
            search_all: bool = False  # Se True, busca TODOS os chunks do banco
            
            def _get_relevant_documents(self, query: str) -> List[Document]:
                import re
                from datetime import datetime
                from pathlib import Path
                from langchain_core.documents import Document as LCDocument
                
                # Detecta referências temporais e converte para formato de arquivo
                date_patterns = {
                    # Padrões com dia específico (ex: "15 de agosto de 2025")
                    r'(\d{1,2})\s+de\s+janeiro\s+(?:de\s+)?(\d{4})': ('01', True),
                    r'(\d{1,2})\s+de\s+fevereiro\s+(?:de\s+)?(\d{4})': ('02', True),
                    r'(\d{1,2})\s+de\s+março\s+(?:de\s+)?(\d{4})': ('03', True),
                    r'(\d{1,2})\s+de\s+abril\s+(?:de\s+)?(\d{4})': ('04', True),
                    r'(\d{1,2})\s+de\s+maio\s+(?:de\s+)?(\d{4})': ('05', True),
                    r'(\d{1,2})\s+de\s+junho\s+(?:de\s+)?(\d{4})': ('06', True),
                    r'(\d{1,2})\s+de\s+julho\s+(?:de\s+)?(\d{4})': ('07', True),
                    r'(\d{1,2})\s+de\s+agosto\s+(?:de\s+)?(\d{4})': ('08', True),
                    r'(\d{1,2})\s+de\s+setembro\s+(?:de\s+)?(\d{4})': ('09', True),
                    r'(\d{1,2})\s+de\s+outubro\s+(?:de\s+)?(\d{4})': ('10', True),
                    r'(\d{1,2})\s+de\s+novembro\s+(?:de\s+)?(\d{4})': ('11', True),
                    r'(\d{1,2})\s+de\s+dezembro\s+(?:de\s+)?(\d{4})': ('12', True),
                    # Padrões apenas com mês e ano (ex: "agosto de 2025")
                    r'janeiro\s+(?:de\s+)?(\d{4})': ('01', False),
                    r'fevereiro\s+(?:de\s+)?(\d{4})': ('02', False),
                    r'março\s+(?:de\s+)?(\d{4})': ('03', False),
                    r'abril\s+(?:de\s+)?(\d{4})': ('04', False),
                    r'maio\s+(?:de\s+)?(\d{4})': ('05', False),
                    r'junho\s+(?:de\s+)?(\d{4})': ('06', False),
                    r'julho\s+(?:de\s+)?(\d{4})': ('07', False),
                    r'agosto\s+(?:de\s+)?(\d{4})': ('08', False),
                    r'setembro\s+(?:de\s+)?(\d{4})': ('09', False),
                    r'outubro\s+(?:de\s+)?(\d{4})': ('10', False),
                    r'novembro\s+(?:de\s+)?(\d{4})': ('11', False),
                    r'dezembro\s+(?:de\s+)?(\d{4})': ('12', False),
                }
                
                # Detecta padrão de data na query
                file_date_filter = None
                query_lower = query.lower()
                for pattern, (month, has_day) in date_patterns.items():
                    match = re.search(pattern, query_lower)
                    if match:
                        if has_day:
                            # Tem dia específico: "15 de agosto de 2025" -> "2025_08_15"
                            day = match.group(1).zfill(2)
                            year = match.group(2)
                            file_date_filter = f"{year}_{month}_{day}"
                        else:
                            # Só mês e ano: "agosto de 2025" -> "2025_08"
                            year = match.group(1)
                            file_date_filter = f"{year}_{month}"
                        print(f"   🔍 Filtro de data detectado: {file_date_filter}_*.md" if not has_day else f"   🔍 Filtro de data detectado: {file_date_filter}.md")
                        break
                
                # Se search_all=True, pega TODOS os chunks. Senão, usa limite
                if self.search_all:
                    # Pega todos os documentos do vectorstore
                    all_data = self.vectorstore.get()
                    all_results = [
                        (LCDocument(page_content=all_data['documents'][i], 
                                 metadata=all_data['metadatas'][i] if all_data['metadatas'] else {}), 0)
                        for i in range(len(all_data['documents']))
                    ]
                else:
                    # Busca semântica limitada
                    all_results = self.vectorstore.similarity_search_with_score(query, k=min(self.k*50, 2000))
                
                # Filtra por data se detectado padrão
                if file_date_filter:
                    all_results = [
                        (doc, score) for doc, score in all_results
                        if file_date_filter in doc.metadata.get('source', '')
                    ]
                    print(f"   📁 Documentos após filtro de data: {len(all_results)}")
                
                # Tokeniza query em palavras relevantes (> 2 caracteres)
                query_words = [word.lower() for word in query.split() if len(word) > 2]
                
                # Identifica palavras-chave principais (nomes próprios - começam com maiúscula)
                import unicodedata
                import string
                
                def remove_accents(text):
                    return ''.join(c for c in unicodedata.normalize('NFD', text) 
                                 if unicodedata.category(c) != 'Mn')
                
                def clean_word(word):
                    """Remove pontuação do início e fim da palavra"""
                    return word.strip(string.punctuation)
                
                query_words_original = [clean_word(word) for word in query.split() if len(clean_word(word)) > 2]
                primary_keywords = []
                for word in query_words_original:
                    # Se começa com maiúscula, é provável que seja nome próprio
                    if word and word[0].isupper():
                        primary_keywords.append(word.lower())
                        # Adiciona versão sem acentos também
                        primary_keywords.append(remove_accents(word.lower()))
                
                if primary_keywords:
                    print(f"   🎯 Palavras-chave principais detectadas: {primary_keywords}")
                
                # Busca específica por arquivo com nome correspondente
                # Ex: se procura por "Juliana", tenta encontrar "Juliana.md"
                file_specific_results = []
                for keyword in primary_keywords:
                    # Remove duplicatas (versão com/sem acento)
                    if keyword not in [k.lower() for r, k in file_specific_results]:
                        # Busca no vectorstore por arquivos que contenham o nome no caminho
                        for doc, score in all_results[:100]:  # Verifica top 100
                            source_name = Path(doc.metadata.get('source', '')).stem.lower()
                            if keyword in source_name:
                                file_specific_results.append((doc, keyword))
                                print(f"   ⭐ Arquivo específico encontrado: {Path(doc.metadata.get('source', '')).name}")
                                break
                
                # Filtra e pontua por número de palavras encontradas
                scored_results = []
                for idx, (doc, semantic_score) in enumerate(all_results):
                    content_lower = doc.page_content.lower()
                    source_name = doc.metadata.get('source', '')
                    
                    # Conta matches com word boundaries
                    keyword_matches = 0
                    primary_keyword_matches = 0
                    
                    # Checa palavras-chave principais (peso 10x)
                    for word in primary_keywords:
                        pattern = r'\b' + re.escape(word) + r'\b'
                        if re.search(pattern, content_lower):
                            primary_keyword_matches += 1
                    
                    # Checa palavras comuns
                    for word in query_words:
                        if word not in primary_keywords:  # Evita contar duas vezes
                            pattern = r'\b' + re.escape(word) + r'\b'
                            if re.search(pattern, content_lower):
                                keyword_matches += 1
                    
                    # Bonus: se o nome do arquivo contém a palavra-chave principal
                    filename_bonus = 0
                    for word in primary_keywords:
                        if word in source_name.lower():
                            filename_bonus = 1000  # Prioridade máxima
                            break
                    
                    # Se tem matches relevantes, adiciona aos resultados
                    if primary_keyword_matches > 0 or keyword_matches > 0 or filename_bonus > 0:
                        # Score combinado: prioriza matches de palavras-chave principais
                        combined_score = (primary_keyword_matches * 1000) + (keyword_matches * 100) + filename_bonus - (semantic_score * 10) - (idx * 0.1)
                        scored_results.append((doc, combined_score, primary_keyword_matches, keyword_matches))
                
                # Ordena por relevância (maior score primeiro)
                scored_results.sort(key=lambda x: x[1], reverse=True)
                
                # Debug: mostra top 5 resultados
                if scored_results and primary_keywords:
                    print(f"   📄 Top 5 chunks encontrados:")
                    for i, (doc, score, pk_matches, k_matches) in enumerate(scored_results[:5]):
                        source = Path(doc.metadata.get('source', 'Desconhecido')).name
                        preview = doc.page_content[:100].replace('\n', ' ')
                        print(f"      {i+1}. [{source}] Score: {score:.1f} (PK:{pk_matches}, K:{k_matches})")
                        print(f"         {preview}...")
                
                filtered = [doc for doc, score, pk, k in scored_results]
                
                # Se não encontrou nada, tenta busca semântica pura como fallback
                if not filtered:
                    print(f"   ⚠️ Nenhum chunk com keywords encontrado, usando busca semântica pura")
                    top_chunks = [doc for doc, score in all_results[:self.k * 3]]
                else:
                    top_chunks = filtered[:self.k * 3]
                
                print(f"   📊 Recuperando {len(top_chunks)} chunks, buscando {self.k} documentos únicos")
                
                # Melhoria 2: Parent Document Retrieval - recupera documentos completos
                unique_sources = {}
                
                # Prioridade 1: Arquivos específicos encontrados (ex: Juliana.md)
                for doc, keyword in file_specific_results:
                    source = doc.metadata.get('parent_source') or doc.metadata.get('source')
                    if source and source not in unique_sources:
                        if source in self.parent_documents:
                            parent_doc = LCDocument(
                                page_content=self.parent_documents[source],
                                metadata=doc.metadata.copy()
                            )
                            parent_doc.metadata['retrieval_type'] = 'parent_document'
                            parent_doc.metadata['priority'] = 'file_specific'
                            unique_sources[source] = parent_doc
                            print(f"   ⭐ Doc PRIORITÁRIO: {Path(source).name} (arquivo específico)")
                        else:
                            doc.metadata['priority'] = 'file_specific'
                            unique_sources[source] = doc
                            print(f"   ⭐ Doc PRIORITÁRIO: {Path(source).name} (chunk específico)")
                
                # Prioridade 2: Chunks top ranqueados
                for chunk in top_chunks:
                    source = chunk.metadata.get('parent_source') or chunk.metadata.get('source')
                    if source and source not in unique_sources:
                        # Verifica se temos o documento pai em cache
                        if source in self.parent_documents:
                            # Cria um novo documento com o conteúdo completo
                            parent_doc = LCDocument(
                                page_content=self.parent_documents[source],
                                metadata=chunk.metadata.copy()
                            )
                            parent_doc.metadata['retrieval_type'] = 'parent_document'
                            unique_sources[source] = parent_doc
                            print(f"   ✅ Doc: {Path(source).name} (parent document)")
                        else:
                            # Se não tem pai, usa o chunk mesmo
                            unique_sources[source] = chunk
                            print(f"   📄 Doc: {Path(source).name} (chunk only)")
                    
                    # Limita ao número de documentos únicos solicitados
                    if len(unique_sources) >= self.k:
                        break
                
                final_docs = list(unique_sources.values())
                print(f"   🎯 Total de documentos únicos retornados: {len(final_docs)}")
                return final_docs
            
            async def _aget_relevant_documents(self, query: str) -> List[Document]:
                return self._get_relevant_documents(query)
        
        retriever = HybridRetriever(
            vectorstore=self.vectorstore,
            parent_documents=self.parent_documents,  # Passa cache de documentos pais
            k=10,  # Aumentado para 10 documentos únicos (com contexto completo)
            search_all=False  # Mude para True para carregar TODOS os chunks
        )
        
        # Prompt customizado que força a IA a usar o contexto fornecido
        from langchain_core.prompts import PromptTemplate
        
        prompt_template = """Você é um assistente pessoal que responde perguntas com base em uma base de conhecimento pessoal.

INSTRUÇÕES IMPORTANTES:
1. Use SEMPRE e SOMENTE as informações fornecidas no CONTEXTO abaixo para responder
2. Se a pergunta pedir para "olhar", "resumir" ou perguntar "o que aconteceu" em uma data ou arquivo:
   - Resuma TODO o conteúdo disponível no contexto
   - Liste todos os eventos, atividades, pensamentos e informações presentes
   - Seja completo e detalhado, não omita nada
3. Se não houver informação relevante no contexto, diga claramente que não há informações
4. NUNCA invente ou invente informações que não estejam explicitamente no contexto
5. Seja específico e cite detalhes do contexto quando relevantes
6. Se o contexto contiver um arquivo de diário ou notas, resuma todo o seu conteúdo

CONTEXTO:
{context}

PERGUNTA: {question}

RESPOSTA COMPLETA:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
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
    
    def search_hybrid(self, query: str, k: int = 10, file_filter: str = None):
        """Busca híbrida: filtra por palavra-chave + ranking semântico.
        
        Args:
            query: Termo de busca
            k: Número de resultados
            file_filter: Nome do arquivo para filtrar (ex: '2025_08_15.md')
        """
        if not self.vectorstore:
            print("❌ Banco vetorial não carregado")
            return []
        
        import re
        
        # Busca MUITO mais resultados para garantir que keywords sejam encontradas
        # Aumentamos de k*5 para k*50 
        all_results = self.vectorstore.similarity_search_with_score(query, k=min(k*50, 500))
        
        # Filtra por arquivo se especificado
        if file_filter:
            file_filter_lower = file_filter.lower()
            all_results = [
                (doc, score) for doc, score in all_results
                if file_filter_lower in doc.metadata.get('source', '').lower()
            ]
        
        # Filtra por arquivo se especificado
        if file_filter:
            file_filter_lower = file_filter.lower()
            all_results = [
                (doc, score) for doc, score in all_results
                if file_filter_lower in doc.metadata.get('source', '').lower()
            ]
        
        # Tokeniza a query em palavras individuais
        query_words = [word.lower() for word in query.split() if len(word) > 2]
        
        # Filtra e pontua resultados
        scored_results = []
        for idx, (doc, semantic_score) in enumerate(all_results):
            content_lower = doc.page_content.lower()
            
            # Conta matches usando word boundaries (palavras completas)
            keyword_matches = 0
            for word in query_words:
                # Usa regex com word boundary para evitar matches parciais
                pattern = r'\b' + re.escape(word) + r'\b'
                if re.search(pattern, content_lower):
                    keyword_matches += 1
            
            # Se tem matches de keywords
            if keyword_matches > 0:
                # Score combinado: prioriza keyword matches, mas considera posição semântica
                # Menor semantic_score = melhor (distância)
                combined_score = (keyword_matches * 100) - (semantic_score * 10) - (idx * 0.1)
                scored_results.append((doc, combined_score, keyword_matches))
        
        # Ordena por score combinado (maior primeiro)
        scored_results.sort(key=lambda x: x[1], reverse=True)
        
        # Retorna apenas os documentos (sem scores)
        filtered = [doc for doc, score, matches in scored_results]
        
        # Retorna apenas documentos com keyword match
        return filtered[:k]
    
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
        
        # Reformula perguntas de "olhar" para serem mais claras (mas mantém original para busca)
        import re
        reformulated_question = question
        
        # Detecta comandos de "olhar" uma data ou arquivo  
        look_patterns = [
            (r'^(olhe?|veja?|mostre?|me diga sobre)\s+(.+)$', 'O que aconteceu em {}? Resuma todas as informações disponíveis.'),
            (r'^o que (tem|temos|há) (em|sobre|no dia|na data|do dia)\s+(.+)\??$', 'O que aconteceu em {}? Resuma todas as informações disponíveis.'),
        ]
        
        for pattern, template in look_patterns:
            match = re.match(pattern, question.lower().strip())
            if match:
                # Pega o último grupo capturado (que contém a data/termo)
                target = match.group(match.lastindex)
                reformulated_question = template.format(target)
                if reformulated_question != question:
                    print(f"   🔄 Pergunta reformulada para LLM: {reformulated_question}")
                break
        
        # Usa pergunta ORIGINAL para busca (mantém palavras-chave), reformulada para LLM
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
    kb.setup(force_rebuild=False)  # Mude para True para reconstruir o banco
    
    print("\n" + "="*80)
    print("💬 Sistema pronto! Digite suas perguntas (ou 'sair' para encerrar)")
    print("\n📌 Comandos especiais:")
    print("   /listar                      - Lista arquivos indexados")
    print("   /rebuild                     - Reconstrói o banco vetorial")
    print("   sair                         - Encerra o programa")
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
            
            if pergunta.lower() == '/rebuild':
                print("\n🔄 Reconstruindo banco vetorial...")
                old_cache_count = len(kb.parent_documents)
                print(f"   📦 Cache atual: {old_cache_count} documentos pais")
                
                kb.setup(force_rebuild=True)
                
                new_cache_count = len(kb.parent_documents)
                print(f"\n✅ Banco reconstruído com sucesso!")
                print(f"   📦 Cache ANTES:  {old_cache_count} documentos")
                print(f"   📦 Cache DEPOIS: {new_cache_count} documentos")
                
                if new_cache_count > old_cache_count:
                    print(f"   🎯 Parent Document Retrieval ATIVADO! (+{new_cache_count - old_cache_count} docs)")
                    print(f"   💡 Agora a IA receberá documentos COMPLETOS ao invés de fragmentos!\n")
                
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
