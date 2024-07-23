
from llama_index.core.readers import SimpleDirectoryReader
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import LLMRerank, SimilarityPostprocessor
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import  VectorStoreIndex, StorageContext, ServiceContext, load_index_from_storage
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
import re 
import os
import chromadb


class RAG:
    def __init__(self, config_dict) -> None:
        self.config = config_dict
        print(self.config)
        self._build_pipleine()


    def _build_pipleine(self):
        self.embedding_model = HuggingFaceEmbedding(model_name=self.config["embedding_model"],cache_folder='./hf_cache') 
        self.llm = HuggingFaceLLM(
            model_name=self.config["llm"],
            generate_kwargs={
                "do_sample": True,
                "temperature": self.config["temperature"],
                "top_p": self.config["top_p"],
            },
            tokenizer_name="microsoft/phi-2",
        )
        if not os.path.exists(self.config['index_path'] + "/chroma.sqlite3"):
            self.documents = self.ingest(self.config['documents_path'])
            self.index = self.build_index(self.config['index_path'])
        else:
            self.index = self.load_index(self.config['index_path'])

        if self.config['reranker']:
            self.reranker = LLMRerank(
                self.config['reranker'],
                choice_batch_size=5,
                top_n=self.config['top_n'],
            )

    def ingest(self, documents_path):
        documents = SimpleDirectoryReader(documents_path).load_data()
        return documents

    def build_index(self, index_path):
        db = chromadb.PersistentClient(path=index_path)
        chroma_collection = db.get_or_create_collection("rel18")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_documents(self.documents, storage_context=storage_context, show_progress=True, embed_model=self.embedding_model,
            transformations=[SentenceSplitter(chunk_size=self.config['chunk_size'], chunk_overlap=self.config['chunk_overlap'])])
        index.storage_context.persist(persist_dir=index_path)
        return index
    
    def load_index(self, index_path):
        storage_context = StorageContext.from_defaults(persist_dir=index_path)
        index = load_index_from_storage(storage_context)
        return index

    def retrieve(self, query):
        # Configure retriever
        top_k = self.config['top_k']
        service_context = ServiceContext.from_defaults(llm=self.llm, embed_model=self.embedding_model)
        retriever = self.index.as_retriever(service_context=service_context)
        retriever = VectorIndexRetriever(index=self.index, similarity_top_k=top_k,embed_model=self.embedding_model)
        # Assemble query engine
        query_engine = RetrieverQueryEngine(retriever=retriever, node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.5)])

        response = query_engine.query(query)
        context = 'Context:\n'
        source_nodes = response.source_nodes

        if self.config['reranker']:
            reranked_nodes = self.reranker.rerank(query, source_nodes)
            source_nodes = reranked_nodes[:top_k]

        # Iterate over different chunks
        for i in range(len(source_nodes)):
            try:
                context = context + source_nodes[i].text + '\n'
            except:
                # Add empty string in context if none of the chunks was matched
                if i == 0:
                    context = ''
        # Remove unnecessary spaces
        context = re.sub('\s+', ' ', context)
        # Add context to the prompt to get final string
        return context

    def answer(self, query):
        return self.llm.complete(query)

    def _rerank(self, query, source_nodes):
        # rerank the retrieved documents based on query
        reranked_nodes = self.reranker.rerank(query, source_nodes)
        return reranked_nodes