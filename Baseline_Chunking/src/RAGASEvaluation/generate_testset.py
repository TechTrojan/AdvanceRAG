
"""
Generate a RAGAS test set from documents stored in a FAISS vector database.

Flow:
1. Load FAISS vector store
2. Randomly select a subset of documents
3. Split documents into chunks
4. Configure LLM and embedding models
5. Generate synthetic evaluation questions using RAGAS
6. Export results to CSV
"""

import os
import sys
import asyncio
from pathlib import Path

import nest_asyncio

from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.testset import TestsetGenerator
from collections import defaultdict
import random
from ragas.testset.persona import Persona

from ragas.testset.synthesizers.single_hop.specific import (
    SingleHopSpecificQuerySynthesizer,
)

from ragas.testset.synthesizers.multi_hop import (
    MultiHopAbstractQuerySynthesizer,
    MultiHopSpecificQuerySynthesizer,
)

# ------------------------------------------------------------------
# Add parent "src" folder to Python path
# This allows importing DataRetriever when debugging from VS Code
# ------------------------------------------------------------------
sys.path.append(str(Path(__file__).resolve().parent.parent))

from DataRetriever import DataRetriever


# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

FAISS_INDEX_PATH = (
    r"C:\repo\TechTrojan\AdvanceRAG\Baseline_Chunking"
    r"\data\index_data\faiss_index_base_chunk_600"
)

OUTPUT_FILE = "testset_3.csv"

# Number of random source documents to use
DOC_SAMPLE_SIZE = 300

# RAGAS output size
TESTSET_SIZE = 30

# Chunking configuration
CHUNK_SIZE = 600
CHUNK_OVERLAP = 60

# Optional safety limit to avoid excessive API usage
MAX_CHUNKS = 301

# OpenAI model
OPENAI_MODEL = "gpt-4o-mini"


# ------------------------------------------------------------------
# Document Loading
# ------------------------------------------------------------------

# def load_documents(index_path: str, sample_size: int):
#     """
#     Load FAISS database and return a random subset of documents.

#     Using a subset keeps RAGAS generation cost manageable and
#     reduces the chance of hitting rate limits.
#     """

#     retriever = DataRetriever(index_path)

#     if not retriever.LoadDatabase():
#         raise RuntimeError("Failed to load FAISS database.")

#     docs = retriever.RetriveRandomDocs(sample_size)

#     if not docs:
#         raise ValueError("No documents found in vector store.")

#     print(f"Loaded {len(docs)} documents")

#     return docs


def load_documents(index_path: str, sample_size: int):
    """
    Load documents from FAISS and ensure the sample contains
    documents from every source file.

    Example:
        3 source files, sample_size=300
        -> ~100 documents from each source
    """

    retriever = DataRetriever(index_path)

    if not retriever.LoadDatabase():
        raise RuntimeError("Failed to load FAISS database.")

    all_docs = retriever.RetriveAllDocs()

    if not all_docs:
        raise ValueError("No documents found in vector store.")

    # Group documents by source file
    docs_by_source = defaultdict(list)

    for doc in all_docs:
        source = doc.metadata.get("source", "unknown")
        docs_by_source[source].append(doc)

    print(f"Found {len(docs_by_source)} source files")

    # Calculate target docs per source
    docs_per_source = max(
        1,
        sample_size // len(docs_by_source)
    )

    selected_docs = []

    for source, source_docs in docs_by_source.items():

        count = min(
            docs_per_source,
            len(source_docs)
        )

        sampled = random.sample(
            source_docs,
            count
        )

        selected_docs.extend(sampled)

        print(
            f"Selected {count} docs from "
            f"{source} (total available: {len(source_docs)})"
        )

    # Fill remaining slots if sample_size
    # isn't evenly divisible by number of sources
    remaining = sample_size - len(selected_docs)

    if remaining > 0:

        remaining_pool = [
            doc
            for source_docs in docs_by_source.values()
            for doc in source_docs
            if doc not in selected_docs
        ]

        if remaining_pool:
            selected_docs.extend(
                random.sample(
                    remaining_pool,
                    min(remaining, len(remaining_pool))
                )
            )

    random.shuffle(selected_docs)

    print(
        f"Loaded {len(selected_docs)} documents "
        f"from {len(docs_by_source)} source files"
    )

    return selected_docs


# ------------------------------------------------------------------
# Chunking
# ------------------------------------------------------------------
 


def create_chunks(documents):
    """
    Split documents into smaller chunks.

    RAGAS works best on reasonably-sized chunks.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    chunks = splitter.split_documents(documents)

    print(f"Generated {len(chunks)} chunks")

    # Prevent extremely large chunk sets from generating
    # excessive LLM requests.
    if len(chunks) > MAX_CHUNKS:
        print(
            f"Limiting chunks from {len(chunks)} "
            f"to {MAX_CHUNKS}"
        )
        chunks = chunks[:MAX_CHUNKS]

    return chunks


# ------------------------------------------------------------------
# Model Setup
# ------------------------------------------------------------------

def create_llm():
    """
    Create OpenAI LLM wrapper used by RAGAS.
    """


    api_key = os.getenv("OPENAI_API_KEY")
    base_url=os.getenv("OPENAI_API_BASE")

    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set.")

    llm = ChatOpenAI(
        model=OPENAI_MODEL,
        api_key=api_key,
        base_url=base_url,
        max_retries=5,
        temperature=0,
    )

    return LangchainLLMWrapper(llm)


def create_embeddings():
    """
    Local embedding model.

    No OpenAI embedding cost is incurred because
    sentence-transformers runs locally.
    """

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return LangchainEmbeddingsWrapper(embeddings)

def create_personas()-> list[Persona]:
    personas = [
    Persona(
        name="Financial Analyst",
        role_description="Analyzes financial performance and risks"
    ),
    Persona(
        name="Investor",
        role_description="Evaluates company growth and valuation"
    ),
    Persona(
        name="Compliance Officer",
        role_description="Reviews regulatory and legal risks"
    ),
]
    
    return personas 

# ------------------------------------------------------------------
# Testset Generation
# ------------------------------------------------------------------

def generate_testset(generator, chunks):
    """
    Generate synthetic questions for RAG evaluation.
    """

    return generator.generate_with_langchain_docs(
        chunks,
        testset_size=TESTSET_SIZE,
        
        query_distribution=[
            (
                SingleHopSpecificQuerySynthesizer(
                    llm=generator.llm
                ),
                0.4,
            ),
            (
                MultiHopAbstractQuerySynthesizer(
                    llm=generator.llm
                ),
                0.4,
            ),
            (
                MultiHopSpecificQuerySynthesizer(
                    llm=generator.llm
                ),
                0.2,
            ),
        ],
    )


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():

    print("Loading documents...")
    docs = load_documents(
        FAISS_INDEX_PATH,
        DOC_SAMPLE_SIZE,
    )

    print("Creating chunks...")
    chunks = create_chunks(docs)

    print("Initializing models...")
    
    llm = create_llm()
    
    print(type(llm))
    embeddings = create_embeddings()

    print("Creating personas")
    persona_list = create_personas() 
    
    print("Creating RAGAS generator...")
    generator = TestsetGenerator(
        llm=llm,
        embedding_model=embeddings,
        persona_list= persona_list
    )

    print("Generating testset...")
    testset = generate_testset(generator, chunks)

    print("Saving CSV...")
    testset.to_pandas().to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(f"Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    
    main()
