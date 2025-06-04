import asyncio

import os
import inspect
import logging
from lightrag import LightRAG, QueryParam
from lightrag.llm.ollama import ollama_model_complete, ollama_embed
from lightrag.utils import EmbeddingFunc
from lightrag.kg.shared_storage import initialize_pipeline_status
import nest_asyncio
nest_asyncio.apply()

# Configure working directory
WORKING_DIR = "./Agriculture_LightRAG"

# Configure logging
logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)


async def initialize_rag():
    # Reduced context size and optimized parameters to lower memory usage
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=ollama_model_complete,
        llm_model_name="llama2:7b",  # Using a 7B parameter model
        llm_model_max_async=1,  # Reduced from 2 to lower memory usage
        llm_model_max_token_size=4096,  # Reduced from 32768 to lower memory requirements
        llm_model_kwargs={
            "host": "http://localhost:11435",
            "options": {
                "num_ctx": 4096,  # Reduced context window size
                "num_batch": 256,  # Lower batch size for reduced memory usage
                "num_thread": 4  # Control thread usage
            },
        },

        embedding_func=EmbeddingFunc(
            embedding_dim=768,
            max_token_size=2048,  # Reduced from 8192 to lower memory requirements
            func=lambda texts: ollama_embed(
                texts,
                embed_model="nomic-embed-text",  # Using a lighter embedding model
                host="http://localhost:11435"
            ),
        ),
    )

    await rag.initialize_storages()
    await initialize_pipeline_status()

    return rag


async def print_stream(stream):
    async for chunk in stream:
        print(chunk, end="", flush=True)


def process_document():
    # Separate document processing to avoid memory spikes
    try:
        rag = asyncio.run(initialize_rag())
        with open("./agriculture.txt", "r", encoding="utf-8") as f:
            rag.insert(f.read())
        print("Document processed successfully")
        return True
    except Exception as e:
        print(f"Error processing document: {e}")
        return False


def query_document():
    # Separate querying to manage memory better
    try:
        rag = asyncio.run(initialize_rag())

        print("\nHybrid Search:")
        result = rag.query(
            "What are some innovative drone battery charging solutions in agricultural environments?",
            param=QueryParam(mode="global")
        )
        print(result)


        print("\nLocal Mode Query:")
        result3 = rag.query(
            "List the main topics or themes covered in this agricultural document",
            param=QueryParam(mode="local")  # 尝试使用local模式
        )
        print(result3)

        # 尝试更具体的问题
        print("\nMore Specific Query:")
        result4 = rag.query(
            "What are the benefits of drone technology in agriculture mentioned in this document?",
            param=QueryParam(mode="hybrid")
        )
        print(result4)

    except Exception as e:
        print(f"Error during query: {e}")
        import traceback
        traceback.print_exc()


def main():
    # Split workflow into separate steps to manage memory better
    doc_processed = process_document()

    if doc_processed:
        print("\nDocument processing complete. Starting queries...")
        query_document()
    else:
        print("Skipping queries due to document processing failure.")


if __name__ == "__main__":
    main()