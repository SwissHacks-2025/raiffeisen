from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from langchain_openai import AzureOpenAI
from langchain.chains import RetrievalQA
from langchain_openai import AzureOpenAIEmbeddings

from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import Chroma
import os

import hydra
from hydra.utils import get_original_cwd, to_absolute_path, instantiate, get_method

from utils import get_client
from dotenv import load_dotenv

import numpy as np
from collections import defaultdict


def load_split_products(raiffeisen_products_path):
    loader = Docx2txtLoader(raiffeisen_products_path)
    data = loader.load()
    full_text = data[0].page_content

    headers_to_split_on = [
        ("##", "name_product"),
    ]

    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    md_header_splits = markdown_splitter.split_text(full_text)
    return md_header_splits




@hydra.main(version_base=None, config_path="configs", config_name="basic_rag_config")
def main(config):
    load_dotenv()


    # get embeddings
    embeddings = AzureOpenAIEmbeddings(
        model=config.embeddings.model_name,
        azure_endpoint=config.azure_endpoint,
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=config.api_version,
    )

    # load bias and build vectordb
    bias_loader = CSVLoader(config.bias_path)
    bias_docs = bias_loader.load()
    print(f"Loaded {len(bias_docs)} bias documents")
    print(bias_docs[0].page_content)

    ids = [str(i) for i in range(0, len(bias_docs))]
    vectordb = Chroma.from_documents(bias_docs, embedding=embeddings, ids=ids)



    # load dialogue
    loader = Docx2txtLoader(config.dialogue_path)
    data = loader.load()
    full_text = data[0].page_content

    # split dialogue
    text_splitter = RecursiveCharacterTextSplitter(
    separators=["\n", "!", "?", "."],
    chunk_size=200,
    chunk_overlap=50,
    length_function=len,
    # is_separator_regex=False,
    )
    texts = text_splitter.create_documents([full_text])

    # create llm azure openai
    llm = AzureOpenAI(
        model=config.llm.model_name,
        azure_endpoint=config.azure_endpoint,
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=config.api_version,
    )
    

    retrieved_docs = []
    for chunk in texts:
        docs = vectordb.similarity_search_with_score(chunk.page_content, k=3)
        retrieved_docs.extend(docs)

    # Step 4: Aggregate
    score_map = defaultdict(list)
    doc_map = {}

    for doc, score in retrieved_docs:
        doc_id = doc.metadata.get("id", doc.page_content)  # Use your unique key
        score_map[doc_id].append(score)
        doc_map[doc_id] = doc

    # Step 5: Average scores
    avg_scores = {doc_id: np.mean(scores) for doc_id, scores in score_map.items()}

    # Define a threshold (lower = more similar; adjust based on embedding model)
    SIMILARITY_THRESHOLD = 0.5

    if not avg_scores:
        print("No matches found at all.")
    else:
        best_doc_id = min(avg_scores, key=avg_scores.get)
        best_score = avg_scores[best_doc_id]

        if best_score < SIMILARITY_THRESHOLD:
            print("No sufficiently similar match found.", best_score)
        else:
            print("Best matched document:")
            print(doc_map[best_doc_id].page_content)
            print("Average similarity score:", best_score)







    
    





if __name__ == "__main__":
    main()
    print('done')