from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter

from langchain_openai import AzureOpenAI

from langchain_openai import AzureOpenAIEmbeddings

from langchain_community.document_loaders import CSVLoader
from langchain.vectorstores import Chroma
import os

import hydra
from hydra.utils import get_original_cwd, to_absolute_path, instantiate, get_method

from utils import get_client
from dotenv import load_dotenv




@hydra.main(version_base=None, config_path="configs", config_name="basic_rag_config")
def main(config):
    load_dotenv()

    ### Load and split data ###
    loader = Docx2txtLoader(config.raiffeisen_products_path)
    data = loader.load()
    full_text = data[0].page_content

    headers_to_split_on = [
        ("##", "name_product"),
    ]

    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    md_header_splits = markdown_splitter.split_text(full_text)


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
    


    
    





if __name__ == "__main__":
    main()
    print('done')