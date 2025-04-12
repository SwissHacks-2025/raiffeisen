from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter

import hydra
from hydra.utils import get_original_cwd, to_absolute_path, instantiate, get_method





@hydra.main(version_base=None, config_path="configs", config_name="basic_rag_config")
def main(config):

    # Load and split data
    loader = Docx2txtLoader(config.raiffeisen_products_path)
    data = loader.load()
    full_text = data[0].page_content

    headers_to_split_on = [
        ("##", "name_product"),
    ]

    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    md_header_splits = markdown_splitter.split_text(full_text)




if __name__ == "__main__":
    main()
    print('done')