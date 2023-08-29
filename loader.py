from typing import Dict, List, Union
from langchain.docstore.document import Document
from langchain.document_loaders import DirectoryLoader, BSHTMLLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Qdrant


# https://github.com/langchain-ai/langchain/blob/master/libs/langchain/langchain/document_loaders/html_bs.py
class MyBSHTMLLoader(BSHTMLLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def load(self) -> List[Document]:
        """Load HTML document into document objects."""
        from bs4 import BeautifulSoup

        with open(self.file_path, "r", encoding=self.open_encoding) as f:
            soup = BeautifulSoup(f, **self.bs_kwargs)

        if soup.title:
            title = str(soup.title.string)
        else:
            title = ""

        text = title + self.get_text_separator + soup.select("#main .entry")[0].get_text(self.get_text_separator)

        metadata: Dict[str, Union[str, None]] = {
            "source": self.file_path,
            "title": title,
        }
        return [Document(page_content=text, metadata=metadata)]


loader = DirectoryLoader(
    "./data",
    glob="**/entry-*.html",
    loader_cls=MyBSHTMLLoader,
    show_progress=True,
)
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(encoding_name="cl100k_base", chunk_size=500)
docs = text_splitter.split_documents(documents)
embeddings = OpenAIEmbeddings()

qdrant = Qdrant.from_documents(
    docs,
    embeddings,
    collection_name="my_documents",
)
