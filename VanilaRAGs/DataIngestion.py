from langchain_community.document_loaders import PyMuPDFLoader

import os 

def LoadingDocs(math_file_path):
    # Initialize PDF loader using LangChain's PyMuPDFLoader
# This loads the document in a format suitable for text chunking and embedding
    pdf_loader = PyMuPDFLoader(math_file_path)
    page_docs = pdf_loader.load()  # usually 1 Document per page
 

    return page_docs
    



 



cwd = os.getcwd()

print(f'current working directory{cwd}')

file_path = f"{cwd}\\VanilaRAGs\\grade5-teks-062024-0.pdf"

print(file_path)

docs = LoadingDocs(file_path)

print(docs[16:25])

    