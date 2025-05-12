from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader = DirectoryLoader(path = 'books',
                loader_cls= PyPDFLoader,
                show_progress= True,
                glob= '*.pdf'
                )

# docs = loader.load()
docs = loader.lazy_load()

#print(len(docs))
print(docs[325].metadata)

for page in docs:
    print(page.metadata)