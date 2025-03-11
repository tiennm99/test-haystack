from haystack_integrations.components.generators.ollama import OllamaGenerator

from haystack import Pipeline, Document
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack.document_stores.in_memory import InMemoryDocumentStore
import os
import glob

document_store = InMemoryDocumentStore()

documents = []
for txt_file in glob.glob("data/*.txt"):
    with open(txt_file, 'r', encoding='utf-8') as f:
        content = f.read()
        documents.append(Document(content=content))

document_store.write_documents(documents)

prompt_template = """
Dựa trên những tài liệu sau, trả lời câu hỏi.
Tài liệu:
{% for doc in documents %}
    {{ doc.content }}
{% endfor %}
Câu hỏi: {{question}}
Trả lời:
"""

retriever = InMemoryBM25Retriever(document_store=document_store)
prompt_builder = PromptBuilder(template=prompt_template)
llm = OllamaGenerator(url = "http://localhost:11434", model="qwen2.5")

rag_pipeline = Pipeline()
rag_pipeline.add_component("retriever", retriever)
rag_pipeline.add_component("prompt_builder", prompt_builder)
rag_pipeline.add_component("llm", llm)
rag_pipeline.connect("retriever", "prompt_builder.documents")
rag_pipeline.connect("prompt_builder", "llm")

question = "Khan là ai?"
results = rag_pipeline.run(
    {
        "retriever": {"query": question},
        "prompt_builder": {"question": question},
    }
)

print(results["llm"]["replies"])
