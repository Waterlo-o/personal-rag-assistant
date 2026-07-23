import re

import chromadb

from rag_assistant.ingestion.loader import load_file
from rag_assistant.ingestion.chunker import chunk_text
from rag_assistant.retrieval.embedder import embed_texts
from rag_assistant.cli import get_client, load_system_prompt
from rag_assistant.pipeline import make_search_tool, answer_with_tools
from rag_assistant.constants import CHROMA_PATH

client = get_client()

document = load_file("data/test_report.txt")

system_prompt = load_system_prompt("config/system_prompt.txt")

chunks = chunk_text(document)
chunk_embeddings = embed_texts(client, chunks)

chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_or_create_collection("eval_run")

existing_ids = collection.get()["ids"]
if existing_ids:
    collection.delete(ids=existing_ids)

chunk_emb = []

ids = [str(i) for i in range(len(chunks))]
for i, emb in enumerate(chunk_embeddings):
    if emb is not None:
        chunk_emb.append(chunk_embeddings[i])
collection.add(ids=ids, documents=chunks, embeddings=chunk_emb)

search_tool = make_search_tool(client, collection)

eval_qa = [
    {
        "question": "Какой был доход компании 'Ромашка' за 2023 год?",
        "expected": "5 млн рублей",
    },
    {
        "question": "Какой был доход компании 'Василёк' за 2023 год?",
        "expected": "8 млн рублей",
    },
    {
        "question": "Какой был доход компании 'Ромашка' за 2022 год?",
        "expected": "3 млн рублей",
    },
    {
        "question": "Сколько составил совокупный доход всей группы компаний за 2023 год?",
        "expected": "14.5 млн рублей",
    },
    {
        "question": "Какой был доход компании 'Одуванчик' за 2022 год?",
        "expected": [
            "нет данных",
            "нет информации",
            "не содержит",
            "не указан",
            "отсутств",
        ],
    },
]

result = []
bad_answers = []

pattern = r"(\d+),(\d+)"
replacement = r"\1.\2"

for num, case in enumerate(eval_qa, start=1):
    raw_answer = answer_with_tools(client, case["question"], search_tool, system_prompt)

    if raw_answer is None:
        result.append({num: "FAILED (no answer returned)\n"})
        bad_answers.append({"question": case["question"], "answer": None})
        continue

    answer = re.sub(pattern, replacement, raw_answer)

    if isinstance(case["expected"], list):
        passed = any(marker in answer for marker in case["expected"])
    else:
        passed = case["expected"] in answer

    if passed:
        result.append({num: "PASSED\n"})
    else:
        result.append({num: "FAILED\n"})
        bad_answers.append({"question": case["question"], "answer": answer})

print(result)
print(bad_answers)
