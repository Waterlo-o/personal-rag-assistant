import re

from rag_assistant.ingestion.loader import load_file
from rag_assistant.ingestion.chunker import chunk_text
from rag_assistant.retrieval.embedder import embed_texts
from rag_assistant.cli import get_client, load_system_prompt
from rag_assistant.pipeline import answer_question

client = get_client()

document = load_file("data/test_report.txt")

system_prompt = load_system_prompt("config/system_prompt.txt")

chunks = chunk_text(document)

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

embed_txt = embed_texts(client, chunks)

result = []
bad_answers = []

pattern = r"(\d+),(\d+)"

replacement = r"\1.\2"

for num, case in enumerate(eval_qa, start=1):
    raw_answer = answer_question(
        client, case["question"], chunks, embed_txt, system_prompt
    )

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
