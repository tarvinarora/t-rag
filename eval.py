## EVAL SECTION (LLM-as-judge)
# Context Relevance: did the retriever fetch the right chunks?
# Faithfulness: is the answer supported by "those chunks"?
# Answer Relevance: is the answer relevant to the question asked?

eval_questions = [
    "Which school did Tarvin go to?",
    "What internships has Tarvin done?",
    "What projects has Tarvin done?",
    "Which domain did Tarvin complete her Bachelor's in?",
    "What industry is Tarvin interested in?",
    "When is Tarvin going to graduate?",
    "For the PowerBI TRS Dashboard, how many KPIs were used?",
    "What was the benefit of the InfoSec Query chat assistant Tarvin built?",
    "How many patrons has Tarvin managed in her past roles?",
    "List Tarvin's top 3 skills.",
    "Is Tarvin fluent in Agentic frameworks given her technical skills?",
    "What was Tarvin's tutoring focus during her work as a tutor?",
]

#run the rag pipeline
def get_answer(question):
    docs = retriever.invoke(question)
    context = format_docs(docs)
    final_prompt = prompt.format(context = context, question = question)
    answer = llm.invoke(final_prompt)

    content = answer.content
    if isinstance(content, list):
        text = "".join(b["text"] for b in content if b.get("type") == "text")
    else:
        text = content
    return text

#LLM-as-judge
judge_template = """You are evaluating a RAG system's answer.
Rate how well the ANSWER addresses the QUESTION on a scale of 1 to 5:
5 = directly and completely answers the question
4 = answers the question but with minor gaps or extra fluff
3 = partially answers; misses part of the question
2 = mostly off-topic or talks around the question
1 = does not address the question at all

Respond in exactly this format:
Score: <number>
Reason: <one sentence>

QUESTION: {question}
ANSWER: {answer}"""

def judge_relevance(question, answer):
    j_prompt = judge_template.format(question=question, answer=answer)
    result = llm.invoke(j_prompt)

    content = result.content
    if isinstance(content, list):
        text = "".join(b["text"] for b in content if b.get("type") == "text")
    else:
        text = content
    return text

import re

scores = []
for q in eval_questions:
    answer = get_answer(q)
    verdict = judge_relevance(q, answer)

    # pull the number out of "Score: 4"
    match = re.search(r"Score:\s*(\d)", verdict)
    score = int(match.group(1)) if match else None
    scores.append(score)

    print(f"\nQ: {q}")
    print(f"A: {answer[:120]}...")
    print(verdict)
    print("-" * 40)

valid = [s for s in scores if s is not None]
print(f"\nAverage relevance: {sum(valid)/len(valid):.2f} / 5 over {len(valid)} questions")