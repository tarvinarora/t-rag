import gradio as gr
from rag_setup import get_answer

def answer_question(question, history):
    text, docs = get_answer(question)
    sources = "\n".join(
        f"- {d.metadata['source']} p.{d.metadata['page']}" for d in docs
    )
    return f"{text}\n\n---\n**Sources:**\n{sources}"

demo = gr.ChatInterface(
    fn=answer_question,
    title="Tarvin's RAG Assistant",
    description="Ask questions about the loaded documents.",
)

if __name__ == "__main__":
    demo.launch()