from utils.model_loader import load_qa_model


def answer_question(query: str, context: str) -> str:
    model = load_qa_model()
    # Use the loaded model to generate an answer
    answer = model.answer_question(query, context)
    return answer
