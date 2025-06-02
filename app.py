import gradio as gr
from huggingface_hub import InferenceClient
import os

# Configuration - using environment variables for security
model_name = os.getenv("MODEL_NAME", "jacobpmeyer/book-advisor-lora")
hf_token = os.getenv("HF_TOKEN")

# Initialize client with fallback
def get_client():
    try:
        # Try your LoRA model first
        client = InferenceClient(model=model_name, token=hf_token)
        # Test if it works
        client.text_generation("Test", max_new_tokens=1)
        print(f"✅ Using trained model: {model_name}")
        return client, True
    except Exception as e:
        print(f"⚠️ LoRA model not available: {e}")
        print("📝 Falling back to base model...")
        # Fallback to base model
        base_client = InferenceClient(model="meta-llama/Llama-3.1-8B-Instruct", token=hf_token)
        return base_client, False

client, is_personalized = get_client()

def generate_response(instruction, input_text="", max_length=400, temperature=0.7):
    """Generate response using HuggingFace Inference API"""

    # Format the prompt
    if input_text.strip():
        prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
    else:
        prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"

    try:
        response = client.text_generation(
            prompt,
            max_new_tokens=max_length,
            temperature=temperature,
            repetition_penalty=1.1,
            top_p=0.9,
            stop_sequences=["###", "\n\n###"],
            return_full_text=False
        )
        return response.strip()

    except Exception as e:
        return f"Error: {str(e)}"

def chat_interface(message, history, temperature, max_length):
    """Chat interface for Gradio"""
    response = generate_response(
        instruction=message,
        max_length=max_length,
        temperature=temperature
    )
    return response

# Create the Gradio interface
with gr.Blocks(
    title="📚 Personal Book Advisor",
    theme=gr.themes.Soft(),
) as demo:

    gr.Markdown(f"""
    # 📚 Personal Book Advisor
    ### Powered by HuggingFace Inference API ⚡

    {"🎯 **Personalized**: Trained on Jacob's book library!" if is_personalized else "📖 **General**: Using base Llama model (train your LoRA for personalization)"}
    """)

    # Shared controls
    with gr.Row():
        temperature = gr.Slider(0.1, 1.0, value=0.7, step=0.1, label="Temperature")
        max_length = gr.Slider(100, 500, value=300, step=50, label="Response Length")

    with gr.Tabs():
        # General Chat Tab
        with gr.TabItem("💬 General Chat"):
            chatbot = gr.ChatInterface(
                fn=lambda msg, hist: chat_interface(msg, hist, temperature.value, max_length.value),
                examples=[
                    "What's the most interesting book in my collection?",
                    "Tell me about the themes in my library",
                    "What would you recommend for a rainy weekend?",
                    "Summarize the key ideas from my philosophy books"
                ],
                title="Ask me anything about your books!"
            )

        # Book Recommendations Tab
        with gr.TabItem("📖 Book Recommendations"):
            with gr.Row():
                with gr.Column():
                    genre_input = gr.Textbox(
                        label="Genre/Topic of Interest",
                        placeholder="e.g., science fiction, productivity, history",
                        lines=2
                    )
                    situation_input = gr.Textbox(
                        label="Reading Situation (optional)",
                        placeholder="e.g., vacation reading, commute, before bed",
                        lines=2
                    )
                    rec_button = gr.Button("Get Recommendation", variant="primary")

                with gr.Column():
                    recommendation_output = gr.Textbox(
                        label="Recommendation",
                        lines=8,
                        interactive=False
                    )

            def book_recommendation_interface(genre_or_topic, reading_situation):
                instruction = f"Recommend a book from my personal library for someone interested in {genre_or_topic}"
                input_text = f"Context: {reading_situation}" if reading_situation else ""
                return generate_response(instruction, input_text, temperature.value, max_length.value)

            rec_button.click(
                fn=book_recommendation_interface,
                inputs=[genre_input, situation_input],
                outputs=recommendation_output
            )

        # Content Questions Tab
        with gr.TabItem("❓ Content Questions"):
            with gr.Row():
                with gr.Column():
                    question_input = gr.Textbox(
                        label="Your Question",
                        placeholder="e.g., What does the author say about leadership?",
                        lines=3
                    )
                    context_input = gr.Textbox(
                        label="Specific Book/Context (optional)",
                        placeholder="e.g., from my business books, in the chapter about...",
                        lines=2
                    )
                    question_button = gr.Button("Ask Question", variant="primary")

                with gr.Column():
                    answer_output = gr.Textbox(
                        label="Answer",
                        lines=8,
                        interactive=False
                    )

            def content_question_interface(question, book_context):
                instruction = f"Answer this question about the content in my book library: {question}"
                input_text = book_context if book_context else ""
                return generate_response(instruction, input_text, temperature.value, max_length.value)

            question_button.click(
                fn=content_question_interface,
                inputs=[question_input, context_input],
                outputs=answer_output
            )

    gr.Markdown(f"""
    ---
    **Status**: {"✅ Using your personalized book advisor model" if is_personalized else "⚠️ Using base model - complete LoRA training for personalization"}

    *Fast responses powered by HuggingFace Inference API*
    """)

# Launch the app
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port, share=False)
