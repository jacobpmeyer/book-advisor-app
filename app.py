import gradio as gr
from huggingface_hub import InferenceClient
import os

# Configuration - ONLY your trained model (merged version for Inference API)
model_name = "jacobpmeyer/book-advisor-merged"
hf_token = os.getenv("HF_TOKEN")

def get_lora_client():
    """Initialize client with ONLY your LoRA model"""
    if not hf_token:
        return None, "❌ HF_TOKEN environment variable not set"

    try:
        print(f"🔄 Loading your personalized model: {model_name}")
        client = InferenceClient(model=model_name, token=hf_token)
        
        # Just initialize without testing - test on first actual use
        print(f"✅ Client initialized for model: {model_name}")
        return client, "✅ Your personalized book advisor is ready!"

    except Exception as e:
        error_msg = str(e)
        print(f"❌ Failed to load {model_name}: {error_msg}")
        print(f"❌ Full exception: {repr(e)}")

        # Give specific error messages
        if "401" in error_msg or "unauthorized" in error_msg.lower():
            return None, "❌ Authentication failed. Check your HF_TOKEN in Railway environment variables."
        elif "403" in error_msg or "forbidden" in error_msg.lower() or "gated" in error_msg.lower():
            return None, f"❌ Model access denied: {model_name}\n\nYour model appears to be private/gated. To fix this:\n1. Go to https://huggingface.co/{model_name}\n2. Click 'Settings' → 'Visibility' → Make it 'Public'\n3. OR ensure your HF_TOKEN has proper permissions for private models\n4. Refresh this page after changing visibility\n\nFull error: {error_msg}"
        elif "404" in error_msg or "not found" in error_msg.lower():
            return None, f"❌ Model not found: {model_name}\n\nThis means your merged model hasn't been created yet. Please:\n1. Run the merge script in Google Colab\n2. Wait for upload to complete\n3. Verify your model exists at: https://huggingface.co/{model_name}"
        else:
            return None, f"❌ Error loading your model: {error_msg}\n\nFull error details: {repr(e)}\n\nTroubleshooting:\n1. Verify model exists: https://huggingface.co/{model_name}\n2. Check HF_TOKEN permissions\n3. Make model public or ensure token has private model access\n4. Model might still be processing on HuggingFace (wait 5-10 minutes)\n5. Try the refresh button"

# Initialize your LoRA model
client, status_message = get_lora_client()
is_working = client is not None

def generate_response(instruction, input_text="", max_length=400, temperature=0.7):
    """Generate response using ONLY your trained LoRA model"""

    if not is_working:
        return f"🚫 Your personalized model isn't available yet.\n\n{status_message}\n\nThis app only works with your trained LoRA model - no generic substitutes!"

    # Format the prompt for your trained model
    if input_text.strip():
        prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
    else:
        prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"

    try:
        # Try chat completions API instead
        messages = [
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = client.chat_completion(
                messages=messages,
                max_tokens=max_length,
                temperature=temperature
            )
            
            if response and hasattr(response, 'choices') and response.choices:
                return response.choices[0].message.content.strip()
            elif isinstance(response, dict) and 'choices' in response:
                return response['choices'][0]['message']['content'].strip()
            else:
                return str(response).strip()
                
        except Exception as chat_error:
            print(f"⚠️ Chat completion failed: {chat_error}")
            # Fallback to basic text generation without problematic parameters
            response = client.text_generation(
                prompt,
                max_new_tokens=min(max_length, 200),  # Reduce max tokens
                temperature=min(temperature, 0.8),    # Reduce temperature
                return_full_text=False
            )
            
            if response and len(response.strip()) > 0:
                return response.strip()
            else:
                return "I apologize, but I couldn't generate a response. Please try rephrasing your question."

    except Exception as e:
        error_msg = str(e)
        print(f"❌ Generation error: {error_msg}")
        print(f"❌ Full exception: {repr(e)}")
        return f"❌ Error with your model: {error_msg}\n\nFull error: {repr(e)}\n\nYour LoRA model exists but encountered an error. Try refreshing or check the logs."

def chat_interface(message, history, temperature, max_length):
    """Chat interface - only your LoRA model"""
    response = generate_response(
        instruction=message,
        max_length=max_length,
        temperature=temperature
    )
    return response

# Create the Gradio interface
with gr.Blocks(
    title="📚 Jacob's Personal Book Advisor",
    theme=gr.themes.Soft(),
) as demo:

    gr.Markdown(f"""
    # 📚 Jacob's Personal Book Advisor
    ### Trained Exclusively on Your Book Library 🎯

    **Model**: `{model_name}`
    **Status**: {status_message}

    {"This AI knows your personal book collection and can provide personalized recommendations and insights!" if is_working else ""}
    """)

    if not is_working:
        gr.Markdown(f"""
        ### 🔧 Setup Required

        Your personalized book advisor isn't ready yet. Here's what to do:

        #### Debug Information:
        - **Model Name**: `{model_name}`
        - **HF Token**: {"✅ Set" if hf_token else "❌ Missing"} ({hf_token[:10] + "..." if hf_token else "None"})
        - **Model URL**: [Check if model exists](https://huggingface.co/{model_name})

        #### If you haven't created the merged model:
        1. **Your LoRA training is complete** (jacobpmeyer/book-advisor-lora exists)
        2. **Run the merge script** in Google Colab (creates the Inference API compatible version)
        3. **Wait for upload** to finish (creates jacobpmeyer/book-advisor-merged)
        4. **Come back here** and refresh the page

        #### If merge script is complete:
        1. **Check your model exists**: [https://huggingface.co/{model_name}](https://huggingface.co/{model_name})
        2. **Verify HF_TOKEN** in Railway environment variables
        3. **Check model visibility** (should be public or you have access)
        4. **Wait 5-10 minutes** if model was just uploaded (HuggingFace processing time)

        #### Setup Status Check:
        - ✅ Google Colab training completed?
        - ✅ LoRA model uploaded to HuggingFace?
        - ✅ Merge script run successfully?
        - ✅ Merged model visible at the link above?
        - ✅ HF_TOKEN set in Railway?

        **This app ONLY works with your trained model - no substitutes!**

        **Current Error**: {status_message}
        """)

    else:
        # Only show the interface if the model is working

        # Model management
        with gr.Row():
            refresh_btn = gr.Button("🔄 Refresh Model Connection", size="sm")
            model_status = gr.Textbox(
                value=status_message,
                label="Model Status",
                lines=2,
                interactive=False
            )

        def refresh_model():
            global client, status_message, is_working
            client, status_message = get_lora_client()
            is_working = client is not None
            return status_message

        refresh_btn.click(refresh_model, outputs=model_status)

        # Shared controls
        with gr.Row():
            temperature = gr.Slider(0.1, 1.0, value=0.7, step=0.1, label="Temperature (Creativity)")
            max_length = gr.Slider(100, 800, value=400, step=50, label="Response Length")

        with gr.Tabs():
            # General Chat Tab
            with gr.TabItem("💬 General Chat"):
                chatbot = gr.ChatInterface(
                    fn=lambda msg, hist: chat_interface(msg, hist, temperature.value, max_length.value),
                    type="messages",
                    examples=[
                        "What's the most interesting book in my collection?",
                        "Tell me about the themes in my library",
                        "What would you recommend for a rainy weekend?",
                        "Summarize the key ideas from my philosophy books",
                        "Which book changed your perspective the most?"
                    ],
                    title="Ask me anything about YOUR book library!"
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
                            label="Recommendation from Your Library",
                            lines=8,
                            interactive=False
                        )

                def book_recommendation_interface(genre_or_topic, reading_situation):
                    instruction = f"Recommend a book from my personal library for someone interested in {genre_or_topic}. Explain why this book from my collection would be perfect."
                    input_text = f"Reading context: {reading_situation}" if reading_situation else ""
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
                            placeholder="e.g., What does the author say about leadership in my business books?",
                            lines=3
                        )
                        context_input = gr.Textbox(
                            label="Specific Book/Context (optional)",
                            placeholder="e.g., from my business books, in the philosophy section, etc.",
                            lines=2
                        )
                        question_button = gr.Button("Ask Question", variant="primary")

                    with gr.Column():
                        answer_output = gr.Textbox(
                            label="Answer from Your Library",
                            lines=8,
                            interactive=False
                        )

                def content_question_interface(question, book_context):
                    instruction = f"Answer this question based on the content from my personal book library: {question}"
                    input_text = f"Focus on: {book_context}" if book_context else ""
                    return generate_response(instruction, input_text, temperature.value, max_length.value)

                question_button.click(
                    fn=content_question_interface,
                    inputs=[question_input, context_input],
                    outputs=answer_output
                )

        gr.Markdown(f"""
        ---
        **🎯 This AI knows YOUR books**: Responses are based exclusively on Jacob's personal library
        **🚫 No generic responses**: Only recommendations and insights from your actual collection
        **⚡ Powered by**: Your custom merged model (LoRA + Llama-3.1-8B-Instruct)
        """)

# Launch the app
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port, share=False)
