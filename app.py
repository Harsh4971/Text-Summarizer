import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import os

# Load tokenizer and model only once
@st.cache_resource
def load_model():
    hf_token = os.getenv("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained("dps13/text-summarizer-model", token=hf_token)
    model = AutoModelForSeq2SeqLM.from_pretrained("dps13/text-summarizer-model", token=hf_token)
    return tokenizer, model

# Load model and tokenizer
tokenizer, model = load_model()

# App UI
st.set_page_config(page_title="Text Summarizer", page_icon="📝")
st.title("📝 Text Summarizer")
st.markdown("Paste a **blog, article, or long conversation** and generate a clean, concise summary.")

# Input box
dialogue = st.text_area("📥 Input Text", height=300, placeholder="Enter blog or dialogue text here...")

# Generate Summary Button
if st.button("Generate Summary"):
    if not dialogue.strip():
        st.warning("⚠️ Please enter text to summarize.")
    else:
        try:
            # Show waiting message
            status_placeholder = st.empty()
            status_placeholder.info("⏳ Generating summary, please wait...")

            # Tokenization
            inputs = tokenizer(
                dialogue,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            )

            # Generate summary
            summary_ids = model.generate(
                inputs["input_ids"],
                max_length=100,
                min_length=20,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=2,
                forced_bos_token_id=tokenizer.bos_token_id
            )

            # Decode summary
            summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

            status_placeholder.empty()

            # Display results
            if summary.strip():
                st.success("✅ Summary generated successfully!")
                st.subheader("📄 Summary")
                st.write(summary)
            else:
                st.error("❌ Summary is empty. Try longer or more structured input.")

        except Exception as e:
            st.exception(f"An error occurred: {e}")

# Footer
st.markdown("---")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8501))
    st.set_page_config(page_title="Text Summarizer")
    st.run(port=port)
