from huggingface_hub import upload_folder

upload_folder(
    repo_id="dps13/text-summarizer-model",  # ✅ your actual HF repo ID
    folder_path="./model_directory",
    commit_message="Upload fine-tuned model"
)
