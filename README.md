# hf-summarize-cli

CLI tool to summarize German website content into an English paragraph using Hugging Face hosted models.

## Usage

```bash
python hf_summarize_cli.py "https://example.de/article"
```

Optional: set a Hugging Face access token to avoid stricter rate limits.

```bash
export HF_API_TOKEN="your_token"
python hf_summarize_cli.py "https://example.de/article"
```

## Models used

- Translation (German → English): `Helsinki-NLP/opus-mt-de-en`
- Summarization (English): `facebook/bart-large-cnn`
