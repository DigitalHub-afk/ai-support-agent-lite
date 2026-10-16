# AI Support Agent Lite (Python + RAG)

A minimal AI support agent that answers customer questions using **your own documents**. Drop `.md` or `.txt` files in a folder and ask away. No vector database, just 3 dependencies.

```
Customer: Do you offer free shipping?
Agent: Yes! Shipping is free on orders over $50. [shipping.md]
```

## Quick start

```sh
git clone https://github.com/DigitalHub-afk/ai-support-agent-lite.git
cd ai-support-agent-lite
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env    # add your Anthropic API key
python agent.py
```

Requires Python 3.9+ and an [Anthropic API key](https://console.anthropic.com/). You only pay for your API usage.

## How it works

1. Your documents in `knowledge/` are split into chunks and indexed with TF-IDF.
2. For each question, the most relevant chunks are retrieved.
3. Claude answers using only that context and cites the source file.

## Lite vs. full template

| Feature | Lite (free) | Full template |
|---|---|---|
| Answers from your documents (RAG) | Yes | Yes |
| Source citations | Yes | Yes |
| Conversation memory | No | Yes |
| Tool use (the agent decides when to search) | No | Yes |
| Ticket creation for human follow-up | No | Yes |
| Commercial license for client work | MIT | Yes |
| Extended README and customization guide | No | Yes |

**Want the full version?** [Get the AI Support Agent Template](YOUR_STORE_URL).

## Limitations

- Retrieval is keyword-based (TF-IDF), so questions and documents should be in the same language.
- Each question is answered independently (no memory).

## License

MIT. See [LICENSE](LICENSE).
