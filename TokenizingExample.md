import anthropic

client = anthropic.Anthropic(api_key='your_api_key_here')

def count_claude_tokens(text: str, model: str = "claude-3.5-sonnet-20240620") -> int:
    response = client.count_tokens(
        model=model,
        messages=[{"role": "user", "content": text}]
    )
    return response.input_tokens

# Usage example
text_variable = "<your-text-content-here>"
token_count = count_claude_tokens(text_variable)
print(f"Estimated tokens: {token_count}")
