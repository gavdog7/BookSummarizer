#!/usr/bin/env python3
"""
Test script for verifying the tokenizer functionality in the BookSummarizer app.
"""
import anthropic
import os
from dotenv import load_dotenv

# Load API key
load_dotenv()
api_key = os.environ.get("ANTHROPIC_API_KEY")

if not api_key:
    print("Error: No API key found. Set ANTHROPIC_API_KEY in your .env file.")
    exit(1)

# Initialize client
client = anthropic.Anthropic(api_key=api_key)

# Define the model
MODEL = "claude-3-7-sonnet-20250219"

# Test text
TEST_TEXT = """
This is a sample text to test the tokenizer functionality.
We'll see how many tokens this text contains according to Claude's tokenizer.
The number of tokens will be returned by the tokenizer API.
"""

# Function to count tokens using the Anthropic API
def count_claude_tokens(text):
    try:
        print("Attempting to count tokens...")
        
        # For newer Anthropic API versions, we need to make a message with a small output
        # and check the usage information to get token counts
        response = client.messages.create(
            model=MODEL,
            max_tokens=10,
            messages=[
                {"role": "user", "content": text}
            ]
        )
        
        # Extract input token count from usage info
        input_tokens = response.usage.input_tokens
        
        print("Token counting successful!")
        return input_tokens
    except Exception as e:
        print(f"Error counting tokens: {str(e)}")
        return None

# Run the test
token_count = count_claude_tokens(TEST_TEXT)

if token_count is not None:
    print(f"The test text contains {token_count} tokens according to Claude's tokenizer.")
    print("Tokenizer test passed!")
else:
    print("Tokenizer test failed.")