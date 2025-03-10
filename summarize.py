#!/usr/bin/env python3
"""
Simple Book Summarization Tool

Usage: summarize.py <filename>

This script summarizes books using Claude 3.7 API, with minimal dependencies
and a streamlined interface.
"""

import os
import sys
import math
import time
import argparse
import subprocess
import glob
import json
from pathlib import Path
import anthropic
from dotenv import load_dotenv

# Optional rich formatting
RICH_FORMATTING = False
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn
    RICH_FORMATTING = True
except ImportError:
    # Rich is not available, we'll use standard formatting
    pass

# Constants
WORDS_PER_MINUTE = 240
WORD_TO_TOKEN_RATIO = 1.3  # Simple multiplier for estimating tokens from words
CLAUDE_MAX_INPUT_TOKENS = 200000
CLAUDE_MAX_OUTPUT_TOKENS = 128000
CLAUDE_MODEL = "claude-3-7-sonnet-20250219"
CLAUDE_INPUT_PRICE_PER_1M = 15.00  # $15 per million input tokens
CLAUDE_OUTPUT_PRICE_PER_1M = 75.00  # $75 per million output tokens

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Summarize books using Claude 3.7")
    parser.add_argument("filename", nargs="?", help="Path to the file to summarize")
    return parser.parse_args()

def convert_epub_to_text(filename):
    """Convert EPUB to text using pandoc if available."""
    output_file = f"{Path(filename).stem}_text.txt"
    
    try:
        subprocess.run(["pandoc", filename, "-o", output_file], 
                      check=True, capture_output=True)
        with open(output_file, 'r', encoding='utf-8') as f:
            text = f.read()
        os.remove(output_file)  # Clean up temporary file
        return text
    except subprocess.CalledProcessError:
        print("Error: Failed to convert EPUB to text. Is pandoc installed?")
        sys.exit(1)
    except FileNotFoundError:
        print("Error: pandoc not found. Please install pandoc to process EPUB files.")
        print("For macOS: brew install pandoc")
        print("For Ubuntu/Debian: sudo apt-get install pandoc")
        print("For Windows: https://pandoc.org/installing.html")
        sys.exit(1)

def read_file(filename):
    """Read file contents, converting if necessary."""
    file_ext = Path(filename).suffix.lower()
    
    if file_ext == '.txt':
        with open(filename, 'r', encoding='utf-8') as f:
            return f.read()
    elif file_ext == '.epub':
        return convert_epub_to_text(filename)
    else:
        print(f"Unsupported file format: {file_ext}")
        print("Currently supporting: .txt and .epub files")
        sys.exit(1)

def count_words(text):
    """Count the number of words in the text."""
    return len(text.split())

def estimate_tokens(word_count, text=None, client=None):
    """
    Smart token estimation that uses a combination of approaches:
    1. Start with a rough estimation based on word count
    2. For large texts, sample a portion to get an accurate token ratio from API
    3. Scale up the token estimate based on the sample
    4. Fall back to simple word multiplier if API call fails
    """
    # Always start with a rough estimation as baseline
    rough_token_estimate = int(word_count * WORD_TO_TOKEN_RATIO)
    
    # If no text or client provided, just return the rough estimate
    if text is None or client is None:
        return rough_token_estimate
    
    try:
        # For small texts (under ~100K tokens), just use the API directly
        if rough_token_estimate < 100000:
            print("Text is small enough for direct token counting...")
            response = client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=1,
                messages=[{"role": "user", "content": text}]
            )
            return response.usage.input_tokens
        
        # For larger texts, use a sampling approach
        print("Text is large, using sampling approach for token estimation...")
        
        # Calculate how much text to sample (aim for around 50K tokens sample size)
        target_sample_size = 50000
        sample_ratio = min(1.0, target_sample_size / rough_token_estimate)
        
        # Get a representative sample of the text
        words = text.split()
        sample_word_count = int(len(words) * sample_ratio)
        
        # Take sample from different parts of the text for better representation
        samples = []
        
        # Beginning sample (40%)
        begin_size = int(sample_word_count * 0.4)
        samples.append(" ".join(words[:begin_size]))
        
        # Middle sample (30%)
        mid_start = len(words) // 2 - int(sample_word_count * 0.15)
        mid_end = mid_start + int(sample_word_count * 0.3)
        samples.append(" ".join(words[mid_start:mid_end]))
        
        # End sample (30%)
        end_size = int(sample_word_count * 0.3)
        samples.append(" ".join(words[-end_size:]))
        
        # Join samples with markers
        sampled_text = "\n\n[...]\n\n".join(samples)
        
        # Count words in the sample
        sample_word_count = len(sampled_text.split())
        
        # Get token count from API for the sample
        print("Counting tokens in the sample...")
        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=1,
            messages=[{"role": "user", "content": sampled_text}]
        )
        sample_token_count = response.usage.input_tokens
        
        # Calculate token-to-word ratio from the sample
        token_word_ratio = sample_token_count / sample_word_count
        
        # Apply this ratio to the full text
        estimated_tokens = int(word_count * token_word_ratio)
        
        print(f"Sample token-to-word ratio: {token_word_ratio:.3f}")
        print(f"Estimated tokens based on sample: {estimated_tokens:,}")
        
        return estimated_tokens
        
    except Exception as e:
        print(f"Warning: Could not use Anthropic's tokenizer: {str(e)}")
        print("Falling back to simple estimation method.")
        return rough_token_estimate

def estimate_reading_time(word_count):
    """Estimate reading time in hours based on words per minute."""
    minutes = word_count / WORDS_PER_MINUTE
    return minutes / 60  # Convert to hours

def list_books_in_directory(directory="BooksIn"):
    """List all text and epub files in the directory and return their reading times."""
    # Get path to the BooksIn directory
    books_dir = Path(directory)
    
    # Ensure the directory exists
    if not books_dir.exists() or not books_dir.is_dir():
        print(f"Error: {directory} directory not found.")
        sys.exit(1)
    
    # Find all .txt and .epub files
    book_files = []
    for ext in ["*.txt", "*.epub"]:
        book_files.extend(books_dir.glob(ext))
    
    # Sort files by size (largest first as a proxy for longest reading time)
    book_files.sort(key=lambda x: x.stat().st_size, reverse=True)
    
    # No books found
    if not book_files:
        print(f"No books found in {directory} directory.")
        print("Please add .txt or .epub files to this directory.")
        sys.exit(1)
    
    # Process each file to get word count and reading time
    books = []
    for i, file_path in enumerate(book_files):
        try:
            # Read file contents
            text = read_file(file_path)
            
            # Count words and estimate reading time
            word_count = count_words(text)
            reading_hours = estimate_reading_time(word_count)
            
            # Add to books list
            books.append({
                "index": i + 1,
                "filename": file_path.name,
                "path": str(file_path),
                "word_count": word_count,
                "reading_hours": reading_hours
            })
        except Exception as e:
            print(f"Warning: Could not process {file_path.name}: {str(e)}")
    
    return books

def check_for_exit(input_text):
    """Check if the user wants to exit the program."""
    if input_text.lower() in ["exit", "quit", "q"]:
        print("\nExiting program.")
        sys.exit(0)
    return input_text

def select_book_to_summarize():
    """Display a list of books and allow the user to select one."""
    # Get books from directory
    books = list_books_in_directory()
    
    # Display books sorted by reading time (longest first)
    print("\nAvailable books to summarize:")
    print("-" * 50)
    for book in books:
        # Format reading time more concisely
        if book["reading_hours"] < 1:
            # Convert to minutes for very short books
            reading_minutes = book["reading_hours"] * 60
            reading_time = f"~{reading_minutes:.0f}m to read"
        else:
            # Use hours for longer books, with 1 decimal point
            reading_time = f"~{book['reading_hours']:.1f}h to read"
        
        print(f"{book['index']}. {book['filename']}, {reading_time}")
    print("-" * 50)
    print("Type 'exit' at any prompt to quit the program.")
    
    # Ask user to select a book
    while True:
        try:
            selection = input("\nWhich book would you like to summarize? (enter number 1-" + str(len(books)) + "): ")
            
            # Check for exit command
            check_for_exit(selection)
            
            # Check if input is a number
            if not selection.isdigit():
                print("Please enter a number.")
                continue
                
            book_index = int(selection)
            
            # Check if the number is in the valid range
            if book_index < 1 or book_index > len(books):
                print(f"Please enter a number between 1 and {len(books)}.")
                continue
                
            # Return the selected book path
            selected_book = books[book_index - 1]
            print(f"\nSelected: {selected_book['filename']}")
            return selected_book["path"]
            
        except ValueError:
            print("Please enter a valid number.")
            continue

def calculate_output_tokens(hours):
    """Calculate output tokens based on desired reading time."""
    # Convert hours to minutes, then to words, then approximate tokens
    minutes = hours * 60
    words = minutes * WORDS_PER_MINUTE
    return int(words * WORD_TO_TOKEN_RATIO)

def determine_split_strategy(input_tokens, desired_output_tokens):
    """
    Determine if and how to split the file.
    
    Returns a tuple of (num_parts, limiting_factor) where limiting_factor 
    is a string indicating whether input or output tokens are the limiting factor.
    """
    if input_tokens <= CLAUDE_MAX_INPUT_TOKENS and desired_output_tokens <= CLAUDE_MAX_OUTPUT_TOKENS:
        return 1, "none"  # No need to split
    
    # Calculate parts based on input tokens
    parts_by_input = math.ceil(input_tokens / CLAUDE_MAX_INPUT_TOKENS)
    
    # Calculate parts based on output tokens
    parts_by_output = math.ceil(desired_output_tokens / CLAUDE_MAX_OUTPUT_TOKENS)
    
    # Determine the limiting factor
    if parts_by_input >= parts_by_output:
        return parts_by_input, "input"
    else:
        return parts_by_output, "output"

def split_text(text, num_parts, overlap_percentage=5):
    """Split text into parts with overlap."""
    if num_parts == 1:
        return [{"part": 1, "text": text}]
    
    words = text.split()
    parts = []
    
    # Base number of words per part
    base_size = len(words) // num_parts
    
    # Calculate overlap in words
    overlap_size = int(base_size * overlap_percentage / 100)
    
    for i in range(num_parts):
        if i == 0:
            # First part
            start = 0
            end = base_size + overlap_size
            part_words = words[start:end]
            
            part_text = " ".join(part_words)
            parts.append({
                "part": i + 1,
                "text": part_text,
                "new_text": " ".join(words[start:base_size]),
                "overlap_next": " ".join(words[base_size:end])
            })
            
        elif i == num_parts - 1:
            # Last part
            start = i * base_size - overlap_size
            part_words = words[start:]
            
            part_text = " ".join(part_words)
            parts.append({
                "part": i + 1,
                "text": part_text,
                "overlap_prev": " ".join(words[start:start + overlap_size]),
                "new_text": " ".join(words[start + overlap_size:])
            })
            
        else:
            # Middle parts
            start = i * base_size - overlap_size
            end = (i + 1) * base_size + overlap_size
            part_words = words[start:end]
            
            part_text = " ".join(part_words)
            parts.append({
                "part": i + 1,
                "text": part_text,
                "overlap_prev": " ".join(words[start:start + overlap_size]),
                "new_text": " ".join(words[start + overlap_size:start + base_size + overlap_size]),
                "overlap_next": " ".join(words[start + base_size + overlap_size:end])
            })
    
    return parts

def format_part_for_claude(part, num_parts, desired_output_tokens):
    """Format text part for Claude API."""
    if num_parts == 1:
        prompt = f"""
        Summarize the following text in about {desired_output_tokens} tokens. 
        Focus on the main ideas, key arguments, and important details.
        
        TEXT TO SUMMARIZE:
        {part["text"]}
        """
        return prompt
    
    # For multi-part summarization
    if part["part"] == 1:
        prompt = f"""
        This is part {part["part"]} of {num_parts} of a longer text. Summarize this part in about {desired_output_tokens // num_parts} tokens.
        Focus on the main ideas, key arguments, and important details.
        
        TEXT TO SUMMARIZE:
        <Begin new text to be summarized for part {part["part"]} of {num_parts}>
        {part["new_text"]}
        <Begin overlap text for part {part["part"] + 1} of {num_parts}>
        {part["overlap_next"]}
        <End overlap text for part {part["part"] + 1} of {num_parts}>
        <End new text to be summarized for part {part["part"]} of {num_parts}>
        """
    
    elif part["part"] == num_parts:
        prompt = f"""
        This is part {part["part"]} of {num_parts} of a longer text. Summarize this part in about {desired_output_tokens // num_parts} tokens.
        Focus on the main ideas, key arguments, and important details.
        
        TEXT TO SUMMARIZE:
        <Begin overlap text from part {part["part"] - 1} of {num_parts} (5% overlap)>
        {part["overlap_prev"]}
        <Begin new text to be summarized for part {part["part"]} of {num_parts}>
        {part["new_text"]}
        <End new text to be summarized for part {part["part"]} of {num_parts}>
        """
    
    else:
        prompt = f"""
        This is part {part["part"]} of {num_parts} of a longer text. Summarize this part in about {desired_output_tokens // num_parts} tokens.
        Focus on the main ideas, key arguments, and important details.
        
        TEXT TO SUMMARIZE:
        <Begin overlap text from part {part["part"] - 1} of {num_parts} (5% overlap)>
        {part["overlap_prev"]}
        <Begin new text to be summarized for part {part["part"]} of {num_parts}>
        {part["new_text"]}
        <Begin overlap text for part {part["part"] + 1} of {num_parts}>
        {part["overlap_next"]}
        <End overlap text for part {part["part"] + 1} of {num_parts}>
        <End new text to be summarized for part {part["part"]} of {num_parts}>
        """
    
    return prompt

def estimate_cost(num_parts, input_tokens, output_tokens):
    """Estimate the cost of API calls."""
    # Calculate cost per part
    input_cost_per_part = (input_tokens / num_parts) * (CLAUDE_INPUT_PRICE_PER_1M / 1000000)
    output_cost_per_part = (output_tokens / num_parts) * (CLAUDE_OUTPUT_PRICE_PER_1M / 1000000)
    cost_per_part = input_cost_per_part + output_cost_per_part
    
    # Calculate total cost
    total_cost = cost_per_part * num_parts
    
    return cost_per_part, total_cost

def summarize_with_claude(client, prompt, max_tokens):
    """Send text to Claude API for summarization."""
    print(f"  Sending request to Claude API (max tokens: {max_tokens})...")
    
    try:
        # Make the API request with proper error handling
        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=max_tokens,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        # Check for successful response
        if hasattr(response, 'content') and response.content:
            return response.content[0].text
        else:
            print("Warning: Received empty response from Claude API")
            return "Summary unavailable due to API response error."
    except anthropic.APIError as e:
        if e.status_code == 404:
            print(f"Error 404: The requested model '{CLAUDE_MODEL}' was not found.")
            print("Please check if the model name is correct and if you have access to it.")
        elif e.status_code == 401:
            print("Error 401: Authentication failed. Your API key may be invalid.")
        elif e.status_code == 429:
            print("Error 429: Rate limit exceeded. Please try again later.")
        else:
            print(f"API Error ({e.status_code}): {str(e)}")
        raise
    except Exception as e:
        print(f"Error calling Claude API: {str(e)}")
        raise

def show_progress(part, num_parts, start_time=None):
    """Show a simple progress indicator."""
    if start_time:
        elapsed = time.time() - start_time
        print(f"  Part {part}/{num_parts} completed in {elapsed:.2f} seconds.")
    else:
        print(f"  Processing part {part}/{num_parts}...")
        
def save_text_parts_to_temp(filename, parts):
    """Save split text parts to the Temp folder for recovery if needed."""
    try:
        temp_dir = Path("Temp")
        temp_dir.mkdir(exist_ok=True, mode=0o777)  # Create with full permissions if it doesn't exist
        
        # Try to set permissions if directory already exists
        try:
            os.chmod(temp_dir, 0o777)
        except Exception as e:
            print(f"Warning: Could not set permissions on Temp directory: {str(e)}")
            print("Will attempt to continue anyway...")
        
        # Create a base filename
        file_base = Path(filename).stem
        
        # Save each part to a separate file
        saved_files = []
        for i, part in enumerate(parts):
            part_num = i + 1
            part_file = temp_dir / f"{file_base}_part_{part_num}.txt"
            
            try:
                with open(part_file, 'w', encoding='utf-8') as f:
                    f.write(part["text"])
                    
                # Try to set permissions on the file
                try:
                    os.chmod(part_file, 0o666)  # rw-rw-rw-
                except:
                    pass  # Ignore if we can't set permissions
                    
                saved_files.append(str(part_file))
            except PermissionError:
                print(f"Warning: Permission denied when writing to {part_file}")
                print("Will continue without saving this part to temp storage.")
            except Exception as e:
                print(f"Warning: Could not save part {part_num} to temp file: {str(e)}")
        
        # Only save manifest if we successfully saved at least one part
        if saved_files:
            # Also save a manifest file with metadata
            manifest_file = temp_dir / f"{file_base}_parts_manifest.json"
            manifest_data = {
                "original_file": filename,
                "num_parts": len(parts),
                "part_files": saved_files,
                "timestamp": time.time()
            }
            
            try:
                with open(manifest_file, 'w', encoding='utf-8') as f:
                    json.dump(manifest_data, f, indent=2)
                    
                # Try to set permissions on the manifest file
                try:
                    os.chmod(manifest_file, 0o666)  # rw-rw-rw-
                except:
                    pass  # Ignore if we can't set permissions
                    
                print(f"Split text parts saved to Temp folder")
            except Exception as e:
                print(f"Warning: Could not save manifest file: {str(e)}")
        else:
            print("Warning: No parts could be saved to temp storage.")
            print("If the process is interrupted, you may not be able to resume.")
            
        return saved_files
    except Exception as e:
        print(f"Warning: Error saving to temp files: {str(e)}")
        print("Will continue without temp storage (no resume capability if interrupted).")
        return []

def save_to_temp_file(filename, part_num, num_parts, summaries, desired_output_tokens):
    """Save progress to a temporary file for potential recovery."""
    try:
        temp_dir = Path("Temp")
        temp_dir.mkdir(exist_ok=True, mode=0o777)  # Create with full permissions if it doesn't exist
        
        # Try to set permissions if directory already exists
        try:
            os.chmod(temp_dir, 0o777)
        except Exception as e:
            print(f"Warning: Could not set permissions on Temp directory: {str(e)}")
            print("Will attempt to continue anyway...")
        
        # Create a unique temp file name based on the original file
        file_base = Path(filename).stem
        temp_file = temp_dir / f"{file_base}_temp.json"
        
        # Prepare data to save
        temp_data = {
            "filename": filename,
            "part_num": part_num,
            "num_parts": num_parts,
            "summaries": summaries,
            "desired_output_tokens": desired_output_tokens,
            "timestamp": time.time()
        }
        
        # Save to file
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump(temp_data, f, indent=2)
        
        # Try to set permissions on the file
        try:
            os.chmod(temp_file, 0o666)  # rw-rw-rw-
        except:
            pass  # Ignore if we can't set permissions
            
        return temp_file
    except PermissionError:
        print(f"Warning: Permission denied when saving progress to temp file.")
        print("If the process is interrupted, you may not be able to resume.")
        return None
    except Exception as e:
        print(f"Warning: Could not save progress to temp file: {str(e)}")
        print("If the process is interrupted, you may not be able to resume.")
        return None

def check_for_resumable_jobs():
    """Check for resumable summarization jobs in the Temp directory."""
    temp_dir = Path("Temp")
    
    # If Temp directory doesn't exist, no resumable jobs
    if not temp_dir.exists() or not temp_dir.is_dir():
        return None
        
    # Find all JSON files in Temp
    temp_files = list(temp_dir.glob("*_temp.json"))
    
    if not temp_files:
        return None
        
    # Sort by modification time (newest first)
    temp_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    # Return the most recent temp file
    most_recent = temp_files[0]
    
    try:
        with open(most_recent, 'r', encoding='utf-8') as f:
            temp_data = json.load(f)
            
        # Check if the referenced file still exists
        if not Path(temp_data["filename"]).exists():
            print(f"Warning: Original file {temp_data['filename']} not found.")
            return None
            
        return temp_data
        
    except Exception as e:
        print(f"Warning: Could not read temp file {most_recent}: {str(e)}")
        return None

def cleanup_temp_files(filename):
    """Clean up temporary files after successful completion."""
    temp_dir = Path("Temp")
    
    if not temp_dir.exists() or not temp_dir.is_dir():
        return
        
    # Get the file base name
    file_base = Path(filename).stem
    
    # Find all matching temp files
    temp_files = []
    temp_files.extend(temp_dir.glob(f"{file_base}_temp*"))
    temp_files.extend(temp_dir.glob(f"{file_base}_part_*"))
    temp_files.extend(temp_dir.glob(f"{file_base}_parts_manifest*"))
    
    # Delete all matching temp files
    if temp_files:
        print(f"Cleaning up {len(temp_files)} temporary files...")
    
    for temp_file in temp_files:
        try:
            temp_file.unlink()
        except Exception as e:
            print(f"Warning: Could not delete temp file {temp_file}: {str(e)}")

def main():
    args = parse_arguments()
    
    print(f"\n===== Book Summarization Tool =====\n")
    
    # Check for API key in .env file first
    load_dotenv()
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    
    # If API key not found in environment, ask user for it
    if not api_key:
        user_input = input("\nPlease enter your Anthropic API key (or type 'exit' to quit): ")
        api_key = check_for_exit(user_input)
    
    # Basic validation of API key format
    if not api_key.startswith("sk-ant-"):
        print("\nWarning: The API key doesn't follow the expected format (should start with 'sk-ant-').")
        print("This may cause authentication issues.")
        
        while True:
            user_input = input("Do you still want to continue? (y/n): ").lower()
            continue_choice = check_for_exit(user_input)
            if continue_choice in ['y', 'yes']:
                break
            elif continue_choice in ['n', 'no']:
                print("Exiting.")
                sys.exit(0)
            else:
                print("Please enter 'y' or 'n'.")
    
    # Initialize client with proper API key
    try:
        client = anthropic.Anthropic(api_key=api_key)
        
        # Test the API connection with a minimal request
        print("\nVerifying API connection...")
        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=10,
            messages=[
                {"role": "user", "content": "Say 'API connection successful' and nothing else."}
            ]
        )
        print(f"API connection verified: {response.content[0].text.strip()}")
    except Exception as e:
        print(f"\nError connecting to Claude API: {str(e)}")
        print("\nPossible issues:")
        print("1. The API key may be invalid")
        print("2. You may not have access to the specified model")
        print("3. There may be network connectivity issues")
        print("\nPlease check your API key and try again.")
        sys.exit(1)
    
    # Check for resumable jobs
    resumable_job = check_for_resumable_jobs()
    if resumable_job:
        # Ask if user wants to resume
        book_name = Path(resumable_job["filename"]).name
        part_completed = resumable_job["part_num"]
        total_parts = resumable_job["num_parts"]
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(resumable_job["timestamp"]))
        
        print(f"Found a resumable job from {timestamp}:")
        print(f"Book: {book_name}")
        print(f"Progress: {part_completed}/{total_parts} parts completed")
        
        while True:
            user_input = input("\nWould you like to resume this job? (y/n): ").lower()
            resume_choice = check_for_exit(user_input)
            if resume_choice in ['y', 'yes']:
                # Set up variables from the saved job
                filename = resumable_job["filename"]
                start_part = resumable_job["part_num"] + 1
                num_parts = resumable_job["num_parts"]
                summaries = resumable_job["summaries"]
                desired_output_tokens = resumable_job["desired_output_tokens"]
                
                # Read the file
                text = read_file(filename)
                
                # Recalculate some variables
                word_count = count_words(text)
                token_count = estimate_tokens(word_count, text, client)
                
                # Split text
                parts = split_text(text, num_parts)
                
                # Save parts to Temp folder (for consistency)
                save_text_parts_to_temp(filename, parts)
                
                # Skip to output setup
                resume_mode = True
                break
            elif resume_choice in ['n', 'no']:
                # Start a new job
                resume_mode = False
                break
            else:
                print("Please enter 'y' or 'n'.")
    else:
        resume_mode = False
    
    # Normal startup if not resuming
    if not resume_mode:
        # Determine file to summarize
        filename = args.filename
        if not filename:
            # If no file provided via command line, ask the user to select from BooksIn
            filename = select_book_to_summarize()
        
        # Read file
        print(f"Reading file: {filename}")
        text = read_file(filename)
        
        # Count words and estimate tokens
        word_count = count_words(text)
        print("Counting tokens using Anthropic's tokenizer...")
        token_count = estimate_tokens(word_count, text, client)
        reading_hours = estimate_reading_time(word_count)
        
        # Format reading time more concisely
        if reading_hours < 1:
            # Convert to minutes for very short documents
            reading_minutes = reading_hours * 60
            reading_time = f"~{reading_minutes:.0f}m"
        else:
            # Use hours for longer documents, with 1 decimal point
            reading_time = f"~{reading_hours:.1f}h"
            
        print(f"Total words: {word_count:,}")
        print(f"Estimated tokens: {token_count:,}")
        print(f"Estimated reading time: {reading_time}")
        
        # Ask user for desired reading time
        while True:
            try:
                user_input = input("\nHow many hours would you like to spend reading the summary? (or type 'exit' to quit): ")
                check_for_exit(user_input)
                desired_hours = float(user_input)
                if desired_hours <= 0:
                    print("Please enter a positive number.")
                    continue
                
                # Check if user wants to expand the file (make summary longer than original)
                if desired_hours > reading_hours:
                    # Format the times consistently
                    if reading_hours < 1:
                        original_time = f"{reading_hours * 60:.0f}m"
                    else:
                        original_time = f"{reading_hours:.1f}h"
                        
                    if desired_hours < 1:
                        summary_time = f"{desired_hours * 60:.0f}m"
                    else:
                        summary_time = f"{desired_hours:.1f}h"
                        
                    print("\nWARNING: You're attempting to create a summary that would take longer")
                    print(f"to read (~{summary_time}) than the original document (~{original_time}).")
                    print("This is experimental and may not produce good results.")
                    
                    while True:
                        user_input = input("Are you sure you want to proceed? (y/n): ").lower()
                        expand_choice = check_for_exit(user_input)
                        if expand_choice in ['y', 'yes']:
                            break
                        elif expand_choice in ['n', 'no']:
                            print("Please enter a shorter reading time.")
                            break
                        else:
                            print("Please enter 'y' or 'n'.")
                    
                    if expand_choice in ['n', 'no']:
                        continue  # Go back to entering reading time
                
                break
            except ValueError:
                print("Please enter a valid number.")
        
        # Calculate desired output tokens
        desired_output_tokens = calculate_output_tokens(desired_hours)
        
        # Format summary time consistently
        if desired_hours < 1:
            summary_time = f"{desired_hours * 60:.0f}m"
        else:
            summary_time = f"{desired_hours:.1f}h"
            
        print(f"For a ~{summary_time} summary, we'll target approximately {desired_output_tokens:,} tokens.")
        
        # Determine split strategy
        num_parts, limiting_factor = determine_split_strategy(token_count, desired_output_tokens)
        
        # Present approach
        if num_parts == 1:
            print("\nThis summarization task fits within Claude 3.7's context window. No need to split the file.")
        else:
            print(f"\nTo summarize this book, I'll need to split this file into {num_parts} parts")
            
            # Explain the limiting factor
            if limiting_factor == "input":
                print(f"The split is necessary because the book is too large ({token_count:,} tokens) for")
                print(f"Claude's {CLAUDE_MAX_INPUT_TOKENS:,} token input limit.")
            elif limiting_factor == "output":
                print(f"The split is necessary because your desired summary length ({desired_output_tokens:,} tokens) exceeds")
                print(f"Claude's {CLAUDE_MAX_OUTPUT_TOKENS:,} token output limit per request.")
            
            print(f"I'll include a 5% overlap in each part for cohesiveness of the summary.")
        
        # Ask user to proceed or modify
        while True:
            user_input = input("Proceed? (yes/no/modify): ").lower()
            choice = check_for_exit(user_input)
            
            if choice in ["yes", "y"]:
                break
            elif choice in ["no", "n"]:
                print("Exiting.")
                sys.exit(0)
            elif choice in ["modify", "m"]:
                # Allow user to modify parameters
                while True:
                    try:
                        user_input = input(f"Enter new number of parts (current: {num_parts}): ")
                        check_for_exit(user_input)
                        num_parts = int(user_input)
                        if num_parts <= 0:
                            print("Please enter a positive number.")
                            continue
                        break
                    except ValueError:
                        print("Please enter a valid number.")
                
                overlap_percentage = 5  # Default
                while True:
                    try:
                        user_input = input("Enter new overlap percentage (current: 5): ")
                        check_for_exit(user_input)
                        overlap_percentage = float(user_input)
                        if overlap_percentage < 0 or overlap_percentage > 50:
                            print("Please enter a number between 0 and 50.")
                            continue
                        break
                    except ValueError:
                        print("Please enter a valid number.")
                
                print(f"\nUpdated approach: Split into {num_parts} parts with {overlap_percentage}% overlap.")
                break
            else:
                print("Please enter 'yes', 'no', or 'modify'.")
        
        # Estimate cost
        cost_per_part, total_cost = estimate_cost(num_parts, token_count, desired_output_tokens)
        
        print(f"\nCost estimate:")
        print(f"{num_parts} parts with approximately {token_count//num_parts:,} input tokens")
        print(f"and {desired_output_tokens//num_parts:,} output tokens per part")
        print(f"will cost approximately ${cost_per_part:.2f} each.")
        print(f"Total estimated cost: ${total_cost:.2f}")
        
        # Ask user to proceed
        while True:
            user_input = input("Would you like to proceed? (yes/no): ").lower()
            choice = check_for_exit(user_input)
            
            if choice in ["yes", "y"]:
                break
            elif choice in ["no", "n"]:
                print("Exiting.")
                sys.exit(0)
            else:
                print("Please enter 'yes' or 'no'.")
        
        # Initialize empty summaries list and starting part
        summaries = []
        start_part = 1
        
        # Split text
        parts = split_text(text, num_parts)
        
        # Save parts to Temp folder
        save_text_parts_to_temp(filename, parts)
    
    # Process parts with the client initialized earlier
    
    # Process each part
    print("\nSummarizing text...")
    
    for i in range(start_part - 1, num_parts):
        part_num = i + 1
        part = parts[i]
        show_progress(part_num, num_parts)
        
        # Format prompt
        prompt = format_part_for_claude(part, num_parts, desired_output_tokens)
        
        # Prepare tokens for this part
        max_tokens = CLAUDE_MAX_OUTPUT_TOKENS
        if num_parts > 1:
            max_tokens = min(CLAUDE_MAX_OUTPUT_TOKENS, desired_output_tokens // num_parts)
        
        # Start timer
        start_time = time.time()
        
        try:
            # Call API
            summary = summarize_with_claude(client, prompt, max_tokens)
            
            # Add to summaries
            summaries.append(summary)
            
            # Save progress to temp file
            temp_file = save_to_temp_file(filename, part_num, num_parts, summaries, desired_output_tokens)
            # Note: temp_file may be None if saving failed, but we can continue anyway
            
            # Show progress
            show_progress(part_num, num_parts, start_time)
            
            # Remaining time estimate
            if part_num < num_parts:
                elapsed = time.time() - start_time
                est_remaining = elapsed * (num_parts - part_num)
                m, s = divmod(est_remaining, 60)
                print(f"  Estimated time remaining: {int(m)} minutes {int(s)} seconds")
                
        except KeyboardInterrupt:
            print("\nOperation interrupted. Progress saved to temporary file.")
            print(f"You can resume this job later by running the script again.")
            sys.exit(0)
        except Exception as e:
            print(f"\nError: {str(e)}")
            print("\nProgress saved to temporary file.")
            print(f"You can resume this job later by running the script again.")
            sys.exit(1)
    
    # Combine summaries
    combined_summary = "\n\n".join(summaries)
    
    # Save results
    file_base = Path(filename).stem
    
    # Create BooksOut directory if it doesn't exist
    booksout_dir = Path("BooksOut")
    try:
        booksout_dir.mkdir(exist_ok=True, mode=0o777)
        # Try to set permissions if directory already exists
        try:
            os.chmod(booksout_dir, 0o777)
        except:
            pass  # Ignore if we can't set permissions
    except Exception as e:
        print(f"Warning: Could not create or set permissions on BooksOut directory: {str(e)}")
        print("Will attempt to continue anyway...")
    
    # Prepare file paths
    txt_output = booksout_dir / f"{file_base}_summary.txt"
    
    # Save text file
    try:
        with open(txt_output, 'w', encoding='utf-8') as f:
            f.write(combined_summary)
            
        # Try to set permissions on the output file
        try:
            os.chmod(txt_output, 0o666)  # rw-rw-rw-
        except:
            pass  # Ignore if we can't set permissions
            
        print(f"Summary saved as: {txt_output}")
    except PermissionError:
        print(f"Error: Permission denied when writing summary to {txt_output}")
        # Try to save to the current directory as a fallback
        fallback_file = Path(f"{file_base}_summary.txt")
        try:
            with open(fallback_file, 'w', encoding='utf-8') as f:
                f.write(combined_summary)
            print(f"Summary saved to current directory instead: {fallback_file}")
        except Exception as e:
            print(f"Error: Could not save summary file at all: {str(e)}")
    except Exception as e:
        print(f"Error: Could not save summary file: {str(e)}")
    
    # Move the original file to BooksOut (if it's not already there)
    original_file = Path(filename)
    destination_file = booksout_dir / original_file.name
    
    # Only move the file if it's not already in BooksOut
    if str(original_file.parent) != str(booksout_dir):
        try:
            # Copy the original file rather than move it (safer)
            with open(original_file, 'rb') as src, open(destination_file, 'wb') as dst:
                dst.write(src.read())
                
            # Try to set permissions on the destination file
            try:
                os.chmod(destination_file, 0o666)  # rw-rw-rw-
            except:
                pass  # Ignore if we can't set permissions
                
            print(f"Original file copied to: {destination_file}")
        except Exception as e:
            print(f"Warning: Could not copy original file: {str(e)}")
    
    print(f"\nSummarization complete!")
    
    # Optionally create EPUB if pandoc is available
    try:
        epub_output = booksout_dir / f"{file_base}_summary.epub"
        subprocess.run(
            ["pandoc", str(txt_output), "-o", str(epub_output), "--metadata", f"title=Summary of {file_base}"],
            check=True, capture_output=True
        )
        
        # Try to set permissions on the EPUB file
        try:
            os.chmod(epub_output, 0o666)  # rw-rw-rw-
        except:
            pass  # Ignore if we can't set permissions
            
        print(f"EPUB version saved as: {epub_output}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Note: EPUB creation skipped (requires pandoc to be installed)")
    except Exception as e:
        print(f"Warning: Could not create EPUB version: {str(e)}")
        
    # Clean up temp files
    cleanup_temp_files(filename)

if __name__ == "__main__":
    main()