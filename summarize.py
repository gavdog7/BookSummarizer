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

def estimate_tokens(word_count):
    """Estimate the number of tokens using a simple word multiplier."""
    return int(word_count * WORD_TO_TOKEN_RATIO)

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

def select_book_to_summarize():
    """Display a list of books and allow the user to select one."""
    # Get books from directory
    books = list_books_in_directory()
    
    # Display books sorted by reading time (longest first)
    print("\nAvailable books to summarize:")
    print("-" * 50)
    for book in books:
        reading_minutes = book["reading_hours"] * 60
        print(f"{book['index']}. {book['filename']}, Time to read: {book['reading_hours']:.1f} hours ({reading_minutes:.0f} minutes)")
    print("-" * 50)
    
    # Ask user to select a book
    while True:
        try:
            selection = input("\nWhich book would you like to summarize? (enter number 1-" + str(len(books)) + "): ")
            
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
    """Determine if and how to split the file."""
    if input_tokens <= CLAUDE_MAX_INPUT_TOKENS and desired_output_tokens <= CLAUDE_MAX_OUTPUT_TOKENS:
        return 1  # No need to split
    
    # Calculate parts based on input tokens
    parts_by_input = math.ceil(input_tokens / CLAUDE_MAX_INPUT_TOKENS)
    
    # Calculate parts based on output tokens
    parts_by_output = math.ceil(desired_output_tokens / CLAUDE_MAX_OUTPUT_TOKENS)
    
    # Use the maximum of the two
    return max(parts_by_input, parts_by_output)

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
        
def save_to_temp_file(filename, part_num, num_parts, summaries, desired_output_tokens):
    """Save progress to a temporary file for potential recovery."""
    temp_dir = Path("Temp")
    temp_dir.mkdir(exist_ok=True)
    
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
        
    return temp_file

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
    
    # Find matching temp files
    temp_files = list(temp_dir.glob(f"{file_base}_temp*"))
    
    # Delete all matching temp files
    for temp_file in temp_files:
        try:
            temp_file.unlink()
        except Exception as e:
            print(f"Warning: Could not delete temp file {temp_file}: {str(e)}")

def main():
    args = parse_arguments()
    
    print(f"\n===== Book Summarization Tool =====\n")
    
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
            resume_choice = input("\nWould you like to resume this job? (y/n): ").lower()
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
                token_count = estimate_tokens(word_count)
                
                # Split text
                parts = split_text(text, num_parts)
                
                # Skip to API setup
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
        token_count = estimate_tokens(word_count)
        reading_hours = estimate_reading_time(word_count)
        
        print(f"Total words: {word_count:,}")
        print(f"Estimated tokens: {token_count:,}")
        print(f"Estimated full reading time: {reading_hours:.2f} hours ({reading_hours * 60:.0f} minutes)")
        
        # Ask user for desired reading time
        while True:
            try:
                desired_hours = float(input("\nHow many hours would you like to spend reading the summary? "))
                if desired_hours <= 0:
                    print("Please enter a positive number.")
                    continue
                
                # Check if user wants to expand the file (make summary longer than original)
                if desired_hours > reading_hours:
                    print("\nWARNING: You're attempting to create a summary that would take longer")
                    print(f"to read ({desired_hours:.2f} hours) than the original document ({reading_hours:.2f} hours).")
                    print("This is experimental and may not produce good results.")
                    
                    while True:
                        expand_choice = input("Are you sure you want to proceed? (y/n): ").lower()
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
        print(f"For a {desired_hours:.2f} hour summary, we'll target approximately {desired_output_tokens:,} tokens.")
        
        # Determine split strategy
        num_parts = determine_split_strategy(token_count, desired_output_tokens)
        
        # Present approach
        if num_parts == 1:
            print("\nThis summarization task fits within Claude 3.7's context window. No need to split the file.")
        else:
            print(f"\nTo summarize this book, I'll need to split this file into {num_parts} parts")
            print(f"so it can be summarized by Claude 3.7. I'll include a 5% overlap in each part")
            print(f"for cohesiveness of the summary.")
        
        # Ask user to proceed or modify
        while True:
            choice = input("Proceed? (yes/no/modify): ").lower()
            
            if choice in ["yes", "y"]:
                break
            elif choice in ["no", "n"]:
                print("Exiting.")
                sys.exit(0)
            elif choice in ["modify", "m"]:
                # Allow user to modify parameters
                while True:
                    try:
                        num_parts = int(input(f"Enter new number of parts (current: {num_parts}): "))
                        if num_parts <= 0:
                            print("Please enter a positive number.")
                            continue
                        break
                    except ValueError:
                        print("Please enter a valid number.")
                
                overlap_percentage = 5  # Default
                while True:
                    try:
                        overlap_percentage = float(input("Enter new overlap percentage (current: 5): "))
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
            choice = input("Would you like to proceed? (yes/no): ").lower()
            
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
    
    # Check for API key in .env file first
    load_dotenv()
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    
    # If API key not found in environment, ask user for it
    if not api_key:
        api_key = input("\nPlease enter your Anthropic API key: ")
    
    # Basic validation of API key format
    if not api_key.startswith("sk-ant-"):
        print("\nWarning: The API key doesn't follow the expected format (should start with 'sk-ant-').")
        print("This may cause authentication issues.")
        
        while True:
            continue_choice = input("Do you still want to continue? (y/n): ").lower()
            if continue_choice in ['y', 'yes']:
                break
            elif continue_choice in ['n', 'no']:
                print("Exiting.")
                sys.exit(0)
            else:
                print("Please enter 'y' or 'n'.")
    
    # Initialize client with proper API key and base URL
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
    booksout_dir.mkdir(exist_ok=True)
    
    # Prepare file paths
    txt_output = booksout_dir / f"{file_base}_summary.txt"
    
    # Save text file
    with open(txt_output, 'w', encoding='utf-8') as f:
        f.write(combined_summary)
    
    # Move the original file to BooksOut (if it's not already there)
    original_file = Path(filename)
    destination_file = booksout_dir / original_file.name
    
    # Only move the file if it's not already in BooksOut
    if str(original_file.parent) != str(booksout_dir):
        try:
            # Copy the original file rather than move it (safer)
            with open(original_file, 'rb') as src, open(destination_file, 'wb') as dst:
                dst.write(src.read())
            print(f"Original file copied to: {destination_file}")
        except Exception as e:
            print(f"Warning: Could not copy original file: {str(e)}")
    
    print(f"\nSummarization complete!")
    print(f"Summary saved as: {txt_output}")
    
    # Optionally create EPUB if pandoc is available
    try:
        epub_output = booksout_dir / f"{file_base}_summary.epub"
        subprocess.run(
            ["pandoc", str(txt_output), "-o", str(epub_output), "--metadata", f"title=Summary of {file_base}"],
            check=True, capture_output=True
        )
        print(f"EPUB version saved as: {epub_output}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Note: EPUB creation skipped (requires pandoc to be installed)")
        
    # Clean up temp files
    cleanup_temp_files(filename)

if __name__ == "__main__":
    main()