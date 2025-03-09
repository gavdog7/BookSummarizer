# BookSummarizer

A lightweight command-line utility that automates the process of summarizing books and lengthy text documents using Claude 3.7 AI.

## Features

- Interactive book selection from the BooksIn directory
- Estimates reading time of original documents
- Specifies desired reading time for summaries
- Automatically splits large documents when needed
- Generates AI-powered summaries via the Anthropic Claude API
- Saves summaries in both text and EPUB formats (when pandoc is available)
- Interruption recovery via temporary files
- Cost estimation before processing

## Directory Structure

- `BooksIn/`: Place your books here to be summarized
- `BooksOut/`: Generated summaries and original files are saved here after processing
- `Temp/`: Temporary files for recovering interrupted summarization jobs
- `summarize.py`: Main script for generating summaries

## Usage

### Basic Usage

Run the interactive script:

```
python summarize.py
```

The script will:
1. Present a list of available books in the BooksIn directory 
2. Allow you to select a book to summarize
3. Show you information about the book and ask how long you want the summary to be
4. Process the book with Claude 3.7 to create a summary
5. Save the summary and original book to the BooksOut directory

### Command Line Usage

You can also specify a specific file to summarize:

```
python summarize.py /path/to/your/book.txt
```

### File Formats

Currently supported formats:
- Text files (.txt)
- EPUB files (.epub) - requires pandoc to be installed

## Requirements

- Python 3.6+
- Anthropic Python library
- Python dotenv library
- Optional: pandoc (for EPUB support)

## API Key

The script requires an Anthropic API key to function. You can:
1. Set the key in a `.env` file with `ANTHROPIC_API_KEY=sk-ant-your-key-here`
2. Enter the key when prompted by the script

## Resuming Interrupted Jobs

If the summarization process is interrupted, the script saves progress to temporary files. When you run the script again, it will detect any interrupted jobs and offer to resume them.