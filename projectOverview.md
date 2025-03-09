# Book Summarization Tool

## Project Overview

The Book Summarization Tool is a lightweight command-line utility that automates the process of summarizing books and lengthy text documents using Claude 3.7 AI. It enables users to convert large volumes of text into concise, readable summaries tailored to their specific time constraints.

## Core Functionality

This tool allows users to:
- Process text files (.txt) and EPUB documents
- Estimate reading time of the original document 
- Specify their desired reading time for the summary
- Automatically split large documents when needed
- Generate AI-powered summaries via the Anthropic Claude API
- Save summaries in both text and EPUB formats (when pandoc is available)

## Technical Approach

The application follows a minimalist design philosophy focused on simplicity and usability:

- **Text Processing**: Uses a straightforward word-to-token ratio (1.3) for estimating document size
- **Smart Splitting**: Divides documents into parts with configurable overlap for context preservation
- **Context Management**: Carefully handles Claude's context window limitations (200K input/128K output tokens)
- **Cost Estimation**: Calculates and displays estimated API costs before proceeding
- **Robust API Handling**: Validates credentials and provides detailed error handling

## Key Features

- **Minimal Dependencies**: Requires only the Anthropic Python library
- **Interactive UI**: Simple command-line interface with sensible defaults
- **Progress Tracking**: Shows real-time progress and time estimates
- **User Customization**: Allows modification of splitting strategy and overlap percentage
- **Expansion Warning**: Alerts users when attempting to create summaries longer than the original
- **Format Flexibility**: Works with plain text and EPUB formats

## Implementation Details

The tool is implemented as a single Python script with optional pandoc integration for enhanced format support. It leverages the Claude 3.7 API (claude-3-7-sonnet-20250219) to process text in manageable chunks, maintaining coherence through strategic text overlap.

The implementation emphasizes error handling, providing clear feedback on potential issues with API credentials, network connectivity, or model access. The user experience is streamlined with sensible defaults and straightforward prompts.

## Use Case

This tool is particularly valuable for researchers, students, and professionals who need to quickly extract key information from lengthy documents. Instead of skimming or reading selectively, users can obtain comprehensive summaries that preserve the essential content while significantly reducing reading time.
