### Book Summarizer
## Code base model
[X] Code summarizer.py
[X] Upload to GitHub

## Add features
[ ] Update so the first step is to ask 'Which book would you like to summarize?', and the app will look in the BooksIn folder, list all of the books with format '<Book#>. <FileName>, Time to read: <Estimated time to read>. User can then select which book to summarize by entering a number 1-x that corresponds with the book they choose. The books should be listed in order with longest to read books at the top
[ ] Change so that when summarization is complete, the app moves all files to BooksOut folder (including the pre-summarized book)
[ ] Add a Temp folder so data can be retained if the summarization fails midway through
[ ] Add feature to resume if the app launches and there are files in the temp folder
[ ] Add Temp folder contents to .gitignore
[ ] Update so the app uses Anthropic's tokenizer endpoint to estimate tokens rather than rough estimation method
