### Book Summarizer
## Code base model
[X] Code summarizer.py
[X] Upload to GitHub

## Add features
[X] Enhance readme file
[X] Update so the first step is to ask 'Which book would you like to summarize?', and the app will look in the BooksIn folder, list all of the books with format '<Book#>. <FileName>, Time to read: <Estimated time to read>. User can then select which book to summarize by entering a number 1-x that corresponds with the book they choose. The books should be listed in order with longest to read books at the top
[X] Change so that when summarization is complete, the app moves all files to BooksOut folder (including the pre-summarized book)
[X] Add a Temp folder so data can be retained if the summarization fails midway through
[X] Add feature to resume if the app launches and there are files in the temp folder
[X] Add Temp folder contents to .gitignore
[X] Update so the app uses Anthropic's tokenizer endpoint to estimate tokens rather than rough estimation method
[X] Fix error with tokenizer - Warning: Could not use Anthropic's tokenizer: 'Anthropic' object has no attribute 'count_tokens'
[X] Fix error with tokenizer - Warning: Could not use Anthropic's tokenizer: cannot import name 'count_tokens' from 'anthropic' (/usr/local/lib/python3.12/dist-packages/anthropic/__init__.py)
[X] Fix tokenizer behavior - add logic to do a rough estimation of tokens first based on words method, then take a small chunk that should be around 100k tokens. Hit the anthropic token estimator with that chunk. Take the response % of tokens, and scale up based on the % of total file size that the ~100k token chunk represented
[X] Fix Temp folder behavior - This folder should contain the split text files if the API call to summarize is unsuccessful because those files should land there first
[X] At every point in the flow, the user should be able to exit by typing exit. Add that
[X] Add functionality - when the book is chosen and the app suggests how many splits to make, add logic to indicate whether the splits are limited by the size of the book (input tokens) or the size of the intended summary (output tokens).
[X] On the very first page, the interface says Time to read: 55.7 hours (3343 minutes). Remove the minutes part that is in brackets and change the structure so the name is <filename>, ~<hours> to read

*** END, DO NOT DO STEPS BELOW THIS LINE ***

[ ] Update formatting of the app in the terminal interface so it looks great (but make the formatting optional if it needs a dependency so the app still works)
[ ] Fix module not found errors so they are cleaner 'eg. No module named 'dotenv'
[ ] Change gitignore so the BooksIn, BooksOut and Temp folders appear in the uploaded GitHub repo, but the contents of those folders do not appear

