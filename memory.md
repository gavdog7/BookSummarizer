# Developer Notes

## Key Things to Remember

- Use the **correct Claude model string**: `claude-3-7-sonnet-20250219`

- API keys must start with `sk-ant-` - validate this before making requests

- Token estimation uses simple **word count × 1.3** formula - no need for complex tokenizers

- Always **test API connectivity** with a small request before processing large texts

- The **200K input / 128K output token limits** determine when splitting is needed

- Include **5% text overlap** between parts for coherent summaries

- Implement **graceful failures** for missing dependencies (pandoc)

- Check for expansion attempts (summary longer than original) and warn users

- Accept short form responses (`y`, `n`, `m`) for better user experience

- Handle specific API errors (401, 404, 429) with clear user messages

- Remember that EPUB creation is optional and depends on pandoc availability

- Estimate and display costs before processing to avoid surprises
