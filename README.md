# LLM-Powered File System Assistant

A sophisticated file system assistant that integrates structured tool interfaces with Large Language Models (LLMs) for natural language interaction with file operations. This project demonstrates LLM function calling capabilities for resume processing and document analysis.

## 📋 Assignment Overview

This project fulfills the **LLM Function Calling/Tool Use Assignment** requirements:

### Part A: Core File System Tools (60%)
- ✅ **read_file(filepath)** - Reads PDF, TXT, DOCX files with text extraction
- ✅ **list_files(directory, extension)** - Lists files with optional filtering
- ✅ **write_file(filepath, content)** - Creates files with automatic directory creation  
- ✅ **search_in_file(filepath, keyword)** - Searches with context highlighting

### Part B: LLM Integration (40%)
- ✅ **OpenAI/OpenRouter Integration** - Function calling with multiple AI models
- ✅ **Natural Language Processing** - Handles assignment example queries
- ✅ **Multi-step Workflows** - Executes complete tasks automatically

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.8 or higher
- OpenRouter API key (provides access to OpenAI, Anthropic, and other models)

### 2. Installation

```bash
# Clone or navigate to the project directory
cd "LLM-Powered File System Assistant"

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration

Create a `.env` file with your API credentials:

```bash
# OpenRouter API Configuration
OPENAI_API_KEY=sk-or-v1-your-openrouter-key-here
OPENAI_BASE_URL=https://openrouter.ai/api/v1
```

**Note:** OpenRouter provides access to multiple AI models (OpenAI GPT, Anthropic Claude, Google Gemini, etc.) with a single API key.

### 4. Run the Assistant

```bash
# Activate virtual environment
source venv/bin/activate

# Start the interactive assistant
python llm_file_assistant.py
```

## 💬 Usage Examples

The assistant handles natural language queries and automatically executes appropriate file operations:

### Assignment Example Queries

```bash
💬 You: Read all resumes in the resumes folder
🤖 Assistant: [Lists files, reads each resume, presents complete analysis]

💬 You: Find resumes mentioning Python experience  
🤖 Assistant: [Lists resumes, searches each for "Python", shows matches with context]

💬 You: Create a summary file for resume_john_doe.pdf
🤖 Assistant: [Reads the resume, creates formatted summary, saves to file]
```

### Additional Capabilities

```bash
# File exploration
"List all PDF files in the documents folder"
"Show me all text files in the current directory"

# Content analysis  
"Find files mentioning machine learning"
"Search for JavaScript experience across all resumes"

# File creation
"Create a comparison report of all candidates"
"Write a summary of technical skills found in resumes"
```

## 📁 Project Structure

```
LLM-Powered File System Assistant/
├── fs_tools.py              # Core file system operations (Part A)
├── llm_file_assistant.py    # LLM integration & chat interface (Part B)
├── requirements.txt         # Python dependencies
├── README.md               # This documentation
├── .env                    # API configuration (create this)
├── examples/
│   └── resumes/           # Sample resume files (8 total)
│       ├── john_doe_resume.txt
│       ├── sarah_smith_resume.txt
│       ├── mike_johnson_resume.txt
│       ├── emily_chen_resume.txt
│       ├── david_rodriguez_resume.txt
│       ├── alexandra_kim_resume.txt
│       ├── robert_taylor_resume.txt
│       └── lisa_anderson_resume.txt
└── venv/                   # Virtual environment
```

## 🛠️ Core Components

### fs_tools.py - File System Operations

```python
from fs_tools import read_file, list_files, write_file, search_in_file

# Read any supported file type
result = read_file("examples/resumes/john_doe_resume.txt")
print(result['content'])  # Extracted text content
print(result['metadata'])  # File information

# List files with filtering
files = list_files("examples/resumes", ".txt")
for file in files:
    print(f"{file['name']} ({file['size_human']})")

# Write content to file
write_file("output/summary.txt", "Content here")

# Search within files
matches = search_in_file("resume.txt", "Python")
print(f"Found {matches['total_matches']} matches")
```

### llm_file_assistant.py - LLM Integration

```python
from llm_file_assistant import LLMFileAssistant

# Initialize with OpenRouter
assistant = LLMFileAssistant(
    provider="openai",
    model="openai/gpt-4o-mini"  # or "anthropic/claude-3-sonnet"
)

# Process natural language queries
result = assistant.process_query("Find all resumes mentioning Python")
print(result['response'])

# Start interactive chat session
assistant.chat_session()
```

## 📊 Supported File Formats

| Format | Library | Capabilities |
|--------|---------|--------------|
| **TXT** | Built-in | Full text extraction |
| **PDF** | PyPDF2 | Text extraction from most PDFs |
| **DOCX** | python-docx | Microsoft Word documents |

## 🎯 AI Model Options

Through OpenRouter, you can access multiple AI models:

- `openai/gpt-4o-mini` (fast, cost-effective)
- `openai/gpt-4o` (most capable)  
- `anthropic/claude-3-sonnet-20240229` (excellent reasoning)
- `google/gemini-pro` (Google's flagship)
- And many more...

Change models by modifying the initialization:
```python
assistant = LLMFileAssistant(model="anthropic/claude-3-sonnet-20240229")
```

## 🧪 Testing the System

### Test Core Functions
```bash
# Test file system operations directly
python -c "from fs_tools import *; print(list_files('examples/resumes'))"
```

### Test LLM Integration
```bash
# Run the assistant and try these queries:
python llm_file_assistant.py

# Then test:
"List all resume files"
"Find candidates with Python skills"  
"Create a summary of technical backgrounds"
```

## 📋 Dependencies

```txt
# Core file operations
PyPDF2>=3.0.1              # PDF reading
python-docx>=0.8.11         # DOCX reading  
python-dotenv>=1.0.0        # Environment variables

# LLM integration
openai>=1.12.0              # OpenAI/OpenRouter API client
anthropic>=0.18.0           # Anthropic API client

# Development
pytest>=7.0.0               # Testing framework
pytest-cov>=4.0.0           # Coverage reporting
```

## 🔧 Configuration Options

### Environment Variables
```bash
# Required
OPENAI_API_KEY=sk-or-v1-...           # OpenRouter API key
OPENAI_BASE_URL=https://openrouter.ai/api/v1  # OpenRouter endpoint

# Optional
ANTHROPIC_API_KEY=sk-ant-...          # Direct Anthropic access
LOG_LEVEL=INFO                        # Logging level
```

### Model Selection
```python
# In code or via initialization
assistant = LLMFileAssistant(
    provider="openai",                 # or "anthropic"
    model="openai/gpt-4o-mini",       # OpenRouter model format
    api_key="custom_key",             # Optional override
    base_url="custom_endpoint"        # Optional override
)
```

## 🐛 Troubleshooting

### Common Issues

**"No API key found"**
```bash
# Check your .env file exists and contains:
OPENAI_API_KEY=sk-or-v1-your-key-here
OPENAI_BASE_URL=https://openrouter.ai/api/v1
```

**"ModuleNotFoundError: No module named 'PyPDF2'"**
```bash
# Ensure virtual environment is activated and dependencies installed:
source venv/bin/activate
pip install -r requirements.txt
```

**"File not found" errors**
```bash
# Ensure you're running from the project root directory
cd "LLM-Powered File System Assistant"
python llm_file_assistant.py
```

**Function calling not working**
- Verify your OpenRouter API key is valid
- Check that the base URL is set correctly
- Try with a different model if issues persist

## 📚 Learning Objectives Achieved

✅ **Understand LLM function calling/tool use**
- Implemented structured tool interfaces with proper schemas
- Demonstrated automatic tool selection based on user intent

✅ **Implement structured tool interfaces**  
- All functions follow consistent API patterns
- Proper input validation and error handling
- Structured return formats with metadata

✅ **Handle file I/O operations programmatically**
- Multiple file format support (PDF, TXT, DOCX)
- Directory operations with filtering
- Automatic path handling and directory creation

✅ **Parse and validate documents**
- Text extraction from various document formats
- Content validation and error recovery
- Metadata extraction and processing

## 🤝 Contributing

This project was developed for academic purposes. Key areas for potential enhancement:

1. **Additional File Formats** - Add support for Excel, PowerPoint, etc.
2. **Advanced Search** - Implement semantic search and regex patterns
3. **Batch Operations** - Process multiple files simultaneously  
4. **Web Interface** - Create a web-based UI for the assistant
5. **Database Integration** - Store and index processed documents

## 📄 License

This project is developed for educational purposes as part of an LLM Function Calling assignment.

## 🙏 Acknowledgments

- **OpenAI** for GPT models and function calling capabilities
- **Anthropic** for Claude models  
- **OpenRouter** for unified API access to multiple AI providers
- **PyPDF2** and **python-docx** communities for document processing libraries

---

**Ready to explore intelligent file operations with natural language commands!** 🚀

For questions or issues, please ensure your environment is properly configured according to the setup instructions above.