"""
File System Tools Module
========================

Core file system operations for the LLM-Powered File System Assistant.
Part A of the LLM Function Calling Assignment (60% of total grade).

This module provides structured tool interfaces for:
- Reading files (PDF, TXT, DOCX) with metadata extraction
- Listing directory contents with filtering capabilities
- Writing files with automatic directory creation
- Searching file content with context highlighting

Author: Assignment Submission
Course: LLM Function Calling and Tool Use
Date: 2024

Usage:
    from fs_tools import read_file, list_files, write_file, search_in_file
    
    # Read a resume file
    result = read_file("examples/resumes/john_doe_resume.txt")
    
    # List all text files
    files = list_files("examples/resumes", ".txt")
    
    # Search for Python experience
    matches = search_in_file("resume.txt", "Python")
"""

import os
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Document parsing imports
import PyPDF2
from docx import Document


def read_file(filepath: str) -> Dict:
    """
    Read resume files (PDF, TXT, DOCX) and extract text content.
    
    Args:
        filepath (str): Path to the file to read
        
    Returns:
        dict: Structured response with content, metadata, and status
    """
    try:
        file_path = Path(filepath)
        
        # Check if file exists
        if not file_path.exists():
            return {
                "success": False,
                "error": f"File not found: {filepath}",
                "content": None,
                "metadata": None
            }
        
        # Get file metadata
        stat = file_path.stat()
        metadata = {
            "filename": file_path.name,
            "filepath": str(file_path.absolute()),
            "size_bytes": stat.st_size,
            "modified_date": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "extension": file_path.suffix.lower()
        }
        
        # Extract content based on file type
        content = ""
        file_extension = file_path.suffix.lower()
        
        if file_extension == ".txt":
            content = _read_text_file(file_path)
        elif file_extension == ".pdf":
            content = _read_pdf_file(file_path)
        elif file_extension in [".docx", ".doc"]:
            content = _read_docx_file(file_path)
        else:
            # Try to read as text file
            try:
                content = _read_text_file(file_path)
                metadata["note"] = "File read as plain text"
            except Exception as e:
                return {
                    "success": False,
                    "error": f"Unsupported file type: {file_extension}. Error: {str(e)}",
                    "content": None,
                    "metadata": metadata
                }
        
        return {
            "success": True,
            "error": None,
            "content": content.strip(),
            "metadata": metadata
        }
        
    except Exception as e:
        logger.error(f"Error reading file {filepath}: {str(e)}")
        return {
            "success": False,
            "error": f"Error reading file: {str(e)}",
            "content": None,
            "metadata": None
        }


def list_files(directory: str, extension: str = None) -> List[Dict]:
    """
    List all files in a directory with optional extension filtering.
    
    Args:
        directory (str): Directory path to list files from
        extension (str, optional): File extension filter (e.g., '.pdf', '.txt')
        
    Returns:
        list: List of file metadata dictionaries
    """
    try:
        dir_path = Path(directory)
        
        # Check if directory exists
        if not dir_path.exists():
            logger.error(f"Directory not found: {directory}")
            return []
        
        if not dir_path.is_dir():
            logger.error(f"Path is not a directory: {directory}")
            return []
        
        files_list = []
        
        # Iterate through files in directory
        for file_path in dir_path.iterdir():
            if file_path.is_file():
                # Apply extension filter if specified
                if extension and not file_path.suffix.lower() == extension.lower():
                    continue
                
                try:
                    stat = file_path.stat()
                    file_info = {
                        "name": file_path.name,
                        "filepath": str(file_path.absolute()),
                        "size_bytes": stat.st_size,
                        "size_human": _format_file_size(stat.st_size),
                        "modified_date": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        "extension": file_path.suffix.lower()
                    }
                    files_list.append(file_info)
                except Exception as e:
                    logger.warning(f"Error getting info for file {file_path}: {str(e)}")
                    continue
        
        # Sort by name
        files_list.sort(key=lambda x: x["name"].lower())
        
        logger.info(f"Listed {len(files_list)} files from {directory}")
        return files_list
        
    except Exception as e:
        logger.error(f"Error listing files in {directory}: {str(e)}")
        return []


def write_file(filepath: str, content: str) -> Dict:
    """
    Write content to file, creating directories if needed.
    
    Args:
        filepath (str): Path where to write the file
        content (str): Content to write
        
    Returns:
        dict: Success/failure status with details
    """
    try:
        file_path = Path(filepath)
        
        # Create parent directories if they don't exist
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write content to file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        # Get file info after writing
        stat = file_path.stat()
        
        logger.info(f"Successfully wrote file: {filepath}")
        
        return {
            "success": True,
            "error": None,
            "filepath": str(file_path.absolute()),
            "size_bytes": stat.st_size,
            "message": f"File written successfully: {file_path.name}"
        }
        
    except Exception as e:
        logger.error(f"Error writing file {filepath}: {str(e)}")
        return {
            "success": False,
            "error": f"Error writing file: {str(e)}",
            "filepath": filepath,
            "size_bytes": 0,
            "message": "Failed to write file"
        }


def search_in_file(filepath: str, keyword: str) -> Dict:
    """
    Search for keywords in file content with context.
    
    Args:
        filepath (str): Path to the file to search in
        keyword (str): Keyword to search for (case-insensitive)
        
    Returns:
        dict: Search results with matches and context
    """
    try:
        # First read the file content
        file_data = read_file(filepath)
        
        if not file_data["success"]:
            return {
                "success": False,
                "error": file_data["error"],
                "matches": [],
                "total_matches": 0,
                "metadata": None
            }
        
        content = file_data["content"]
        if not content:
            return {
                "success": True,
                "error": None,
                "matches": [],
                "total_matches": 0,
                "metadata": file_data["metadata"]
            }
        
        # Split content into lines for context
        lines = content.split('\n')
        matches = []
        
        # Search for keyword (case-insensitive)
        keyword_lower = keyword.lower()
        
        for line_num, line in enumerate(lines, 1):
            if keyword_lower in line.lower():
                # Get context (previous and next lines)
                start_line = max(0, line_num - 2)
                end_line = min(len(lines), line_num + 1)
                context_lines = lines[start_line:end_line]
                
                # Highlight the keyword in the matching line
                highlighted_line = re.sub(
                    re.escape(keyword), 
                    f"**{keyword}**", 
                    line, 
                    flags=re.IGNORECASE
                )
                
                match_info = {
                    "line_number": line_num,
                    "line_content": line.strip(),
                    "highlighted_content": highlighted_line.strip(),
                    "context": [l.strip() for l in context_lines],
                    "context_start_line": start_line + 1
                }
                matches.append(match_info)
        
        logger.info(f"Found {len(matches)} matches for '{keyword}' in {filepath}")
        
        return {
            "success": True,
            "error": None,
            "matches": matches,
            "total_matches": len(matches),
            "keyword": keyword,
            "metadata": file_data["metadata"]
        }
        
    except Exception as e:
        logger.error(f"Error searching in file {filepath}: {str(e)}")
        return {
            "success": False,
            "error": f"Error searching file: {str(e)}",
            "matches": [],
            "total_matches": 0,
            "metadata": None
        }


# Helper functions for file reading
def _read_text_file(file_path: Path) -> str:
    """Read plain text file."""
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read()


def _read_pdf_file(file_path: Path) -> str:
    """Read PDF file and extract text."""
    
    text = ""
    try:
        with open(file_path, 'rb') as f:
            pdf_reader = PyPDF2.PdfReader(f)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
    except Exception as e:
        raise Exception(f"Error reading PDF: {str(e)}")
    
    return text


def _read_docx_file(file_path: Path) -> str:
    """Read DOCX file and extract text."""
    
    try:
        doc = Document(file_path)
        paragraphs = []
        for para in doc.paragraphs:
            if para.text.strip():
                paragraphs.append(para.text)
        return "\n".join(paragraphs)
    except Exception as e:
        raise Exception(f"Error reading DOCX: {str(e)}")


def _format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"


# Tool definitions for LLM function calling
TOOL_DEFINITIONS = [
    {
        "name": "read_file",
        "description": "Read and extract text content from resume files (PDF, TXT, DOCX)",
        "parameters": {
            "type": "object",
            "properties": {
                "filepath": {
                    "type": "string",
                    "description": "Path to the file to read"
                }
            },
            "required": ["filepath"]
        }
    },
    {
        "name": "list_files",
        "description": "List all files in a directory with optional extension filtering",
        "parameters": {
            "type": "object",
            "properties": {
                "directory": {
                    "type": "string",
                    "description": "Directory path to list files from"
                },
                "extension": {
                    "type": "string",
                    "description": "Optional file extension filter (e.g., '.pdf', '.txt')",
                }
            },
            "required": ["directory"]
        }
    },
    {
        "name": "write_file",
        "description": "Write content to a file, creating directories if needed",
        "parameters": {
            "type": "object",
            "properties": {
                "filepath": {
                    "type": "string",
                    "description": "Path where to write the file"
                },
                "content": {
                    "type": "string",
                    "description": "Content to write to the file"
                }
            },
            "required": ["filepath", "content"]
        }
    },
    {
        "name": "search_in_file",
        "description": "Search for keywords in file content and return matches with context",
        "parameters": {
            "type": "object",
            "properties": {
                "filepath": {
                    "type": "string",
                    "description": "Path to the file to search in"
                },
                "keyword": {
                    "type": "string",
                    "description": "Keyword to search for (case-insensitive)"
                }
            },
            "required": ["filepath", "keyword"]
        }
    }
]


if __name__ == "__main__":
    # Test the functions
    print("Testing File System Tools...")
    
    # Test list_files
    print("\n1. Testing list_files:")
    files = list_files(".", ".py")
    for file in files[:3]:  # Show first 3 files
        print(f"  - {file['name']} ({file['size_human']})")
    
    # Test write_file
    print("\n2. Testing write_file:")
    test_content = "This is a test file created by fs_tools.py\nIt contains sample content for testing."
    result = write_file("test_output.txt", test_content)
    print(f"  Write result: {result['message']}")
    
    # Test read_file
    print("\n3. Testing read_file:")
    if result["success"]:
        read_result = read_file("test_output.txt")
        if read_result["success"]:
            print(f"  File content: {read_result['content'][:50]}...")
            print(f"  File size: {read_result['metadata']['size_bytes']} bytes")
    
    # Test search_in_file
    print("\n4. Testing search_in_file:")
    if result["success"]:
        search_result = search_in_file("test_output.txt", "test")
        if search_result["success"]:
            print(f"  Found {search_result['total_matches']} matches for 'test'")
            for match in search_result['matches']:
                print(f"    Line {match['line_number']}: {match['highlighted_content']}")
    
    print("\nAll tests completed!")