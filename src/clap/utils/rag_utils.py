
import csv 
from typing import List, Dict, Any, Tuple, Optional, Union


try:
    import pypdf
except ImportError:
    raise ImportError(
        "pypdf not found. Please install it for PDF loading: pip install pypdf"
    )



def load_text_file(file_path: str) -> str:
    """Loads text content from a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Error loading text file {file_path}: {e}")
        return ""

def load_pdf_file(file_path: str) -> str:
    """Loads text content from a PDF file."""
    text = ""
    try:
        with open(file_path, 'rb') as f:
            reader = pypdf.PdfReader(f)
            print(f"Loading PDF '{file_path}' with {len(reader.pages)} pages...")
            for i, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n" 
                else:
                    print(f"Warning: No text extracted from page {i+1} of {file_path}")
        print(f"Finished loading PDF '{file_path}'.")
        return text.strip()
    except FileNotFoundError:
        print(f"Error: PDF file not found at {file_path}")
        return ""
    except Exception as e:
        print(f"Error loading PDF file {file_path}: {e}")
        return ""


def load_csv_file(
    file_path: str,
    content_column: Union[str, int],
    metadata_columns: Optional[List[Union[str, int]]] = None,
    delimiter: str = ',',
    encoding: str = 'utf-8'
) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Loads data from a CSV file, extracting content and metadata.

    Each row is treated as a potential document/chunk.

    Args:
        file_path: Path to the CSV file.
        content_column: The name (string) or index (int) of the column containing the main text content.
        metadata_columns: Optional list of column names (string) or indices (int)
                          to include as metadata for each row.
        delimiter: CSV delimiter (default ',').
        encoding: File encoding (default 'utf-8').

    Returns:
        A list of tuples, where each tuple contains:
        (document_text: str, metadata: dict)
    """
    data = []
    metadata_columns = metadata_columns or []
    try:
        with open(file_path, mode='r', encoding=encoding, newline='') as f:
            
            has_header = isinstance(content_column, str) or any(isinstance(mc, str) for mc in metadata_columns)

            if has_header:
                reader = csv.DictReader(f, delimiter=delimiter)
                headers = reader.fieldnames
                if headers is None:
                     print(f"Error: Could not read headers from CSV {file_path}")
                     return []

                
                if isinstance(content_column, str) and content_column not in headers:
                    raise ValueError(f"Content column '{content_column}' not found in CSV headers: {headers}")
                for mc in metadata_columns:
                    if isinstance(mc, str) and mc not in headers:
                        raise ValueError(f"Metadata column '{mc}' not found in CSV headers: {headers}")

                content_key = content_column
                meta_keys = [mc for mc in metadata_columns if isinstance(mc, str)]

            else:
                
                reader = csv.reader(f, delimiter=delimiter)
                
                content_key = int(content_column)
                meta_keys = [int(mc) for mc in metadata_columns]


            print(f"Loading CSV '{file_path}'...")
            for i, row in enumerate(reader):
                try:
                    if has_header:
                        
                        doc_text = row.get(content_key, "").strip()
                        metadata = {key: row.get(key, "") for key in meta_keys}
                    else:
                        
                        if content_key >= len(row): continue 
                        doc_text = row[content_key].strip()
                        metadata = {}
                        for key_index in meta_keys:
                            if key_index < len(row):
                               
                                metadata[f"column_{key_index}"] = row[key_index]


                    if doc_text: 
                        metadata["source_row"] = i + (1 if has_header else 0)
                        data.append((doc_text, metadata))
                except IndexError:
                     print(f"Warning: Skipping row {i} due to index out of bounds (check column indices).")
                except Exception as row_e:
                     print(f"Warning: Skipping row {i} due to error: {row_e}")


        print(f"Finished loading CSV '{file_path}', processed {len(data)} rows with content.")
        return data

    except FileNotFoundError:
        print(f"Error: CSV file not found at {file_path}")
        return []
    except ValueError as ve: 
        print(f"Error processing CSV header/indices for {file_path}: {ve}")
        return []
    except Exception as e:
        print(f"Error loading CSV file {file_path}: {e}")
        return []




def chunk_text_by_fixed_size(
    text: str, chunk_size: int, chunk_overlap: int = 0
) -> List[str]:
    """Chunks text into fixed size blocks with optional overlap."""
    if not isinstance(text, str):
        print(f"Warning: chunk_text_by_fixed_size expected string, got {type(text)}. Skipping.")
        return []
    if chunk_overlap >= chunk_size:
        
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    if chunk_size <= 0:
         raise ValueError("chunk_size must be positive")

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        
        step = chunk_size - chunk_overlap
        if step <= 0:
             
             step = 1 

        start += step

    return [chunk for chunk in chunks if chunk.strip()] 


def chunk_text_by_separator(text: str, separator: str = "\n\n") -> List[str]:
    """Chunks text based on a specified separator."""
    if not isinstance(text, str):
        print(f"Warning: chunk_text_by_separator expected string, got {type(text)}. Skipping.")
        return []
    chunks = text.split(separator)
    return [chunk for chunk in chunks if chunk.strip()] 


