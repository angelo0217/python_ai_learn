#  uv pip install pypdf
import os
from pypdf import PdfReader, PdfWriter

def split_pdf(input_pdf_path, output_dir, pages_per_split=8):
    """
    Splits a PDF into multiple files, each with a specified number of pages.

    Args:
        input_pdf_path (str): The path to the input PDF file.
        output_dir (str): The directory where the split PDFs will be saved.
        pages_per_split (int): The number of pages for each split file.
    """
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    try:
        reader = PdfReader(input_pdf_path)
        total_pages = len(reader.pages)
        print(f"Total pages in the PDF: {total_pages}")

        for i in range(0, total_pages, pages_per_split):
            writer = PdfWriter()
            start_page = i
            end_page = min(i + pages_per_split, total_pages)
            
            print(f"Processing pages {start_page + 1} to {end_page}...")

            for page_num in range(start_page, end_page):
                writer.add_page(reader.pages[page_num])

            output_filename = f"part_{i // pages_per_split + 1}.pdf"
            output_filepath = os.path.join(output_dir, output_filename)

            with open(output_filepath, "wb") as output_pdf:
                writer.write(output_pdf)
            
            print(f"Successfully created {output_filepath}")

        print("\nPDF splitting complete.")

    except FileNotFoundError:
        print(f"Error: The file was not found at {input_pdf_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # NOTE: You might need to install the pypdf library first.
    # You can do this by running: uv pip install pypdf
    
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define the input PDF file path relative to the script's location
    pdf_file_path = os.path.join(script_dir, "saa_c03_.pdf")
    
    # Define the output directory path relative to the script's location
    output_directory = os.path.join(script_dir, "split_pdfs")
    
    # Number of pages per new PDF file
    pages_per_file = 110
    
    split_pdf(pdf_file_path, output_directory, pages_per_file)
