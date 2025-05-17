import json
import datetime
from fpdf import FPDF
from io import BytesIO

# Chat Export Functions
def generate_chat_pdf(chat_history):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    
    # Use standard PDF core font instead of Arial
    pdf.set_font("helvetica", size=12)
    
    for message in chat_history:
        role = message["role"].capitalize()
        content = message["content"]
        
        # Handle encoding and special characters
        try:
            content = content.encode('latin-1', 'replace').decode('latin-1')
        except:
            content = content.encode('utf-8', 'replace').decode('latin-1', 'replace')
        
        # Add message to PDF
        pdf.multi_cell(0, 10, f"{role}: {content}\n")
    
    # Create bytes buffer (updated method)
    pdf_buffer = BytesIO()
    pdf_output = pdf.output()  # Removed deprecated 'dest' parameter
    pdf_buffer.write(pdf_output)
    pdf_buffer.seek(0)
    return pdf_buffer

def export_chat_history(history, fmt):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"chat_export_{timestamp}"
    
    if fmt == "json":
        return json.dumps(history, indent=2), "application/json", f"{base_name}.json"
    elif fmt == "txt":
        text = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in history])
        return text, "text/plain", f"{base_name}.txt"
    elif fmt == "pdf":
        pdf_bytes = generate_chat_pdf(history)
        return pdf_bytes, "application/pdf", f"{base_name}.pdf"