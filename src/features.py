import base64
import os
import requests
import fitz

'''
This file defines the features of the application, such as images coder, and other features that we want to add in the future.

'''  

def pdf_to_image_codified(path):
    '''
    This function converts PDF files to images, which it then converts to base64. The result is used for data entry in LLM.

    Args:
        -path: absolute path to PDF file

    Return:
        -base64 encoding for all pages of PDF file.
    '''

    images_base64 = []
    doc = fitz.open(path)
    
    for page_number in range(len(doc)):
        page = doc.load_page(page_number)
        pix = page.get_pixmap(dpi=500) 
        
        img_bytes = pix.tobytes("jpeg")
        img_b64 = base64.b64encode(img_bytes).decode('utf-8')
        images_base64.append(img_b64)
        
    return images_base64



def to_llm(codified_content, API_URL):
    """
    Send the content to LLM
    Args: 
        -codified_content: content codified
        -API-URL: endpoint access to LLM
     
    Return: Answer LLM.
    """
    
    imagen_data_uri = f"data:image/jpeg;base64,{codified_content}"
    
    payload = {
        "model": "local-model", 
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text", 
                        "text": "Trancribe all the text in the image whith original format, dont add descriptions"
                    },
                    {
                        "type": "image_url", 
                        "image_url": {
                            "url": imagen_data_uri
                        }
                    }
                ]
            }
        ],
        "temperature": 0.1,
        "max_tokens": 8192,
        "stream": False
    }

    try:
        answer = requests.post(API_URL, json=payload, timeout=600) 
        answer.raise_for_status()
        
        data = answer.json()
        return data['choices'][0]['message']['content'].strip()
        
    except requests.exceptions.ConnectionError:
        print("   [ERROR]: Error to conect. The server is working?")
        return None
    except Exception as e:
        print(f"   [ERROR API]: {e}")
        return None


def process_full_dir(dir_in, API_URL, dir_out):
    '''
    This function transcribe all .pdf, .png, .jpeg, .jpg files to text.

    Args:
        -dir_in: absolute path directory when the files are located
        -dir_out: absolute path directory when the files transcription are saved
        -API_URL: endpoin to connect whith LLM
    '''
    
    print(f"Directory in: {dir_in}")
    
    archives = os.listdir(dir_in)
    total_archives = len(archives)
    
    for idx, name in enumerate(archives, 1):
        full_path = os.path.join(dir_in, name)
        
        if os.path.isdir(full_path):
            continue

        base_name, ext = os.path.splitext(name)
        ext = ext.lower()
        
        path_txt_out = os.path.join(dir_out, f"{base_name}.txt")
        
        if os.path.exists(path_txt_out):
            print(f"[{idx}/{total_archives}] Skipping '{name}' ( .txt already exist)")
            continue

        print(f"\n[{idx}/{total_archives}] Processing: {name}...")
        final_text = ""

        try:
            if ext in ['.png', '.jpg', '.jpeg']:
                data = image_process_streamlit(full_path)
                final_text = to_llm(data, API_URL)
            
            elif ext == '.pdf':
                pages_list_b64 = pdf_to_image_codified(full_path)
                text_pages = [] 
                print(f"   [~] The PDF have {len(pages_list_b64)} pages.")
                
                for i, pag_b64 in enumerate(pages_list_b64, 1):
                    print(f"   [→] Transcribing page {i} of {len(pages_list_b64)}...")
                    page_text = to_llm(pag_b64, API_URL)
                    
                    if page_text:
                        text_pages.append(page_text)
                    else:
                        text_pages.append(f"[Error processing the page {i}]")
                        
                final_text = "\n\n--- Next page ---\n\n".join(text_pages)
            
            else:
                print(f"   [-] Format '{ext}' ignored.")
                continue

            if final_text:
                with open(path_txt_out, 'w', encoding='utf-8') as f:
                    f.write(final_text)
                print(f"   [✓] Save succesfully.")

        except Exception as e:
            print(f"  Error {name}: {e}")