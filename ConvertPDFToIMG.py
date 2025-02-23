from pdf2image import convert_from_path
import os

# Create directory for converted images
os.makedirs('old_photos', exist_ok=True)

if __name__ == "__main__":
    # Convert PDFs to images
    for pdf_file in os.listdir('pdf_folder'):
        if pdf_file.endswith('.pdf'):
            images = convert_from_path(f'pdf_folder/{pdf_file}')
            for i, image in enumerate(images):
                image.save(f'old_photos/{pdf_file[:-4]}_page_{i+1}.jpg', 'JPEG')