import os
from pdf2image import convert_from_path
from PIL import Image


def convert_pdf_to_images(pdf_folder, output_folder):
    pdf_files = [file for file in os.listdir(pdf_folder) if file.endswith('.pdf')]
    for pdf_file in pdf_files:
        images = convert_from_path(os.path.join(pdf_folder, pdf_file))
        for i, image in enumerate(images):
            image_path = os.path.join(output_folder, f"{pdf_file[:-4]}_{i}.png")
            image.save(image_path, 'PNG')


def create_gif_from_images(image_folder, gif_path, duration=300):
    image_files = [file for file in os.listdir(image_folder) if file.endswith('.png')]
    image_files.sort()  # Sort the files to ensure the correct sequence in the GIF
    images = [Image.open(os.path.join(image_folder, file)) for file in image_files]
    images[0].save(gif_path, save_all=True, append_images=images[1:], duration=duration, loop=0)


if __name__ == "__main__":
    # Define the input and output folders
    input_folder = "pics/iris_gif/"
    output_folder = "pics/iris_gif_out/"
    gif_path = output_folder + "gif.gif"

    # Step 1: Convert PDFs to images
    convert_pdf_to_images(input_folder, output_folder)

    # Step 2: Create GIF from the images
    create_gif_from_images(output_folder, gif_path)
