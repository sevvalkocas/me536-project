import os
from rembg import remove
from PIL import Image

# Get the directory where this script is located
current_dir = os.path.dirname(os.path.abspath(__file__))

# Set both input and output to the current directory
# (You can change image_change if you want the result in a different folder)
image_input_dir = current_dir
image_output_dir = current_dir

def process_single_image(input_dir, output_dir, target_file):
    """Removes background from a specific image file."""
    input_path = os.path.join(input_dir, target_file)
    # Renaming the output so you don't overwrite the original immediately
    output_path = os.path.join(output_dir, f"no_bg_{target_file}")

    if os.path.exists(input_path):
        try:
            with open(input_path, "rb") as inp_file:
                input_data = inp_file.read()

            print(f"Processing: {target_file}...")
            output_data = remove(input_data)  # Remove background

            with open(output_path, "wb") as out_file:
                out_file.write(output_data)
            
            print(f"Success! Saved to: {output_path}")
        except Exception as e:
            print(f"Failed to process {target_file}: {e}")
    else:
        print(f"Error: {target_file} not found in {input_dir}")

# Execute for knife.png
process_single_image(image_input_dir, image_output_dir, "banana.png")

# import os
# from rembg import remove
# from PIL import Image

# # Directories for your data
# current_dir = os.path.dirname(os.path.abspath(__file__))
# image= os.path.join(current_dir, "fruitCrash", "fruitCrash_dataset","level2_original gercek")
# image_change = os.path.join(current_dir, "fruitCrash", "fruitCrash_dataset", "level2_son2")

# os.makedirs(image_change, exist_ok=True)


# def process_images(input_dir, output_dir):
#     """Remove background from all images in the input directory and save to output directory."""
#     for file_name in os.listdir(input_dir):
#         input_path = os.path.join(input_dir, file_name)
#         output_path = os.path.join(output_dir, file_name)

#         if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
#             try:
#                 with open(input_path, "rb") as inp_file:
#                     input_data = inp_file.read()

#                 output_data = remove(input_data)  # Remove background

#                 with open(output_path, "wb") as out_file:
#                     out_file.write(output_data)
#                 print(f"Processed: {file_name}")
#             except Exception as e:
#                 print(f"Failed to process {file_name}: {e}")

# process_images(image, image_change)

# print("Background removal complete for all datasets.")
