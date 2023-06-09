# Script to correct RAW files, extract GPS data from jpg and export in format that can be georectified

from MapIR import MapIR_RAW
from pathlib import Path
import os


base_directory = '../Data/AC Summer 23/Wheat Field/6-8'

# Create a new directory to contain processed files if doesnt exist
processed_directory = base_directory + '/_processed'

bd = Path(base_directory)

# Rename all files in base directory for easier processing
# for filename in os.listdir(base_directory):
#     if filename.endswith('.JPG') or filename.endswith('.RAW'):
#         # Get the file extension (suffix)
#         suffix = os.path.splitext(filename)[1]
#
#         # Get the last 3 characters of the filename, excluding the suffix
#         new_name = filename[-len(suffix) - 3:-len(suffix)] + suffix
#
#         # Create the full old and new file paths
#         old_file_path = os.path.join(base_directory, filename)
#         new_file_path = os.path.join(base_directory, new_name)
#
#         # Rename the file
#         os.rename(old_file_path, new_file_path)


# pd = Path(processed_directory)
# if not pd.exists():
#     pd.mkdir()


for file in bd.iterdir():
    if file.suffix == '.RAW':
        image = MapIR_RAW(file, stats=False)
        image.extract_GPS()




# file = '../Data/AC Summer 23/Wheat Field/6-8/2023_0608_091139_081.RAW'
# image = MapIR_RAW(file, stats=True)
# image.display()
# image.NDVI()



