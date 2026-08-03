import pandas as pd
from phenome import make_dataframe_metadata_fn, get_metadata_from_path
from .image_viewer import display_images

def load_data_csv(parent):
    """
    Find the image files and corresponding masks, and load the metadata
    into the metadata table.
    """
    images_folder = parent.params['paths']['images']
    masks_folder = parent.params['paths']['masks']
    metadata_csv = parent.params['paths']['metadata']

    if not images_folder or not masks_folder or not metadata_csv:
        print("Please select all required directories and files.")
        return

    metadata_df = pd.read_csv(metadata_csv)
    metadata_fn = make_dataframe_metadata_fn(metadata_df)

    parent.metadata_df = parent.pheno.find_files(images_folder, mask_dir=masks_folder, metadata_fn=metadata_fn)
    update_metadata_table(parent)

def load_data_path(parent):
    """
    Find files using the paths in the parameters and load them into the parent object.
    """
    metadata_fn = get_metadata_from_path(parent.params['paths']['template'])
    parent.metadata_df = parent.pheno.find_files(parent.params['paths']['data_path'], metadata_fn=metadata_fn)
    update_metadata_table(parent)

def update_metadata_table(parent):
    """
    Update the metadata table in the GUI with the current metadata dataframe.
    """
    if parent.metadata_df is None or parent.metadata_df.empty:
        parent._metadata_table.value = pd.DataFrame()  # Clear the table if no data is loaded
        raise ValueError("Could not load data. Please check the provided paths and files.")
    else:
        parent._metadata_table.value = parent.metadata_df

def display_table_image(parent):
    """
    Display corresponding image and mask when a row is selected from the table.
    """
    row = parent._metadata_table.native.currentRow()
    row_data = parent._metadata_table.to_dataframe().iloc[row]

    if row < 0:
        return

    image_path = row_data['file_path']
    if 'mask_path' in row_data.keys():
        mask_path = [row_data['mask_path']]
    else:
        mask_path = None

    display_images(parent, [image_path], mask_path, [row_data.to_dict()])