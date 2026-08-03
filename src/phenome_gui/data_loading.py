import pandas as pd
from phenome import make_dataframe_metadata_fn, get_metadata_from_path

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
    if parent.metadata_df is not None:
        parent._metadata_table.value = parent.metadata_df
    else:
        parent._metadata_table.value = pd.DataFrame()  # Clear the table if no data is loaded