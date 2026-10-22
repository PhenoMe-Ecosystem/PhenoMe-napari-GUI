# phenome-gui

[![License MIT](https://img.shields.io/pypi/l/phenome-gui.svg?color=green)](https://github.com//phenome-gui/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/phenome-gui.svg?color=green)](https://pypi.org/project/phenome-gui)
[![Python Version](https://img.shields.io/pypi/pyversions/phenome-gui.svg?color=green)](https://python.org)
[![tests](https://github.com//phenome-gui/workflows/tests/badge.svg)](https://github.com//phenome-gui/actions)
[![codecov](https://codecov.io/gh//phenome-gui/branch/main/graph/badge.svg)](https://codecov.io/gh//phenome-gui)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/phenome-gui)](https://napari-hub.org/plugins/phenome-gui)
[![npe2](https://img.shields.io/badge/plugin-npe2-blue?link=https://napari.org/stable/plugins/index.html)](https://napari.org/stable/plugins/index.html)
[![Copier](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/copier-org/copier/master/img/badge/badge-grayscale-inverted-border-purple.json)](https://github.com/copier-org/copier)

GUI integration of the [PhenoMe] phenotyping framework.

----------------------------------

This [napari] plugin was generated with [copier] using the [napari-plugin-template].

<!--
Don't miss the full getting started guide to set up your new package:
https://github.com/napari/napari-plugin-template#getting-started

and review the napari docs for plugin developers:
https://napari.org/stable/plugins/index.html
-->

## Installation

You can install `phenome-gui` in your napari environment via [pip]:

```bash
pip install phenome-gui
```

If napari is not already installed, you can install `phenome-gui` with napari and Qt via:

```bash
pip install "phenome-gui[all]"
```

For now, follow these instructions:

1. Clone the phenome-gui repo:
    
    `git clone --recurse-submodules git@git.valeria.science:predis/flc-phenome-gui.git`

2. Install napari as a Python package following the instructions on the website: https://napari.org/stable/getting_started/installation.html

    We recommend [uv] to manage environments.

3. Activate the napari environment: 

    `source .venv/bin/activate`

4. Install phenome-gui into the napari environment: 
    
    `uv pip install <path to phenome-gui>/.`

The PhenoMe GUI plugin will now be available in the Plugins tab of napari.

## Usage

This napari plugin integrates the [PhenoMe] phenotyping pipeline in a user-friendly graphical interface. To summarize, PhenoMe is a framework for phenotyping biological image data using embeddings extracted using vision models and physical properties. For in-depth information, see the PhenoMe documentation.

### Getting started

Open the plugin from napari's menu bar: **Plugins → PhenoMe (PhenoMe GUI)**. The widget docks on the side of the napari window and has three tabs, one per step of the pipeline:

1. **Data loading**: point the plugin to your images and their metadata.
2. **Processing**: extract the embeddings and compute the properties of every image.
3. **Analysis**: cluster the images, retrieve typical and unusual ones, and explore them in an interactive plot.

Work through the tabs in order: each step needs the results of the one before it.

A few things apply everywhere in the plugin:

- **Coloured outlines** show the state of each step. A group's outline turns **yellow** while its task runs, **green** once it has finished, and **red** if something went wrong.
- **Greyed-out buttons** are waiting for an earlier step. Hover over one to see what it needs (for example *"Please load data first."*).

### 1. Data loading

#### 1.1. Metadata structure

Choose the way metadata is extracted from your data: either a **.CSV file** is provided, or meta data is generated based on the provided file structure (**Path-based**).

If **Path-based** is selected, a path template must be provided, with the shape `.../(drug_name)/(time)/(filename).*`, where 'drug_name', 'time' and 'filename' will become metadata columns, with the values being the folder and file names.

#### 1.2. Data loading

Fill in the paths to your data. Type them, use the buttons to browse, or drag a folder or file from your file explorer onto the box.

- **Images Directory...** (required): the folder containing the images to analyse.
- **Masks Directory...** (optional): the folder containing the masks. Masks are needed to compute the shape properties.
- **Metadata Path...** (optional, **.CSV file** mode only): the CSV file describing the images.
- **Template** (**Path-based** mode only): the path template described above.

Each box is outlined in green when its path is valid and in red when it is not. Once every box is green, press **Load Data**. **Loaded images** shows how many images were found.

#### 1.3. Metadata table

Once loaded, the images' metadata is displayed in this table. Click a row to display the corresponding image, and its mask if there is one, in the napari viewer.

The table can be detached from the widget into its own window, which is handy for large datasets. Cluster labels are added to it as a new column once a clustering has been computed (see [3.1](#31-clustering)).

### 2. Processing

#### 2.1. Embeddings

- **Processing Device**: choose **CPU**, or **CUDA** to use a GPU. **CUDA** can only be selected if a compatible GPU is detected on your machine.
- **Select Model...**: choose the foundation model that extracts the embeddings. The selected model is downloaded the first time it is used, which can take a few minutes.
- **Current progress**: how many of the loaded images already have their embeddings.

**Advanced options...** opens additional settings. The defaults work for most datasets.

- *Resource management*
    - **Batch size**: how many images are sent to the model at once. Larger is faster, but uses more memory; lower it if you run out of memory.
    - **Num. workers**: how many extra processes read the images from disk.
- *Image pre-processing*
    - **Channel mode**: how images with several channels are handled. **split** extracts embeddings for each channel separately and joins them; **combined** sends all channels to the model as a single image.
    - **Resize size**: when ticked, every image is resized to a square of this size before reaching the model.
    - **Pad size**: when ticked, smaller images are padded with zeros up to this size before being resized.
    - **L2-normalize channels**: in **split** mode, gives every channel the same weight in the final embedding.

#### 2.2. Properties

Properties are measurements taken directly on the images and masks (intensity, shape, texture...). They are computed after the embeddings.

- **Select Preset...**: choose which properties are computed. The preset is shown under **Selected preset**.
    - **Basic**: essential intensity statistics and mask shape properties.
    - **Standard**: all shape and intensity statistics, plus concentric ring features.
    - **Complete**: everything, including texture and image quality measurements.
    - **Shape**: mask shape properties only. Requires masks.
    - **Intensity**: image intensity statistics only. Works without masks.
- **Compute properties after embeddings**: when checked, the properties are computed right after the embeddings, so the whole task finishes without further input. This is useful when the embedding extraction is long.
- **Compute Properties**: computes the properties alone, from the existing embeddings. Use it to compute them later, or to compute them again with a different preset.

#### 2.3. Processing

Press **Process Images** to extract the embeddings, followed by the properties if **Compute properties after embeddings** is checked. The progress bar and **Elapsed time** follow the task as it runs.

- **Stop** interrupts the running task. The embeddings computed so far are kept, and automatically saved to a file named `autosave.hdf5` in the folder napari was started from. The same happens if the extraction fails partway.
- **Resume progress**: when checked, a new run continues from the existing embeddings and only processes the images that are missing. When unchecked, every image is processed again.

#### 2.4. Save / Load

- **Save results...** writes the embeddings and properties to a checkpoint file (`.hdf5`), so they never have to be computed again.
- **Load results...** reads a checkpoint back. Load the same dataset in the **Data loading** tab first. Once loaded, the **Analysis** tab is available without processing the images again.

**Current checkpoint** shows the file in use; hover over it to see its full path.

To continue an interrupted run, load the autosave (`autosave.hdf5`) with **Load results...**, check **Resume progress**, and press **Process Images**.

### 3. Analysis

The analysis tools are available once the embeddings have been extracted or loaded from a checkpoint.

Most tools offer a **Source** option that chooses which data they work on: the **embeddings**, the **properties**, or both **combined**. The last two are only available once properties have been computed.

#### 3.1. Clustering

Clustering sorts the images into groups of similar images.

**Options...** sets the clustering algorithm and its corresponding parameters.

Once the options are set, press **Compute Clustering** to cluster the data. The cluster of each image is added to the metadata table, and can then be used like any other metadata column: to group images, to colour the interactive explorer, or to filter it.

#### 3.2. Image retrieval

- **Prototype Images...** finds the most representative images of each group.
    - **Group By**: the metadata column defining the groups (for example an experimental condition, or the clusters).
    - **Images per Group**: how many prototypes to retrieve for each group.
    - **Source**: the data the images are compared on.
- **Outlier Images...** finds the images that differ the most from the rest of their group.
    - **Group By**: the metadata column defining the groups.
    - **Method**: **z-score** or **iqr** (interquartile range), two ways of measuring how far an image is from the rest of its group.
    - **Threshold**: how far an image must be to count as an outlier. A higher threshold returns fewer images. The same number means different things for the two methods.
    - **Source**: the data the images are compared on.

Press **OK** to run the search. The images found are listed in the table below the buttons. Click a row to display the image in the napari viewer; the layer is named after the image's group.

#### 3.3. Interactive Explorer

**Interactive Explorer...** opens the explorer in a separate window (see [section 4](#4-interactive-explorer)).

### 4. Interactive Explorer

The Interactive Explorer places every image as a point on a 2D map, where similar images end up close to each other. It lets you see how your conditions and clusters spread out, and look at the images behind any point. The window can stay open next to napari while you work.

The plot is on the left, and the controls are on the right, in four groups.

#### 4.1. The plot

- **Hover** over a point to see the image's name and metadata.
- **Click** a point to display its image in the napari viewer. Each click replaces the image shown before, so you can browse through the points one by one.
- **Scroll** to zoom and **drag** to move around.

#### 4.2. Embedding

These options decide how the map is calculated. Changing them does nothing until you press **Compute**.

- **Method**: the dimensionality reduction technique used to project the data onto a 2D map.
    - **t-SNE** (default) is good at showing groups of similar images, but distances between groups are not meaningful.
    - **PCA** is the fastest, and keeps the overall structure of the data, but separates groups less clearly.
    - **UMAP** shows groups as clearly as t-SNE while keeping more of the overall structure.
- **Source**: whether the map is built from the **embeddings**, the **properties**, or both **combined**.
- **Filter...**: restricts the map to the images having chosen metadata values.
    1. Choose a **Field** (a metadata column).
    2. Select one or several values in the **Values** list and press **Add**. Use **Remove** to take values back.
    3. Repeat for other fields if needed. Fields with values chosen are shown in green, with the number of values chosen.
    4. **Clear** removes every value chosen, on every field.

    An image is kept only if it matches every filtered field.
- **Exclude...**: the opposite of **Filter...**. It works the same way, but removes the images having the chosen values. An image is removed if it matches any of the excluded fields.
- **Compute**: calculates the map and draws it.

#### 4.3. Appearance

- **Color by**: the metadata column or property the points are coloured by.
    - Columns holding labels (conditions, clusters, or numbers with only a few different values) give each value its own colour, listed in a legend.
    - Columns holding measurements (such as the properties) colour the points along a gradient, with a colour bar to read the values.

#### 4.4. Highlight

Highlighting fades every point except the ones you choose, to see where they sit on the map.

- **Field**: the metadata column or property to highlight on.
- Choose what to highlight:
    - for a column of labels, pick a **value** from the list;
    - for a column of measurements, set a **range** with the two handles of the slider.
- Check **Highlight** to apply it. Any change to the field, value or range is applied immediately.

#### 4.5. Selection

Select points to list the images behind them, look at them, or export them.

- **Box**: press it, then drag a rectangle over the plot.
- **Lasso**: press it, then draw a free-hand shape around the points.
- **Highlight**: selects the points matching the current **Highlight** field and value or range.
- **Clear**: removes the selection.

Selected points are shown with a yellow overlay, and the number of points selected is shown below the buttons.

**View selection...** opens a table listing the selected images, with their position on the map, their metadata and their properties. Click a row to display the image in the napari viewer. From this window:

- **Export Indices...** saves the indices of the selected images as a comma-separated list in a `.txt` file.
- **Export CSV...** saves the whole table as a `.csv` file.

#### 4.6. Export graph

**Export graph...** saves the plot, with its legend or colour bar, as an image: **SVG** (best for publications and editing) or **PNG**.

## Contributing

Contributions are very welcome. Tests can be run with [tox], please ensure
the coverage at least stays the same before you submit a pull request.

## License

Distributed under the terms of the [MIT] license,
"phenome-gui" is free and open source software

## Issues

If you encounter any problems, please file an issue along with a detailed description.

[napari]: https://github.com/napari/napari
[copier]: https://copier.readthedocs.io/en/stable/
[MIT]: http://opensource.org/licenses/MIT
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[GNU GPL v3.0]: http://www.gnu.org/licenses/gpl-3.0.txt
[GNU LGPL v3.0]: http://www.gnu.org/licenses/lgpl-3.0.txt
[Apache Software License 2.0]: http://www.apache.org/licenses/LICENSE-2.0
[Mozilla Public License 2.0]: https://www.mozilla.org/media/MPL/2.0/index.txt
[napari-plugin-template]: https://github.com/napari/napari-plugin-template
[PhenoMe]: https://github.com/AAitorG/PhenoMe/tree/main

[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/
[uv]:  https://docs.astral.sh/uv/
