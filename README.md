# PhenoMe GUI - A napari plugin

[![License MIT](https://img.shields.io/pypi/l/phenome-gui.svg?color=green)](https://github.com/PhenoMe-Ecosystem/PhenoMe-napari-GUI/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/phenome-gui.svg?color=green)](https://pypi.org/project/phenome-gui)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/phenome-gui)](https://napari-hub.org/plugins/phenome-gui)
[![npe2](https://img.shields.io/badge/plugin-npe2-blue?link=https://napari.org/stable/plugins/index.html)](https://napari.org/stable/plugins/index.html)
[![Copier](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/copier-org/copier/master/img/badge/badge-grayscale-inverted-border-purple.json)](https://github.com/copier-org/copier)

GUI integration of the [PhenoMe] phenotyping framework as a plugin for the [napari] image viewer.


This [napari] plugin was generated with [copier] using the [napari-plugin-template].

<!--
Don't miss the full getting started guide to set up your new package:
https://github.com/napari/napari-plugin-template#getting-started

and review the napari docs for plugin developers:
https://napari.org/stable/plugins/index.html
-->

## Installation

If you you already have napari installed, this plugin can be installed directly from the napari hub. Otherwise, follow these instructions to install napari:

https://napari.org/stable/getting_started/installation.html

You can also install `phenome-gui` in your napari environment via [pip]:

```bash
pip install phenome-gui
```

If napari is not already installed, you can install `phenome-gui` with napari and Qt via:

```bash
pip install "napari[all]" phenome-gui
```

## Usage

This napari plugin integrates the [PhenoMe] phenotyping pipeline in a user-friendly graphical interface. To summarize, PhenoMe is a framework for phenotyping biological image data using embeddings extracted using vision models and physical properties. 

As a first step, take a look at the [tutorial](docs/TUTORIAL.md) for a simple step-by-step explanation.

A [manual](docs/MANUAL.md) is provided for more specific explanations about every functionality.

This plugin allows users to easily execute the PhenoMe pipeline in a guided manner, following the three main steps:

1. **Data loading:** Point to the folders containing the images to analyse
2. **Image processing:** Extract the embeddings and properties from the images
3. **Analysis:** Explore the resulting embeddings

For in-depth information about the inner workings of the plugin, see the [PhenoMe] documentation.

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
