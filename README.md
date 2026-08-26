# phenome-gui

[![License MIT](https://img.shields.io/pypi/l/phenome-gui.svg?color=green)](https://github.com//phenome-gui/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/phenome-gui.svg?color=green)](https://pypi.org/project/phenome-gui)
[![Python Version](https://img.shields.io/pypi/pyversions/phenome-gui.svg?color=green)](https://python.org)
[![tests](https://github.com//phenome-gui/workflows/tests/badge.svg)](https://github.com//phenome-gui/actions)
[![codecov](https://codecov.io/gh//phenome-gui/branch/main/graph/badge.svg)](https://codecov.io/gh//phenome-gui)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/phenome-gui)](https://napari-hub.org/plugins/phenome-gui)
[![npe2](https://img.shields.io/badge/plugin-npe2-blue?link=https://napari.org/stable/plugins/index.html)](https://napari.org/stable/plugins/index.html)
[![Copier](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/copier-org/copier/master/img/badge/badge-grayscale-inverted-border-purple.json)](https://github.com/copier-org/copier)

GUI integration of the PhenoMe phenotyping framework

----------------------------------

This [napari] plugin was generated with [copier] using the [napari-plugin-template].

<!--
Don't miss the full getting started guide to set up your new package:
https://github.com/napari/napari-plugin-template#getting-started

and review the napari docs for plugin developers:
https://napari.org/stable/plugins/index.html
-->

## Installation

Once this plugin is published, you will be able to install `phenome-gui` via [pip]:

```bash
pip install phenome-gui
```

If napari is not already installed, you can install `phenome-gui` with napari and Qt via:

```bash
pip install "phenome-gui[all]"
```

For now, follow these instructions:

1. Install napari following the instructions on the website: https://napari.org/stable/getting_started/installation.html

    We recommend [uv] to manage environments.

2. Activate the napari environment: 

    `source .venv/bin/activate`

3. Run the following command: 
    
    `uv add "git+https://git.valeria.science/predis/flc-phenome-gui.git"`


## Contributing

Contributions are very welcome. Tests can be run with [tox], please ensure
the coverage at least stays the same before you submit a pull request.

## License

Distributed under the terms of the [MIT] license,
"phenome-gui" is free and open source software

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

[napari]: https://github.com/napari/napari
[copier]: https://copier.readthedocs.io/en/stable/
[MIT]: http://opensource.org/licenses/MIT
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[GNU GPL v3.0]: http://www.gnu.org/licenses/gpl-3.0.txt
[GNU LGPL v3.0]: http://www.gnu.org/licenses/lgpl-3.0.txt
[Apache Software License 2.0]: http://www.apache.org/licenses/LICENSE-2.0
[Mozilla Public License 2.0]: https://www.mozilla.org/media/MPL/2.0/index.txt
[napari-plugin-template]: https://github.com/napari/napari-plugin-template

[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/
[uv]:  https://docs.astral.sh/uv/
