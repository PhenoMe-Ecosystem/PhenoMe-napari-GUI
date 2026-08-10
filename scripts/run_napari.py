"""
This script runs napari with the plugin already launched, for developping purposes
"""

from napari import Viewer, run


def main() -> None:
    """ Open a napari viewer with the PhenoMe widget already docked, and run the event loop.
    """
    viewer = Viewer()
    viewer.window.add_plugin_dock_widget('phenome-gui', 'PhenoMe')
    run()


if __name__ == "__main__":
    main()
