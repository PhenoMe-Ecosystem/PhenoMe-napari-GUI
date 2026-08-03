# launch_napari.py
from napari import Viewer, run

"""
This scripts run napari with the plugin already launched, for developping purposes
"""

def main():
    viewer = Viewer()
    dock_widget, plugin_widget = viewer.window.add_plugin_dock_widget(
        'phenome-gui', 'PhenoMe'
    )
    run()


if __name__ == "__main__":
    main()