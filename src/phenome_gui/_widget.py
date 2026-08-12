import logging
from typing import TYPE_CHECKING

from qtpy.QtWidgets import (
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .ui.analysis_panel import AnalysisPanel
from .ui.loading_panel import LoadingPanel
from .ui.processing_panel import ProcessingPanel
from .utils.image_viewer import ImageViewer
from .utils.phenome_interface import PhenoMeInterface

logger = logging.getLogger("main_logger")
logging.basicConfig(filename='phenome_log.log', level=logging.DEBUG, filemode='w')

if TYPE_CHECKING:
    import napari

class PhenoMeWidget(QWidget):
    """ The napari dock widget of the PhenoMe plugin.

    Holds one panel per step of the workflow: data loading, processing and analysis. Owns what they share
    - the interface to PhenoMe and the image viewer - and connects them to each other, the panels
    themselves knowing nothing about their siblings.
    """

    def __init__(self, viewer: 'napari.viewer.Viewer') -> None:
        """ Build the widget, its interface to PhenoMe and its three panels.

        Args:
            viewer (napari.viewer.Viewer): The napari viewer the widget is docked in.
        """
        super().__init__()
        self._viewer = viewer

        # Interface to PhenoMe, holding the pipeline object and its parameters
        self._phenome = PhenoMeInterface(seed=42)
        self._image_viewer = ImageViewer(viewer)

        # Panel 1: Data loading and table view of metadata
        self.loading_panel = LoadingPanel(self._phenome, self._image_viewer)

        # Panel 2: Image processing and properties computing
        self.processing_panel = ProcessingPanel(self._phenome)

        # Panel 3: Analysis and plotting
        self.analysis_panel = AnalysisPanel(self._phenome, self._image_viewer)

        self._connect_panels()

        # Build the panel structure of the widget
        self._tabs = QTabWidget()
        self._tabs.addTab(self.loading_panel, "Data loading")
        self._tabs.addTab(self.processing_panel, "Processing")
        self._tabs.addTab(self.analysis_panel, "Analysis")

        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self._tabs)

    def _connect_panels(self) -> None:
        """ Connect each panel to the ones its work affects.

        Every step feeds the next: loading gives the processing step its images, processing gives the
        analysis step its embeddings, and the clustering writes labels the metadata table then shows. The
        panels announce what they have done rather than reaching for each other, so the whole cycle is
        declared here and nowhere else.
        """
        self.loading_panel.data_loaded.connect(self.processing_panel.refresh_buttons)
        self.processing_panel.task_running_changed.connect(self.analysis_panel.set_task_running)
        self.analysis_panel.clustering_finished.connect(self.loading_panel.refresh_table)
