import logging

import pandas as pd
from magicgui.widgets import create_widget
from napari.qt.threading import thread_worker
from qtpy.QtCore import Signal
from qtpy.QtWidgets import (
    QAbstractItemView,
    QGroupBox,
    QHBoxLayout,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ..utils.image_viewer import ImageViewer
from ..utils.phenome_interface import PhenoMeInterface
from .clustering_params import ClusterParamsWindow
from .prototypes_outliers_params import (
    OutliersParamsWindow,
    PrototypesParamsWindow,
)

logger = logging.getLogger("main_logger")


class AnalysisPanel(QWidget):
    """ The analysis tab of the PhenoMe widget.

    Holds the clustering step and the prototype and outlier image retrieval, along with the table the
    retrieved images are listed in.
    """

    # Emitted once the clustering has written its labels into the metadata, so whoever shows the metadata
    # can pick the new column up.
    clustering_finished = Signal()

    # Emitted when the explorer button is pressed. The window is shown alongside the widget rather than
    # inside the tab, so it is owned by whoever owns the panels rather than by the panel itself.
    explorer_requested = Signal()

    def __init__(
        self,
        phenome: PhenoMeInterface,
        image_viewer: ImageViewer,
        parent: QWidget | None = None
    ) -> None:
        """ Build the panel and its widgets.

        Args:
            phenome (PhenoMeInterface): The interface every call to PhenoMe goes through.
            image_viewer (ImageViewer): The viewer the selected image is displayed in.
            parent (QWidget): The widget the panel is nested in.
        """
        super().__init__(parent)
        self._phenome = phenome
        self._image_viewer = image_viewer

        # Every step here runs on the embeddings the processing tasks produce, so they all stay locked
        # while one is running. Kept in step with the processing panel through set_task_running().
        self._task_running = False

        self._build_ui()

    def _build_ui(self) -> None:
        """ Build the analysis tab and its widgets.

        Includes a group to configure and run the clustering, one to view the prototype and outlier
        images of a chosen metadata group, and the button opening the interactive explorer.
        """
        layout = QVBoxLayout()

        # Group for the clustering step
        clustering_group = QGroupBox("Clustering")
        clustering_layout = QHBoxLayout()
        clustering_group.setLayout(clustering_layout)

        # Clustering parameters button, opening a form rebuilt on every use
        self._clustering_params_dialog = None
        self._cluster_options_button = QPushButton("Options...")
        self._cluster_options_button.clicked.connect(self._on_cluster_options_clicked)
        clustering_layout.addWidget(self._cluster_options_button)

        # Clustering button, enabled only once the images have been processed
        self._run_clustering_button = QPushButton("Compute Clustering")
        self._run_clustering_button.clicked.connect(self._on_run_clustering_clicked)
        clustering_layout.addWidget(self._run_clustering_button)

        # Group for the image inspection steps
        images_group = QGroupBox("Image Retrieval")
        images_layout = QVBoxLayout()
        images_group.setLayout(images_layout)

        # Prototype button, enabled only once the images have been processed. It opens the parameter
        # form rather than searching straight away, the search being what confirming the form does.
        self._prototypes_params_dialog = None
        self._prototype_button = QPushButton("Prototype Images...")
        self._prototype_button.clicked.connect(self._on_prototype_clicked)
        images_layout.addWidget(self._prototype_button)

        # Outlier button, enabled only once the images have been processed. It opens the parameter form
        # rather than detecting straight away, the detection being what confirming the form does.
        self._outliers_params_dialog = None
        self._outlier_button = QPushButton("Outlier Images...")
        self._outlier_button.clicked.connect(self._on_outlier_clicked)
        images_layout.addWidget(self._outlier_button)

        # Data table to view the retrieved images, one image per row. Holds the frame of whichever step
        # ran last, kept alongside it so the selection can be resolved back to a full row.
        self._images_df = None
        self._images_group_column = None
        self._images_table = create_widget(
            label='Images Table', widget_type='Table', options={'value': None}
        )
        self._images_table.native.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers) # Prevent editing
        self._images_table.native.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._images_table.native.selectionModel().selectionChanged.connect(self._on_image_table_selection)

        images_layout.addWidget(self._images_table.native, 1)

        # Explorer button, kept out of both groups: it opens a window of its own rather than belonging
        # to the clustering or the retrieval step. Opening it is left to whoever owns the window.
        self._explorer_button = QPushButton("Interactive Explorer...")
        self._explorer_button.clicked.connect(self.explorer_requested.emit)

        # Refreshed here rather than next to each button, as it needs all of them to exist
        self._refresh_analysis_buttons()

        layout.addWidget(clustering_group, 0)
        layout.addWidget(images_group, 1)
        layout.addWidget(self._explorer_button, 0)
        self.setLayout(layout)

    def clear_results(self) -> None:
        """ Let go of the retrieved images, a new dataset having taken the place of the one they came from.

        The buttons are refreshed alongside: the embeddings they run on went with the previous dataset,
        so every step here closes back down until the new one has been processed.
        """
        self._update_images_table(None, None)
        self._refresh_analysis_buttons()

    def set_task_running(self, running: bool) -> None:
        """ Take note that a processing task has started or finished, and lock the buttons accordingly.

        Args:
            running (bool): True when a task has just been started, False once it has finished.
        """
        self._task_running = running
        self._refresh_analysis_buttons()

    def _refresh_analysis_buttons(self) -> None:
        """ Enable the analysis buttons only once the images have been processed, and no task is running.

        The clustering, prototype and outlier steps all run on the embeddings held by the pipeline, and
        raise when there are none, so the state is read from the interface the same way the properties
        button does. The explorer projects those same embeddings, so it waits on them too. They share
        one method as their condition is the same. Called when building the tab and whenever a
        processing task starts or finishes.
        """
        has_processed_images = self._phenome.has_processed_images

        if self._task_running:
            tooltip = 'Please wait for the running task to finish.'
        else:
            tooltip = '' if has_processed_images else 'Please extract embeddings first.'

        for button in (
            self._run_clustering_button,
            self._prototype_button,
            self._outlier_button,
            self._explorer_button,
        ):
            button.setEnabled(has_processed_images and not self._task_running)
            button.setToolTip(tooltip)

    def _on_cluster_options_clicked(self) -> None:
        """ Open the clustering parameter form.

        The form is rebuilt on every use, as its fields are filled when it is constructed, so a cancelled
        edit is never kept nor shown again. The parameters are only written back when the form is confirmed.
        """
        self._clustering_params_dialog = ClusterParamsWindow(
            self, self._phenome.params.clustering, self._phenome.has_properties
        )

        if self._clustering_params_dialog.exec_():
            self._phenome.params.clustering = self._clustering_params_dialog.get_params()

    def _on_run_clustering_clicked(self) -> None:
        """ Begin the clustering task."""
        logger.info("Compute Clustering clicked, with parameters %s", self._phenome.params.clustering)
        worker = self._compute_clustering_worker()
        worker.returned.connect(self._on_finish_clustering)
        worker.start()

    @thread_worker
    def _compute_clustering_worker(self) -> None:
        """ Create worker to compute the properties on a parallel thread. """
        self._phenome.compute_clustering()

    def _on_finish_clustering(self) -> None:
        """ Once clustering is finished, announce it so the metadata table picks up the cluster numbers.

        The cluster column also becomes a grouping option, which needs no work here: the parameter forms
        read the metadata columns when they are built, and they are rebuilt on every use.
        """
        self.clustering_finished.emit()

    def _on_prototype_clicked(self) -> None:
        """ Open the prototype parameter form, and begin the search once it is confirmed.

        The form is rebuilt on every use, as its fields are filled when it is constructed, so a cancelled
        edit is never kept nor shown again. Cancelling it runs no search and changes no parameter.
        """
        self._prototypes_params_dialog = PrototypesParamsWindow(
            self, self._phenome.params.prototypes, self._phenome.get_metadata_columns(), self._phenome.has_properties
        )

        if not self._prototypes_params_dialog.exec_():
            return

        self._phenome.params.prototypes = self._prototypes_params_dialog.get_params()

        logger.info("View Prototype Images confirmed, with parameters %s", self._phenome.params.prototypes)
        worker = self._prototypes_worker()
        worker.returned.connect(self._on_finish_prototypes)
        worker.start()

    @thread_worker
    def _prototypes_worker(self) -> None:
        """ Create worker to find the prototype images on a parallel thread. """
        self._phenome.get_prototype_images()

    def _on_finish_prototypes(self) -> None:
        """ Called when the prototype search is done. Shows the prototypes in the images table. """
        prototypes_df = self._phenome.prototypes_df
        logger.info("Found %d prototype images.", 0 if prototypes_df is None else len(prototypes_df))
        self._update_images_table(prototypes_df, self._phenome.params.prototypes.group_by)

    def _on_outlier_clicked(self) -> None:
        """ Open the outlier parameter form, and begin the detection once it is confirmed.

        The form is rebuilt on every use, as its fields are filled when it is constructed, so a cancelled
        edit is never kept nor shown again. Cancelling it runs no detection and changes no parameter.
        """
        self._outliers_params_dialog = OutliersParamsWindow(
            self, self._phenome.params.outliers, self._phenome.get_metadata_columns(), self._phenome.has_properties
        )

        if not self._outliers_params_dialog.exec_():
            return

        self._phenome.params.outliers = self._outliers_params_dialog.get_params()

        logger.info("View Outlier Images confirmed, with parameters %s", self._phenome.params.outliers)
        worker = self._outliers_worker()
        worker.returned.connect(self._on_finish_outliers)
        worker.start()

    @thread_worker
    def _outliers_worker(self) -> None:
        """ Create worker to detect the outlier images on a parallel thread. """
        self._phenome.get_outlier_images()

    def _on_finish_outliers(self) -> None:
        """ Called when the outlier detection is done. Shows the outliers in the images table. """
        outliers_df = self._phenome.outliers_df
        logger.info("Found %d outlier images.", 0 if outliers_df is None else len(outliers_df))
        self._update_images_table(outliers_df, self._phenome.params.outliers.group_by)

    def _update_images_table(self, images_df: pd.DataFrame | None, group_by: str | None) -> None:
        """ Show the images retrieved by the prototype or outlier step, one image per row.

        The frame is kept whole for the selection handler, while the table only shows the columns worth
        reading: the image index and path are dropped from the view but stay available behind it. The
        rows are not reordered, so a table row still addresses the same row of the kept frame.

        Args:
            images_df (pd.DataFrame): The retrieved images. None or empty clears the table, which is a
                normal outcome for the outlier step when nothing is past the threshold.
            group_by (str | None): The grouping key the step was given, used to find the group
                column. None when the step ran without one.
        """
        self._images_df = images_df

        if images_df is None or images_df.empty:
            self._images_group_column = None
            self._images_table.value = pd.DataFrame()
            return

        # The two PhenoMe calls disagree on the group column: both name it after the grouping key when it
        # resolved, but find_prototypes falls back to a 'group' column holding 'All' while detect_outliers
        # leaves it out entirely.
        if group_by in images_df.columns:
            self._images_group_column = group_by
        elif 'group' in images_df.columns:
            self._images_group_column = 'group'
        else:
            self._images_group_column = None

        self._images_table.value = images_df.drop(columns=['image_index', 'image_path'], errors='ignore')

    def _on_image_table_selection(self) -> None:
        """ Display the corresponding image and mask when a row is selected from the images table.

        The layer is named after the group the image belongs to, so the groups stay tellable apart in the
        napari layer list. Falls back to the image name alone when the step ran without a grouping key.
        """
        row = self._images_table.native.currentRow()
        if row < 0 or self._images_df is None or self._images_df.empty:
            return

        row_data = self._images_df.iloc[row]
        name = str(row_data['image_name'])
        if self._images_group_column:
            name = f"{row_data[self._images_group_column]} - {name}"

        image_paths, mask_paths, metadata = self._phenome.get_image_display_data(row_data['image_path'])
        self._image_viewer.display_images(image_paths, mask_paths, metadata, names=[name])
