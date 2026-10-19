from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QSpinBox,
    QWidget,
)

from ..utils.params import CHANNEL_MODE_CHOICES, MAX_WORKERS, ProcessParams

# What an optional size shows when it is off, so ticking its box offers a size worth running with
DEFAULT_SIZE = 224

# Largest size the resize and pad spinboxes offer
MAX_SIZE = 4096


class AdvancedProcessingParams(QDialog):
    """ Parameter form for the advanced settings of the embedding extraction.

    Edits its own copy of the processing parameters, so the caller only picks up the new values through
    get_params() once the form has been confirmed.

    For now, the advanced parameters are the batch size and the number of loader workers, and how the
    images are prepared: the channel mode, the resize and pad sizes, and whether each channel's
    embedding is normalised. The two sizes are optional, each turned on and off by a checkbox beside it.
    TODO: Think about which parameters should be here. Some parameters would probably instead be tied to the model used (resize, force_rgb, etc)?
    """

    def __init__(self, parent: QWidget, params: ProcessParams) -> None:
        """ Build the form and fill it with the given parameters.

        Args:
            parent: The widget the dialog is parented to, so it closes with it.
            params: The current processing parameters. Copied, never modified in place.
        """
        super().__init__(parent)
        self.setWindowTitle("Processing parameters")
        #self.setFixedSize(300, 160)

        self.params = params.model_copy(deep=True)

        form_layout = QFormLayout()

        # Resource management options
        resource_group = QGroupBox("Resource management")
        resource_layout = QFormLayout()
        resource_group.setLayout(resource_layout)

        # Number of images PhenoMe feeds to the model at once
        self._batch_size_spinbox = QSpinBox()
        self._batch_size_spinbox.setRange(1, 1024)
        self._batch_size_spinbox.setValue(self.params.batch_size)
        resource_layout.addRow("Batch size", self._batch_size_spinbox)

        # Number of processes the images are read in, none of them by default: a worker is a Python
        # started from nothing, which imports torch before it reads its first image, and the run waits
        # on all of them before its first batch. Worth raising where starting a process is cheap and
        # the images are slow to read.
        self._num_workers_spinbox = QSpinBox()
        self._num_workers_spinbox.setRange(0, MAX_WORKERS)
        self._num_workers_spinbox.setValue(self.params.num_workers)
        self._num_workers_spinbox.setToolTip(
            'Processes the images are read in. 0 reads them in the processing thread, which starts '
            'right away; more can read faster once started, at a wait before every run.'
        )
        resource_layout.addRow("Num. workers", self._num_workers_spinbox)


        # Image pre-processing options
        image_group = QGroupBox("Image pre-processing")
        image_layout = QFormLayout()
        image_group.setLayout(image_layout)

        # What a multi-channel image is embedded as. First of the three, being the only one that
        # decides what is embedded rather than how quickly: the other two leave the same embeddings.
        self._channel_mode_combo = QComboBox()
        self._channel_mode_combo.addItems(CHANNEL_MODE_CHOICES)
        self._channel_mode_combo.setCurrentText(self.params.channel_mode)
        self._channel_mode_combo.setToolTip(
            "How a multi-channel image reaches the model. 'split' embeds each channel on its own and "
            "joins the results; 'combined' feeds the channels as one image. A checkpoint filled under "
            "one is not resumed under the other."
        )
        image_layout.addRow("Channel mode", self._channel_mode_combo)

        # The square size the images are resized to before the model reads them
        self._resize_checkbox, self._resize_spinbox = self._optional_size(
            self.params.resize_size,
            'Resize every image to this size, square, before it reaches the model. Unticked, the images '
            'keep their own size.',
        )
        image_layout.addRow("Resize size", self._row(self._resize_checkbox, self._resize_spinbox))

        # The size the images are padded up to, before they are resized
        self._pad_checkbox, self._pad_spinbox = self._optional_size(
            self.params.pad_size,
            'Pad a smaller image with zeros up to this size on each side, before it is resized. '
            'Unticked, nothing is padded.',
        )
        image_layout.addRow("Pad size", self._row(self._pad_checkbox, self._pad_spinbox))

        # Whether each channel weighs the same in a split embedding
        self._l2_checkbox = QCheckBox("L2-normalize channels")
        self._l2_checkbox.setChecked(self.params.l2_normalize_channels)
        self._l2_checkbox.setToolTip(
            "Scale each channel's embedding to unit length before they are joined, so every channel "
            "weighs the same. Only applies in 'split' mode."
        )
        image_layout.addRow(self._l2_checkbox)

        form_layout.addRow(resource_group)
        form_layout.addRow(image_group)

        # OK closes the form with an accepted result, which is what tells the caller to apply the values
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        form_layout.addRow(button_box)

        self.setLayout(form_layout)

    @staticmethod
    def _optional_size(size: int | None, tooltip: str) -> tuple[QCheckBox, QSpinBox]:
        """ Build a size that can be turned off: a checkbox, and the spinbox it enables.

        Off is a size of None, which is what PhenoMe reads as skipping the step. The spinbox keeps
        its value while disabled, so ticking the box again brings back the size it held.

        Args:
            size: The current size, or None for the step being off.
            tooltip: What the size does, shown over both the checkbox and the spinbox.
        Returns:
            The checkbox, ticked when there is a size, and the spinbox showing it.
        """
        spinbox = QSpinBox()
        spinbox.setRange(1, MAX_SIZE)
        spinbox.setValue(DEFAULT_SIZE if size is None else size)
        spinbox.setToolTip(tooltip)

        checkbox = QCheckBox()
        checkbox.setToolTip(tooltip)
        checkbox.toggled.connect(spinbox.setEnabled)
        checkbox.setChecked(size is not None)
        spinbox.setEnabled(checkbox.isChecked())

        return checkbox, spinbox

    @staticmethod
    def _row(checkbox: QCheckBox, spinbox: QSpinBox) -> QHBoxLayout:
        """ Lay a checkbox out on the left of the spinbox it turns on and off.

        Args:
            checkbox: The checkbox turning the size on and off.
            spinbox: The size itself.
        Returns:
            The row holding both.
        """
        row = QHBoxLayout()
        row.addWidget(checkbox)
        row.addWidget(spinbox, 1)
        return row

    def get_params(self) -> ProcessParams:
        """ Return the edited parameters.

        The fields the form does not show, such as the model and the device, are carried over as they
        were given. A size whose box is unticked is handed back as None.

        Returns:
            The local copy of the processing parameters, holding the values currently shown in the form.
        """
        self.params.channel_mode = self._channel_mode_combo.currentText()
        self.params.batch_size = self._batch_size_spinbox.value()
        self.params.num_workers = self._num_workers_spinbox.value()
        self.params.resize_size = self._resize_spinbox.value() if self._resize_checkbox.isChecked() else None
        self.params.pad_size = self._pad_spinbox.value() if self._pad_checkbox.isChecked() else None
        self.params.l2_normalize_channels = self._l2_checkbox.isChecked()
        return self.params
