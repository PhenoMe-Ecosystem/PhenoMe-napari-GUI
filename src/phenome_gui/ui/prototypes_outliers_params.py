from magicgui.widgets import create_widget
from qtpy.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from ..utils.params import (
    METHOD_CHOICES,
    SOURCE_CHOICES,
    OutliersParams,
    PrototypesParams,
)


class PrototypesParamsWindow(QDialog):
    """ Parameter form for the prototype images step.

    Edits its own copy of the prototype parameters, so the caller only picks up the new values through
    get_params() once the form has been confirmed. Confirming it is also what runs the search, as the
    form is opened by the 'View Prototype Images' button rather than by a separate options button.

    TODO: Add advanced parameters such as the distance metric and the choice of property subset.
    """

    def __init__(
        self,
        parent: QWidget,
        params: PrototypesParams,
        columns: list[str],
        properties_available: bool=False,
    ) -> None:
        """ Build the form and fill it with the given parameters.

        Args:
            parent: The widget the dialog is parented to, so it closes with it.
            params: The current prototype parameters. Copied, never modified in place.
            columns: Metadata columns the images can be grouped by. Empty until data is loaded.
        """
        super().__init__(parent)
        self.setWindowTitle("Prototype Images Options")
        self.resize(400, 200)

        self.params = params.model_copy(deep=True)

        layout = QVBoxLayout(self)

        # Metadata column the images are grouped by. The stored value is None until a column has been
        # picked once, and create_widget rejects a value absent from its choices, so it is only passed
        # through when it is still one of the loaded columns.
        group_by_row = QHBoxLayout()
        group_by_row.addWidget(QLabel("Group By"))
        group_by_options = {'choices':columns}
        if self.params.group_by in columns:
            group_by_options['value'] = self.params.group_by
        self._group_by_dropdown = create_widget(
            label='Group By', widget_type='ComboBox', options=group_by_options
        )
        self._group_by_dropdown.native.setEnabled(bool(columns))
        self._group_by_dropdown.native.setToolTip('' if columns else 'Please load data first.')
        group_by_row.addWidget(self._group_by_dropdown.native)

        n_prototypes_row = QHBoxLayout()
        n_prototypes_row.addWidget(QLabel("Images per Group"))
        self._n_prototypes_spin = create_widget(
            widget_type='SpinBox', options={'value':self.params.n_prototypes, 'min':1, 'max':100}
        )
        n_prototypes_row.addWidget(self._n_prototypes_spin.native)

        source_row = QHBoxLayout()
        source_row.addWidget(QLabel("Source"))
        self._source_dropdown = create_widget(
            label='Source', widget_type='ComboBox', options={
                'choices':SOURCE_CHOICES, 'value':self.params.source}
        )
        source_row.addWidget(self._source_dropdown.native)
        if not properties_available:
            self._source_dropdown.native.setEnabled(False)

        layout.addLayout(group_by_row)
        layout.addLayout(n_prototypes_row)
        layout.addLayout(source_row)

        self._cancel_btn = create_widget(label='Cancel', widget_type='PushButton')
        self._ok_btn = create_widget(label='OK', widget_type='PushButton')
        buttons_row = QHBoxLayout()
        buttons_row.addWidget(self._cancel_btn.native)
        buttons_row.addWidget(self._ok_btn.native)
        layout.addLayout(buttons_row)

        # The dropdown may have landed on the first column rather than on the stored value, so the copy
        # is squared with the form before the user touches anything.
        self._on_value_change()

        self._ok_btn.changed.connect(self._accept)
        self._cancel_btn.changed.connect(self._cancel)
        self._group_by_dropdown.changed.connect(self._on_value_change)
        self._n_prototypes_spin.changed.connect(self._on_value_change)
        self._source_dropdown.changed.connect(self._on_value_change)

    def _on_value_change(self) -> None:
        """ Write the widget values back into the local parameter copy when any of them changes. """
        self.params.group_by = self._group_by_dropdown.value or None
        self.params.n_prototypes = self._n_prototypes_spin.value
        self.params.source = self._source_dropdown.value

    def _accept(self) -> None:
        """ Close the form with an accepted result, which is what tells the caller to apply the values. """
        self.accept()

    def _cancel(self) -> None:
        """ Close the form with a rejected result, leaving the caller's parameters untouched. """
        self.reject()

    def get_params(self) -> PrototypesParams:
        """ Return the edited parameters.

        Returns:
            PrototypesParams: The local copy of the prototype parameters, holding the values currently shown in the form.
        """
        return self.params


class OutliersParamsWindow(QDialog):
    """ Parameter form for the outlier images step.

    Edits its own copy of the outlier parameters, so the caller only picks up the new values through
    get_params() once the form has been confirmed. Confirming it is also what runs the detection, as the
    form is opened by the 'Outlier Images...' button rather than by a separate options button.

    Unlike the prototype step there is no count to set: PhenoMe returns every image past the threshold,
    so how many outliers come back is decided by the method and the threshold together.
    """

    def __init__(
        self,
        parent: QWidget,
        params: OutliersParams,
        columns: list[str],
        properties_available: bool=False,
    ) -> None:
        """ Build the form and fill it with the given parameters.

        Args:
            parent: The widget the dialog is parented to, so it closes with it.
            params: The current outlier parameters. Copied, never modified in place.
            columns: Metadata columns the images can be grouped by. Empty until data is loaded.
            properties_available: Whether the properties have been computed, which the source choices need.
        """
        super().__init__(parent)
        self.setWindowTitle("Outlier Images Options")
        self.resize(400, 250)

        self.params = params.model_copy(deep=True)

        layout = QVBoxLayout(self)

        # Metadata column the images are grouped by. The stored value is None until a column has been
        # picked once, and create_widget rejects a value absent from its choices, so it is only passed
        # through when it is still one of the loaded columns.
        group_by_row = QHBoxLayout()
        group_by_row.addWidget(QLabel("Group By"))
        group_by_options = {'choices':columns}
        if self.params.group_by in columns:
            group_by_options['value'] = self.params.group_by
        self._group_by_dropdown = create_widget(
            label='Group By', widget_type='ComboBox', options=group_by_options
        )
        self._group_by_dropdown.native.setEnabled(bool(columns))
        self._group_by_dropdown.native.setToolTip('' if columns else 'Please load data first.')
        group_by_row.addWidget(self._group_by_dropdown.native)

        method_row = QHBoxLayout()
        method_row.addWidget(QLabel("Method"))
        self._method_dropdown = create_widget(
            label='Method', widget_type='ComboBox', options={
                'choices':METHOD_CHOICES, 'value':self.params.method}
        )
        method_row.addWidget(self._method_dropdown.native)

        # Read as a number of standard deviations for 'z-score', and as a multiple of the interquartile
        # range for 'iqr', so the same value means different things from one method to the other.
        threshold_row = QHBoxLayout()
        threshold_row.addWidget(QLabel("Threshold"))
        self._threshold_spin = create_widget(
            widget_type='FloatSpinBox',
            options={'value':self.params.threshold, 'min':0.1, 'max':100, 'step':0.1}
        )
        threshold_row.addWidget(self._threshold_spin.native)

        source_row = QHBoxLayout()
        source_row.addWidget(QLabel("Source"))
        self._source_dropdown = create_widget(
            label='Source', widget_type='ComboBox', options={
                'choices':SOURCE_CHOICES, 'value':self.params.source}
        )
        source_row.addWidget(self._source_dropdown.native)
        if not properties_available:
            self._source_dropdown.native.setEnabled(False)

        layout.addLayout(group_by_row)
        layout.addLayout(method_row)
        layout.addLayout(threshold_row)
        layout.addLayout(source_row)

        self._cancel_btn = create_widget(label='Cancel', widget_type='PushButton')
        self._ok_btn = create_widget(label='OK', widget_type='PushButton')
        buttons_row = QHBoxLayout()
        buttons_row.addWidget(self._cancel_btn.native)
        buttons_row.addWidget(self._ok_btn.native)
        layout.addLayout(buttons_row)

        # The dropdown may have landed on the first column rather than on the stored value, so the copy
        # is squared with the form before the user touches anything.
        self._on_value_change()

        self._ok_btn.changed.connect(self._accept)
        self._cancel_btn.changed.connect(self._cancel)
        self._group_by_dropdown.changed.connect(self._on_value_change)
        self._method_dropdown.changed.connect(self._on_value_change)
        self._threshold_spin.changed.connect(self._on_value_change)
        self._source_dropdown.changed.connect(self._on_value_change)

    def _on_value_change(self) -> None:
        """ Write the widget values back into the local parameter copy when any of them changes. """
        self.params.group_by = self._group_by_dropdown.value or None
        self.params.method = self._method_dropdown.value
        self.params.threshold = self._threshold_spin.value
        self.params.source = self._source_dropdown.value

    def _accept(self) -> None:
        """ Close the form with an accepted result, which is what tells the caller to apply the values. """
        self.accept()

    def _cancel(self) -> None:
        """ Close the form with a rejected result, leaving the caller's parameters untouched. """
        self.reject()

    def get_params(self) -> OutliersParams:
        """ Return the edited parameters.

        Returns:
            OutliersParams: The local copy of the outlier parameters, holding the values currently shown in the form.
        """
        return self.params
