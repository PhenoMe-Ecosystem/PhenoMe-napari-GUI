from copy import deepcopy
from typing import Any

from magicgui.widgets import create_widget
from qtpy.QtWidgets import (
    QDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)


class ClusterParamsWindow(QDialog):
    """ Parameter form for the clustering step.

    Edits its own copy of the clustering parameters, so the caller only picks up the new values through
    get_params() once the form has been confirmed.

    TODO: Add advanced parameters such as number of dimensions, dimensionality reduction algorithm and choice to use/include properties for clustering
    """

    def __init__(
        self,
        parent: QWidget,
        params: dict[str, Any]
    ):
        """ Build the form and fill it with the given parameters.

        Args:
            parent (QWidget): The widget the dialog is parented to, so it closes with it.
            params (dict[str, Any]): The current clustering parameters. Copied, never modified in place.
        """
        super().__init__(parent)
        self.setWindowTitle("Clustering Options")
        self.resize(400, 300)

        self.params = deepcopy(params)

        layout = QVBoxLayout(self)
        method_row = QHBoxLayout()
        method_row.addWidget(QLabel("Clustering Method"))
        self._method_dropdown = create_widget(
            label='Clustering Method', widget_type='ComboBox', options={
                'choices':['kmeans', 'dbscan', 'gmm'], 'value':self.params['clustering_method']}
        )
        method_row.addWidget(self._method_dropdown.native)

        # Parameters depending on the selected method
        self._parameters_group = QGroupBox(self._method_dropdown.value + " parameters")
        group_layout = QVBoxLayout()
        self._parameters_group.setLayout(group_layout)

        n_clusters_row = QHBoxLayout()
        n_clusters_label = QLabel("Number of clusters")
        self._n_clusters_spin = create_widget(widget_type='SpinBox', options={'value':self.params['n_clusters']})
        n_clusters_row.addWidget(n_clusters_label)
        n_clusters_row.addWidget(self._n_clusters_spin.native)
        group_layout.addLayout(n_clusters_row)

        dbscan_eps_row = QHBoxLayout()
        dbscan_eps_label = QLabel("DBSCAN EPS")
        self._dbscan_eps_spin = create_widget(widget_type='FloatSpinBox', options={'value':self.params['dbscan_eps']})
        dbscan_eps_row.addWidget(dbscan_eps_label)
        dbscan_eps_row.addWidget(self._dbscan_eps_spin.native)
        group_layout.addLayout(dbscan_eps_row)

        min_samples_row = QHBoxLayout()
        min_samples_label = QLabel("Minimum samples")
        self._min_samples_spin = create_widget(widget_type='SpinBox', options={'value':self.params['dbscan_min_samples']})
        min_samples_row.addWidget(min_samples_label)
        min_samples_row.addWidget(self._min_samples_spin.native)
        group_layout.addLayout(min_samples_row)

        layout.addLayout(method_row)
        layout.addWidget(self._parameters_group)

        self._cancel_btn = create_widget(label='Cancel', widget_type='PushButton')
        self._ok_btn = create_widget(label='OK', widget_type='PushButton')
        buttons_row = QHBoxLayout()
        buttons_row.addWidget(self._cancel_btn.native)
        buttons_row.addWidget(self._ok_btn.native)
        layout.addLayout(buttons_row)

        self._check_method()

        self._ok_btn.changed.connect(self._accept)
        self._cancel_btn.changed.connect(self._cancel)
        self._method_dropdown.changed.connect(self._on_value_change)
        self._dbscan_eps_spin.changed.connect(self._on_value_change)
        self._n_clusters_spin.changed.connect(self._on_value_change)
        self._min_samples_spin.changed.connect(self._on_value_change)

    def _check_method(self) -> None:
        """ Enable the parameters that apply to the selected clustering method, and disable the others.

        Reads the method straight from the dropdown rather than from the parameters, as _on_value_change
        calls this before it has written the new value back.
        """
        if self._method_dropdown.value == 'dbscan':
            self._dbscan_eps_spin.native.setEnabled(True)
            self._min_samples_spin.native.setEnabled(True)
            self._n_clusters_spin.native.setEnabled(False)
        else:
            self._dbscan_eps_spin.native.setEnabled(False)
            self._min_samples_spin.native.setEnabled(False)
            self._n_clusters_spin.native.setEnabled(True)

    def _on_value_change(self) -> None:
        """ Write the widget values back into the local parameter copy when any of them changes. """
        widget_params = {
            'clustering_method':self._method_dropdown.value,
            'n_clusters':self._n_clusters_spin.value,
            'dbscan_eps':self._dbscan_eps_spin.value,
            'dbscan_min_samples':self._min_samples_spin.value
        }

        self._parameters_group.setTitle(self._method_dropdown.value + " parameters")

        for k, v in widget_params.items():
            self.params[k] = v
        self._check_method()

    def _accept(self) -> None:
        """ Close the form with an accepted result, which is what tells the caller to apply the values. """
        self.accept()

    def _cancel(self) -> None:
        """ Close the form with a rejected result, leaving the caller's parameters untouched. """
        self.reject()

    def get_params(self) -> dict[str, Any]:
        """ Return the edited parameters.

        Returns:
            dict[str, Any]: The local copy of the clustering parameters, holding the values currently shown in the form.
        """
        return self.params
