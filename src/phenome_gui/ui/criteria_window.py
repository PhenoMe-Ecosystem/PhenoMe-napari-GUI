"""The forms picking which images a projection is computed from.

One form serves both: a filter names the values an image has to have to be kept, an exclusion the
values that drop it. Only the name they go under differs, so they share everything but the two words
that say which way their picks are read.
"""

import logging

import pandas as pd
from qtpy.QtCore import Qt
from qtpy.QtGui import QBrush, QColor
from qtpy.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QStyledItemDelegate,
    QVBoxLayout,
    QWidget,
)

from .styles import COLOR_OK

logger = logging.getLogger("main_logger")


class _CriteriaWindow(QDialog):
    """ Form picking metadata values, whichever way the projection then reads them.

    One field at a time: the values it holds are listed on the left, the ones already picked on the
    right, and the two buttons move a selection from one list to the other. The picks are kept per
    field rather than in one heap, so leaving a field and coming back to it shows what was picked
    there - which is also the shape PhenoMe wants them in, a mapping of field to values. A field holding
    picks has its row in the field list shown in green, with the number of values picked beside its
    name, so the picks of the fields not shown are not out of sight.

    Each row of the field list carries the field's name as its data, its text being free to say more:
    the name is what every lookup reads.

    Never used directly: the two subclasses below name it, and nothing else about them differs.
    """

    # What the window is called, what the list of picked values is headed with, and what is said of
    # the values picked on a field, beside its name in the field list
    WINDOW_TITLE = ''
    PICKED_LABEL = ''
    PICKED_VERB = ''

    def __init__(
        self,
        parent: QWidget,
        columns: list[str],
        metadata_df: pd.DataFrame | None = None,
        criteria: dict[str, list] | None = None,
    ) -> None:
        """ Build the form and fill it with the values of the first field.

        Args:
            parent: The widget the dialog is parented to, so it closes with it.
            columns: Fields the images can be picked on, which are the metadata columns
                and the cluster labels among them. Empty until data has been loaded.
            metadata_df: The metadata the values of each field are read from. None
                leaves every field empty, as does a field the frame does not carry.
            criteria: Values already picked, per field, as the metadata holds them.
                Copied, never modified in place.
        """
        super().__init__(parent)
        self.setWindowTitle(self.WINDOW_TITLE)
        self.resize(500, 400)

        self._columns = columns
        self._metadata_df = metadata_df

        # Kept per field, so switching fields neither loses a pick nor shows it under another name
        self._criteria = {field: list(values) for field, values in (criteria or {}).items()}

        self._build_ui()

    def _build_ui(self) -> None:
        """ Build the form and its widgets.

        The field is chosen at the top, and the two lists sit side by side underneath it with the
        button that moves a selection into the other one beneath each.
        """
        layout = QVBoxLayout(self)

        field_row = QHBoxLayout()
        field_row.addWidget(QLabel("Field: "))
        self._field_combo = QComboBox()
        for field in self._columns:
            self._field_combo.addItem(field, field)
        self._field_combo.setEnabled(bool(self._columns))
        self._field_combo.setToolTip('' if self._columns else 'Please load data first.')

        # Drawn by a plain item delegate, which paints a row's own background. The macOS style draws
        # the list as a native menu otherwise, and leaves out the colour marking a field holding picks.
        self._field_combo.setItemDelegate(QStyledItemDelegate(self._field_combo))
        field_row.addWidget(self._field_combo, 1)
        layout.addLayout(field_row)

        # The values of the chosen field, and the ones picked out of them
        self._values_list = QListWidget()
        self._picked_list = QListWidget()
        self._add_button = QPushButton("Add")
        self._remove_button = QPushButton("Remove")

        lists_row = QHBoxLayout()
        lists_row.addLayout(self._build_list_column("Values", self._values_list, self._add_button), 1)
        lists_row.addLayout(
            self._build_list_column(self.PICKED_LABEL, self._picked_list, self._remove_button), 1
        )
        layout.addLayout(lists_row, 1)

        # Kept out of both columns, and off to one side: the two buttons above act on the selection
        # of the list they sit under, where this one lets go of every field at once.
        self._clear_button = QPushButton("Clear")
        clear_row = QHBoxLayout()
        clear_row.addStretch(1)
        clear_row.addWidget(self._clear_button)
        layout.addLayout(clear_row)

        # Several values are usually wanted at once, a criterion rarely coming down to a single one
        for values in (self._values_list, self._picked_list):
            values.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)

        # On the row rather than on its text, which changes whenever the field's picks are counted
        self._field_combo.currentIndexChanged.connect(self._on_field_changed)
        self._add_button.clicked.connect(self._on_add_clicked)
        self._remove_button.clicked.connect(self._on_remove_clicked)
        self._clear_button.clicked.connect(self._on_clear_clicked)

        # Filled here rather than next to each list, as it needs both of them to exist
        self._refresh_lists()

    def _build_list_column(self, title: str, values: QListWidget, button: QPushButton) -> QVBoxLayout:
        """ Stack a list under its title and over the button acting on it.

        Args:
            title: Name shown above the list.
            values: The list itself.
            button: The button moving the list's selection into the other list.
        Returns:
            The three of them, stacked.
        """
        column = QVBoxLayout()
        column.addWidget(QLabel(title))
        column.addWidget(values, 1)
        column.addWidget(button)
        return column

    def _field_values(self) -> list:
        """ Every value the chosen field holds.

        Sorted on the values themselves rather than on their text, so cluster 10 follows cluster 9
        instead of cluster 1. A frame mixing types in one column has no such order and falls back to
        the text.

        Returns:
            The distinct values of the field, as the metadata holds them, empty when there is
            no field or no data.
        """
        field = self._current_field()
        if not field or self._metadata_df is None or field not in self._metadata_df.columns:
            return []

        distinct = self._metadata_df[field].dropna().unique()
        try:
            return sorted(distinct)
        except TypeError:
            return sorted(distinct, key=str)

    def _refresh_lists(self) -> None:
        """ Fill both lists for the chosen field, and square the Clear button with what is picked.

        A value sits in one list or the other, never in both: what has been picked is taken out of
        the values it was picked from, so the left list always reads as what is left to pick.

        Each row shows its value as text but carries the value itself, PhenoMe matching a criterion
        by equality against a metadata that holds the cluster labels as numbers: a cluster picked as
        '1' would go down as text and match no image at all.
        """
        picked = self._criteria.get(self._current_field(), [])

        self._values_list.clear()
        for value in self._field_values():
            if value not in picked:
                self._add_value(self._values_list, value)

        self._picked_list.clear()
        for value in picked:
            self._add_value(self._picked_list, value)

        # Closed when there is nothing anywhere to let go of
        self._clear_button.setEnabled(bool(self.get_criteria()))

        self._refresh_field_marks()

    def _current_field(self) -> str:
        """ The name of the field currently shown, read from its row rather than from the text it shows.

        Returns:
            The field's name, empty when there is no field to show.
        """
        return self._field_combo.currentData() or ''

    def _refresh_field_marks(self) -> None:
        """ Mark every field holding picks: its row in green, and the number of values picked beside its name.

        The other fields are shown by their bare name, as the style draws them. Gone over whole after
        every change, the fields being a handful: a pick, a removal and a clear can each change which
        fields hold picks, the clear all of them at once.
        """
        for index in range(self._field_combo.count()):
            field = self._field_combo.itemData(index)
            n_picked = len(self._criteria.get(field, []))

            label = field
            if n_picked:
                label = f"{field} ({n_picked} {'value' if n_picked == 1 else 'values'} {self.PICKED_VERB})"

            self._field_combo.setItemText(index, label)
            self._field_combo.setItemData(
                index, QBrush(QColor(COLOR_OK)) if n_picked else None, Qt.ItemDataRole.BackgroundRole
            )

    def _add_value(self, values: QListWidget, value) -> None:
        """ List one value, keeping it alongside the text it is shown as.

        Args:
            values: The list to add the value to.
            value: The value itself, as the metadata holds it.
        """
        item = QListWidgetItem(str(value))
        item.setData(Qt.ItemDataRole.UserRole, value)
        values.addItem(item)

    def _on_field_changed(self) -> None:
        """ Show the values of the newly chosen field, and what has already been picked on it. """
        self._refresh_lists()

    def _on_add_clicked(self) -> None:
        """ Pick the values selected on the left, moving them into the list on the right. """
        self._move_selection(self._values_list, pick=True)

    def _on_remove_clicked(self) -> None:
        """ Let go of the values selected on the right, moving them back into the left list. """
        self._move_selection(self._picked_list, pick=False)

    def _on_clear_clicked(self) -> None:
        """ Let go of every value picked, on every field.

        The picks of the fields not currently shown go with it, which is the whole point of the
        button: they are the ones there is otherwise no way to see, let alone take back, without
        walking the whole combo box.
        """
        # Stringified rather than passed as a mapping: logging reads a lone mapping argument as the
        # values to interpolate by name, and raises out of the handler rather than logging anything
        logger.info("Cleared %s.", str(self._criteria) if self._criteria else 'nothing')

        self._criteria.clear()
        self._refresh_lists()

    def _move_selection(self, source: QListWidget, pick: bool) -> None:
        """ Move the values selected in a list into the other one.

        Args:
            source: The list the selection is taken from.
            pick: True when the selection is being picked, False when it is being let go of.
        """
        selected = [item.data(Qt.ItemDataRole.UserRole) for item in source.selectedItems()]
        if not selected:
            return

        field = self._current_field()
        picked = self._criteria.setdefault(field, [])

        if pick:
            picked.extend(value for value in selected if value not in picked)
        else:
            self._criteria[field] = [value for value in picked if value not in selected]

        logger.info("Picked %s on %s.", self._criteria[field] or 'nothing', field)
        self._refresh_lists()

    def get_criteria(self) -> dict[str, list]:
        """ Return the values picked on each field.

        The fields nothing was picked on are left out, an empty list of values reading as a criterion
        no image can answer rather than as no criterion at all. The values are typed as the metadata
        holds them, which is how PhenoMe has to be given them.

        Returns:
            The picked values, per field.
        """
        return {field: list(values) for field, values in self._criteria.items() if values}


class FilterWindow(_CriteriaWindow):
    """ Form choosing the metadata values a projection is kept to.

    An image is projected when it answers every field picked on, so filtering two fields narrows the
    projection twice over.
    """

    WINDOW_TITLE = 'Filter Images'
    PICKED_LABEL = 'Filtered Values'
    PICKED_VERB = 'filtered'


class ExcludeWindow(_CriteriaWindow):
    """ Form choosing the metadata values a projection leaves out.

    Read the other way round from the filter, and combined the other way round too: an image is
    dropped for answering any one of the fields picked on, rather than having to answer them all.
    """

    WINDOW_TITLE = 'Exclude Images'
    PICKED_LABEL = 'Excluded Values'
    PICKED_VERB = 'excluded'
