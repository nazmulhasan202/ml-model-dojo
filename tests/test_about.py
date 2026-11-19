from PySide6.QtWidgets import QLabel, QTabWidget

from ml_model_dojo.ui.dialogs import HelpDialog


def test_credit_in_the_status_bar(qtbot, win):
    credit = win.findChild(QLabel, "credit")
    assert credit is not None and credit.isVisible()
    assert "Created by" in credit.text() and "Nazmul Hasan" in credit.text()
    assert "nazmul.ipe.buet@gmail.com" in credit.text() and "+calude" not in credit.text()
    assert credit.openExternalLinks()


def test_about_tab_lists_the_author_and_degrees(qtbot, win):
    dlg = HelpDialog(win, lambda: None)
    qtbot.addWidget(dlg)
    tabs = dlg.findChild(QTabWidget)
    names = [tabs.tabText(i) for i in range(tabs.count())]
    assert names == ["Overview", "Glossary", "About"]
    text = tabs.widget(names.index("About")).toPlainText()
    for needle in ("Nazmul Hasan", "nazmul.ipe.buet@gmail.com",
                   "PhD in Systems and Industrial Engineering (ongoing), University of Arizona",
                   "Master of Science in Industrial and Production Engineering",
                   "Bachelor of Science in Industrial and Production Engineering",
                   "Bangladesh University of Engineering and Technology (BUET)", "MIT license",
                   "github.com/nazmulhasan202/ml-model-dojo", "beta"):
        assert needle in text, needle
    assert "—" not in text                      # no em dashes
    assert "+calude" not in text
