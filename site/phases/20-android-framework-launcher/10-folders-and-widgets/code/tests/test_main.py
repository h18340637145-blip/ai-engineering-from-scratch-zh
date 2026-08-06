import unittest, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from main import LauncherDatabase, FolderItem, CONTAINER_DESKTOP, ITEM_TYPE_FOLDER

class TestFolderOperations(unittest.TestCase):
    def setUp(self):
        self.db = LauncherDatabase()

    def test_create_folder_item_type(self):
        f = self.db.create_folder("Test", 0, 0)
        self.assertEqual(f.item_type, ITEM_TYPE_FOLDER)

    def test_create_folder_on_desktop(self):
        f = self.db.create_folder("Test", 0, 0)
        self.assertEqual(f.container, CONTAINER_DESKTOP)

    def test_add_child_updates_container(self):
        f = self.db.create_folder("F", 0, 0)
        app = self.db.create_app("App", "com.test", 1, 0)
        f.add_child(app)
        self.assertEqual(app.container, f.id)

    def test_folder_has_correct_children(self):
        f = self.db.create_folder("F", 0, 0)
        a1 = self.db.create_app("A1", "pkg1", 0, 0)
        a2 = self.db.create_app("A2", "pkg2", 1, 0)
        f.add_child(a1)
        f.add_child(a2)
        self.assertEqual(len(f.children), 2)

    def test_rename_updates_title(self):
        f = self.db.create_folder("Old", 0, 0)
        f.rename("New")
        self.assertEqual(f.title, "New")

    def test_unique_ids(self):
        ids = [self.db.create_folder(f"F{i}", 0, i).id for i in range(5)]
        self.assertEqual(len(ids), len(set(ids)))
