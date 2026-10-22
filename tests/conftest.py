"""NiceGUI ships a pytest harness that drives the app's pages in-process; loading its plugin
here is what gives tests the `user` fixture. It only adds fixtures — tests that never ask for
them are unaffected.

Steps open their review pages in the browser when they finish (`core/console.reveal`); a test
run would open one per step it runs, so it never does. Set before any test or child process."""
import os

os.environ["TOOLKIT_NO_OPEN"] = "1"

pytest_plugins = ["nicegui.testing.user_plugin"]
