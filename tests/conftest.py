"""Point the tests at a throwaway database before any project module is imported."""
import os
import tempfile

os.environ["DB_PATH"] = os.path.join(tempfile.mkdtemp(), "test.db")
os.environ["LLM_PROVIDER"] = "mock"
os.environ["MOCK_FAIL_RATE"] = "0"
