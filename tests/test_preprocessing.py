from src.preprocessing import clean_text
import pytest
import sys
import os
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")))


def test_removes_html():
    assert "<" not in clean_text("<b>Hello</b>")


def test_remove_urls():
    assert "http" not in clean_text("Visit https://example.com today")


def test_empty():
    assert clean_text("") == ""


def test_collapses_whitespace():
    assert "  " not in clean_text("hello     world")
