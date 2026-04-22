"""
Text cleaning, URL scraping, and tokenization.
"""
import re
import logging
from typing import Optional
from transformers import DistilBertTokenizerFast

logger = logging.getLogger(__name__)


def clean_text(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)          # strip HTML
    text = re.sub(r"http\S+|www\.\S+", " ", text)  # remove URLs
    text = re.sub(r"[^\w\s.,!?;:'\"-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def scrape_article(url: str) -> Optional[str]:
    try:
        from newspaper import Article
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except Exception as e:
        logger.warning(f"Failed to scrape {url}: {e}")
        return None


class BiasTokenizer:
    def __init__(self, model_name: str = "distilbert-base-uncased", max_length: int = 512):
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
        self.max_length = max_length

    def encode(self, text: str) -> dict:
        return self.tokenizer(
            clean_text(text),
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

    def encode_batch(self, texts: list[str]) -> dict:
        return self.tokenizer(
            [clean_text(t) for t in texts],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
