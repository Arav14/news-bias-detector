"""
Uses Google Gemini API (free tier) to generate human-readable
explanations of why an article was classified as biased.
"""
import os
import logging
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
logger = logging.getLogger(__name__)

# PROMPTS

SYSTEM_PROMPT = """You are an expert media analyst specializing in identifying political bias
in news articles. You are objective, analytical, and cite specific evidence from the text.
Your explanation are concise (3-5 sentences), readable by a general audience, and avoid 
taking a political stance yourself. Always structure your response with:
1. A one-sentence verdict
2. 2-3 specific pieces of textual evidence
3. A brief note on what a more neutral version might look like"""

USER_TEMPLATE = """A news bias classifier labeled the following article excerpt as : **{label}** (confidence: {confidence:.0})

Article text:
\"\"\"
{text}
\"\"\"

Please explain why this article might be considered {label}-learning. Cite specific words,
phrases, or framing choices from the text as evidence. Be analytical and fair."""

# EXPLAINER


class BiasExplainer:
    """
    Calls Google Gemini to generate a natural-language explanation
    for a bias classification.
    """

    def __init__(self, model: str = "gemini-1.5-flash"):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set. Check your .env file.")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(
            model_name=model,
            system_instruction=SYSTEM_PROMPT,
        )

    def explain(self, text: str, label: str, confidence: float, max_chars: int = 1500) -> str:
        """
        Generate an explanation for why `text` was classified as `label`.

        Args:
            text: The article text (truncated if too long)
            label: Predicted label - "Left", "Center", or "Right"
            confidence: Model confidence score (0-1)
            max_chars: Max characters sent to the API

        Return:
            Explanation string from Gemini
        """
        truncated = text[:max_chars] + ("..." if len(text) > max_chars else "")
        prompt = USER_TEMPLATE.format(
            label=label,
            confidence=confidence,
            text=truncated,
        )

        try:
            response = self.model.generate_content(prompt)
            return response.text

        except Exception as e:
            error_msg = str(e)
            if "API_KEY" in error_msg or "authentication" in error_msg.lower():
                logger.error("Invalid Gemini API key")
                return "Explanation unavailable: invalid API key"
            elif "quota" in error_msg.lower():
                logger.warning("Gemini free tier quota exceeded")
                return "Daily free quota reached. Try again tomorrow."
            else:
                logger.error(f"Gemini API error: {e}")
                return f"Explanation unavailable: {error_msg}"


# QUICK TEST

if __name__ == "__main__":
    explainer = BiasExplainer()
    sample = (
        "The radical left's reckless spending spree is destroying the economy. "
        "Hardworking Indians are being crushed by skyrocketing inflation caused "
        "by the Biden administration's failed policies."
    )
    explanation = explainer.explain(sample, label="Right", confidence=0.91)
    print(explanation)
