
import time
import logging

import config

logger = logging.getLogger(__name__)

ANSWER_PROMPT = """You are a research analyst translating AI research papers into business insights.

Answer the question using ONLY the numbered excerpts from research papers below.

Rules:
- Write for a business audience: plain language, short paragraphs, explain technical terms in passing.
- Cite every claim with the excerpt it came from, like [1] or [2][3].
- If the excerpts do not answer the question, say so plainly. Do not use outside knowledge or invent findings.
- End with a one-line "Bottom line:" takeaway a busy executive could quote.

Excerpts:
{context}

Question: {question}"""

_client = None


def _gemini():
    global _client
    if _client is None:
        from google import genai
        _client = genai.Client(api_key=config.GEMINI_API_KEY)
    return _client


def format_source(i: int, hit: dict) -> str:
    start = hit.get("page_start", "?")
    end = hit.get("page_end", start)
    pages = f"p.{start}"
    if end != start:
        pages += f"-{end}"

    return (
        f"[{i}] {hit.get('title', 'Untitled')} - "
        f"{hit.get('section_path', '')} ({pages})"
    )


def build_context(hits: list[dict]) -> str:
    return "\n\n".join(
        f"{format_source(i, h)}\n{h.get('text', '')}"
        for i, h in enumerate(hits, 1)
    )


def generate_answer(question: str, hits: list[dict], retries: int = 3) -> str:
    if not hits:
        return "No matching chunks were found in the corpus for this question."

    if not config.GEMINI_API_KEY:
        return "Gemini configuration error: GEMINI_API_KEY is missing."

    from google.genai import errors, types

    gen_config = types.GenerateContentConfig(temperature=0.2)
    prompt = ANSWER_PROMPT.format(
        context=build_context(hits),
        question=question,
    )

    for attempt in range(retries):
        try:
            resp = _gemini().models.generate_content(
                model=config.GEMINI_MODEL,
                contents=prompt,
                config=gen_config,
            )

            answer = (resp.text or "").strip()

            if answer:
                return answer

            logger.warning(
                "Gemini returned no text. Model: %s; response: %s",
                config.GEMINI_MODEL,
                resp,
            )
            return "Gemini returned no text. Check Streamlit logs for response details."

        except errors.ServerError as e:
            logger.exception("Gemini server error")
            if attempt == retries - 1:
                return f"Gemini server error: {e}"
            time.sleep(2 ** attempt)

        except Exception as e:
            logger.exception("Gemini answer generation failed")
            return f"Gemini API error ({type(e).__name__}): {e}"

    return "Gemini failed after all retry attempts."
