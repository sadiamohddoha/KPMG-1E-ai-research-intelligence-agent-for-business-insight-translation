
import logging

import config

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

logger = logging.getLogger(__name__)


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


def generate_answer(
    question: str,
    hits: list[dict],
    retries: int = 1,
) -> str:

    if not hits:
        return "No matching research sources were found."

    if not config.GEMINI_API_KEY:
        return "Gemini API key is missing."

    from google import genai
    from google.genai import errors, types

    try:
        client = genai.Client(
            api_key=config.GEMINI_API_KEY,
            http_options=types.HttpOptions(
                timeout=30000,
                retry_options=types.HttpRetryOptions(
                    attempts=1
                ),
            ),
        )

        prompt = ANSWER_PROMPT.format(
            context=build_context(hits),
            question=question,
        )

        response = client.models.generate_content(
            model=config.GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.2,
                max_output_tokens=1024,
            ),
        )

        answer = (response.text or "").strip()

        if answer:
            return answer

        return "Gemini returned an empty response. Please try again."

    except errors.ServerError as e:
        logger.warning("Gemini server error: %s", e)
        return (
            "Gemini is temporarily overloaded (503). "
            "Please try again in a few minutes."
        )

    except Exception as e:
        logger.exception("Gemini generation failed")
        return (
            f"Gemini request failed ({type(e).__name__}). "
            "Check Streamlit logs for details."
        )
