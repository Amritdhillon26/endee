import io
import json
from groq import Groq

try:
    import pdfplumber
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False


class ResumeParser:
    """Extracts structured information from a resume PDF using Groq LLM."""

    def parse(self, file, groq_api_key: str) -> dict:
        text = self._extract_text(file)
        return self._analyze(text, groq_api_key)

    def _extract_text(self, file) -> str:
        raw = file.read()
        if not PDF_SUPPORT:
            raise RuntimeError("Run: pip install pdfplumber")
        pages = []
        with pdfplumber.open(io.BytesIO(raw)) as pdf:
            for page in pdf.pages:
                pages.append(page.extract_text() or "")
        return "\n".join(pages)

    def _analyze(self, text: str, groq_api_key: str) -> dict:
        client = Groq(api_key=groq_api_key)
        prompt = f"""Analyze this resume and return ONLY a valid JSON object with these fields:
{{
  "name": "candidate full name",
  "education": "highest degree and field",
  "experience_years": <integer, 0 if fresher>,
  "skills": ["skill1", "skill2", ...],
  "domains": ["domain1", ...],
  "summary": "2 sentence professional summary"
}}

Resume text:
{text[:4000]}

Return ONLY the JSON. No explanation, no markdown, no backticks."""

        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=600,
            temperature=0.1,
        )
        raw = resp.choices[0].message.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        try:
            return json.loads(raw)
        except Exception:
            return {
                "name": "Candidate",
                "education": "Not detected",
                "experience_years": 0,
                "skills": [],
                "domains": [],
                "summary": text[:200],
            }