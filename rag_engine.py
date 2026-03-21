import json
from typing import List, Dict, Optional
from groq import Groq
from question_bank import QuestionBank

MODEL = "llama-3.3-70b-versatile"


class RAGEngine:
    def __init__(self):
        self._groq:  Optional[Groq]         = None
        self._bank:  Optional[QuestionBank] = None

    def set_config(self, groq_api_key: str, endee_url: str):
        self._groq = Groq(api_key=groq_api_key)
        self._bank = QuestionBank()
        self._bank.set_config(endee_url)

    def get_questions(self, role: str, resume_data: dict, n: int = 5) -> List[str]:
        skills  = ", ".join(resume_data.get("skills",  [])[:8])
        domains = ", ".join(resume_data.get("domains", [])[:4])

        search_query = (
            f"{role} interview questions. "
            f"Skills: {skills}. "
            f"Domains: {domains}."
        )

        retrieved    = self._bank.search(search_query, top_k=20)
        behavioural  = self._bank.search("behavioural interview tell me about yourself", top_k=5)
        all_candidates = retrieved + behavioural

        seen, unique = set(), []
        for q in all_candidates:
            if q["text"] not in seen:
                seen.add(q["text"])
                unique.append(q)

        candidates_text = "\n".join(
            f"{i+1}. [{q['role']} / {q['category']}] {q['text']}"
            for i, q in enumerate(unique[:25])
        )

        prompt = f"""You are an expert technical interviewer.
Candidate profile:
- Target Role: {role}
- Skills: {skills}
- Domains: {domains}
- Experience: {resume_data.get('experience_years', 0)} years
- Education: {resume_data.get('education', 'N/A')}

From the candidate list below, select exactly {n} questions that are MOST relevant
to this candidate's profile and target role. Mix technical and behavioural questions.
Return ONLY a JSON array of the selected question strings, nothing else.

Candidate questions:
{candidates_text}

Return format: ["question 1", "question 2", ...]"""

        resp = self._groq.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=800,
            temperature=0.3,
        )
        raw = resp.choices[0].message.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()

        try:
            questions = json.loads(raw)
            return questions[:n]
        except Exception:
            return [q["text"] for q in unique[:n]]

    def evaluate(self, question: str, answer: str, resume_data: dict) -> dict:
        prompt = f"""You are a senior interviewer evaluating a candidate's answer.

Candidate Profile:
- Experience: {resume_data.get('experience_years', 0)} years
- Skills: {', '.join(resume_data.get('skills', [])[:6])}

Question: {question}

Candidate's Answer: {answer}

Evaluate and return ONLY a valid JSON object:
{{
  "score": <integer 1-10>,
  "feedback": "2-3 sentences of constructive positive feedback on what they did well",
  "improvement": "2-3 sentences on specific ways to improve this answer",
  "keywords_missing": ["keyword1", "keyword2"]
}}

Scoring guide:
9-10: Exceptional, shows deep expertise
7-8:  Good answer with minor gaps
5-6:  Adequate but lacks depth or specifics
3-4:  Basic, misses key concepts
1-2:  Off-topic or very weak

Return ONLY the JSON. No extra text."""

        resp = self._groq.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400,
            temperature=0.2,
        )
        raw = resp.choices[0].message.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()

        try:
            result = json.loads(raw)
            result["score"] = max(1, min(10, int(result.get("score", 5))))
            return result
        except Exception:
            return {
                "score": 5,
                "feedback": "Your answer was received.",
                "improvement": "Try to be more specific and use concrete examples.",
                "keywords_missing": [],
            }

    def overall_recommendation(
        self, avg_score: float, resume_data: dict, improvements: str
    ) -> str:
        prompt = f"""Based on a mock interview session, provide a 3-4 sentence personalised
career advice paragraph for this candidate.

Average score: {avg_score}/10
Skills: {', '.join(resume_data.get('skills', [])[:6])}
Experience: {resume_data.get('experience_years', 0)} years
Common improvement areas from the session: {improvements[:500]}

Give actionable, encouraging advice. Be specific. No bullet points, just a paragraph."""

        resp = self._groq.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=250,
            temperature=0.5,
        )
        return resp.choices[0].message.content.strip()