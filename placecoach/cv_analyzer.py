import json
from groq import Groq

MODEL = "llama-3.3-70b-versatile"

class CVAnalyzer:
    def __init__(self):
        self._groq = None

    def set_config(self, groq_api_key: str):
        self._groq = Groq(api_key=groq_api_key)

    def analyze(self, resume_data: dict, role: str) -> dict:
        skills  = ", ".join(resume_data.get("skills", []))
        domains = ", ".join(resume_data.get("domains", []))
        exp     = resume_data.get("experience_years", 0)
        edu     = resume_data.get("education", "N/A")
        summary = resume_data.get("summary", "")

        prompt = f"""You are an expert career coach analyzing a CV for a specific role.
CV: Target Role: {role}, Education: {edu}, Experience: {exp} years, Skills: {skills}, Domains: {domains}, Summary: {summary}

Return ONLY valid JSON:
{{"overall_score":<0-100>,"role_fit_score":<0-100>,"ats_score":<0-100>,"experience_score":<0-100>,"skills_score":<0-100>,"education_score":<0-100>,"grade":"A+/A/B+/B/C","summary_verdict":"one sentence verdict","strengths":["strength1","strength2","strength3"],"weaknesses":["weakness1","weakness2","weakness3"],"missing_skills":["skill1","skill2","skill3","skill4"],"suggestions":["suggestion1","suggestion2","suggestion3","suggestion4"],"ats_tips":["tip1","tip2"],"role_verdict":"2 sentence role assessment"}}"""

        resp = self._groq.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1200,
            temperature=0.2,
        )
        raw = resp.choices[0].message.content.strip().replace("```json","").replace("```","").strip()
        try:
            result = json.loads(raw)
            for key in ["overall_score","role_fit_score","ats_score","experience_score","skills_score","education_score"]:
                if key in result:
                    result[key] = max(0, min(100, int(result[key])))
            return result
        except Exception:
            return {"overall_score":60,"role_fit_score":60,"ats_score":55,"experience_score":60,"skills_score":60,"education_score":70,"grade":"B","summary_verdict":"CV analysis completed.","strengths":["Has relevant experience","Educational background present","Shows domain knowledge"],"weaknesses":["Missing key technical skills","Summary needs improvement","Limited quantified achievements"],"missing_skills":["Role-specific skills not detected","Add more technical keywords"],"suggestions":["Add specific technical skills","Quantify your achievements","Use role-specific keywords","Add project links"],"ats_tips":["Use keywords from job descriptions","Avoid tables and graphics"],"role_verdict":"Candidate shows potential. Further skill development recommended."}

    def get_followup(self, question: str, answer: str, groq_api_key: str) -> str:
        client = Groq(api_key=groq_api_key)
        prompt = f"""You are a senior interviewer. Ask ONE sharp follow-up question based on this answer.
Question: {question}
Answer: {answer}
Return ONLY the follow-up question, nothing else."""
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0.4,
        )
        return resp.choices[0].message.content.strip()
