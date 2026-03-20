from sentence_transformers import SentenceTransformer
from endee import Endee, Precision

EMBED_MODEL = "all-MiniLM-L6-v2"
EMBED_DIM   = 384
INDEX_NAME  = "interview_questions"

QUESTIONS = [
    # Software Engineer
    {"id":"se_001","text":"Explain the difference between a stack and a queue with real-world examples.","role":"Software Engineer","category":"Data Structures"},
    {"id":"se_002","text":"What is time complexity? Explain Big O notation with examples.","role":"Software Engineer","category":"Algorithms"},
    {"id":"se_003","text":"Describe the SOLID principles and give an example of each.","role":"Software Engineer","category":"OOP"},
    {"id":"se_004","text":"What is the difference between SQL and NoSQL databases? When would you use each?","role":"Software Engineer","category":"Databases"},
    {"id":"se_005","text":"Explain RESTful API design principles. What makes an API RESTful?","role":"Software Engineer","category":"APIs"},
    {"id":"se_006","text":"What is Git branching strategy? Explain GitFlow.","role":"Software Engineer","category":"Version Control"},
    {"id":"se_007","text":"How do you approach debugging a production issue?","role":"Software Engineer","category":"Problem Solving"},
    {"id":"se_008","text":"Explain the concept of microservices vs monolithic architecture.","role":"Software Engineer","category":"Architecture"},
    {"id":"se_009","text":"What is CI/CD? How have you implemented it in past projects?","role":"Software Engineer","category":"DevOps"},
    {"id":"se_010","text":"Describe a challenging technical problem you solved and your approach.","role":"Software Engineer","category":"Behavioral"},
    {"id":"se_011","text":"What is memory management and how does garbage collection work?","role":"Software Engineer","category":"Systems"},
    {"id":"se_012","text":"Explain multithreading and concurrency. What are race conditions?","role":"Software Engineer","category":"Systems"},
    # Frontend Developer
    {"id":"fe_001","text":"Explain the virtual DOM in React and why it improves performance.","role":"Frontend Developer","category":"React"},
    {"id":"fe_002","text":"What is CSS specificity and how does the cascade work?","role":"Frontend Developer","category":"CSS"},
    {"id":"fe_003","text":"Explain event bubbling and event delegation in JavaScript.","role":"Frontend Developer","category":"JavaScript"},
    {"id":"fe_004","text":"What are React hooks? Explain useState and useEffect with examples.","role":"Frontend Developer","category":"React"},
    {"id":"fe_005","text":"How do you optimize web performance? List at least 5 techniques.","role":"Frontend Developer","category":"Performance"},
    {"id":"fe_006","text":"What is responsive design? Explain CSS Flexbox vs Grid.","role":"Frontend Developer","category":"CSS"},
    {"id":"fe_007","text":"Explain the browser's critical rendering path.","role":"Frontend Developer","category":"Browser"},
    {"id":"fe_008","text":"What is CORS and how do you handle it?","role":"Frontend Developer","category":"APIs"},
    # Backend Developer
    {"id":"be_001","text":"Explain database indexing and when you would use it.","role":"Backend Developer","category":"Databases"},
    {"id":"be_002","text":"What is ORM? Explain its advantages and disadvantages.","role":"Backend Developer","category":"Databases"},
    {"id":"be_003","text":"Describe different authentication strategies: JWT, OAuth, sessions.","role":"Backend Developer","category":"Security"},
    {"id":"be_004","text":"What is caching? Explain Redis and when to use it.","role":"Backend Developer","category":"Performance"},
    {"id":"be_005","text":"How do you design a rate-limiting system for an API?","role":"Backend Developer","category":"System Design"},
    {"id":"be_006","text":"Explain ACID properties in databases with examples.","role":"Backend Developer","category":"Databases"},
    {"id":"be_007","text":"What is the N+1 query problem and how do you solve it?","role":"Backend Developer","category":"Databases"},
    {"id":"be_008","text":"How would you design a URL shortener like bit.ly?","role":"Backend Developer","category":"System Design"},
    # Data Scientist
    {"id":"ds_001","text":"Explain the bias-variance tradeoff with examples.","role":"Data Scientist","category":"ML Theory"},
    {"id":"ds_002","text":"What is overfitting? How do you prevent it?","role":"Data Scientist","category":"ML Theory"},
    {"id":"ds_003","text":"Explain the difference between supervised, unsupervised, and reinforcement learning.","role":"Data Scientist","category":"ML Theory"},
    {"id":"ds_004","text":"What is cross-validation and why is it important?","role":"Data Scientist","category":"Model Evaluation"},
    {"id":"ds_005","text":"Explain precision, recall, F1 score and when to use each.","role":"Data Scientist","category":"Model Evaluation"},
    {"id":"ds_006","text":"How do you handle missing data in a dataset?","role":"Data Scientist","category":"Data Preprocessing"},
    {"id":"ds_007","text":"Explain feature engineering and feature selection techniques.","role":"Data Scientist","category":"Feature Engineering"},
    {"id":"ds_008","text":"What is regularization? Explain L1 vs L2 regularization.","role":"Data Scientist","category":"ML Theory"},
    {"id":"ds_009","text":"Explain how a Random Forest works and its advantages.","role":"Data Scientist","category":"Algorithms"},
    {"id":"ds_010","text":"What is gradient descent? Explain variants: SGD, Adam, RMSProp.","role":"Data Scientist","category":"Optimization"},
    {"id":"ds_011","text":"Explain PCA and when you would use dimensionality reduction.","role":"Data Scientist","category":"Unsupervised Learning"},
    {"id":"ds_012","text":"How do you deal with imbalanced datasets?","role":"Data Scientist","category":"Data Preprocessing"},
    {"id":"ds_013","text":"Explain the transformer architecture and attention mechanism.","role":"Data Scientist","category":"Deep Learning"},
    {"id":"ds_014","text":"What is A/B testing and how would you design one?","role":"Data Scientist","category":"Statistics"},
    # ML Engineer
    {"id":"ml_001","text":"How do you deploy a machine learning model to production?","role":"ML Engineer","category":"MLOps"},
    {"id":"ml_002","text":"What is model drift and how do you monitor for it?","role":"ML Engineer","category":"MLOps"},
    {"id":"ml_003","text":"Explain the difference between batch and real-time inference.","role":"ML Engineer","category":"Deployment"},
    {"id":"ml_004","text":"What is MLflow? How do you track experiments?","role":"ML Engineer","category":"Tools"},
    {"id":"ml_005","text":"How would you build a recommendation system from scratch?","role":"ML Engineer","category":"System Design"},
    {"id":"ml_006","text":"Explain vector embeddings and their role in modern AI applications.","role":"ML Engineer","category":"Embeddings"},
    {"id":"ml_007","text":"What is RAG (Retrieval Augmented Generation) and how does it work?","role":"ML Engineer","category":"LLMs"},
    {"id":"ml_008","text":"How do you evaluate a large language model?","role":"ML Engineer","category":"LLMs"},
    # Product Manager
    {"id":"pm_001","text":"How do you prioritize features when you have limited engineering resources?","role":"Product Manager","category":"Prioritization"},
    {"id":"pm_002","text":"Describe a product you admire and how you would improve it.","role":"Product Manager","category":"Product Sense"},
    {"id":"pm_003","text":"How do you define success metrics for a new feature?","role":"Product Manager","category":"Metrics"},
    {"id":"pm_004","text":"Walk me through how you would launch a new product from zero to one.","role":"Product Manager","category":"Strategy"},
    {"id":"pm_005","text":"How do you handle disagreements between engineering and design teams?","role":"Product Manager","category":"Stakeholder Management"},
    {"id":"pm_006","text":"Explain the RICE and MoSCoW prioritization frameworks.","role":"Product Manager","category":"Frameworks"},
    {"id":"pm_007","text":"How do you gather and synthesize customer feedback into product decisions?","role":"Product Manager","category":"User Research"},
    {"id":"pm_008","text":"Describe a time you made a difficult product decision with incomplete data.","role":"Product Manager","category":"Behavioral"},
    {"id":"pm_009","text":"How do you write a Product Requirements Document (PRD)?","role":"Product Manager","category":"Documentation"},
    {"id":"pm_010","text":"What is the difference between product vision, strategy and roadmap?","role":"Product Manager","category":"Strategy"},
    {"id":"pm_011","text":"How would you improve the onboarding experience for a SaaS product?","role":"Product Manager","category":"Product Sense"},
    {"id":"pm_012","text":"Explain North Star Metric and how you would identify it for a product.","role":"Product Manager","category":"Metrics"},
    # Marketing Manager
    {"id":"mk_001","text":"How do you build a go-to-market strategy for a new product?","role":"Marketing Manager","category":"Strategy"},
    {"id":"mk_002","text":"Explain the difference between inbound and outbound marketing.","role":"Marketing Manager","category":"Concepts"},
    {"id":"mk_003","text":"How do you measure the ROI of a marketing campaign?","role":"Marketing Manager","category":"Analytics"},
    {"id":"mk_004","text":"Describe your experience with SEO/SEM. How do you optimize content?","role":"Marketing Manager","category":"Digital Marketing"},
    {"id":"mk_005","text":"How do you build and segment a target audience?","role":"Marketing Manager","category":"Strategy"},
    {"id":"mk_006","text":"Explain the marketing funnel from awareness to conversion.","role":"Marketing Manager","category":"Concepts"},
    {"id":"mk_007","text":"How do you use data analytics to improve marketing performance?","role":"Marketing Manager","category":"Analytics"},
    {"id":"mk_008","text":"Describe a successful campaign you led. What made it work?","role":"Marketing Manager","category":"Behavioral"},
    {"id":"mk_009","text":"How do you manage a brand during a PR crisis?","role":"Marketing Manager","category":"Brand Management"},
    {"id":"mk_010","text":"What metrics do you track for email marketing campaigns?","role":"Marketing Manager","category":"Analytics"},
    # Business Analyst
    {"id":"ba_001","text":"How do you gather and document business requirements from stakeholders?","role":"Business Analyst","category":"Requirements"},
    {"id":"ba_002","text":"Explain the difference between functional and non-functional requirements.","role":"Business Analyst","category":"Requirements"},
    {"id":"ba_003","text":"How do you perform a gap analysis?","role":"Business Analyst","category":"Analysis"},
    {"id":"ba_004","text":"Describe your experience with process mapping and BPMN diagrams.","role":"Business Analyst","category":"Process"},
    {"id":"ba_005","text":"How do you prioritize conflicting requirements from different stakeholders?","role":"Business Analyst","category":"Stakeholder Management"},
    {"id":"ba_006","text":"Explain how you would use SQL to analyze business data.","role":"Business Analyst","category":"Data"},
    {"id":"ba_007","text":"What is a feasibility study and when do you conduct one?","role":"Business Analyst","category":"Analysis"},
    {"id":"ba_008","text":"How do you create a business case for a new project?","role":"Business Analyst","category":"Documentation"},
    # DevOps Engineer
    {"id":"do_001","text":"Explain Kubernetes architecture and key components.","role":"DevOps Engineer","category":"Containers"},
    {"id":"do_002","text":"What is Infrastructure as Code? Explain Terraform.","role":"DevOps Engineer","category":"IaC"},
    {"id":"do_003","text":"Explain blue-green deployment and canary releases.","role":"DevOps Engineer","category":"Deployment"},
    {"id":"do_004","text":"How do you monitor a production system? What tools do you use?","role":"DevOps Engineer","category":"Monitoring"},
    {"id":"do_005","text":"What is the difference between Docker and a virtual machine?","role":"DevOps Engineer","category":"Containers"},
    {"id":"do_006","text":"How do you handle secrets management in a cloud environment?","role":"DevOps Engineer","category":"Security"},
    # Finance Analyst
    {"id":"fa_001","text":"Explain DCF (Discounted Cash Flow) analysis and when to use it.","role":"Finance Analyst","category":"Valuation"},
    {"id":"fa_002","text":"What is EBITDA and why is it important?","role":"Finance Analyst","category":"Financial Metrics"},
    {"id":"fa_003","text":"How do you build a financial model from scratch?","role":"Finance Analyst","category":"Modeling"},
    {"id":"fa_004","text":"Explain the three financial statements and how they connect.","role":"Finance Analyst","category":"Accounting"},
    {"id":"fa_005","text":"What is working capital and how do you manage it?","role":"Finance Analyst","category":"Financial Metrics"},
    {"id":"fa_006","text":"How do you perform a variance analysis?","role":"Finance Analyst","category":"Analysis"},
    # HR Manager
    {"id":"hr_001","text":"How do you design an effective onboarding program for new employees?","role":"HR Manager","category":"Onboarding"},
    {"id":"hr_002","text":"Explain your approach to performance management and appraisals.","role":"HR Manager","category":"Performance"},
    {"id":"hr_003","text":"How do you handle conflict resolution between employees?","role":"HR Manager","category":"Employee Relations"},
    {"id":"hr_004","text":"What strategies do you use for talent acquisition and retention?","role":"HR Manager","category":"Recruitment"},
    {"id":"hr_005","text":"How do you build and maintain a positive company culture?","role":"HR Manager","category":"Culture"},
    # Sales Executive
    {"id":"sl_001","text":"Walk me through your sales process from prospecting to closing.","role":"Sales Executive","category":"Process"},
    {"id":"sl_002","text":"How do you handle objections from potential clients?","role":"Sales Executive","category":"Objection Handling"},
    {"id":"sl_003","text":"Describe your most successful sale. What made it work?","role":"Sales Executive","category":"Behavioral"},
    {"id":"sl_004","text":"How do you build and maintain long-term client relationships?","role":"Sales Executive","category":"Relationship Management"},
    {"id":"sl_005","text":"What CRM tools have you used and how do you leverage them?","role":"Sales Executive","category":"Tools"},
    # UI/UX Designer
    {"id":"ux_001","text":"Walk me through your design process from research to final delivery.","role":"UI/UX Designer","category":"Process"},
    {"id":"ux_002","text":"How do you conduct user research and usability testing?","role":"UI/UX Designer","category":"Research"},
    {"id":"ux_003","text":"Explain the difference between UX and UI design.","role":"UI/UX Designer","category":"Concepts"},
    {"id":"ux_004","text":"How do you design for accessibility?","role":"UI/UX Designer","category":"Accessibility"},
    {"id":"ux_005","text":"Describe a design decision you made based on user feedback.","role":"UI/UX Designer","category":"Behavioral"},
    # Cybersecurity Analyst
    {"id":"cy_001","text":"Explain the CIA triad in cybersecurity.","role":"Cybersecurity Analyst","category":"Fundamentals"},
    {"id":"cy_002","text":"What is penetration testing and how do you perform it?","role":"Cybersecurity Analyst","category":"Testing"},
    {"id":"cy_003","text":"How do you respond to a security incident or data breach?","role":"Cybersecurity Analyst","category":"Incident Response"},
    {"id":"cy_004","text":"Explain common web vulnerabilities: SQL injection, XSS, CSRF.","role":"Cybersecurity Analyst","category":"Web Security"},
    {"id":"cy_005","text":"What is zero-trust security architecture?","role":"Cybersecurity Analyst","category":"Architecture"},
    # Cloud Architect
    {"id":"ca_001","text":"Compare AWS, Azure and GCP. When would you choose each?","role":"Cloud Architect","category":"Cloud Platforms"},
    {"id":"ca_002","text":"How do you design for high availability and fault tolerance in the cloud?","role":"Cloud Architect","category":"Architecture"},
    {"id":"ca_003","text":"What is serverless architecture? What are its tradeoffs?","role":"Cloud Architect","category":"Architecture"},
    {"id":"ca_004","text":"How do you optimize cloud costs in a large-scale deployment?","role":"Cloud Architect","category":"Cost Optimization"},
    {"id":"ca_005","text":"Explain VPC, subnets, and security groups in AWS.","role":"Cloud Architect","category":"Networking"},
    # General Behavioral
    {"id":"gen_001","text":"Tell me about yourself and your professional journey.","role":"General","category":"Behavioral"},
    {"id":"gen_002","text":"Where do you see yourself in 5 years?","role":"General","category":"Behavioral"},
    {"id":"gen_003","text":"Describe a time you failed and what you learned from it.","role":"General","category":"Behavioral"},
    {"id":"gen_004","text":"How do you handle working under pressure and tight deadlines?","role":"General","category":"Behavioral"},
    {"id":"gen_005","text":"Tell me about a time you worked in a difficult team. How did you handle it?","role":"General","category":"Behavioral"},
    {"id":"gen_006","text":"What are your greatest strengths and areas for improvement?","role":"General","category":"Behavioral"},
    {"id":"gen_007","text":"Why do you want to work at this company?","role":"General","category":"Behavioral"},
    {"id":"gen_008","text":"How do you stay updated with the latest trends in your field?","role":"General","category":"Behavioral"},
    {"id":"gen_009","text":"Describe your ideal work environment and team culture.","role":"General","category":"Behavioral"},
    {"id":"gen_010","text":"What motivates you professionally?","role":"General","category":"Behavioral"},
]


class QuestionBank:
    def __init__(self):
        self._model  = SentenceTransformer(EMBED_MODEL)
        self._client = None
        self._index  = None

    def set_config(self, endee_url: str):
        client = Endee()
        client.set_base_url(f"{endee_url}/api/v1")
        self._client = client
        try:
            existing = [idx.name for idx in client.list_indexes()]
        except Exception:
            existing = []
        if INDEX_NAME not in existing:
            client.create_index(
                name=INDEX_NAME,
                dimension=EMBED_DIM,
                space_type="cosine",
                precision=Precision.INT8,
            )
        self._index = client.get_index(name=INDEX_NAME)

    def index_all(self):
        texts = [q["text"] for q in QUESTIONS]
        vectors = self._model.encode(texts, show_progress_bar=False)
        items = [
            {
                "id": q["id"],
                "vector": vectors[i].tolist(),
                "meta": {
                    "text":     q["text"],
                    "role":     q["role"],
                    "category": q["category"],
                },
            }
            for i, q in enumerate(QUESTIONS)
        ]
        for i in range(0, len(items), 100):
            self._index.upsert(items[i: i + 100])

    def search(self, query: str, top_k: int = 10):
        vec = self._model.encode(query).tolist()
        results = self._index.query(vector=vec, top_k=top_k)
        return [
            {
                "text":     r.meta.get("text", ""),
                "role":     r.meta.get("role", ""),
                "category": r.meta.get("category", ""),
                "score":    round(r.similarity, 4),
            }
            for r in results
        ]