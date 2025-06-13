import os
from crewai import Agent, Task, Crew
from crewai_tools import Tool
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from typing import List, Dict, Any
from dotenv import load_dotenv

# --- Load environment variables ---
load_dotenv()

# --- LLM Setup ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    # This check should ideally catch if the key is missing from .env
    print("ERROR: GROQ_API_KEY environment variable not set.")
    raise ValueError(
        "GROQ_API_KEY environment variable not set. "
        "Please ensure it's in your .env file (e.g., GROQ_API_KEY='your_key_here') "
        "and that the .env file is in the root of your project."
    )

# Sanity check: print a masked version of the key to confirm it's loaded
print(f"GROQ_API_KEY loaded: {'*****' + GROQ_API_KEY[-4:] if GROQ_API_KEY else 'None (ERROR - API Key missing)'}")

groq_llm = None # Initialize to None to be explicit

try:
    groq_llm = ChatGroq(
        temperature=0.7,
        model_name="groq/llama3-8b-8192", # This was the previous fix for litellm
        api_key=GROQ_API_KEY
    )
    print("ChatGroq LLM initialized successfully.")

    # CRITICAL NEW CHECK: Verify the LLM object is not None immediately after initialization
    if groq_llm is None:
        raise RuntimeError("ChatGroq LLM object is None after successful initialization block. This is unexpected.")
    
    # Optional: Further checks if LLM object is actually callable or has expected attributes
    # For example, hasattr(groq_llm, 'invoke') would check if it has a basic method
    # The 'function_calling_llm' attribute error is typically internal to CrewAI's use
    # of the LLM, so ensuring 'groq_llm' itself is not None is the primary goal.

except Exception as e:
    print(f"ERROR: Failed to initialize ChatGroq LLM: {e}")
    # Re-raise the exception to prevent the application from starting
    # with a non-functional LLM.
    raise RuntimeError(f"Failed to initialize Groq LLM: {e}. Check your API key, model name, and network connectivity.")


# --- Pydantic Models for Structured Output ---
class CareerGuidanceDetails(BaseModel):
    career_path_suggestion: str = Field(description="A personalized suggestion for a career path.")
    relevant_skills_gap: str = Field(description="Specific skills the user needs to acquire or improve to achieve their career goal.")
    actionable_steps: str = Field(description="Detailed and practical steps to bridge identified skills gaps, including courses, projects, and networking.")
    potential_job_titles: List[str] = Field(description="A list of 5-10 job titles that align with the user's profile and suggested career path.")

class JobMatch(BaseModel):
    title: str = Field(description="Job title.")
    company: str = Field(description="Company offering the job.")
    location: str = Field(description="Job location.")
    skills_required: List[str] = Field(description="List of skills required for the job.")
    description: str = Field(description="Brief description of the job.")

class FinalCrewOutput(BaseModel):
    guidance: CareerGuidanceDetails = Field(description="Detailed career guidance for the user based on their profile and goals.")
    matched_jobs: List[JobMatch] = Field(description="List of highly relevant job matches from the filtered job opportunities.")


# --- Custom Tools (Mock Implementations for demonstration) ---
class ResumeProcessingTool:
    name: str = "Resume Processing Tool"
    description: str = "Processes a PDF resume to extract key information like summary and skills."
    
    def _run(self, pdf_path: str) -> Dict[str, Any]:
        print(f"DEBUG: Mocking ResumeProcessingTool for {pdf_path}. (In a real app, this parses the PDF)")
        import time
        time.sleep(1)
        return {
            "status": "success",
            "skills": ["Python", "SQL", "Data Analysis", "Cloud Computing", "Machine Learning Fundamentals", "Project Management"],
            "resume_summary": "Experienced professional with a strong background in data analysis, cloud computing, and a passion for machine learning. Proven ability to lead projects and deliver insights from complex datasets."
        }

class JobFilteringTool:
    name: str = "Job Filtering Tool"
    description: str = "Filters a list of predefined jobs based on user skills to find relevant opportunities."
    
    def _run(self, user_skills: List[str]) -> List[Dict[str, Any]]:
        print(f"DEBUG: Mocking JobFilteringTool with user skills: {user_skills}. (In a real app, this fetches and filters jobs)")
        import time
        time.sleep(1.5)

        all_mock_jobs = [
            {"title": "Data Scientist", "company": "Tech Innovations", "location": "Remote", "skills_required": ["Python", "Machine Learning", "SQL", "Deep Learning", "TensorFlow"], "description": "Develop and deploy machine learning models to solve complex business problems."},
            {"title": "Software Engineer (Backend)", "company": "Global Solutions", "location": "Hyderabad", "skills_required": ["Python", "Java", "APIs", "Microservices", "AWS"], "description": "Design and implement scalable backend services for large-scale applications."},
            {"title": "Cloud Architect", "company": "Cloud Builders", "location": "Bangalore", "skills_required": ["AWS", "Azure", "Cloud Security", "Terraform", "Solution Design"], "description": "Design and implement secure and scalable cloud infrastructure."},
            {"title": "DevOps Engineer", "company": "CI/CD Masters", "location": "Pune", "skills_required": ["Linux", "Docker", "Kubernetes", "CI/CD", "Ansible", "Jenkins"], "description": "Automate deployment pipelines and manage infrastructure as code."},
            {"title": "Business Analyst", "company": "Consulting Group", "location": "Mumbai", "skills_required": ["SQL", "Data Modeling", "Business Process Mapping", "Stakeholder Management"], "description": "Analyze business needs and propose solutions."},
            {"title": "Machine Learning Engineer", "company": "AI Driven Inc.", "location": "Seattle", "skills_required": ["Python", "TensorFlow", "PyTorch", "MLOps", "Model Deployment"], "description": "Build, optimize, and deploy machine learning models into production environments."},
            {"title": "Data Analyst", "company": "Insightful Analytics", "location": "Chennai", "skills_required": ["SQL", "Excel", "Tableau", "Data Visualization", "Statistical Analysis"], "description": "Extract, clean, and analyze data to provide actionable business insights."},
            {"title": "Product Manager (AI/ML)", "company": "Future Tech", "location": "San Francisco", "skills_required": ["Product Management", "AI/ML Concepts", "Market Research", "Roadmapping"], "description": "Define and launch AI/ML products that meet market needs and business goals."},
            {"title": "Operations Research Analyst", "company": "Supply Chain Solutions", "location": "Atlanta", "skills_required": ["Python", "Optimization", "Statistics", "Simulation", "Decision Science"], "description": "Apply mathematical modeling and optimization techniques to improve operational efficiency."},
            {"title": "Quantitative Analyst", "company": "Fintech Innovations", "location": "New York", "skills_required": ["Python", "R", "Statistics", "Financial Modeling", "Time Series Analysis"], "description": "Develop quantitative models for financial markets and risk management."}
        ]
        
        filtered_jobs = []
        user_skills_lower = {s.lower() for s in user_skills}
        for job in all_mock_jobs:
            job_skills_lower = {s.lower() for s in job["skills_required"]}
            if any(skill in job_skills_lower for skill in user_skills_lower):
                filtered_jobs.append(job)
        
        if len(filtered_jobs) > 7:
            filtered_jobs = filtered_jobs[:7]

        return filtered_jobs


# --- Define Agents ---
# The career_assistant_agent will use the groq_llm for its reasoning and Pydantic output.
# IMPORTANT: 'groq_llm' MUST be initialized and not None here
career_assistant_agent = Agent(
    role='Personalized Career Advisor',
    goal='Provide tailored career path suggestions, identify skill gaps, offer actionable steps, and list potential job titles based on user resume and career goals.',
    backstory="""You are an expert career consultant with a deep understanding of industry trends, job market demands, and skill development strategies. 
    You are adept at analyzing individual profiles and providing highly personalized and actionable guidance, guiding individuals towards successful careers.""",
    verbose=True,
    allow_delegation=False,
    llm=groq_llm # This is where the 'NoneType' error occurs if groq_llm is not set
)

# --- Define Tasks ---
career_guidance_task = Task(
    description=(
        "Given the `context_for_guidance` (which is a JSON string including resume summary, user skills, user query, and a list of filtered job matches), "
        "generate comprehensive career guidance. This guidance should include:\n"
        "1. **Career Path Suggestion:** A personalized suggestion for a career path that aligns with the user's resume and their stated career goal (`user_query`).\n"
        "2. **Relevant Skills Gap:** Identify specific skills the user needs to acquire or improve to achieve their career goal and match the provided job opportunities. Be precise.\n"
        "3. **Actionable Steps:** Detailed and practical steps to bridge the identified skills gaps. This can include specific online courses, certifications, "
        "personal projects, networking strategies, or professional development activities.\n"
        "4. **Potential Job Titles:** A list of 5-10 job titles that align with the user's current profile, the suggested career path, and the matched job opportunities.\n"
        "The final output MUST be a Pydantic object of type `FinalCrewOutput`. The `guidance` field of this object must contain the career path suggestion, skills "
        "gap, actionable steps, and potential job titles. The `matched_jobs` field of this object MUST be directly populated from the `filtered_jobs_list` provided "
        "in the input `context_for_guidance`. Do NOT re-generate or modify the `matched_jobs` list; simply embed it. "
        "Ensure there are no preambles, explanations, or extraneous text outside the Pydantic object."
    ),
    expected_output="A comprehensive FinalCrewOutput Pydantic object containing detailed career guidance and a list of highly relevant job matches based on the user's resume and goals.",
    agent=career_assistant_agent,
    output_pydantic=FinalCrewOutput
)