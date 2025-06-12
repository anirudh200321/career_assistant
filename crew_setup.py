import os
import logging
from crewai import Agent, Task, Crew
from crewai_tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type
import PyPDF2
from dotenv import load_dotenv
import json # ADD THIS: For parsing LLM responses within the tool
from litellm import completion # ADD THIS: For making direct LLM calls in the tool

load_dotenv()

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
os.environ["CREW_LOGGING_LEVEL"] = "DEBUG"

class ResumeProcessingToolInput(BaseModel):
    """Input for ResumeProcessingTool."""
    pdf_path: str = Field(description="The file path to the PDF resume.")

class ResumeProcessingTool(BaseTool):
    name: str = "resume_processing_tool"
    description: str = "Processes a PDF resume to extract text and identify skills."
    args_schema: Type[BaseModel] = ResumeProcessingToolInput

    def _run(self, pdf_path: str) -> dict:
        extracted_text = ""
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page_num in range(len(reader.pages)):
                    extracted_text += reader.pages[page_num].extract_text() or ""
            if not extracted_text.strip():
                raise ValueError("No readable text found in PDF.")
        except Exception as e:
            logging.error(f"Error reading or processing PDF: {e}")
            return {"error": f"Failed to read or process PDF: {e}", "extracted_text": "", "skills": []}

        # --- NEW: Use LLM for enhanced skill extraction and resume summarization ---
        logging.info("Using LLM for enhanced skill extraction and resume summarization...")
        try:
            llm_prompt = f"""
            Analyze the following resume text.
            Extract all unique skills (programming languages, frameworks, tools, soft skills, domain knowledge) present in the resume.
            Also, provide a concise summary of the resume's objective, education, and relevant experience.

            Format your response as a JSON object with two keys:
            "skills": a list of strings representing the extracted skills.
            "summary": a string containing the concise resume summary.

            Resume Text:
            ---
            {extracted_text}
            ---
            """
            
            # Make a direct LLM call using litellm
            response = completion(
                model=os.environ["OPENAI_MODEL_NAME"], # Uses Groq model specified in env
                messages=[
                    {"role": "system", "content": "You are an expert resume analyst, highly skilled in extracting detailed information."},
                    {"role": "user", "content": llm_prompt}
                ],
                response_format={"type": "json_object"}, # Ensure JSON response
                temperature=0.2 # Keep temperature low for factual extraction
            )
            
            # Extract content from the LLM response
            llm_content = response.choices[0].message.content
            parsed_llm_output = json.loads(llm_content)
            
            found_skills = parsed_llm_output.get("skills", [])
            resume_summary = parsed_llm_output.get("summary", "No summary provided.")

            logging.info(f"LLM extracted {len(found_skills)} skills and a summary.")
            return {
                "status": "success",
                "extracted_text": extracted_text, # Keep original text for context if needed
                "skills": found_skills,
                "resume_summary": resume_summary # New: Add resume summary
            }

        except Exception as llm_e:
            logging.error(f"Error during LLM skill extraction: {llm_e}")
            # Fallback to basic keyword matching if LLM extraction fails
            common_skills = [
                "Python", "Java", "C++", "JavaScript", "React", "Angular", "Vue.js",
                "SQL", "NoSQL", "MongoDB", "PostgreSQL", "MySQL", "Docker", "Kubernetes",
                "AWS", "Azure", "GCP", "Machine Learning", "Deep Learning", "Data Analysis",
                "Pandas", "NumPy", "Scikit-learn", "TensorFlow", "PyTorch", "Git", "API",
                "FastAPI", "Streamlit", "LangChain", "CrewAI", "Communication", "Teamwork",
                "Project Management", "Data Science", "Web Development", "Cloud Computing"
            ]
            found_skills = []
            text_lower = extracted_text.lower()
            for skill in common_skills:
                if skill.lower() in text_lower:
                    found_skills.append(skill)
            found_skills = sorted(list(set(found_skills)))
            logging.warning("Falling back to basic keyword skill extraction.")
            return {
                "status": "partial_success",
                "extracted_text": extracted_text,
                "skills": found_skills,
                "resume_summary": "LLM extraction failed, basic skills extracted."
            }


# Initialize the custom tool
resume_processing_tool = ResumeProcessingTool()

os.environ["OPENAI_API_BASE"] = "https://api.groq.com/openai/v1"
os.environ["OPENAI_API_KEY"] = os.getenv("GROQ_API_KEY", "")
os.environ["OPENAI_MODEL_NAME"] = "llama3-8b-8192"

resume_agent = Agent(
    role='Resume Processor',
    goal='Extract key information, especially skills and a summary, from a PDF resume and prepare it for further analysis.',
    backstory=(
        "An expert in document parsing and information extraction, specialized in converting "
        "complex resume formats into structured data. Highly efficient and accurate in "
        "identifying relevant skills and experiences. Utilizes advanced LLM capabilities for deeper insights.\n\n"
        "Your primary goal is to use the 'resume_processing_tool' to extract comprehensive text, skills, and a summary "
        "from the provided PDF path and then return the output in a structured JSON format."
    ),
    verbose=True,
    allow_delegation=False,
    tools=[resume_processing_tool]
)

research_agent = Agent(
    role='Job Researcher',
    goal='Find and shortlist relevant job openings based on provided skills from a resume.',
    backstory=(
        "A meticulous job market analyst who tirelessly searches various job boards and "
        "platforms to identify the best opportunities matching a candidate's profile. "
        "**Note: Since I do not have direct access to live job boards, I will intelligently simulate "
        "finding relevant jobs based on the provided skills and industry trends.**" # Added simulation note
    ),
    verbose=True,
    allow_delegation=True
)

career_assistant_agent = Agent(
    role='Career Advisor',
    goal='Provide personalized career guidance and answer specific questions about job matches.',
    backstory=(
        "A compassionate and knowledgeable career counselor, dedicated to helping individuals "
        "navigate their professional journey and make informed decisions. "
        "Your final response MUST be a direct, professional, and conversational advice message. "
        "It MUST NOT contain any internal 'Thought:', 'Action:', 'Action Input:', or 'Observation:' steps. "
        "Start directly with the advice."
    ),
    verbose=True,
    allow_delegation=True
)

# --- Define Tasks ---

# MODIFIED: Expected output now includes 'resume_summary'
resume_processing_task = Task(
    description=(
        "Process the PDF resume located at '{pdf_path}'. "
        "Extract all readable text and then identify a comprehensive list of skills present in the resume. "
        "Also, provide a concise summary of the resume's objective, education, and relevant experience. "
        "The output should be a JSON object containing the 'extracted_text', a 'skills' array, and a 'resume_summary'."
    ),
    expected_output="A JSON object with 'extracted_text' (string), 'skills' (list of strings), and 'resume_summary' (string).",
    agent=resume_agent,
)

# MODIFIED: Task description to emphasize simulation and use richer input
job_search_task = Task(
    description=(
        "Given the JSON output from the resume analysis: {processed_resume_output}, "
        "which contains 'extracted_text', a 'skills' array, and 'resume_summary'. "
        "First, parse this JSON string to extract the 'skills' list and the 'resume_summary'. "
        "Then, **simulate searching** for relevant job openings that directly align with these identified skills and the resume summary. "
        "Provide a concise summary of the top 5 *plausible* job titles, their companies, and a generic placeholder link. "
        "Format the output as a clear, readable markdown list. Ensure the job titles and companies sound realistic."
    ),
    expected_output="A markdown list of top 5 plausible job openings (Title, Company, Link), derived from the provided skills and resume summary.",
    agent=research_agent,
)

career_guidance_task = Task(
    description=(
        "Based on the identified job openings (markdown list): {job_matches}, "
        "and the user's initial query: '{user_query}', "
        "provide personalized career advice and suggest next steps for the user. "
        "Answer the user's specific query related to the jobs or their career path. "
        "Your final response MUST be a direct, friendly, and professional conversational advice message. "
        "It MUST NOT contain any internal 'Thought:', 'Action:', 'Action Input:', or 'Observation:' steps. "
        "Start directly with the advice. Respond as if you are the final output of the entire system."
    ),
    expected_output="A friendly and professional conversational career advice message, formatted as a readable string or markdown, directly addressing the user's query and insights from job matches.",
    agent=career_assistant_agent,
)

# --- Define the Crew creation function ---
def create_career_crew():
    return Crew(
        agents=[resume_agent, research_agent, career_assistant_agent],
        tasks=[resume_processing_task, job_search_task, career_guidance_task],
        verbose=True
    )

if __name__ == "__main__":
    dummy_pdf_path = "dummy_resume.pdf"
    if not os.path.exists(dummy_pdf_path):
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
        c = canvas.Canvas(dummy_pdf_path, pagesize=letter)
        c.drawString(100, 750, "Resume of John Doe")
        c.drawString(100, 730, "Skills: Python, Data Analysis, SQL, Machine Learning, Communication")
        c.drawString(100, 710, "Experience: Software Engineer at Tech Solutions")
        c.drawString(100, 690, "Education: Master of Science in Computer Science")
        c.save()
        logging.info(f"Created dummy PDF: {dummy_pdf_path}")

    logging.info("Creating the career crew for direct test...")
    crew = create_career_crew()

    logging.info("## Running the Career Guidance Crew with a dummy resume...")
    crew_result = crew.kickoff(
        inputs={
            'pdf_path': dummy_pdf_path,
            'user_query': 'What are the best job titles for my skills and how can I improve my resume?'
        }
    )
    logging.info("########################")
    logging.info("## Crew Execution Finished")
    logging.info("########################\n")
    logging.info(crew_result)