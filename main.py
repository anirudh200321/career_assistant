import os
import shutil
import tempfile
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import logging
import traceback
import json

# Configure Logging for FastAPI and general Python modules
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Import only the agents and tasks needed for the CREWAI part of the chain
from crew_setup import research_agent, career_assistant_agent, \
                       job_search_task, career_guidance_task, \
                       ResumeProcessingTool

from crewai import Crew

# Initialize FastAPI app
app = FastAPI(
    title="AI Career Assistant API",
    description="API for processing resumes and providing career guidance using CrewAI.",
    version="1.0.0"
)

# CORS (Cross-Origin Resource Sharing) middleware settings
origins = [
    "http://localhost",
    "http://localhost:8501", # Default Streamlit port
    "http://127.0.0.1:8501",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define a directory for temporary file storage
TEMP_FILES_DIR = "temp_uploads"
os.makedirs(TEMP_FILES_DIR, exist_ok=True)
logging.info(f"Temporary file directory set to: {TEMP_FILES_DIR}")

@app.get("/")
async def root():
    return {"message": "Welcome to the AI Career Assistant API. Use /process_career_request/ to get started."}


@app.post("/process_career_request/")
async def process_career_request(
    resume_file: UploadFile = File(..., description="The PDF resume file to upload."),
    user_query: Optional[str] = Form("Tell me about job opportunities related to my skills."),
):
    """
    Receives a PDF resume and a user query, processes them through the CrewAI agents,
    and returns career guidance and job matches.
    """
    pdf_path = None
    try:
        # 1. Save the uploaded PDF temporarily
        # Use tempfile to create a secure temporary file.
        # This ensures unique file names and handles cleanup in the finally block.
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf", dir=TEMP_FILES_DIR) as tmp_file:
            shutil.copyfileobj(resume_file.file, tmp_file)
            pdf_path = tmp_file.name
        
        logging.info(f"Received PDF: {resume_file.filename} saved to {pdf_path}")

        # --- Directly execute ResumeProcessingTool ---
        # This bypasses the LLM's tool-calling logic for the first step,
        # ensuring the PDF processing is always executed directly and reliably.
        logging.info("Directly executing ResumeProcessingTool...")
        # Instantiate the tool
        resume_processing_tool_instance = ResumeProcessingTool()
        # Call the _run method of the tool directly with the PDF path
        processed_resume_data = resume_processing_tool_instance._run(pdf_path=pdf_path)
        logging.info(f"Resume Processing Tool raw output: {processed_resume_data}")
        
        # Check for errors from the tool itself (e.g., if PDF reading failed)
        if processed_resume_data.get("status") == "error":
            raise Exception(f"Resume processing failed: {processed_resume_data.get('error')}")

        # Convert the dictionary output from the tool to a JSON string.
        # Subsequent CrewAI tasks expect string inputs for interpolation.
        processed_resume_output_str = json.dumps(processed_resume_data)
        logging.info("Resume processing finished. Proceeding with job search.")

        # 2. Run Job Search Task
        # Create a mini-crew for just the job search task.
        # This agent will use the LLM to 'think' and 'generate' job matches based on the provided data.
        job_search_crew = Crew(
            agents=[research_agent],
            tasks=[job_search_task],
            verbose=True # Enable verbose logging for this crew's execution
        )
        logging.info("Starting job search...")
        # Kickoff the job search crew, passing the processed resume output string as an input.
        job_matches_output_obj = job_search_crew.kickoff(inputs={'processed_resume_output': processed_resume_output_str})
        
        # Convert the CrewOutput object to a string. This is typically how to get the final task output.
        job_matches_output_str = str(job_matches_output_obj) 
        
        logging.info(f"Job search finished. Output: {job_matches_output_str}")

        # 3. Run Career Guidance Task
        # Create a mini-crew for just the career guidance task.
        # This agent will use the LLM to generate personalized advice based on job matches and user query.
        career_guidance_crew = Crew(
            agents=[career_assistant_agent],
            tasks=[career_guidance_task],
            verbose=True # Enable verbose logging for this crew's execution
        )
        logging.info("Starting career guidance generation...")
        final_crew_output_obj = career_guidance_crew.kickoff(
            inputs={
                'job_matches': job_matches_output_str, # Now a plain string
                'user_query': user_query
            }
        )
        logging.info("Career guidance generation finished.")

        # NEW FIX: Convert the final CrewOutput object to a string and clean it up.
        # This ensures it's a plain string and removes any unwanted internal monologue.
        final_crew_output_str = str(final_crew_output_obj)
        # Apply string cleaning to remove common CrewAI internal thought process tags if they appear
        final_crew_output_str = final_crew_output_str.replace("Thought:", "").replace("Action:", "").replace("Action Input:", "").replace("Observation:", "").strip()

        # Return the final output from the career guidance agent as a JSON response
        return JSONResponse(content={
            "status": "success",
            "message": "Career guidance generated successfully.",
            "crew_output": final_crew_output_str # Now definitively a clean string
        })

    except Exception as e:
        # Log the full traceback for detailed debugging on the server side
        logging.exception(f"An error occurred during career request processing: {e}") 
        # Return a meaningful error message in case of an exception
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")
    finally:
        # 4. Clean up the temporary PDF file
        if pdf_path and os.path.exists(pdf_path):
            os.remove(pdf_path)
            logging.info(f"Cleaned up temporary file: {pdf_path}")
