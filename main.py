import os
import json
import logging
import re
import tempfile
import shutil
from typing import Optional, Any, Dict, List

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError

from crewai import Crew

# Import your crew setup components
from crew_setup import (
    career_assistant_agent,
    career_guidance_task,
    ResumeProcessingTool,
    JobFilteringTool,
    FinalCrewOutput
)

# --- Configure Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.getLogger('crewai').setLevel(logging.DEBUG)
logging.getLogger('langchain_core').setLevel(logging.DEBUG) # Crucial for LLM errors
logging.getLogger('langchain').setLevel(logging.DEBUG) # Also helpful for broader LangChain issues
logging.getLogger('uvicorn.access').setLevel(logging.INFO)
logging.getLogger('uvicorn.error').setLevel(logging.INFO)

# --- FastAPI App Initialization ---
app = FastAPI(
    title="AI Career Assistant API",
    description="API for processing resumes and providing career guidance using CrewAI.",
    version="1.0.0"
)

# --- CORS Middleware Configuration ---
origins = [
    "http://localhost",
    "http://localhost:8501", # For Streamlit frontend
    "http://127.0.0.1:8501", # For Streamlit frontend
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Temporary File Directory ---
TEMP_FILES_DIR = "temp_uploads"
os.makedirs(TEMP_FILES_DIR, exist_ok=True)
logging.info(f"Temporary file directory set to: {TEMP_FILES_DIR}")

@app.get("/")
async def root():
    """Root endpoint for health check and welcome message."""
    return {"message": "Welcome to the AI Career Assistant API. Use /process_career_request/ to get started."}

@app.post("/process_career_request/")
async def process_career_request(
    resume_file: UploadFile = File(..., description="The PDF resume file to upload."),
    user_query: Optional[str] = Form("Tell me about job opportunities related to my skills and advice on career progression."),
):
    """
    Processes an uploaded resume and user query to provide career guidance and job matches.
    """
    pdf_path = None
    try:
        # Save uploaded PDF to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf", dir=TEMP_FILES_DIR) as tmp_file:
            shutil.copyfileobj(resume_file.file, tmp_file)
            pdf_path = tmp_file.name

        logging.info(f"Received PDF: {resume_file.filename} saved to {pdf_path}")

        # --- 1. Execute ResumeProcessingTool directly ---
        logging.info("Directly executing ResumeProcessingTool...")
        resume_processing_tool_instance = ResumeProcessingTool()
        processed_resume_data: Dict[str, Any] = resume_processing_tool_instance._run(pdf_path=pdf_path)
        
        logging.debug(f"Resume Processing Tool raw output: {processed_resume_data}")

        # Handle potential string output from mock tool if it's not a dict
        if isinstance(processed_resume_data, str):
            try:
                processed_resume_data = json.loads(processed_resume_data)
            except json.JSONDecodeError:
                # Fallback for mock tool returning non-JSON string, try regex for key pieces
                logging.warning("ResumeProcessingTool returned a string that is not valid JSON. Attempting regex extraction for mock data.")
                skills_match = re.search(r"'skills':\s*\[([^\]]+)\]", processed_resume_data)
                user_skills = [s.strip().strip("'\"") for s in skills_match.group(1).split(',')] if skills_match else []
                summary_match = re.search(r"'resume_summary':\s*'(.*?)'", processed_resume_data)
                resume_summary = summary_match.group(1) if summary_match else "No summary extracted."
                processed_resume_data = {"skills": user_skills, "resume_summary": resume_summary}
        
        if processed_resume_data.get("status") == "error":
            raise HTTPException(status_code=400, detail=f"Resume processing failed: {processed_resume_data.get('error', 'Unknown error during PDF processing.')}")
        
        user_skills = processed_resume_data.get("skills", [])
        resume_summary = processed_resume_data.get("resume_summary", "No summary extracted.")

        logging.info(f"Resume processing finished. Extracted skills: {user_skills[:5]}... Summary length: {len(resume_summary)}.")

        # --- 2. Execute JobFilteringTool directly ---
        logging.info("Directly executing JobFilteringTool...")
        job_filtering_tool_instance = JobFilteringTool()
        filtered_jobs_list: List[Dict[str, Any]] = job_filtering_tool_instance._run(user_skills=user_skills)
        
        logging.info(f"Job filtering finished. Found {len(filtered_jobs_list)} jobs.")

        # --- 3. Run Career Guidance Task using CrewAI ---
        combined_context_for_llm = {
            "resume_summary": resume_summary,
            "user_extracted_skills": user_skills,
            "user_query": user_query,
            "filtered_jobs_list": filtered_jobs_list
        }
        combined_context_str = json.dumps(combined_context_for_llm, indent=2)

        career_guidance_crew = Crew(
            agents=[career_assistant_agent],
            tasks=[career_guidance_task],
            verbose=True,
            output_pydantic=FinalCrewOutput # Instruct CrewAI for Pydantic output
        )
        
        logging.info("Starting career guidance generation with CrewAI...")
        
        crew_raw_output_from_kickoff: Any
        try:
            # CrewAI's kickoff can return the Pydantic object directly if successful,
            # or a RawOutput (which has a .raw string attribute), or sometimes just a string.
            crew_raw_output_from_kickoff = career_guidance_crew.kickoff(
                inputs={
                    'context_for_guidance': combined_context_str,
                }
            )
            logging.info("Career guidance generation finished successfully.")
        except Exception as e:
            logging.error(f"CrewAI kickoff failed: {e}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"CrewAI execution failed: {e}. Check server logs for detailed LLM errors."
            )

        final_pydantic_result: Optional[FinalCrewOutput] = None
        full_string_output: str = ""
        
        # Determine the raw string output from CrewAI's kickoff
        if isinstance(crew_raw_output_from_kickoff, str):
            full_string_output = crew_raw_output_from_kickoff
        elif hasattr(crew_raw_output_from_kickoff, 'raw') and isinstance(crew_raw_output_from_kickoff.raw, str):
            full_string_output = crew_raw_output_from_kickoff.raw
        elif isinstance(crew_raw_output_from_kickoff, FinalCrewOutput):
            # If it's already the Pydantic object, no string parsing needed
            final_pydantic_result = crew_raw_output_from_kickoff
            logging.debug("CrewAI kickoff returned FinalCrewOutput directly.")
        else:
            # Fallback for unexpected types
            full_string_output = str(crew_raw_output_from_kickoff)
            logging.warning(f"CrewAI kickoff returned unexpected type: {type(crew_raw_output_from_kickoff)}. Attempting string conversion.")

        # Attempt to parse the string output into the Pydantic model, if not already a Pydantic object
        if not final_pydantic_result and full_string_output:
            # Use regex to find the outermost JSON object in the string
            # This regex looks for a string that starts with '{' and ends with '}'
            # and is as short as possible (non-greedy) but still captures the whole JSON.
            # re.DOTALL ensures '.' matches newlines as well.
            json_match = re.search(r'{.*}', full_string_output.strip(), re.DOTALL)
            
            if json_match:
                extracted_json_string = json_match.group(0)
                logging.debug(f"Successfully extracted potential JSON string (first 200 chars): {extracted_json_string[:200]}...")
                
                try:
                    # Attempt to parse the extracted JSON string with Pydantic
                    final_pydantic_result = FinalCrewOutput.model_validate_json(extracted_json_string)
                    logging.info("Successfully parsed extracted JSON string to FinalCrewOutput.")
                except (json.JSONDecodeError, ValidationError) as e:
                    logging.error(f"Failed to parse extracted JSON string as FinalCrewOutput (JSON or Pydantic error): {e}. Extracted string: '{extracted_json_string}'", exc_info=True)
                    raise HTTPException(
                        status_code=500,
                        detail=f"CrewAI output parsing error: Extracted content was not valid for FinalCrewOutput. Raw output: {extracted_json_string}"
                    )
            else:
                logging.error(f"Could not extract a JSON object from the CrewAI output string. Full raw output: '{full_string_output}'")
                raise HTTPException(
                    status_code=500,
                    detail=f"CrewAI output format error: The AI did not produce a recognizable JSON output. Raw output: {full_string_output}"
                )
        
        # Final check: If after all attempts, final_pydantic_result is still not a FinalCrewOutput instance
        if not isinstance(final_pydantic_result, FinalCrewOutput):
            error_content = full_string_output if full_string_output else str(crew_raw_output_from_kickoff)
            logging.error(f"Final object is not FinalCrewOutput after all parsing attempts. Actual type: {type(final_pydantic_result)}. Content: {error_content}")
            raise HTTPException(
                status_code=500,
                detail=f"CrewAI output format error: Final result is not a FinalCrewOutput instance. This indicates a deeper issue. Raw initial output: {error_content}"
            )
        
        # Now, `final_pydantic_result` is guaranteed to be a FinalCrewOutput instance
        return JSONResponse(content={
            "status": "success",
            "message": "Career guidance generated successfully.",
            # Ensure model_dump() is used for nested Pydantic models when converting to dict
            "crew_output": final_pydantic_result.guidance.model_dump(),
            "matched_jobs": [job.model_dump() for job in final_pydantic_result.matched_jobs]
        })

    except HTTPException as e:
        logging.error(f"HTTPException caught: {e.detail}", exc_info=True)
        raise
    except Exception as e:
        logging.exception(f"An unexpected error occurred during career request processing: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {e}. Please check server logs for details."
        )
    finally:
        # Ensure temporary PDF file is cleaned up
        if pdf_path and os.path.exists(pdf_path):
            try:
                os.remove(pdf_path)
                logging.info(f"Cleaned up temporary file: {pdf_path}")
            except Exception as e:
                logging.error(f"Failed to delete temporary file {pdf_path}: {e}")