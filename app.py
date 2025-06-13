import streamlit as st
import requests
import json
import datetime
import pytz # For accurate timezone handling

# --- Configuration ---
# Assuming your FastAPI backend is running locally on port 8000
BACKEND_URL = "http://localhost:8000/process_career_request/"

# --- Helper Classes to structure received data (optional, but good practice) ---
# These classes help you access data from the JSON response in an organized way.
# They reflect the structure of your Pydantic models in crew_setup.py
class CareerGuidanceDetails:
    def __init__(self, data: dict):
        self.career_path_suggestion = data.get("career_path_suggestion", "N/A")
        self.relevant_skills_gap = data.get("relevant_skills_gap", "N/A")
        self.actionable_steps = data.get("actionable_steps", "N/A")
        self.potential_job_titles = data.get("potential_job_titles", [])

class JobMatch:
    def __init__(self, data: dict):
        self.title = data.get("title", "N/A")
        self.company = data.get("company", "N/A")
        self.location = data.get("location", "N/A")
        self.skills_required = data.get("skills_required", [])
        self.description = data.get("description", "N/A")

# --- Function to format the output ---
def format_career_guidance_output(guidance_data: dict, matched_jobs_data: list) -> str:
    """
    Formats the JSON guidance and job data received from the backend
    into a human-readable text string.
    """
    # Create instances of helper classes for easier access
    guidance = CareerGuidanceDetails(guidance_data)
    matched_jobs = [JobMatch(job) for job in matched_jobs_data]

    # Get current date, time, and location dynamically
    # Use 'Asia/Kolkata' for IST (Indian Standard Time)
    tz = pytz.timezone('Asia/Kolkata')
    now = datetime.datetime.now(tz)
    formatted_time = now.strftime("%A, %B %d, %Y at %I:%M:%S %p %Z")
    location = "Hyderabad, Telangana, India" # Hardcoded as requested

    output_text = f"**Generated on:** {formatted_time} in {location}.\n\n"
    output_text += "---\n\n" # Separator

    # --- Career Guidance Section ---
    output_text += "### Your Personalized Career Guidance\n\n"
    output_text += f"**Career Path Suggestion:**\nBased on your profile, a highly suitable career path for you is **{guidance.career_path_suggestion}**. This path aligns with your current skills and offers significant opportunities for growth.\n\n"

    output_text += "**Relevant Skills Gap:**\nTo excel in the {guidance.career_path_suggestion} role and enhance your capabilities, you should focus on acquiring or improving skills in:\n"
    # Assuming relevant_skills_gap might be a comma-separated string or a list.
    # If it's a string, split and format it. If it's a list, iterate directly.
    if isinstance(guidance.relevant_skills_gap, str):
        skills_list = [s.strip() for s in guidance.relevant_skills_gap.split(',')]
    else: # Assume it's already a list
        skills_list = guidance.relevant_skills_gap
    for skill in skills_list:
        if skill: # Ensure skill is not empty
            output_text += f"* {skill.capitalize()}\n"
    output_text += "\n"

    output_text += "**Actionable Steps to Bridge Skills Gaps:**\nHere are detailed steps you can take to develop the identified skills:\n"
    # Assuming actionable_steps might be a period-separated string.
    if isinstance(guidance.actionable_steps, str):
        steps_list = [s.strip() for s in guidance.actionable_steps.split('. ') if s.strip()] # Split and filter empty
    else: # Assume it's already a list of steps
        steps_list = guidance.actionable_steps
    for step_line in steps_list:
        if step_line: # Ensure step is not empty
            output_text += f"* {step_line.capitalize()}.\n" # Re-add period for clarity
    output_text += "\n"

    output_text += "**Potential Job Titles:**\nHere are some job titles that align well with your profile and the suggested career path:\n"
    for title in guidance.potential_job_titles:
        output_text += f"* {title}\n"
    output_text += "\n"

    output_text += "---\n\n" # Separator

    # --- Matched Job Opportunities Section ---
    output_text += "### Highly Relevant Job Opportunities\n\n"
    if matched_jobs:
        for i, job in enumerate(matched_jobs):
            output_text += f"* **{job.title}** at **{job.company}** ({job.location})\n"
            output_text += f"    * **Skills Required:** {', '.join(job.skills_required)}\n"
            output_text += f"    * **Description:** {job.description}\n"
            if i < len(matched_jobs) - 1:
                output_text += "\n" # Add a newline between jobs for better readability
    else:
        output_text += "No relevant job opportunities found at this time.\n"

    return output_text

# --- Streamlit User Interface (UI) ---
st.set_page_config(page_title="AI Career Assistant", layout="wide")

st.title("Your Personalized Career Guidance")
st.markdown("Upload your resume and tell us your career goals to get tailored advice and job matches.")

# File uploader widget
uploaded_file = st.file_uploader("Upload your Resume (PDF)", type=["pdf"])

# Text area for user's career query
user_query = st.text_area(
    "Tell us about your career goals or what kind of guidance you need:",
    value="Tell me about job opportunities related to my skills and advice on career progression.",
    height=100
)

# Button to trigger the backend request
if st.button("Get Guidance"):
    if uploaded_file is not None:
        st.info("Processing your resume and generating guidance... This might take a moment.")
        
        # Prepare the file and form data for the POST request
        files = {"resume_file": ("resume.pdf", uploaded_file.getvalue(), "application/pdf")}
        data = {"user_query": user_query}

        try:
            # Make the API call to your FastAPI backend
            response = requests.post(BACKEND_URL, files=files, data=data)
            
            # Check if the request was successful (HTTP status code 200)
            if response.status_code == 200:
                result = response.json() # Parse the JSON response
                st.success("Guidance generated successfully!")
                
                # Check the 'status' field from your custom backend response
                if result.get("status") == "success":
                    guidance_output = result.get("crew_output", {})
                    matched_jobs_output = result.get("matched_jobs", [])
                    
                    # Format the parsed data into a readable string
                    formatted_text = format_career_guidance_output(guidance_output, matched_jobs_output)
                    st.markdown(formatted_text) # Display the formatted text using Markdown
                else:
                    # If backend indicates a non-success status, show error and raw JSON for debugging
                    st.error(f"Backend returned an error: {result.get('message', 'Unknown error')}")
                    st.json(result)
            else:
                # Handle HTTP errors (e.g., 500 Internal Server Error, 400 Bad Request)
                error_detail = response.json().get("detail", "No specific error message from backend.")
                st.error(f"Error from backend: {response.status_code} - {error_detail}")
                st.json(response.json()) # Display raw JSON for debugging
        except requests.exceptions.ConnectionError:
            # Handle cases where the Streamlit app cannot connect to the FastAPI backend
            st.error("Could not connect to the backend API. Please ensure your FastAPI server is running.")
        except json.JSONDecodeError:
            # Handle cases where the backend response is not valid JSON
            st.error("Received an invalid response from the backend (not JSON).")
            st.code(response.text) # Show raw text for debugging
        except Exception as e:
            # Catch any other unexpected errors during the process
            st.error(f"An unexpected error occurred: {e}")
            st.exception(e) # Display full traceback in Streamlit
    else:
        st.warning("Please upload a resume to get guidance.")