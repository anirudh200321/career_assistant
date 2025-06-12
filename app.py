import streamlit as st
import httpx # For making HTTP requests to the FastAPI backend
import io
import json
import asyncio # For running async functions in Streamlit

# --- Streamlit App Configuration ---
st.set_page_config(
    page_title="AI Career Assistant",
    page_icon="💼",
    layout="centered",
    initial_sidebar_state="auto"
)

# --- FastAPI Backend URL ---
# IMPORTANT: Adjust this if your FastAPI server is not on localhost:8000
FASTAPI_URL = "http://localhost:8000/process_career_request/"

# --- Helper Function to Call FastAPI ---
async def call_fastapi_backend(resume_file_bytes_io: io.BytesIO, user_query: str):
    """
    Sends the resume file and user query to the FastAPI backend.
    """
    files = {'resume_file': (resume_file_bytes_io.name, resume_file_bytes_io.getvalue(), 'application/pdf')}
    data = {'user_query': user_query}

    # Use httpx for making async requests
    async with httpx.AsyncClient() as client:
        try:
            # Set a longer timeout for the request, as LLM operations can take time
            response = await client.post(FASTAPI_URL, files=files, data=data, timeout=300.0) 
            response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)
            return response.json()
        except httpx.RequestError as e:
            st.error(f"Network error communicating with the backend: {e}")
            return None
        except httpx.HTTPStatusError as e:
            st.error(f"Backend returned an error: {e.response.status_code} - {e.response.text}")
            return None
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
            return None

# --- Streamlit UI ---
st.title("💼 AI Career Assistant")
st.markdown("Upload your resume (PDF) and ask a question to get personalized career guidance and job matches.")

# Resume Upload
uploaded_file = st.file_uploader("Upload your resume (PDF only)", type=["pdf"])

# User Query
user_question = st.text_area(
    "Ask your career assistant a question:",
    "Tell me about job opportunities based on my skills and how I can improve my resume for these roles."
)

# Process Button
if st.button("Get Guidance"):
    if uploaded_file is not None:
        st.info("Processing your request... This may take a moment as agents work on it.")
        
        # Create a BytesIO object from the uploaded file
        pdf_bytes_io = io.BytesIO(uploaded_file.getvalue())
        # Preserve the original file name for the multipart form data
        pdf_bytes_io.name = uploaded_file.name 
        
        with st.spinner("Agents are thinking and processing..."):
            # Streamlit runs in its own thread, so we need to run async calls using asyncio
            try:
                result = asyncio.run(call_fastapi_backend(pdf_bytes_io, user_question))
            except Exception as e:
                st.error(f"Error during async execution: {e}")
                result = None

            if result:
                if result.get("status") == "success":
                    st.success("Guidance generated successfully!")
                    st.subheader("Your Career Assistant's Advice:")
                    st.markdown(result.get("crew_output", "No output provided."))
                    
                else:
                    st.error(f"Failed to get guidance: {result.get('message', 'Unknown error')}")
            else:
                st.error("Could not get a response from the backend. Please check server logs for errors.")
    else:
        st.warning("Please upload a PDF resume to proceed.")

st.markdown("---")
st.markdown("Built with CrewAI (powered by Groq), FastAPI, and Streamlit.")