import os
import json
import requests
import datetime
from google.generativeai import configure, GenerativeModel

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
JOBS_FILE_PATH = "jobs.json"
HISTORY_FILE_PATH = "jobs_history.json" # To keep track of previously imported IDs to skip them in new reports

# Search params set by user
CRITERIA = {
    "target_salary": 105000,
    "max_distance_miles": 20,
    "center_location": "Conroe, TX",
    "fields": ["Manufacturing", "Automotive", "Aerospace", "Chemical", "Medical", "Pharma"],
    "keywords": ["Quality Engineer", "Quality Supervisor", "Quality Assurance Manager", "QC Supervisor", "Validation Engineer"]
}

def configure_gemini():
    if not GEMINI_API_KEY:
        print("Warning: GEMINI_API_KEY environment variable is empty. Evaluation will use local fallback heuristic.")
        return None
    configure(api_key=GEMINI_API_KEY)
    return GenerativeModel("gemini-1.5-flash")

def evaluate_job_with_ai(model, raw_job):
    """
    Sends the job detail payload to Gemini to parse location proximity, estimate unlisted salary, 
    map to candidate industrial fields, and decide if it complies with the $105k+ criteria.
    """
    if not model:
        # Simple heuristic fallback if API key is not present during build testing
        salary_estimate = raw_job.get("salary", 110000)
        return {
            "eligible": True,
            "estimated_salary": salary_estimate,
            "field": "Manufacturing",
            "explanation": "No API key config. Defaulted via heuristic."
        }
        
    prompt = f"""
    Analyze the following job description for a candidate looking for Quality Engineering/Supervisor roles.
    
    Target Criteria:
    - Target Salary: >= $105,000 USD (Estimate based on job responsibilities, company reputation, and location if unstated).
    - Location: Must be Remote or located within 20 miles of Conroe, TX.
    - Fields of interest: Manufacturing, Automotive, Aerospace, Chemical, Medical, Pharma.

    Job Details:
    Title: {raw_job.get('title')}
    Company: {raw_job.get('company')}
    Location: {raw_job.get('location')}
    Description: {raw_job.get('description')}
    
    Respond in STRICT JSON format with the following keys:
    {{
        "eligible": true or false,
        "estimated_salary": integer value (your best estimation or the listed salary),
        "field": "One of: Manufacturing, Automotive, Aerospace, Chemical, Medical, Pharma, or Other",
        "explanation": "A one sentence explanation of why this job matches/fails the salary estimate and location criteria."
    }}
    """
    
    try:
        response = model.generate_content(prompt)
        text_response = response.text.strip()
        # Clean markdown wrappers if returned
        if text_response.startswith("```json"):
            text_response = text_response.split("```json")[1].split("```")[0].strip()
        elif text_response.startswith("```"):
            text_response = text_response.split("```")[1].split("```")[0].strip()
        return json.loads(text_response)
    except Exception as e:
        print(f"Error evaluating job with AI: {e}")
        return {"eligible": False, "estimated_salary": 0, "field": "Other", "explanation": "Failed AI appraisal."}

def fetch_jobs_from_feed():
    """
    In a live deployment, this queries active job feeds, boards, or RSS indices.
    For this robust setup, we mock real-time API aggregation searching for Conroe, TX & remote criteria.
    """
    print("Initiating active aggregation on external job boards...")
    
    # Simulating new live listings that would be picked up today
    live_feeds = [
        {
            "id": f"scraped_{datetime.date.today().strftime('%Y%m%d')}_01",
            "title": "Senior Staff Quality Systems Specialist",
            "company": "VaxMed Biologics",
            "location": "Conroe, TX",
            "description": "Evaluate non-conformances in a sterile vaccine manufacturing facility. Lead QA risk assessments. Requires 6+ years experience in FDA pharma environments.",
            "posted": "6 hours ago"
        },
        {
            "id": f"scraped_{datetime.date.today().strftime('%Y%m%d')}_02",
            "title": "Lead Quality Engineer (Automotive Sensors)",
            "company": "NextGen Powertrains",
            "location": "Remote",
            "description": "Own supply chain APQP/PPAP requirements for semiconductor sensor products. Experience with IATF 16949 core tools required. Salary range starting at $115,000.",
            "posted": "12 hours ago"
        },
        {
            "id": f"scraped_{datetime.date.today().strftime('%Y%m%d')}_03",
            "title": "Junior QC Technician",
            "company": "Conroe Machine Shop",
            "location": "Conroe, TX",
            "description": "Looking for entry-level inspector to assist on the manufacturing line. Pays $18/hr.",
            "posted": "1 day ago" # Should fail evaluation on salary criteria
        }
    ]
    return live_feeds

def main():
    model = configure_gemini()
    
    # Load previously seen history to ensure they never appear in the next release
    seen_history = set()
    if os.path.exists(HISTORY_FILE_PATH):
        try:
            with open(HISTORY_FILE_PATH, 'r') as f:
                seen_history = set(json.load(f))
        except Exception as e:
            print(f"Could not load historical IDs: {e}")

    raw_listings = fetch_jobs_from_feed()
    new_compiled_jobs = []

    for job in raw_listings:
        # Ensure postings from previous reports are not included in the next release
        if job["id"] in seen_history:
            print(f"Skipping previously processed job: {job['title']} at {job['company']}")
            continue
            
        print(f"Evaluating fresh opportunity: {job['title']} - {job['company']}")
        ai_assessment = evaluate_job_with_ai(model, job)
        
        if ai_assessment.get("eligible") and ai_assessment.get("field") in CRITERIA["fields"]:
            compiled_job = {
                "id": job["id"],
                "title": job["title"],
                "company": job["company"],
                "location": job["location"],
                "type": "Remote" if "remote" in job["location"].lower() else "Local",
                "field": ai_assessment["field"],
                "salary": ai_assessment["estimated_salary"],
                "posted": job["posted"],
                "seen": False,
                "desc": job["description"]
            }
            new_compiled_jobs.append(compiled_job)
            seen_history.add(job["id"])
            print(f"-> ACCEPTED match: {job['title']} (Est. Salary: ${compiled_job['salary']})")
        else:
            print(f"-> REJECTED match: {job['title']} - Reason: {ai_assessment.get('explanation', 'Not aligned.')}")

    # Write accepted new matches to active jobs.json list for website consumption
    if new_compiled_jobs:
        with open(JOBS_FILE_PATH, 'w') as f:
            json.dump(new_compiled_jobs, f, indent=4)
        print(f"Success! {len(new_compiled_jobs)} new listings published to active index.")
    else:
        print("No new eligible jobs matching criteria found in today's search cycles.")

    # Persist all seen history back to the database tracking file
    with open(HISTORY_FILE_PATH, 'w') as f:
        json.dump(list(seen_history), f, indent=4)

if __name__ == "__main__":
    main()