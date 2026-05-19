import os
import sys
import json
import hashlib
import datetime
from google import genai
from google.genai import types

JOBS_FILE_PATH = "jobs.json"
HISTORY_FILE_PATH = "jobs_history.json"

SEARCH_PROMPT = """
Search Google for active, actual job listings posted recently (ideally within the last 14 days) matching these criteria:
- Titles: Quality Engineer, Senior Quality Engineer, Quality Supervisor, QC Supervisor, Quality Assurance Manager, Validation Engineer, or similar roles.
- Locations: Must be located within 20 miles of Conroe, TX (including nearby towns like The Woodlands, Spring, Montgomery, Huntsville) OR be fully Remote.
- Salary: Base compensation should be estimated or stated at $105,000 USD or more per year. (If not explicitly listed in the posting, make an intelligent estimation based on company size, job seniority, and standard regional or remote market averages).
- Industry Fields: Manufacturing, Automotive, Aerospace, Chemical, Medical, or Pharma.

Please locate real listings from job boards (such as LinkedIn, Indeed, ZipRecruiter, Glassdoor) or direct company career portals. 
For each matching job, extract:
1. Exact Job Title.
2. Company Name.
3. Specific Location (e.g., 'Conroe, TX' or 'Remote').
4. A brief description (3-4 sentences detailing key tasks and requirements).
5. Salary figure (estimated or listed base as an integer).
6. Real source apply URL (avoid generic search landing pages, search for the direct job ID or listing link).
"""

PARSING_SCHEMA = {
    "type": "ARRAY",
    "description": "A structured array containing real job listings matching search guidelines.",
    "items": {
        "type": "OBJECT",
        "properties": {
            "title": {"type": "STRING", "description": "Exact job title of the listing."},
            "company": {"type": "STRING", "description": "Company hiring for the position."},
            "location": {"type": "STRING", "description": "Listing location, e.g., 'Conroe, TX', 'Spring, TX', 'Remote'."},
            "type": {"type": "STRING", "description": "Must be exactly 'Remote' or 'Local'."},
            "field": {"type": "STRING", "description": "Must be exactly one of: Manufacturing, Automotive, Aerospace, Chemical, Medical, Pharma."},
            "salary": {"type": "INTEGER", "description": "Yearly base salary, estimated or listed (must be >= 105000)."},
            "posted": {"type": "STRING", "description": "Estimated elapsed posting time, e.g., '1 day ago', '3 days ago', 'Yesterday'."},
            "desc": {"type": "STRING", "description": "A highly readable, concise summary of duties and qualification mandates."},
            "url": {"type": "STRING", "description": "A direct URL matching the specific post or career portal."}
        },
        "required": ["title", "company", "location", "type", "field", "salary", "posted", "desc", "url"]
    }
}

def calculate_stable_id(job):
    hash_payload = f"{job['company']}_{job['title']}_{job['url']}".encode('utf-8')
    return hashlib.md5(hash_payload).hexdigest()

def main():
    print("Initiating Google GenAI client structure...")
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable is missing. Halting execution.")
        return 1

    client = genai.Client(api_key=api_key)

    print("Executing Google Search Grounding to discover real openings...")
    try:
        search_response = client.models.generate_content(
            model='gemini-3-flash-preview',
            contents=SEARCH_PROMPT,
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())],
                temperature=0.2
            )
        )
        raw_grounding_text = search_response.text
        print("Successfully obtained grounded search information. Now parsing to JSON...")
    except Exception as e:
        print(f"Exception triggered during search execution: {e}")
        return 1

    try:
        structuring_prompt = f"""
        Extract and parse the following live search details into a completely valid JSON array matching the target schema.
        Filter out any job entry that clearly does not match the geographic boundaries (must be Conroe area or Remote), target sectors, or the minimum $105,000 threshold.

        Search Data:
        {raw_grounding_text}
        """

        parse_response = client.models.generate_content(
            model='gemini-3-flash-preview',
            contents=structuring_prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=PARSING_SCHEMA,
                temperature=0.1
            )
        )
        parsed_results = json.loads(parse_response.text)
        print(f"Discovered {len(parsed_results)} matching jobs in current cycle.")
    except Exception as e:
        print(f"Exception encountered during structured JSON extraction: {e}")
        return 1

    seen_history = set()
    if os.path.exists(HISTORY_FILE_PATH):
        try:
            with open(HISTORY_FILE_PATH, 'r') as f:
                seen_history = set(json.load(f))
        except Exception as e:
            print(f"Warning: Could not load history logs. Starting fresh: {e}")

    final_output_list = []
    for job in parsed_results:
        job_id = calculate_stable_id(job)
        if job_id in seen_history:
            print(f"Skipping already compiled job: {job['title']} at {job['company']}")
            continue
        
        job['id'] = job_id
        job['seen'] = False
        final_output_list.append(job)
        seen_history.add(job_id)
        print(f"-> ACCEPTED match: {job['title']} - {job['company']} (${job['salary']}/yr)")

    if final_output_list:
        with open(JOBS_FILE_PATH, 'w') as f:
            json.dump(final_output_list, f, indent=4)
        print(f"Success! Saved {len(final_output_list)} actual, new listings to {JOBS_FILE_PATH}.")
    else:
        # Save empty array so dashboard states respond accurately to empty cycles
        with open(JOBS_FILE_PATH, 'w') as f:
            json.dump([], f, indent=4)
        print("No fresh opportunities matched the criteria in today's search window.")

    with open(HISTORY_FILE_PATH, 'w') as f:
        json.dump(list(seen_history), f, indent=4)
    print("Historical registry successfully saved.")
    return 0

if __name__ == "__main__":
    sys.exit(main())