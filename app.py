from flask import Flask, render_template, request, jsonify
from functools import lru_cache
import requests
from datetime import datetime
import logging
from ratelimit import limits, sleep_and_retry
import os
from dotenv import load_dotenv
from typing import Optional, Dict, List
from openai import OpenAI

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("app.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration
class Config:
    NBA_API_KEY = os.getenv(
        "NBA_API_KEY", "fe7ac125c5msh94f9c196609b1eep12fb18jsndc6f9e5920c3"
    )
    SEARCH_API_KEY = os.getenv(
        "SEARCH_API_KEY", "fe7ac125c5msh94f9c196609b1eep12fb18jsndc6f9e5920c3"
    )
    BASE_URL = "https://api.aimlapi.com/v1"
    API_KEY = os.getenv("AI_API_KEY", "my_key")
    CACHE_TIMEOUT = 3600  # 1 hour
    CALLS_PER_MINUTE = 30
    MAX_RETRIES = 3
    RETRY_DELAY = 5  # seconds


class APIError(Exception):
    """Custom exception for API related errors"""

    pass


def load_system_prompt(file_path: str) -> str:
    """Load system prompt from a text file."""
    with open(file_path, "r") as file:
        return file.read().strip()


# Rate limiting decorators
@sleep_and_retry
@limits(calls=Config.CALLS_PER_MINUTE, period=60)
def rate_limited_api_call(url: str, headers: Dict, params: Dict) -> requests.Response:
    """Make a rate-limited API call"""
    response = requests.get(url, headers=headers, params=params, timeout=10)
    if response.status_code != 200:
        raise APIError(f"API call failed with status {response.status_code}")
    return response


def get_nba_schedule(date: str) -> Optional[Dict]:
    """Get NBA schedule from the API for a specific date with caching"""
    url = "https://api-nba-v1.p.rapidapi.com/games"
    headers = {
        "x-rapidapi-key": Config.NBA_API_KEY,
        "x-rapidapi-host": "api-nba-v1.p.rapidapi.com",
    }

    try:
        response = rate_limited_api_call(url, headers=headers, params={"date": date})
        logger.info(f"Successfully fetched NBA schedule for {date}")
        return response.json()
    except (requests.RequestException, APIError) as e:
        logger.error(f"Error fetching NBA schedule: {str(e)}")
        return None


class MatchDataProcessor:
    """Class to handle match data processing"""

    PRIORITY_DOMAINS = {
        "nba.com": 5,
        "espn.com": 4,
        "basketball-reference.com": 4,
        "aiscore.com": 3,
        "landofbasketball.com": 3,
        "sofascore.com": 4,
    }

    @staticmethod
    @lru_cache(maxsize=64)
    def get_match_data(teams: str, num_results: int = 25) -> str:
        """Get match data from web search API with caching"""
        url = "https://real-time-web-search.p.rapidapi.com/search"
        headers = {
            "x-rapidapi-key": Config.SEARCH_API_KEY,
            "x-rapidapi-host": "real-time-web-search.p.rapidapi.com",
        }
        params = {"q": f"{teams} NBA match statistics", "limit": str(num_results)}

        try:
            response = rate_limited_api_call(url, headers=headers, params=params)
            data = response.json()

            if data.get("status") == "OK" and "data" in data:
                return MatchDataProcessor.process_match_data(data["data"], teams)

            logger.warning(f"No data found for match: {teams}")
            return f"Tidak dapat menemukan data spesifik untuk pertandingan {teams}."

        except Exception as e:
            logger.error(f"Error fetching match data: {str(e)}")
            return f"Terjadi kesalahan saat mengambil data {teams}."

    @staticmethod
    def process_match_data(search_results: List[Dict], teams: str) -> str:
        """Process search results with improved relevance scoring"""
        match_info = []
        seen_urls = set()

        for item in search_results:
            url = item.get("url", "").strip()
            if not url or url in seen_urls:
                continue

            domain = item.get("domain", "")
            priority = next(
                (
                    value
                    for key, value in MatchDataProcessor.PRIORITY_DOMAINS.items()
                    if key in domain
                ),
                0,
            )

            if item.get("snippet"):
                relevance_score = MatchDataProcessor.calculate_relevance_score(
                    item.get("title", ""), item.get("snippet", ""), teams, priority
                )

                match_info.append(
                    {
                        "url": url,
                        "snippet": item.get("snippet", "").strip(),
                        "relevance_score": relevance_score,
                    }
                )
                seen_urls.add(url)

        match_info.sort(key=lambda x: x["relevance_score"], reverse=True)
        return MatchDataProcessor.format_match_data(match_info[:15], teams)

    @staticmethod
    def calculate_relevance_score(
        title: str, snippet: str, teams: str, priority: int
    ) -> float:
        """Calculate relevance score based on content and priority"""
        score = priority * 2.0

        team_names = teams.lower().split()
        content = (title + " " + snippet).lower()
        for team in team_names:
            if team in content:
                score += 1.0

        if any(
            str(year) in snippet
            for year in range(datetime.now().year - 1, datetime.now().year + 1)
        ):
            score += 2.0

        return score

    @staticmethod
    def format_match_data(match_info: List[Dict], teams: str) -> str:
        """Format match data into readable content with 15 URLs"""
        context = f"Data Pertandingan {teams}:\n\n"

        for i, info in enumerate(match_info, 1):
            context += f"{i}. {info['url']}\n"
            context += f"   {info['snippet']}\n\n"

        return context


class AimlClientManager:
    """Class to manage API client using the specified base_url and api_key"""

    def __init__(self):
        self.api = OpenAI(api_key=Config.API_KEY, base_url=Config.BASE_URL)

    async def get_prediction(self, system_prompt: str, user_prompt: str) -> str:
        """Query the API with a prompt and return the response"""
        try:
            completion = self.api.chat.completions.create(
                model=os.getenv("DEFAULT_MODEL"),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.7,
                max_tokens=256,
            )
            return completion.choices[0].message.content
        except Exception as e:
            logger.error(f"Error in API call: {str(e)}")
            return "Maaf, terjadi kesalahan saat menghasilkan prediksi. Silakan coba lagi nanti."


class PredictionGenerator:
    """Class to handle prediction generation using the custom API"""

    def __init__(self):
        self.api_manager = AimlClientManager()
        self.max_context_length = 2000

    @staticmethod
    def create_prediction_prompt(match_data: str, teams: str) -> str:
        """Create an enhanced structured prompt for the AI"""
        return f"""Analisis Prediksi NBA: {teams}

DATA PERTANDINGAN:
{match_data}

Analisis lah berdasarkan data di atas.
"""

    def _truncate_prompt(self, prompt: str) -> str:
        """Truncate prompt to fit within context length while maintaining coherence"""
        if len(prompt) > self.max_context_length:
            truncated = prompt[: self.max_context_length]
            last_period = truncated.rfind(".")
            if last_period > 0:
                truncated = truncated[: last_period + 1]
            return truncated
        return prompt

    @staticmethod
    def _format_response(response: str) -> str:
        """Format the AI response for better readability"""
        formatted = "\n".join(line for line in response.splitlines() if line.strip())
        formatted = formatted.replace("# ", "\n# ")
        formatted = formatted.replace("## ", "\n## ")

        lines = formatted.splitlines()
        formatted_lines = []
        for line in lines:
            if line.strip().startswith("•"):
                formatted_lines.append("  " + line.strip())
            elif line.strip().startswith("-"):
                formatted_lines.append(line.strip())
            else:
                formatted_lines.append(line)

        return "\n".join(formatted_lines)

    async def get_prediction(self, match_data: str, teams: str) -> str:
        """Get prediction from the custom API"""
        try:
            prompt = self.create_prediction_prompt(match_data, teams)
            prompt = self._truncate_prompt(prompt)

            response_text = await self.api_manager.get_prediction(
                load_system_prompt("system_prompt.txt"), prompt
            )
            return self._format_response(response_text)

        except Exception as e:
            logger.error(f"Error generating prediction: {str(e)}")
            return "Maaf, terjadi kesalahan saat menghasilkan prediksi. Silakan coba lagi nanti."


@app.route("/", methods=["GET", "POST"])
async def index():
    """Main route handler with improved error handling and response structure"""
    response_data = {
        "schedule": None,
        "prediction_result": None,
        "start_message": None,
        "date_message": None,
        "error_message": None,
        "current_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    if request.method == "POST":
        try:
            if "start" in request.form:
                response_data[
                    "start_message"
                ] = "Selamat datang! Silakan masukkan tanggal (YYYY-MM-DD) untuk melihat jadwal pertandingan NBA."

            elif "date" in request.form:
                date = request.form.get("date")
                if date:
                    response_data[
                        "date_message"
                    ] = f"Jadwal pertandingan NBA tanggal {date}:"
                    response_data["schedule"] = get_nba_schedule(date)
                    if not response_data["schedule"] or not response_data[
                        "schedule"
                    ].get("response"):
                        response_data[
                            "error_message"
                        ] = "Tidak ada pertandingan pada tanggal tersebut atau terjadi kesalahan mengambil data."
                else:
                    response_data[
                        "error_message"
                    ] = "Mohon masukkan tanggal yang valid."

            elif "match_choice" in request.form:
                user_choice = request.form.get("match_choice")
                match_data = MatchDataProcessor.get_match_data(user_choice)
                prediction_generator = PredictionGenerator()
                response_data[
                    "prediction_result"
                ] = await prediction_generator.get_prediction(match_data, user_choice)

            elif "user_prompt" in request.form:
                user_prompt = request.form.get("user_prompt")
                if user_prompt:
                    prediction_generator = PredictionGenerator()
                    response_data[
                        "prediction_result"
                    ] = await prediction_generator.get_prediction(
                        user_prompt, "Custom Query"
                    )

        except Exception as e:
            logger.error(f"Error in route handler: {str(e)}")
            response_data["error_message"] = f"Terjadi kesalahan: {str(e)}"

    return render_template("index.html", **response_data)


@app.route("/health", methods=["GET"])
async def health_check():
    """Health check endpoint"""
    try:
        return (
            jsonify(
                {
                    "status": "healthy",
                    "api_status": "connected",
                    "timestamp": datetime.now().isoformat(),
                }
            ),
            200,
        )
    except Exception as e:
        return (
            jsonify(
                {
                    "status": "unhealthy",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat(),
                }
            ),
            500,
        )


if __name__ == "__main__":
    # Run with Hypercorn for async support
    import hypercorn.asyncio
    import hypercorn.config

    config = hypercorn.config.Config()
    config.bind = ["localhost:5000"]
    asyncio.run(hypercorn.asyncio.serve(app, config))
