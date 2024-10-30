from flask import Flask, render_template, request, jsonify
from functools import lru_cache
import requests
from poe_api_wrapper import AsyncPoeApi,PoeApi
from datetime import datetime, timedelta
import json
import asyncio
import logging
from ratelimit import limits, sleep_and_retry
import os
from dotenv import load_dotenv
from typing import Optional, Dict, List, Any
import time
import markdown
from whitenoise import WhiteNoise

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("app.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

user_client = PoeApi(tokens={"p-b": os.getenv("POE_TOKEN_P_B"), "p-lat": os.getenv("POE_TOKEN_P_LAT")})
app = Flask(__name__)
app.wsgi_app = WhiteNoise(app.wsgi_app, root="static/")


@app.template_filter("format_date")
def format_date(date_string):
    try:
        if isinstance(date_string, str):
            date = datetime.fromisoformat(date_string.replace("Z", "+00:00"))
            return date.strftime("%d %b %Y")  # Format: 25 Oct 2024
        return date_string
    except Exception as e:
        logger.error(f"Error formatting date: {str(e)}")
        return date_string


# Configuration
class Config:
    NBA_API_KEY = os.getenv("NBA_API_KEY")
    SEARCH_API_KEY = os.getenv("SEARCH_API_KEY")
    POE_TOKEN = {
        "p-b": os.getenv("POE_TOKEN_P_B"),
        "p-lat": os.getenv("POE_TOKEN_P_LAT"),
    }
    POE_PROXY = os.getenv("POE_PROXY", None)
    CACHE_TIMEOUT = 3600  # 1 hour
    CALLS_PER_MINUTE = 30
    MAX_RETRIES = 3
    RETRY_DELAY = 5  # seconds


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
        logger.info("Successfully fetched NBA schedule for %s", date)
        return response.json()
    except (requests.RequestException, APIError) as e:
        logger.error(f"Error fetching NBA schedule: {str(e)}")
        return None


@lru_cache(maxsize=1)
def get_nba_teams() -> Optional[Dict]:
    """Get NBA teams from the API with caching"""
    url = "https://api-nba-v1.p.rapidapi.com/teams"
    headers = {
        "x-rapidapi-key": Config.NBA_API_KEY,
        "x-rapidapi-host": "api-nba-v1.p.rapidapi.com",
    }

    try:
        response = rate_limited_api_call(url, headers=headers, params={})
        logger.info("Successfully fetched NBA teams")
        return response.json()
    except (requests.RequestException, APIError) as e:
        logger.error(f"Error fetching NBA teams: {str(e)}")
        return None


def get_team_schedule(team_id: str) -> Optional[Dict]:
    """Get NBA schedule for a specific team"""
    url = "https://api-nba-v1.p.rapidapi.com/games"
    headers = {
        "x-rapidapi-key": Config.NBA_API_KEY,
        "x-rapidapi-host": "api-nba-v1.p.rapidapi.com",
    }
    params = {"season": "2024", "team": team_id}  # Current season

    try:
        response = rate_limited_api_call(url, headers=headers, params=params)
        logger.info(f"Successfully fetched schedule for team {team_id}")
        return response.json()
    except (requests.RequestException, APIError) as e:
        logger.error(f"Error fetching team schedule: {str(e)}")
        return None


class MatchDataProcessor:
    """Class to handle match data processing"""

    PRIORITY_DOMAINS = {
        "nba.com": 3,
        "espn.com": 4,
        "basketball-reference.com": 4,
        "aiscore.com": 3,
        "landofbasketball.com": 5,
        "sofascore.com": 4,
        "sports.yahoo.com": 4,
        "foxsports.com": 3,
        "www.statmuse.com": 4,
    }

    @staticmethod
    @lru_cache(maxsize=64)
    def get_match_data(
        teams: str, num_results: int = 25
    ) -> str:  # Increased from 10 to 25 to ensure enough quality results
        """Get match data from web search API with caching"""
        url = "https://real-time-web-search.p.rapidapi.com/search"
        headers = {
            "x-rapidapi-key": Config.SEARCH_API_KEY,
            "x-rapidapi-host": "real-time-web-search.p.rapidapi.com",
        }
        params = {
            "q": f"{teams} NBA match statistics",
            "limit": str(num_results),  # Fetch more results initially
        }

        try:
            response = rate_limited_api_call(url, headers=headers, params=params)
            data = response.json()

            if data.get("status") == "OK" and "data" in data:
                return MatchDataProcessor.process_match_data(data["data"], teams)

            logger.warning("No data found for match: %s", teams)
            return f"Tidak dapat menemukan data spesifik untuk pertandingan {teams}."

        except Exception as e:
            logger.error(f"Error fetching match data: {str(e)}")
            return f"Terjadi kesalahan saat mengambil data {teams}."

    @staticmethod
    def process_match_data(search_results: List[Dict], teams: str) -> str:
        """Process search results with improved relevance scoring"""
        match_info = []
        seen_urls = set()  # Track unique URLs to avoid duplicates

        for item in search_results:
            # Skip if URL is already processed
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

        # Sort by relevance score
        match_info.sort(key=lambda x: x["relevance_score"], reverse=True)
        return MatchDataProcessor.format_match_data(
            match_info[:15], teams
        )  # Take top 15 results

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

        # Number each URL for better readability
        for i, info in enumerate(match_info, 1):
            context += f"{i}. {info['url']}\n"
            context += f"   {info['snippet']}\n\n"

        return context


class PoeClientManager:
    _instance = None
    _client = None
    _last_init_time = None
    _tokens = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(PoeClientManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._tokens:
            self._tokens = Config.POE_TOKEN
            if not self._tokens.get("p-b") or not self._tokens.get("p-lat"):
                raise ValueError("POE tokens not found or invalid")

    async def _initialize_client(self):
        try:
            self._client = await AsyncPoeApi(tokens=self._tokens).create()
            self._last_init_time = time.time()
            logger.info("Poe client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Poe client: {str(e)}")
            raise

    async def get_client(self):
        current_time = time.time()

        if (
            not self._client
            or not self._last_init_time
            or current_time - self._last_init_time > 3600
        ):
            await self._initialize_client()

        return self._client


class PredictionGenerator:
    """Class to handle prediction generation using Poe API"""

    def __init__(self):
        self.poe_manager = PoeClientManager()
        self.default_bot = os.getenv("DEFAULT_MODEL")
        self.fallback_bot = "sage"
        self.max_context_length = 2000

    @staticmethod
    def create_prediction_prompt(match_data: str, teams: str) -> str:
        """Create an enhanced structured prompt for the AI"""
        return f"""Analisis Prediksi NBA: {teams}

DATA PERTANDINGAN:
{match_data}

Analisis Lah Berdasarkan Data Diatas
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

    import markdown

    def _format_response(self, response_text: str) -> str:
        """Format the AI response for better readability"""
        # Convert Markdown to HTML
        html_response = markdown.markdown(response_text)
        return html_response

    @staticmethod
    def handle_rate_limit(func):
        """Decorator to handle rate limiting and retries"""

        async def wrapper(*args, **kwargs):
            retries = 0
            while retries < Config.MAX_RETRIES:
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    if "rate" in str(e).lower():
                        retries += 1
                        if retries < Config.MAX_RETRIES:
                            logger.warning(
                                "Rate limit hit, waiting %s seconds. Retry %d/%d",
                                Config.RETRY_DELAY, retries, Config.MAX_RETRIES
                            )
                            await asyncio.sleep(Config.RETRY_DELAY)
                            continue
                    raise
            return None

        return wrapper

    @handle_rate_limit
    async def get_prediction(self, match_data: str, teams: str) -> str:
        """Get prediction from AI with error handling and retry logic"""
        try:
            client = await self.poe_manager.get_client()
            prompt = self.create_prediction_prompt(match_data, teams)
            prompt = self._truncate_prompt(prompt)

            try:
                response_text = ""
                async for chunk in client.send_message(
                    bot=self.default_bot, message=prompt
                ):
                    if chunk.get("response"):
                        response_text += chunk["response"]

                # Format the response before returning
                return self._format_response(response_text)

            except Exception as e:
                logger.warning(f"Error with primary bot, trying fallback: {str(e)}")
                response_text = ""
                async for chunk in client.send_message(
                    bot=self.fallback_bot, message=prompt
                ):
                    if chunk.get("response"):
                        response_text += chunk["response"]

                # Format the response before returning
                return self._format_response(response_text)

        except Exception as e:
            logger.error(f"Error generating prediction: {str(e)}")
            return "Maaf, terjadi kesalahan saat menghasilkan prediksi. Silakan coba lagi nanti."

    async def get_chat_history(self) -> List[Dict]:
        """Get chat history for debugging or monitoring"""
        try:
            client = await self.poe_manager.get_client()
            return await client.get_chat_history()
        except Exception as e:
            logger.error(f"Error getting chat history: {str(e)}")
            return []


@app.route("/", methods=["GET", "POST"])
async def index():
    """Modified route handler with team-based filtering"""
    response_data = {
        "schedule": None,
        "prediction_result": None,
        "start_message": None,
        "date_message": None,
        "team_message": None,
        "error_message": None,
        "teams": None,
        "current_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "filter_type": "date",  # Default filter type
    }
    

    if request.method == "POST":
        try:
            if "start" in request.form:
                response_data[
                    "start_message"
                ] = "Selamat datang! Silakan pilih metode pencarian jadwal:"
                teams_data = get_nba_teams()
                if teams_data and "response" in teams_data:
                    response_data["teams"] = teams_data["response"]

            elif "filter_type" in request.form:
                response_data["filter_type"] = request.form.get("filter_type")
                teams_data = get_nba_teams()
                if teams_data and "response" in teams_data:
                    response_data["teams"] = teams_data["response"]
                if response_data["filter_type"] == "team":
                    response_data["team_message"] = "Pilih tim untuk melihat jadwal:"
                else:
                    response_data[
                        "date_message"
                    ] = "Masukkan tanggal untuk melihat jadwal:"

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

            elif "team_id" in request.form:
                team_id = request.form.get("team_id")
                if team_id:
                    teams_data = get_nba_teams()
                    team_name = next(
                        (
                            team["name"]
                            for team in teams_data["response"]
                            if str(team["id"]) == team_id
                        ),
                        None,
                    )
                    response_data[
                        "team_message"
                    ] = f"Jadwal pertandingan untuk {team_name}:"
                    response_data["schedule"] = get_team_schedule(team_id)
                    if not response_data["schedule"] or not response_data[
                        "schedule"
                    ].get("response"):
                        response_data[
                            "error_message"
                        ] = "Tidak ada pertandingan untuk tim ini atau terjadi kesalahan mengambil data."

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
        poe_manager = PoeClientManager()
        client = await poe_manager.get_client()

        return (
            jsonify(
                {
                    "status": "healthy",
                    "poe_api": "connected",
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


@app.route("/manifest.json")
def serve_manifest():
    return send_file("manifest.json", mimetype="application/manifest+json")

@app.route("/api-info")
def get_token():
    return jsonify({"data": user_client.get_settings()})
    


if __name__ == "__main__":
    # Run with Hypercorn for async support
    import hypercorn.asyncio
    import hypercorn.config

    config = hypercorn.config.Config()
    config.bind = ["localhost:5000"]
    asyncio.run(hypercorn.asyncio.serve(app, config))
