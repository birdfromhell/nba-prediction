# app.py

from flask import Flask, render_template, request
import requests
from poe_api_wrapper import AsyncPoeApi
from datetime import datetime
import json

app = Flask(__name__)

def get_nba_schedule(date):
    """
    Get NBA schedule from the API for a specific date
    """
    url = "https://api-nba-v1.p.rapidapi.com/games"
    querystring = {"date": date}
    headers = {
        "x-rapidapi-key": "fe7ac125c5msh94f9c196609b1eep12fb18jsndc6f9e5920c3",
        "x-rapidapi-host": "api-nba-v1.p.rapidapi.com"
    }
    try:
        response = requests.get(url, headers=headers, params=querystring)
        print(f"Schedule API Response Status: {response.status_code}")
        return response.json() if response.status_code == 200 else None
    except Exception as e:
        print(f"Error fetching NBA schedule: {e}")
        return None

def get_match_data(teams, num_results=10):
    """
    Get match data and process the API response to extract relevant information.
    Returns structured match data and relevant URLs.
    """
    url = "https://real-time-web-search.p.rapidapi.com/search"
    querystring = {
        "q": f"{teams} NBA match statistics head to head",
        "limit": str(num_results)
    }
    
    headers = {
        "x-rapidapi-key": "fe7ac125c5msh94f9c196609b1eep12fb18jsndc6f9e5920c3",
        "x-rapidapi-host": "real-time-web-search.p.rapidapi.com"
    }

    try:
        response = requests.get(url, headers=headers, params=querystring, timeout=10)
        print(f"Search API Response Status: {response.status_code}")
        print(f"Search Query: {teams}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Raw API Response: {json.dumps(data, indent=2)}")
            
            if data.get("status") == "OK" and "data" in data:
                # Prioritize certain domains for better statistics
                priority_domains = {
                    'nba.com': 5,
                    'espn.com': 4,
                    'basketball-reference.com': 4,
                    'aiscore.com': 3,
                    'landofbasketball.com': 3,
                    'sportsbettingdime.com': 2,
                    'flashscore.com': 2,
                    'sofascrore.com': 2,
                }
                
                # Extract and structure the information
                match_info = []
                for item in data['data']:
                    domain = item.get('domain', '')
                    priority = 0
                    
                    # Assign priority based on domain
                    for key_domain, value in priority_domains.items():
                        if key_domain in domain:
                            priority = value
                            break
                    
                    # Only include items with snippets
                    if item.get('snippet'):
                        match_info.append({
                            'title': item.get('title', '').strip(),
                            'snippet': item.get('snippet', '').strip(),
                            'url': item.get('url', ''),
                            'domain': domain,
                            'priority': priority
                        })
                
                # Sort by priority (higher first)
                match_info.sort(key=lambda x: x['priority'], reverse=True)
                
                # Create context string from the collected data
                context_data = f"Data Pertandingan {teams}:\n\n"
                
                # Add structured information from snippets
                for info in match_info:
                    if info['priority'] > 0:  # Only include from priority domains
                        context_data += f"Sumber: {info['domain']}\n"
                        context_data += f"Judul: {info['title']}\n"
                        context_data += f"Informasi: {info['snippet']}\n"
                        context_data += f"URL: {info['url']}\n\n"
                
                # Add summary statistics if available
                summary_stats = extract_summary_stats(match_info)
                if summary_stats:
                    context_data += "\nStatistik Ringkas:\n"
                    context_data += summary_stats
                
                return context_data
            else:
                print("API response tidak sesuai format yang diharapkan")
                return f"Tidak dapat menemukan data spesifik untuk pertandingan {teams}. Menggunakan analisis umum."
                
        else:
            print(f"API request gagal dengan status code: {response.status_code}")
            return f"Gagal mendapatkan data untuk pertandingan {teams}. Menggunakan analisis umum."
            
    except Exception as e:
        print(f"Error saat mengambil data pertandingan: {e}")
        return f"Terjadi kesalahan saat mengambil data {teams}. Menggunakan analisis umum."

def extract_summary_stats(match_info):
    """
    Extract and summarize statistics from match information
    """
    try:
        stats = []
        for info in match_info:
            snippet = info['snippet'].lower()
            
            # Extract scoring statistics
            if 'ppg' in snippet:
                ppg_stats = snippet.split('ppg')
                for stat in ppg_stats:
                    if any(char.isdigit() for char in stat):
                        stats.append(f"PPG: {stat.strip()}")
            
            # Extract win/loss records
            if 'won' in snippet and any(char.isdigit() for char in snippet):
                win_stats = snippet.split('won')
                for stat in win_stats:
                    if any(char.isdigit() for char in stat):
                        stats.append(f"Wins: {stat.strip()}")
        
        if stats:
            return "\n".join(stats)
        return ""
        
    except Exception as e:
        print(f"Error extracting summary stats: {e}")
        return ""

@app.route("/", methods=["GET", "POST"])
async def index():
    schedule = None
    prediction_result = None
    start_message = None
    date_message = None
    error_message = None

    if request.method == "POST":
        try:
            if 'start' in request.form:
                start_message = "Selamat datang! Silakan masukkan tanggal (YYYY-MM-DD) untuk melihat jadwal pertandingan NBA."
            
            elif 'date' in request.form:
                date = request.form.get("date")
                if date:
                    date_message = f"Jadwal pertandingan NBA tanggal {date}:"
                    schedule = get_nba_schedule(date)
                    if not schedule or not schedule.get('response'):
                        error_message = "Tidak ada pertandingan pada tanggal tersebut atau terjadi kesalahan mengambil data."
                else:
                    error_message = "Mohon masukkan tanggal yang valid."
            
            elif 'match_choice' in request.form:
                user_choice = request.form.get("match_choice")
                print(f"Processing match: {user_choice}")
                
                # Get processed match data
                context_data = get_match_data(user_choice)
                print(f"Context data generated: {bool(context_data)}")
                
                # Create detailed prompt
                prompt = f"""Berikan analisis prediksi untuk pertandingan NBA {user_choice} berdasarkan data berikut:

{context_data}

Mohon berikan analisis dalam format berikut:
1. Head-to-Head Record
2. Performa Terkini Kedua Tim
3. Statistik Kunci
4. Faktor Penting (injuries, home/away, dll)
5. Prediksi dengan Justifikasi

Berikan analisis yang objektif dan detail berdasarkan data di atas.
"""
                print("Sending prompt to POE...")
                
                # Initialize Poe API client
                tokens = {
                    'p-b': 'Xim1r52Px8L0ESP8GawH4w%3D%3D',
                    'p-lat': 'i%2BWCgz%2FdMa9g6MoX4DjdHrSYU6sDYOoT06Hi8XaHBw%3D%3D',
                    'formkey': '6afefe7956afbec62bf474bf2b0bc961f9',
                }
                client = await AsyncPoeApi(tokens=tokens).create()
                
                # Get AI analysis
                complete_response = ""
                async for chunk in client.send_message(bot="gemini_pro_search", message=prompt):
                    complete_response = chunk["response"]
                
                prediction_result = complete_response
                print("Prediction generated successfully")
                
            elif 'user_prompt' in request.form:
                user_prompt = request.form.get("user_prompt")
                if user_prompt:
                    # Handle user questions here
                    tokens = {
                        'p-b': 'IUIM3rd2DL9lIgfr324otg%3D%3D',
                        'p-lat': 'N41dYMp4bd%2FloCfYAKpTDXq3ZSG3CjWtjXm04dP38A%3D%3D',
                        'formkey': '6afefe7956afbec62bf474bf2b0bc961f9',
                    }
                    client = await AsyncPoeApi(tokens=tokens).create()
                    
                    complete_response = ""
                    async for chunk in client.send_message(bot="gemini_pro_search", message=user_prompt):
                        complete_response = chunk["response"]
                    
                    prediction_result = complete_response
                
        except Exception as e:
            error_message = f"Terjadi kesalahan: {str(e)}"
            print(f"Error in route handler: {e}")

    return render_template("index.html", 
                         start_message=start_message, 
                         date_message=date_message, 
                         schedule=schedule, 
                         prediction_result=prediction_result,
                         error_message=error_message,
                         current_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

if __name__ == "__main__":
    app.run(debug=True)