"""
weather_service.py — Weather Forecast & Location Services
==========================================================

Provides real-time weather data via Open-Meteo (free, no API key):
  • Geocoding: convert city/district names to coordinates
  • 7-day weather forecast with daily temperatures and precipitation
  • Weather-aware crop suitability summaries for the LLM
"""

from datetime import datetime
from typing import Optional

import requests

from src.config import OPEN_METEO_FORECAST_URL, OPEN_METEO_GEOCODING_URL


# ── Weather Code Mapping ────────────────────────────────────────────

WEATHER_CODES: dict[int, dict] = {
    0:  {"emoji": "☀️", "en": "Clear sky", "hi": "Saaf aasmaan"},
    1:  {"emoji": "🌤️", "en": "Mainly clear", "hi": "Zyaadatar saaf"},
    2:  {"emoji": "⛅", "en": "Partly cloudy", "hi": "Aadha badli"},
    3:  {"emoji": "☁️", "en": "Overcast", "hi": "Poora badli"},
    45: {"emoji": "🌫️", "en": "Fog", "hi": "Kohra"},
    48: {"emoji": "🌫️", "en": "Depositing rime fog", "hi": "Geela kohra"},
    51: {"emoji": "🌦️", "en": "Light drizzle", "hi": "Halki boondi"},
    53: {"emoji": "🌦️", "en": "Moderate drizzle", "hi": "Boondi"},
    55: {"emoji": "🌦️", "en": "Dense drizzle", "hi": "Tez boondi"},
    61: {"emoji": "🌧️", "en": "Slight rain", "hi": "Halki baarish"},
    63: {"emoji": "🌧️", "en": "Moderate rain", "hi": "Baarish"},
    65: {"emoji": "🌧️", "en": "Heavy rain", "hi": "Tez baarish"},
    71: {"emoji": "🌨️", "en": "Slight snow", "hi": "Halki barf"},
    73: {"emoji": "🌨️", "en": "Moderate snow", "hi": "Barf"},
    75: {"emoji": "🌨️", "en": "Heavy snow", "hi": "Tez barf"},
    80: {"emoji": "🌦️", "en": "Slight rain showers", "hi": "Halki bauchhar"},
    81: {"emoji": "🌧️", "en": "Moderate rain showers", "hi": "Bauchhar"},
    82: {"emoji": "⛈️", "en": "Violent rain showers", "hi": "Tez bauchhar"},
    95: {"emoji": "⛈️", "en": "Thunderstorm", "hi": "Aandhi-toofaan"},
    96: {"emoji": "⛈️", "en": "Thunderstorm with hail", "hi": "Ole ke saath toofaan"},
    99: {"emoji": "⛈️", "en": "Thunderstorm with heavy hail", "hi": "Bade ole ke saath toofaan"},
}


def _get_weather_description(code: int) -> dict:
    """Get weather description for a WMO weather code."""
    return WEATHER_CODES.get(code, {"emoji": "❓", "en": "Unknown", "hi": "Pata nahi"})


# ── Geocoding ───────────────────────────────────────────────────────

def get_coordinates(location_text: str) -> Optional[dict]:
    """
    Convert a location name to coordinates using Open-Meteo geocoding.

    Args:
        location_text: City, district, or area name (e.g., "Karnal", "Ludhiana").

    Returns:
        Dict with keys: latitude, longitude, name, state, country.
        None if location not found.
    """
    try:
        location_text = location_text.strip()
        
        # If it's a 6-digit Indian PIN code, resolve to district name first
        if location_text.isdigit() and len(location_text) == 6:
            try:
                pin_resp = requests.get(f"https://api.postalpincode.in/pincode/{location_text}", timeout=10)
                if pin_resp.status_code == 200:
                    pin_data = pin_resp.json()
                    if pin_data and pin_data[0].get("Status") == "Success":
                        district = pin_data[0]["PostOffice"][0]["District"]
                        print(f"[WEATHER] Resolved PIN {location_text} -> {district}")
                        location_text = district
            except Exception as e:
                print(f"[WEATHER] PIN code resolution error: {e}")

        params = {
            "name": location_text,
            "count": 5,
            "language": "en",
            "format": "json",
        }
        response = requests.get(OPEN_METEO_GEOCODING_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        if "results" not in data or not data["results"]:
            print(f"[WEATHER] No results for location: {location_text}")
            return None

        # Prefer Indian results
        india_results = [r for r in data["results"] if r.get("country_code") == "IN"]
        result = india_results[0] if india_results else data["results"][0]

        location_info = {
            "latitude": result["latitude"],
            "longitude": result["longitude"],
            "name": result.get("name", location_text),
            "state": result.get("admin1", ""),
            "country": result.get("country", "India"),
        }
        safe_name = location_info['name'].encode('ascii', 'ignore').decode('ascii')
        safe_state = location_info['state'].encode('ascii', 'ignore').decode('ascii')
        print(f"[WEATHER] Resolved: {location_text} -> {safe_name}, "
              f"{safe_state} ({location_info['latitude']}, {location_info['longitude']})")
        return location_info

    except Exception as e:
        print(f"[WEATHER] Geocoding error: {repr(e)}")
        return None


# ── Weather Forecast ────────────────────────────────────────────────

def get_weather_forecast(lat: float, lon: float) -> Optional[dict]:
    """
    Fetch 7-day weather forecast from Open-Meteo.

    Args:
        lat: Latitude.
        lon: Longitude.

    Returns:
        Dict with structured forecast data, or None on failure.
    """
    try:
        params = {
            "latitude": lat,
            "longitude": lon,
            "daily": ",".join([
                "temperature_2m_max",
                "temperature_2m_min",
                "precipitation_sum",
                "weathercode",
                "wind_speed_10m_max",
            ]),
            "current": "temperature_2m,relative_humidity_2m,wind_speed_10m",
            "timezone": "Asia/Kolkata",
            "forecast_days": 7,
        }
        response = requests.get(OPEN_METEO_FORECAST_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        daily = data.get("daily", {})
        current = data.get("current", {})

        forecast = {
            "current": {
                "temperature": current.get("temperature_2m"),
                "humidity": current.get("relative_humidity_2m"),
                "wind_speed": current.get("wind_speed_10m"),
            },
            "daily": [],
        }

        dates = daily.get("time", [])
        for i, date_str in enumerate(dates):
            day_data = {
                "date": date_str,
                "day_name": _format_day_name(date_str),
                "temp_max": daily["temperature_2m_max"][i],
                "temp_min": daily["temperature_2m_min"][i],
                "precipitation": daily["precipitation_sum"][i],
                "weather_code": daily["weathercode"][i],
                "weather": _get_weather_description(daily["weathercode"][i]),
                "wind_speed": daily["wind_speed_10m_max"][i],
            }
            forecast["daily"].append(day_data)

        return forecast

    except Exception as e:
        print(f"[WEATHER] Forecast error: {e}")
        return None


def _format_day_name(date_str: str) -> str:
    """Convert 'YYYY-MM-DD' to a short day name in Hinglish."""
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        day_names = {
            0: "Mon (Somvar)",
            1: "Tue (Mangal)",
            2: "Wed (Budh)",
            3: "Thu (Guru)",
            4: "Fri (Shukra)",
            5: "Sat (Shani)",
            6: "Sun (Ravi)",
        }
        return day_names.get(dt.weekday(), dt.strftime("%a"))
    except Exception:
        return date_str


# ── Formatted Output ────────────────────────────────────────────────

def format_weather_card(forecast: dict, location_name: str = "") -> str:
    """
    Format forecast data into a Markdown card for display in Gradio.

    Returns a clean Markdown string with weather icons, temps, and rain data.
    """
    if not forecast:
        return "⚠️ Weather data unavailable."

    lines = []
    header = f"### 🌤️ 7-Day Weather — {location_name}" if location_name else "### 🌤️ 7-Day Weather"
    lines.append(header)
    lines.append("")

    # Current conditions
    curr = forecast.get("current", {})
    if curr.get("temperature") is not None:
        lines.append(f"**Abhi:** {curr['temperature']}°C | "
                     f"Humidity: {curr.get('humidity', '—')}% | "
                     f"Hawa: {curr.get('wind_speed', '—')} km/h")
        lines.append("")

    # Daily forecast table
    lines.append("| Din | Mausam | Max°C | Min°C | Baarish (mm) |")
    lines.append("|-----|--------|-------|-------|--------------|")

    for day in forecast.get("daily", []):
        weather = day["weather"]
        rain = day["precipitation"]
        rain_str = f"{rain:.1f}" if rain > 0 else "—"
        lines.append(
            f"| {day['day_name'][:3]} | "
            f"{weather['emoji']} {weather['hi']} | "
            f"{day['temp_max']:.0f} | "
            f"{day['temp_min']:.0f} | "
            f"{rain_str} |"
        )

    return "\n".join(lines)


def get_weather_summary_for_llm(forecast: dict, location_name: str = "") -> str:
    """
    Generate a concise text summary of the forecast for the LLM to reason about.

    This is injected into the session state so the LLM can make weather-aware
    crop recommendations.
    """
    if not forecast:
        return ""

    daily = forecast.get("daily", [])
    if not daily:
        return ""

    # Calculate aggregates
    temps_max = [d["temp_max"] for d in daily]
    temps_min = [d["temp_min"] for d in daily]
    total_rain = sum(d["precipitation"] for d in daily)
    rainy_days = sum(1 for d in daily if d["precipitation"] > 1.0)

    avg_max = sum(temps_max) / len(temps_max)
    avg_min = sum(temps_min) / len(temps_min)

    # Determine season from date
    now = datetime.now()
    month = now.month
    if month in (11, 12, 1, 2):
        season = "Rabi (winter)"
    elif month in (3, 4, 5):
        season = "Zaid (summer)"
    elif month in (6, 7, 8, 9, 10):
        season = "Kharif (monsoon)"
    else:
        season = "Transition"

    summary = (
        f"Location: {location_name}. Season: {season}. "
        f"Next 7 days: Avg max temp {avg_max:.0f}°C, avg min temp {avg_min:.0f}°C. "
        f"Total expected rainfall: {total_rain:.1f}mm over {rainy_days} day(s). "
        f"Temperature range: {min(temps_min):.0f}°C to {max(temps_max):.0f}°C."
    )

    # Add specific weather alerts
    if total_rain > 50:
        summary += " Heavy rainfall expected — avoid sowing water-sensitive crops."
    elif total_rain < 5 and month in (6, 7, 8, 9):
        summary += " Dry spell during monsoon — ensure irrigation availability."

    if max(temps_max) > 40:
        summary += " Heat wave conditions — crops need shade/mulching protection."
    elif min(temps_min) < 5:
        summary += " Frost risk — protect tender seedlings with mulch cover."

    return summary
