"""
raw_check.py -- print the completely unparsed response for ONE station,
no assumptions about structure. Run this before trusting any further
vetting-script output -- two guesses about what was wrong have both
missed, so the next step is looking at the real response directly
instead of guessing a third time.
"""
import requests

TOKEN = "07276f01747ee257d146e319fa6faff0460e46b3"
UID = 401143

url=f"https://aqicn.org/station/@401143/"
#url = f"https://api.waqi.info/feed/@{UID}/?token={TOKEN}"
print(f"Requesting: {url.replace(TOKEN, '<token>')}")
print()

resp = requests.get(url, timeout=10)
print(f"HTTP status code: {resp.status_code}")
print()
print("Raw response body:")
print(resp.text)