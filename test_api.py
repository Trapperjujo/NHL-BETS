import requests
import json

url = "https://api-web.nhle.com/v1/standings/now"
r = requests.get(url)
print(json.dumps(r.json()['standings'][0], indent=2))
