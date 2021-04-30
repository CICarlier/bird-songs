import requests

# API url must include query parameters. See https://www.xeno-canto.org/help/search
url = 'https://www.xeno-canto.org/api/2/recordings?query=cnt:brazil&page=5'
r = requests.get(url, allow_redirects=True)

for result in r.json()['recordings'][:100]:
    f = requests.get(f"http:{result['file']}", allow_redirects=True)
    id = result['id']
    open(f'audio/{id}.mp3', 'wb').write(f.content)