import requests
class VNPTEmbedding:
    def __init__(self,
                 authorization  :str,
                 tokenKey       :str,
                 tokenId        :str,
                 ):
        self._headers = {
            "Authorization": authorization,
            "Token-id": tokenId,
            "Token-key": tokenKey,
            "Content-Type": "application/json",
        }
    def embed_query(self, texts:list):
        json_data = { 
            'model': 'vnptai_hackathon_embedding', 
            'input': texts, 
            'encoding_format': 'float', 
        }
        response = requests.post('https://api.idg.vnpt.vn/data-service/vnptai-hackathon-embedding', 
                             headers=self._headers, json=json_data)
        
        data = response.json()['data']
        data.sort(key=lambda x: x['index'])
        return [item['embedding'] for item in data]

