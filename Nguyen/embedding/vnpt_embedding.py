import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

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
        
        # Create session with optimized connection pool
        self._session = requests.Session()
        self._session.headers.update(self._headers)
        
        # Configure HTTPAdapter with larger pool size for concurrent requests
        # pool_connections: số connection pools
        # pool_maxsize: số connections tối đa trong mỗi pool
        # pool_block: block khi pool đầy thay vì raise exception
        adapter = HTTPAdapter(
            pool_connections=10,
            pool_maxsize=20,  # Tăng lên 20 để handle 10 workers + buffer
            pool_block=True
        )
        
        # Mount adapter cho cả http và https
        self._session.mount('http://', adapter)
        self._session.mount('https://', adapter)
        
    def embed_query(self, texts:list):
        json_data = { 
            'model': 'vnptai_hackathon_embedding', 
            'input': texts, 
            'encoding_format': 'float', 
        }
        response = self._session.post('https://api.idg.vnpt.vn/data-service/vnptai-hackathon-embedding', 
                             json=json_data)
        
        data = response.json()['data']
        data.sort(key=lambda x: x['index'])
        return [item['embedding'] for item in data]

