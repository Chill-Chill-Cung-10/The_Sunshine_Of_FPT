from typing import Dict, List, Optional, Union
import requests
import json

with open('api-keys.json','r', encoding='utf-8') as f:
    api_keys = json.load(f)
llm_embeddings = api_keys[0]
llm_small = api_keys[1]
llm_large = api_keys[2]

vnpt_small_headers = {
    'Authorization': llm_small['authorization'],
    'Token-id': llm_small['tokenId'],
    'Token-key': llm_small['tokenKey'],
    'Content-Type':'application/json'
}
vnpt_large_headers = {
    'Authorization': llm_large['authorization'],
    'Token-id': llm_large['tokenId'],
    'Token-key': llm_large['tokenKey'],
    'Content-Type':'application/json'
}
vnpt_embedding_headers = {
    'Authorization': llm_embeddings['authorization'],
    'Token-id': llm_embeddings['tokenId'],
    'Token-key': llm_embeddings['tokenKey'],
    'Content-Type':'application/json'

}
models = {
    "small":"https://api.idg.vnpt.vn/data-service/v1/chat/completions/vnptai-hackathon-small",
    "large":"https://api.idg.vnpt.vn/data-service/v1/chat/completions/vnptai-hackathon-large",
    "embedding":"https://api.idg.vnpt.vn/data-service/vnptai-hackathon-embedding"
}
def show_body_sml(
    messages: List[Dict[str, str]], 
    model: str = 'vnptai_hackathon_small',
    temperature: float = 0.7, 
    top_p: float = 0.9, 
    top_k: Optional[int] = None,
    n: int = 1, 
    stop: Optional[Union[str, List[str]]] = None,
    max_completion_tokens: int = 4096,
    presence_penalty: Optional[float] = None,
    frequency_penalty: Optional[float] = None,
    response_format: Optional[object] = None,
    seed: Optional[int] = None,
    tools: Optional[List] = None,
    tool_choice: Optional[Union[str, object]] = None,
    logprobs: Optional[bool] = None,
    top_logprobs: Optional[int] = None
) -> Dict:
    """
    Tạo body JSON cho LLM API (Small hoặc Large).

    Parameters:
    - messages (List[Dict]): Danh sách tin nhắn với format [{"role": "user/assistant", "content": "..."}]
    - model (str): Tên mô hình ('vnptai_hackathon_small' hoặc 'vnptai_hackathon_large')
    - temperature (float): Độ ngẫu nhiên (0.0-1.0), default 0.7
    - top_p (float): Nucleus sampling (0.0-1.0), default 0.9
    - top_k (int): Top-k sampling, optional
    - stop (String/List): Khi mô hình sinh ra chuỗi này, nó sẽ lập tức dừng việc tạo văn bản. Hữu ích để kiểm soát cấu trúc output hoặc ngăn mô hình nói quá dài
    - n (int): Số lượng câu trả lời, default 1
    - max_completion_tokens (int): Giới hạn token đầu ra, default 4096
    - presence_penalty (float): [-2.0, 2.0] => Phạt các token dựa trên việc chúng đã xuất hiện trong văn bản hay chưa
    - frequency_penalty (float): [-2.0, 2.0] => Phạt các token dựa trên tần suất xuất hiện của chúng. Giá trị dương làm giảm khả năng mô hình lặp lại nguyên văn một câu
    - response_format (object): Chỉ định định dạng đầu ra, dành cho ứng dụng yêu cầu cấu trúc dữ liệu nghiêm
    - seed (int): Hộ trợ tính năng "reproducible outputs"
    - tools (list): Danh sách functiosn mà mô hình có thể gọi
    - tool_choice (string/object): kiểm soắt việc mô hình có bắt buộc phải gọi tool hay không, auto để mô hình tự quyết định, none để ép trả về text, hoặc định tên hàm cụ thể để ép gọi hàm đó
    - logprobs (bool): Nếu true, trả về thông tin log probabilities của các token đuầ ra, hữu ích để phân tích độ tự tin của mô hình đối với câu trả lời
    - top_logprobs (int): Số lượng token có xác suất cao nhất cần trả về tại mỗi vị trí logprobs (0-20)
    Returns:
    - Dict: Body JSON để gửi request
    """
    json_data = {
        'model': model,
        'messages': messages,
        'temperature': temperature,
        'top_p': top_p,
        'n': n,
        'max_completion_tokens': max_completion_tokens,
    }
    
    # Chỉ thêm top_k nếu được cung cấp
    if top_k is not None:
        json_data['top_k'] = top_k
    
    return json_data

def show_body_e(model, input_text, encoding_format='float'):
    """
    Tạo body JSON cho Embedding API.

    Parameters:
    - model (str): Tên mô hình ('vnptai_hackathon_embedding').
    - input_text (str): Văn bản đầu vào cần vector hoá.
    - encoding_format (str): Định dạng mã hoá (ví dụ: 'float' hoặc 'base64').
    """
    json_data = {
        'model': model, # vnptai_hackathon_embedding 
        'input': input_text, # Văn bản đầu vào 
        'encoding_format': encoding_format # Định dạng mã hoá 
    }
    return json_data
def call_api(models, headers, body_data):
    response = requests.post(models, headers=headers, json=body_data)
    return response.json()

if __name__ == "__main__":
    json_data = { 
        'model': 'vnptai_hackathon_embedding', 
        'input': 'Xin chào, mình là VNPT AI.', 
        'encoding_format': 'float', 
    }
    json_data_1 = { 
        'model': 'vnptai_hackathon_large', 
        'messages': [ 
            { 
                'role': 'user', 
                'content': 'Hi, VNPT AI.', 
            }, 
        ], 
        'temperature': 1.0, 
        'top_p': 1.0, 
        'top_k': 20, 
        'n': 1, 
        'max_completion_tokens': 10, 
    } 
    json_data_2 = { 
        'model': 'vnptai_hackathon_small', 
        'messages': [ 
        { 
            'role': 'user', 
            'content': 'Hi, VNPT AI.', 
        }, 
        ], 
        'temperature': 1.0, 
        'top_p': 1.0, 
        'top_k': 20, 
        'n': 1, 
        'max_completion_tokens': 10, 
    }
    body_1 = call_api(models['embedding'], vnpt_embedding_headers, json_data)
    body_2 = call_api(models['large'], vnpt_large_headers, json_data_1)
    body_3 = call_api(models['small'], vnpt_small_headers, json_data_2)
    # print(body_1)
    print(body_2)
    # print(body_3)
