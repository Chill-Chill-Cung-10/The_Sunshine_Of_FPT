from setup import setup_model_and_graph
import json
import sys
import argparse
from pathlib import Path
import pandas as pd
import time

def load_dataset(file_path: str = None, split: str = "val"):
    """
    Load dataset - có thể gọi trước khi tính thời gian
    
    Args:
        file_path: Đường dẫn trực tiếp đến file test (cho hackathon)
        split: Loại split (val/test) nếu không có file_path
    """
    if file_path:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        current_dir = Path(__file__).parent
        data_path = current_dir / ".." / "data" / f"{split}.json"
        if not data_path.exists():
            data_path = current_dir / "data" / f"{split}.json"
        with open(data_path, "r", encoding="utf-8") as f:
            return json.load(f)

def format_message(item: dict) -> str:
    """Format câu hỏi thành message"""
    question_text = item.get("question", "").strip()
    choices = item.get("choices", [])
    
    parts = [question_text]
    for idx, choice in enumerate(choices):
        letter = chr(ord("A") + idx)
        parts.append(f"{letter}. {choice}")
    
    return "\n".join(parts)

def main(test_file: str = None, split: str = "val", message: str = None):
    """
    Main prediction function
    
    Args:
        test_file: Path to test file (for hackathon submission)
        split: Split type if test_file not provided (for local dev)
    """
    graph = setup_model_and_graph()
    if not message:
        if test_file:
            print(f"Loading test data from: {test_file}")
            dataset = load_dataset(file_path=test_file)
        else:
            print(f"Loading data from split: {split}")
            dataset = load_dataset(split=split)
        
        print(f"Loaded {len(dataset)} questions")

        if test_file and test_file.startswith("/code/"):
            # Hackathon mode
            submission_path = Path("/code/submission.csv")
            submission_time_path = Path("/code/submission_time.csv")
        else:
            # Local dev mode
            submission_path = Path(__file__).parent / "submission.csv"
            submission_time_path = Path(__file__).parent / "submission_time.csv"
        
        results_batch = []

        for idx, item in enumerate(dataset, 1):
            question_start_time = time.time()
            qid = item.get("qid")
            message = format_message(item)
            
            print(f"[{idx}/{len(dataset)}] Processing {qid}...")
            
            graph_input = {
                "qid": qid,
                "message": message
            }
            
            try:
                result = graph.invoke(graph_input)
                execution_time = time.time() - question_start_time
                
                answer = result.get("answer", {}).get("answer", "")
                
                results_batch.append({
                    "qid": qid,
                    "answer": answer,
                    "time": execution_time
                })
                
                print(f"{qid} completed in {execution_time:.2f}s")
            except Exception as e:
                execution_time = time.time() - question_start_time
                print(f"{qid} failed: {e}")
                results_batch.append({
                    "qid": qid,
                    "answer": "",
                    "time": execution_time
                })
        
        if results_batch:
            df_results = pd.DataFrame(results_batch)
            
            # Write submission.csv (qid, answer only)
            df_results[["qid", "answer"]].to_csv(submission_path, index=False, encoding='utf-8')
            
            # Write submission_time.csv (qid, answer, time)
            df_results.to_csv(submission_time_path, index=False, encoding='utf-8')
    else:
        graph_input = {
                "qid": "test",
                "message": message
            }
            
        for chunk in graph.stream(graph_input):
            print(chunk)
        

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run prediction on test data')
    parser.add_argument('--test-file', type=str, default=None,
                        help='Path to test file (for hackathon submission)')
    parser.add_argument('--split', type=str, default='val',
                        choices=['val', 'test'],
                        help='Split to use if test-file not provided (default: val)')
    
    args = parser.parse_args()
    message = "Trong một thí nghiệm phân tán hạt, tiết diện vi phân $ \\frac{d\\sigma}{d\\Omega} $ cho sự phân tán vào một góc khối $ d\\Omega $ được cho bởi công thức Rutherford:\n$$\n\\frac{d\\sigma}{d\\Omega} = \\left( \\frac{Z_1 Z_2 e^2}{4 \\pi \\epsilon_0 E} \\right)^2 \\frac{1}{\\sin^4(\\theta/2)}\n$$\nNếu số nguyên tử $ Z_1 $ của hạt tới được gấp đôi trong khi tất cả các tham số khác không đổi, thì tiết diện vi phân tại một góc cố định $ \\theta $ thay đổi theo nhân tử nào? A.2 B.4 C.1/2 D.1/4"
    main(test_file=args.test_file, split=args.split, message=message)