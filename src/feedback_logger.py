"""
Feedback Logger
Logs user feedback (false positives/negatives) for future retraining.
"""
import json

def log_feedback(sample_id, true_label, predicted_label, feedback, log_path='feedback_log.json'):
    entry = {
        'sample_id': sample_id,
        'true_label': true_label,
        'predicted_label': predicted_label,
        'feedback': feedback
    }
    try:
        with open(log_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        data = []
    data.append(entry)
    with open(log_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Logged feedback for sample {sample_id}.")
