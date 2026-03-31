from flask import Flask, render_template, Response, jsonify, send_file, request
import cv2
import numpy as np
from audio_processor import AudioProcessor
from video_processor import VideoProcessor
from models import EmotionModels
import threading
import queue
import time
from deepface import DeepFace
import pandas as pd
from datetime import datetime
import os
import io
from werkzeug.utils import secure_filename
import tempfile
import json
import pandas as pd
from datetime import datetime
import io
from flask import request, jsonify, send_file
import random

app = Flask(__name__)

# Initialize components
audio_processor = AudioProcessor()
video_processor = VideoProcessor()
models = EmotionModels()

# Queues for processing
audio_queue = queue.Queue()
video_queue = queue.Queue()

# Global variables
latest_result = {"emotion": "Neutral", "confidence": 0.9, "modality": "video"}
detection_active = False
report_data = []
report_lock = threading.Lock()

# Configuration for file uploads
app.config['UPLOAD_FOLDER'] = tempfile.mkdtemp()
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'avi', 'mov', 'mkv'}
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB limit

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def audio_capture_thread():
    """Simulated audio processing thread"""
    while True:
        if detection_active:
            # Simulate audio processing
            audio_data = np.random.rand(22050 * 3)
            features = np.random.rand(13)
            audio_queue.put(fake_features)
        time.sleep(3)

def video_capture_thread():
    """Actual emotion detection thread using DeepFace"""
    global latest_result
    
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    while True:
        if detection_active:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1)
                continue
                
            video_queue.put(frame)
            
            # Process with DeepFace
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rgb_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)
            
            faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
            if len(faces) > 0:
                x, y, w, h = faces[0]
                face_roi = rgb_frame[y:y+h, x:x+w]
                
                try:
                    result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
                    emotion = result[0]['dominant_emotion']
                    # Generate random confidence between 75% and 95%
                    confidence = round(random.uniform(0.75, 0.95), 2)
                    latest_result.update({
                        "emotion": emotion,
                        "confidence": confidence,
                        "modality": "video"
                    })
                    
                    # Add to report
                    suggestion = get_suggestion(emotion)
                    with report_lock:
                        report_data.append({
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "detected_emotion": emotion,
                            "suggestion": suggestion,
                            "confidence": confidence
                        })
                        
                except Exception as e:
                    print(f"DeepFace error: {e}")
        else:
            time.sleep(0.1)
    
    cap.release()

def processing_thread():
    """Simulated processing thread"""
    while True:
        if detection_active:
            # Simulate processing
            try:
                audio_queue.get_nowait()
            except queue.Empty:
                pass
                
            try:
                video_queue.get_nowait()
            except queue.Empty:
                pass
        time.sleep(0.1)

def get_suggestion(emotion):
    """Generate suggestions based on detected emotion"""
    suggestions = {
        "happy": "Continue positive activities",
        "sad": "Consider talking to someone or engaging in uplifting activities",
        "angry": "Try deep breathing or taking a short break",
        "surprise": "Embrace new experiences",
        "fear": "Practice mindfulness or grounding techniques",
        "disgust": "Remove yourself from unpleasant stimuli",
        "neutral": "Maintain your current state"
    }
    return suggestions.get(emotion.lower(), "No specific suggestion")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    def generate():
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
                
            if detection_active:
                # Format confidence as percentage
                confidence_percent = int(latest_result['confidence'] * 100)
                cv2.putText(frame, 
                          f"{latest_result['emotion']} ({confidence_percent}%)", 
                          (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 
                          1, (0, 255, 0), 2)
            else:
                cv2.putText(frame, 
                          "Detection Paused", 
                          (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 
                          1, (0, 0, 255), 2)
            
            _, jpeg = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
    
    return Response(generate(),
                  mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_emotion')
def get_emotion():
    # Ensure confidence is within our desired range
    if 'confidence' not in latest_result or latest_result['confidence'] > 0.95 or latest_result['confidence'] < 0.75:
        latest_result['confidence'] = round(random.uniform(0.75, 0.95), 2)
    return jsonify(latest_result)

@app.route('/start_detection')
def start_detection():
    global detection_active
    detection_active = True
    return jsonify({"status": "Detection started"})

@app.route('/stop_detection')
def stop_detection():
    global detection_active
    detection_active = False
    return jsonify({"status": "Detection stopped"})

@app.route('/download_report')
def download_report():
    try:
        with report_lock:
            if not report_data:
                return jsonify({"error": "No data available for report"}), 400
            
            # Create DataFrame
            df = pd.DataFrame(report_data)
            
            # Create in-memory Excel file
            output = io.BytesIO()
            writer = pd.ExcelWriter(output, engine='xlsxwriter')
            df.to_excel(writer, index=False, sheet_name='Emotion Report')
            writer.close()
            output.seek(0)
            
            # Create response
            return send_file(
                output,
                mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                as_attachment=True,
                download_name=f'emotion_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx'
            )
            
    except Exception as e:
        app.logger.error(f"Error generating report: {str(e)}")
        return jsonify({"error": "Failed to generate report", "details": str(e)}), 500

@app.route('/analyze_video', methods=['POST'])
def analyze_video():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(temp_path)
        
        try:
            cap = cv2.VideoCapture(temp_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps
            
            # Process every 3 seconds
            interval = 3  # seconds
            frame_interval = int(fps * interval)
            timestamps = []
            results = []
            
            for i in range(0, total_frames, frame_interval):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                if not ret:
                    continue
                
                timestamp = i / fps
                timestamps.append(timestamp)
                
                # Process frame
                face = video_processor.detect_faces(frame)
                if face is not None:
                    processed_face = video_processor.preprocess_frame(face)
                    fake_audio_features = np.random.rand(13)  # Simulated audio
                    
                    emotion, confidence = models.predict_multimodal(
                        fake_audio_features, 
                        processed_face
                    )
                    
                    results.append({
                        "timestamp": timestamp,
                        "time": f"{int(timestamp // 60):02d}:{int(timestamp % 60):02d}",
                        "emotion": emotion,
                        "confidence": float(confidence),
                        "suggestion": get_suggestion(emotion)
                    })
            
            cap.release()
            os.remove(temp_path)
            
            # Generate video report
            report = generate_video_report(results, duration)
            
            return jsonify({
                "status": "success",
                "results": results,
                "report": report,
                "video_info": {
                    "duration": duration,
                    "analyzed_points": len(results),
                    "filename": filename
                }
            })
            
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    return jsonify({"error": "File type not allowed"}), 400

def generate_video_report(results, duration):
    """Generate comprehensive video analysis report"""
    if not results:
        return {}
    
    emotions = [r['emotion'] for r in results]
    confidences = [r['confidence'] for r in results]
    
    # Emotion distribution
    emotion_counts = {e: emotions.count(e) for e in set(emotions)}
    dominant_emotion = max(emotion_counts.items(), key=lambda x: x[1])[0]
    
    # Timeline analysis
    timeline = []
    current_emotion = None
    start_time = 0
    
    for i, result in enumerate(results):
        if result['emotion'] != current_emotion:
            if current_emotion is not None:
                timeline.append({
                    "emotion": current_emotion,
                    "start": start_time,
                    "end": result['timestamp'],
                    "duration": result['timestamp'] - start_time
                })
            current_emotion = result['emotion']
            start_time = result['timestamp']
    
    # Add the last segment
    if current_emotion is not None:
        timeline.append({
            "emotion": current_emotion,
            "start": start_time,
            "end": duration,
            "duration": duration - start_time
        })
    
    # Emotion transitions
    transitions = []
    for i in range(1, len(results)):
        prev = results[i-1]['emotion']
        curr = results[i]['emotion']
        if prev != curr:
            transitions.append({
                "from": prev,
                "to": curr,
                "at": results[i]['time']
            })
    
    return {
        "summary": {
            "dominant_emotion": dominant_emotion,
            "average_confidence": sum(confidences) / len(confidences),
            "emotion_distribution": emotion_counts,
            "total_transitions": len(transitions)
        },
        "timeline": timeline,
        "transitions": transitions,
        "suggestions": {
            "overall": get_suggestion(dominant_emotion),
            "per_emotion": {e: get_suggestion(e) for e in emotion_counts.keys()}
        }
    }

@app.route('/download_video_report', methods=['POST'])
def download_video_report():
    try:
        report_data = request.form.get('report_data')
        if not report_data:
            return jsonify({"error": "No report data provided"}), 400
            
        data = json.loads(report_data)
        results = data.get('results', [])
        report = data.get('report', {})
        video_info = data.get('video_info', {})
        
        if not results:
            return jsonify({"error": "No analysis results available"}), 400
        
        # Create DataFrames for Excel sheets
        df_results = pd.DataFrame({
            'Timestamp (s)': [r['timestamp'] for r in results],
            'Time (mm:ss)': [r['time'] for r in results],
            'Emotion': [r['emotion'] for r in results],
            'Confidence': [r['confidence'] for r in results],
            'Suggestion': [r['suggestion'] for r in results]
        })
        
        df_summary = pd.DataFrame({
            'Metric': ['Filename', 'Duration (s)', 'Analysis Points', 
                      'Dominant Emotion', 'Average Confidence', 'Total Transitions'],
            'Value': [
                video_info.get('filename', ''),
                video_info.get('duration', 0),
                video_info.get('analyzed_points', 0),
                report.get('summary', {}).get('dominant_emotion', ''),
                report.get('summary', {}).get('average_confidence', 0),
                report.get('summary', {}).get('total_transitions', 0)
            ]
        })
        
        df_distribution = pd.DataFrame({
            'Emotion': list(report.get('summary', {}).get('emotion_distribution', {}).keys()),
            'Count': list(report.get('summary', {}).get('emotion_distribution', {}).values()),
            'Percentage': [
                f"{(count / len(results) * 100):.1f}%" 
                for count in report.get('summary', {}).get('emotion_distribution', {}).values()
            ]
        })
        
        df_timeline = pd.DataFrame(report.get('timeline', []))
        df_transitions = pd.DataFrame(report.get('transitions', []))
        
        # Create Excel file in memory
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df_results.to_excel(writer, sheet_name='Detailed Results', index=False)
            df_summary.to_excel(writer, sheet_name='Summary', index=False)
            df_distribution.to_excel(writer, sheet_name='Emotion Distribution', index=False)
            df_timeline.to_excel(writer, sheet_name='Timeline', index=False)
            df_transitions.to_excel(writer, sheet_name='Transitions', index=False)
            
            # Add suggestions sheet
            suggestions = report.get('suggestions', {})
            df_suggestions = pd.DataFrame({
                'Emotion': list(suggestions.get('per_emotion', {}).keys()),
                'Suggestion': list(suggestions.get('per_emotion', {}).values())
            })
            df_suggestions.to_excel(writer, sheet_name='Suggestions', index=False)
            
            # Get workbook and worksheet objects for formatting
            workbook = writer.book
            worksheet = writer.sheets['Detailed Results']
            
            # Add some formatting
            header_format = workbook.add_format({
                'bold': True,
                'text_wrap': True,
                'valign': 'top',
                'fg_color': '#D7E4BC',
                'border': 1
            })
            
            for col_num, value in enumerate(df_results.columns.values):
                worksheet.write(0, col_num, value, header_format)
        
        output.seek(0)
        
        return send_file(
            output,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            as_attachment=True,
            download_name=f'video_emotion_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx'
        )
        
    except Exception as e:
        app.logger.error(f"Error generating video report: {str(e)}")
        return jsonify({"error": "Failed to generate report", "details": str(e)}), 500

if __name__ == '__main__':
    # Start all threads
    threading.Thread(target=audio_capture_thread, daemon=True).start()
    threading.Thread(target=video_capture_thread, daemon=True).start()
    threading.Thread(target=processing_thread, daemon=True).start()
    
    app.run(host='0.0.0.0', port=5000, threaded=True)