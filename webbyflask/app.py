import sys
import os
from flask import Flask, request, jsonify, send_from_directory
import cv2
import dlib
import numpy as np
from flask_cors import CORS, cross_origin  # cross_origin 임포트 추가
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from src.personal_color_analysis.personal_color import analysis

app = Flask(__name__)
CORS(app)  # 앱 전체에 대해 CORS 허용

#-------------------------------------------------#
#----------------personal_color-------------------#
#-------------------------------------------------#

# 퍼스널 컬러 분석 API
@app.route('/analyze', methods=['POST'])
@cross_origin()
def analyze_personal_color():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    file_path = os.path.join("uploads", file.filename)
    file.save(file_path)

    try:
        # 퍼스널 컬러 분석 실행
        color_type = analysis(file_path)
        os.remove(file_path)  # 처리 후 파일 삭제
        print(f"분석 완료: {color_type}")  # 결과를 콘솔에 출력
        return jsonify({'colorType': color_type})  # 결과를 클라이언트로 JSON 형태로 반환
    except Exception as e:
        return jsonify({'error': str(e)}), 500

#-------------------------------------------------#
#----------------- LIP_Makeup --------------------#
#-------------------------------------------------#

# 얼굴 검출기와 랜드마크 모델 불러오기 (립 메이크업 기능)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(os.path.abspath("shape_predictor_68_face_landmarks.dat"))

def apply_lipstick(image_path, color):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    
    for face in faces:
        landmarks = predictor(gray, face)
        lip_points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(48, 61)]

        mask = np.zeros_like(gray)
        points = np.array(lip_points, dtype=np.int32)
        cv2.fillPoly(mask, [points], 255)

        # 색상 처리 (색상 값이 '#RRGGBB' 형태일 때 처리)
        if color.startswith('#') and len(color) == 7:
            try:
                # '#RRGGBB' 색상을 (B, G, R) 순서의 RGB로 변환
                lip_color = tuple(int(color[i:i+2], 16) for i in (5, 3, 1))  # BGR 순서
            except ValueError:
                raise ValueError(f"Invalid color format: {color}")
        else:
            raise ValueError(f"Invalid color format: {color}")

        # 입술 색상 적용
        colored_lips = np.copy(image)
        colored_lips[mask == 255] = lip_color

        alpha = 0.3
        blended = cv2.addWeighted(image, 1 - alpha, colored_lips, alpha, 0)

        result_path = "static/result.jpg"
        cv2.imwrite(result_path, blended)

        return result_path

# 립 메이크업 이미지 업로드 및 처리 API
@app.route('/upload', methods=['POST'])
@cross_origin()
def upload_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    color = request.form.get('color', '#FF6F61')  # 기본값으로 립스틱 색상 설정
    file_path = os.path.join("uploads", file.filename)
    file.save(file_path)

    try:
        result_image_path = apply_lipstick(file_path, color)
        return jsonify({'result': f'/static/{os.path.basename(result_image_path)}'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# 정적 파일 제공
@app.route('/static/<path:filename>')
@cross_origin()  # CORS 허용
def serve_static(filename):
    return send_from_directory('static', filename)

# 기본 라우트
@app.route('/')
@cross_origin()  # CORS 허용
def index():
    return 'Welcome to the Personal Color API!'

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    if not os.path.exists('static'):
        os.makedirs('static')
    app.run(debug=True)