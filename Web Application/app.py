import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

app = Flask(__name__, template_folder="templates")

# 모델 불러오기
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = Net()
model.load_state_dict(torch.load("mnist_cnn.pth"))
model.eval()

# 이미지 업로드 경로 설정
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 이미지 업로드 허용 확장자 확인 함수
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# 루트 경로
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # 업로드된 파일 검사
        if 'file' not in request.files:
            return render_template('index.html', result="파일을 선택해주세요.")
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', result="파일을 선택해주세요.")
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # 이미지 전처리
            image = Image.open(filepath).convert('L')
            transform = transforms.Compose([transforms.Resize((28, 28)),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.5,), (0.5,))])
            image = transform(image).unsqueeze(0)
            
            # 모델 추론
            with torch.no_grad():
                output = model(image)
            _, predicted = torch.max(output, 1)
            
            result = f"예측 레이블: {predicted.item()}"
            return render_template('index.html', result=result)
        else:
            return render_template('index.html', result="올바른 이미지 파일을 선택해주세요.")
    return render_template('index.html', result="")

if __name__ == '__main__':
    app.run()
