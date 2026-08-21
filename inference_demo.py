import runpy, torch, random
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms

# 加载 FDS 模块（无扩展名，用 runpy）
fds = runpy.run_path("FDS")
config = fds["Config"]()

# 加载训练好的模型
model = fds["FallDetectionModel"]()
ckpt = torch.load("checkpoints/fall_detection_model.pth", map_location=config.DEVICE)
model.load_state_dict(ckpt["model_state_dict"])
model.to(config.DEVICE)
model.eval()
print("模型加载成功 | 训练时验证准确率: %.2f%% | 设备: %s" % (ckpt.get("val_acc", 0), config.DEVICE))

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 从验证集随机抽10张（5正常+5摔倒）
import xml.etree.ElementTree as ET
def get_label(xml_path):
    try:
        for obj in ET.parse(xml_path).getroot().findall('object'):
            n = obj.find('name')
            if n is not None and n.text and n.text.strip().lower() in ('down','fall','fallen'):
                return 1
    except Exception:
        pass
    return 0

normal, fall = [], []
for x in sorted(Path("dataset/val/images").glob("*.xml")):
    if get_label(x) == 1:
        fall.append(x)
    else:
        normal.append(x)

random.seed(42)
samples = random.sample(normal, 5) + random.sample(fall, 5)
random.shuffle(samples)

correct = 0
print("\n" + "=" * 68)
print("推理演示（验证集 5 正常 + 5 摔倒）")
print("=" * 68)
for xml in samples:
    img_path = xml.with_suffix(".jpg")
    true_label = get_label(xml)
    image = Image.open(img_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(config.DEVICE)
    with torch.no_grad():
        probs = torch.softmax(model(tensor, torch.zeros(1, 34).to(config.DEVICE)), dim=1)[0]
    pred = int(probs.argmax())
    p_normal, p_fall = probs[0].item(), probs[1].item()
    ok = (pred == true_label)
    correct += ok
    print(f"{'✔' if ok else '✘'} {img_path.name:22s} 真实={'摔倒' if true_label else '正常'} 预测={'摔倒' if pred else '正常'} | 正常 {p_normal:.1%} / 摔倒 {p_fall:.1%}")

print("=" * 68)
print(f"演示准确率: {correct}/{len(samples)} ({100.0*correct/len(samples):.0f}%)")
