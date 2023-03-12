import os
import json

import torch
from PIL import Image
import streamlit as st
from torchvision import transforms
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为 SimHei
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

from model import swin_tiny_patch4_window7_224 as create_model

img_path = st.file_uploader("请选择一张图片上传", type=['jpg','png'])

def main():
    # 显示图片
    st.markdown("### 用户上传图片，显示如下: ")
    st.image(img_path, channels="RGB")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    img_size = 224
    data_transform = transforms.Compose(
        [transforms.Resize(int(img_size * 1.14)),
         transforms.CenterCrop(img_size),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # load image

    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path)
    # [N, C, H, W]
    img = data_transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    with open(json_path, "r") as f:
        class_indict = json.load(f)

    # create model
    model = create_model(num_classes=10).to(device)
    # load model weights
    model_weight_path = "./weights/model-9.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()
    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()

    print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
                                                 predict[predict_cla].numpy())
    plt.title(print_res)
    for i in range(len(predict)):
        if str(i) in class_indict:
            print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
                                                      predict[i].numpy()))
        else:
            print("Key not found in class_indict: {}".format(i))
    st.markdown("**请点击按钮开始预测**")
    predict = st.button("类别预测")
    if predict:
        st.image(plt.show())
        st.title("图片中的树木类别是: {}".format(class_indict[str(predict_cla)]))


if img_path  is not None:
    main()
