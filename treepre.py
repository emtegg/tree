import zipfile
import torch
import streamlit as st
from torchvision import transforms

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

    img = img_path
    # [N, C, H, W]
    img = data_transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    # read class_indict
    class_indict = {
          "0": "\u5973\u8d1e",
          "1": "\u6768\u6811",
          "2": "\u67ab\u6811",
          "3": "\u67f3\u6811",
          "4": "\u6842\u82b1\u6811",
          "5": "\u68a7\u6850\u6811",
          "6": "\u6a1f\u6811\uff08\u9999\u6a1f\uff09",
          "7": "\u6d77\u6850\u6811",
          "8": "\u6d77\u68e0\u6811",
          "9": "\u94f6\u674f\u6811"
                    }

    # create model
    model = create_model(num_classes=10).to(device)
    # load model weights
    with zipfile.ZipFile('./weights/model-9.zip', 'r') as zip_ref:
        zip_ref.extractall('extracted')
    model_weight_path = "./extracted/model-9.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()
    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()

    for i in range(len(predict)):
        if str(i) in class_indict:
            print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
                                                      predict[i].numpy()))
        else:
            print("Key not found in class_indict: {}".format(i))
    st.markdown("**请点击按钮开始预测**")
    predict = st.button("类别预测")
    if predict:
        st.title("图片中的树木类别是: {}".format(class_indict[str(predict_cla)]))


if img_path  is not None:
    main()
