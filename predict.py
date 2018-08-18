# Programed by Enrique Corpa Rios
# Udacity Introduction to AI programming with python
# Final project 17 Aug 2018

import numpy as np
import torch
import argparse
import json

from torchvision import transforms
from PIL import Image

def main():

    model = load_model(in_arg.model)
    model.eval()

    img_path = in_arg.dir

    with torch.no_grad():
        probs, classes = predict(image_path=img_path, model=model,k=in_arg.k)
        real_labels = [model.idx_to_class[x] for x in classes] # output vector index to class
        if in_arg.dict == None:
            label_name = [model.class_to_name[x].title() for x in real_labels]
        else:
            with open(in_arg.dict, 'r') as f:
                cat_to_name = json.load(f)
            label_name = [cat_to_name[x].title() for x in real_labels]

        for i in range(len(probs)):
            print('{:20} with probability: {:.2f}%'.format(label_name[i], probs[i] * 100))
        print('Program terminated')
    return 


def load_model(path):
    checkpoint = torch.load(path, map_location=lambda storage, loc: storage)

    model = checkpoint['pretrained_model']
    model.classifier = checkpoint['classifier']
    model.classifier.load_state_dict(checkpoint['state_dict'])
    model.class_to_name = checkpoint['class_to_name']
    model.class_to_idx = checkpoint['class_to_idx']
    model.idx_to_class = dict([(v, k) for k, v in model.class_to_idx.items()])
    
    model.to(device)
    return model

def get_input_args():

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dir', type=str, default='./flowers/test/1/image_06743.jpg', 
    help='path to an image')

    parser.add_argument('--model', type=str, default='checkpoint_final_v1.pth', 
    help='model checkpoint to load the NN')

    parser.add_argument('--k', type=int, default=1, 
    help='number of top predictions shown to the users')

    parser.add_argument('--cuda', type=bool, default=True, 
    help='Execute code on GPU')

    parser.add_argument('--dict', type=str, default=None, 
    help='Determines the names of the outputs')

    return parser.parse_args()

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    process_trans = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
    processed_image = process_trans(image)
    processed_image = np.asarray(processed_image) # imshow expects a numpy array
    
    return processed_image

def predict(image_path, model, k=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model.eval()
    
    pil_img = Image.open(image_path).convert('RGB')
    processed_img = process_image(pil_img)
    torch_img = torch.from_numpy(processed_img)
    torch_img = torch_img.unsqueeze_(0)
    torch_img = torch_img.float().to(device)  # image is ready to be loaded to the model
    
    output = model(torch_img).exp_().topk(k) # I am using log_softmax, so I must exp beforeand

    classes = output[1].cpu().numpy()   
    probs = output[0].cpu().numpy()
    
    return np.reshape(probs, (k)), np.reshape(classes, (k))

if __name__ == "__main__":

    in_arg = get_input_args()

    if(in_arg.cuda and torch.cuda.is_available()):
        device = 'cuda'
        print('Executing model in GPU')
    elif (not torch.cuda.is_available):
        device = 'cpu'
        print('Your system is not compatible with CUDA')
    else:
        device = 'cpu'
        print('Executing model in CPU')

    main()
    