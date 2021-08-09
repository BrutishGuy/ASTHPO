import argparse
import sys
import numpy as np
import json
import os
from os.path import isfile, join
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.spatial import distance
import pandas as pd

import torch 
import torchvision.models as models
from torchvision import transforms
#
# Load the image
#
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision.utils import make_grid
from torchvision.transforms import ToTensor
from torchvision import datasets, transforms

from tqdm import tqdm

#
# Create a preprocessing pipeline
#
preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )])



def process_arguments(args):
    parser = argparse.ArgumentParser(description='tSNE on audio')
    parser.add_argument('--images_path', action='store', help='path to directory of images')
    parser.add_argument('--dataset_name', action='store', help='dataset to process')
    parser.add_argument('--output_path', action='store', help='path to where to put output json file')
    parser.add_argument('--prepend_info', action='store', default='', help='a cutsy name to prepend to your files (default is None)')
    parser.add_argument('--num_dimensions', action='store', default=2, help='dimensionality of t-SNE points (default 2)')
    parser.add_argument('--perplexity', action='store', default=30, help='perplexity of t-SNE (default 30)')
    parser.add_argument('--learning_rate', action='store', default=150, help='learning rate of t-SNE (default 150)')
    params = vars(parser.parse_args(args))
    return params

def get_image(path, input_shape):
    img = image.load_img(path, target_size=input_shape)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

def analyze_images_via_pca(images_path, prepend_info, output_path = './'):
    # make feature_extractor
    model = keras.applications.VGG16(weights='imagenet', include_top=True)
    feat_extractor = Model(inputs=model.input, outputs=model.get_layer("fc2").output)
    input_shape = model.input_shape[1:3]
    # get images
    candidate_images = [f for f in os.listdir(images_path) if os.path.splitext(f)[1].lower() in ['.jpg','.png','.jpeg']]
    # analyze images and grab activations
    activations = []
    images = []
    for idx,image_path in enumerate(candidate_images):
        file_path = join(images_path,image_path)
        img = get_image(file_path, input_shape);
        if img is not None:
            print("getting activations for %s %d/%d" % (image_path,idx,len(candidate_images)))
            acts = feat_extractor.predict(img)[0]
            activations.append(acts)
            images.append(image_path)
    # run PCA firt
    # first save the features separately
    features = np.array(activations)
    np.save(output_path + prepend_info + '_PCA_VGG16_features.npy', features)
    print("Running PCA on %d images..." % len(activations))
    pca = PCA(n_components=300)
    pca.fit(features)
    pca_features = pca.transform(features)
    return images, pca_features

def analyze_images(images_path, prepend_info, output_path = './'):
    # make feature_extractor
    model_ft = models.resnet50(pretrained=True)
    ### strip the last layer
    feature_extractor = torch.nn.Sequential(*list(model_ft.children())[:-1])
    ### check this works

    # get images
    #candidate_images = [f for f in os.listdir(images_path) if os.path.splitext(f)[1].lower() in ['.jpg','.png','.jpeg']]
    # analyze images and grab activations
    activations = []
    image_names = []
    
    ### Making datasets ###
    train_ds = ImageFolder(images_path, preprocess)
    batch_size = 1
    t = 1
    ### PyTorch data loaders ###
    train_dl = DataLoader(train_ds, batch_size, shuffle=False, num_workers=1, pin_memory=True)
    for i, (images, labels) in enumerate(tqdm(train_dl, desc="Image Num. {}/{}".format(t, len(train_dl))), 0):
        #file_path = join(images_path,image_path)
        #image_x = Image.open(file_path).convert('RGB')
        # Pass the image for preprocessing and the image preprocessed
        #
        #image_x = preprocess(image_x)
        #
        # Reshape, crop, and normalize the input tensor for feeding into network for evaluation
        #
        #image_x = torch.unsqueeze(image_x, 0)
        outputs = feature_extractor(images)
        outputs = outputs.cpu().detach().numpy().squeeze()
        sample_fname, _ = train_dl.dataset.samples[i]
        print(sample_fname)
        activations.append(outputs)
        image_names.append(sample_fname)
        t += 1
            
    # first save the features separately
    features = np.array(activations)
    np.save(output_path + prepend_info + '_Resnet50_features.npy', features)
    images_names = np.array(image_names)
    np.save(output_path + prepend_info + '_Resnet50_image_names.npy', images_names)
    df_features = pd.DataFrame(data = features)
    df_features['Image'] = image_names
    df_features.to_csv(output_path + prepend_info + '_Resnet50_features_dataframe.csv')
    return image_names, features

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()



def analyze_dataset(dataset_name, prepend_info, output_path = './'):
    if prepend_info is None:
        prepend_info = dataset_name
    # make feature_extractor
    model_ft = models.resnet50(pretrained=True)
    ### strip the last layer
    feature_extractor = torch.nn.Sequential(*list(model_ft.children())[:-1])
    ### check this works

    # get images
    #candidate_images = [f for f in os.listdir(images_path) if os.path.splitext(f)[1].lower() in ['.jpg','.png','.jpeg']]
    # analyze images and grab activations
    activations = []
    image_names = []
    
    
    ### Making datasets ###
    if dataset_name == 'cifar10' or dataset_name == 'cifar10subset':
        all_transforms = transforms.Compose(
            [transforms.Resize(224),
             transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        train_ds = datasets.CIFAR10(root='../../data/cifar10/train/', train=True , download=True, transform=all_transforms)
    else:
        return
    batch_size = 1
    t = 1
    ### PyTorch data loaders ###
    train_dl = DataLoader(train_ds, batch_size, shuffle=False, num_workers=1, pin_memory=True)
        # get some random training images
    #dataiter = iter(train_dl)
    #images, labels = dataiter.next()
    
    # show images
    #imshow(torchvision.utils.make_grid(images))
    for i, (images, labels) in enumerate(tqdm(train_dl, desc="Image Num. {}/{}".format(t, len(train_dl))), 0):
        #file_path = join(images_path,image_path)
        #image_x = Image.open(file_path).convert('RGB')
        # Pass the image for preprocessing and the image preprocessed
        #
        #image_x = preprocess(image_x)
        #
        # Reshape, crop, and normalize the input tensor for feeding into network for evaluation
        #
        #image_x = torch.unsqueeze(image_x, 0)
        outputs = feature_extractor(images)
        outputs = outputs.cpu().detach().numpy().squeeze()

        activations.append(outputs)
        image_names.append(i)
        t += 1
            
    # first save the features separately
    features = np.array(activations)
    np.save(output_path + prepend_info + '_Resnet50_features.npy', features)
    images_names = np.array(image_names)
    np.save(output_path + prepend_info + '_Resnet50_image_names.npy', images_names)
    df_features = pd.DataFrame(data = features)
    df_features['Image'] = image_names
    df_features.to_csv(output_path + prepend_info + '_Resnet50_features_dataframe.csv')
    return image_names, features

def run_tsne(images_path, output_path, prepend_info, tsne_dimensions, tsne_perplexity, tsne_learning_rate):
    images, pca_features = analyze_images(images_path, prepend_info)
    print("Running t-SNE on %d images..." % len(images))
    X = np.array(pca_features)
    tsne = TSNE(n_components=tsne_dimensions, learning_rate=tsne_learning_rate, perplexity=tsne_perplexity, verbose=2).fit_transform(X)
    # save data to json
    data = []
    for i,f in enumerate(images):
        point = [ (tsne[i,k] - np.min(tsne[:,k]))/(np.max(tsne[:,k]) - np.min(tsne[:,k])) for k in range(tsne_dimensions) ]
        data.append({"path":os.path.abspath(join(images_path,images[i])), "point":point})
    with open(output_path, 'w') as outfile:
        json.dump(data, outfile)


if __name__ == '__main__':
    # './Downloads/imagenette/', 'Extracted_Features_', './Downloads/'
    params = process_arguments(sys.argv[1:])
    images_path = params['images_path']
    dataset_name = params['dataset_name']
    
    output_path = params['output_path']
    prepend_info = params['prepend_info']
    if images_path:
        images, features = analyze_images(images_path, prepend_info, output_path)
    else:
        images, features = analyze_dataset(dataset_name, prepend_info, output_path)
        
    #tsne_dimensions = int(params['num_dimensions'])
    #tsne_perplexity = int(params['perplexity'])
    #tsne_learning_rate = int(params['learning_rate'])
    #run_tsne(images_path, output_path, prepend_info, tsne_dimensions, tsne_perplexity, tsne_learning_rate)
    print("Finished saving %s" % output_path)
