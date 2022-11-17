import torch
import numpy as np
import cv2
from os import path, listdir
from tqdm import tqdm
import time
import copy

class ImageGenerator():
    
    def __init__(self, directory, batch_size=16, shuffle=False, max_dimension=None, ):        
        
        self.directories = directory
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.max_dimension = max_dimension
        self.image_paths = []
        self.class_labels = []
        
        #create list of image file paths and class target labels
        for class_label, class_dir in enumerate(listdir(directory)):
            self.image_paths += [path.join(directory,class_dir,f) for f in listdir(path.join(directory,class_dir))]
            self.class_labels += [class_label for _ in listdir(path.join(directory,class_dir))]

        self.image_paths = np.array(self.image_paths)
        self.class_labels = np.array(self.class_labels)

        #index array for shuffling data
        self.idx = np.arange(len(self.image_paths))
        
    
    def __len__(self):
        
        #number of batches in an epoch
        return int(np.ceil(len(self.image_paths)/float(self.batch_size)))
    
    
    def _load_image(self,img_path):
        
        #load image from path and convert to array
        img = cv2.imread(img_path)
        
        #downsample image if above allowed size if specified
        max_dim = max(img.shape) 
        if self.max_dimension:
            if max_dim > self.max_dimension:
                new_dim = tuple(d*self.max_dimension//max_dim for d in img.shape[1::-1])
                img= cv2.resize(img,new_dim)
            
        #scale image values
        # img = preprocess_input(img)
        return img
    
    
    def _pad_images(self,img,shape):
        #pad images to match largest image in batch
        img = np.pad(img,(*[((shape[i]-img.shape[i])//2,
                            ((shape[i]-img.shape[i])//2) + ((shape[i]-img.shape[i])%2)) for i in range(2)],
                                (0,0)),mode='constant',constant_values=0.)

        #convert image's shape from (W,H,D) to (D,W,H)
        img = np.einsum('ijk->kij',img)
        return img


    def __iter__(self):
        #shuffle index
        if self.shuffle:
            np.random.shuffle(self.idx)
        
        #generate batches
        for batch in range(len(self)):

            batch_image_paths = self.image_paths[self.idx[batch*self.batch_size:(batch+1)*self.batch_size]]

            batch_class_labels = self.class_labels[self.idx[batch*self.batch_size:(batch+1)*self.batch_size]]
            batch_class_labels = torch.Tensor(np.array(batch_class_labels)).to(torch.long)

            batch_images = [self._load_image(image_path) for image_path in batch_image_paths]

            max_resolution = tuple(max([img.shape[i] for img in batch_images]) for i in range(2))

            batch_images = [self._pad_images(image,max_resolution) for image in batch_images]
            batch_images = torch.Tensor(np.array(batch_images))

            yield batch_images, batch_class_labels

def train(num_epochs, model, train_loader, valid_loader, loss_fn, optimizer):
    since = time.time()
    best_model = model
    best_acc = 0.0

    train_losses=[]
    train_accus=[]
    valid_losses = []
    valid_accus = []
    for epoch in range(1,num_epochs+1):
        print('--------------------')
        print(f'Epoch {epoch}/{num_epochs}')

        model.train()
        train_running_loss = 0.0
        train_correct = 0
        train_total = 0

        # Iterate over data.
        for data in tqdm(train_loader):
                
            inputs,labels=data[0],data[1]
            
            # predict classes using images from the training set
            outputs=model(inputs)
            
            # compute the loss based on model output and real labels
            loss=loss_fn(outputs,labels)
            
            #Replaces pow(2.0) with abs() for L1 regularization
            
            l2_lambda = 0.001
            l2_norm = sum(p.pow(2.0).sum()
                        for p in model.parameters())

            loss = loss + l2_lambda * l2_norm

            # zero the parameter gradients
            optimizer.zero_grad()
            # backpropagate the loss
            loss.backward()
            # adjust parameters based on the calculated gradients
            optimizer.step()

            train_running_loss += loss.item()
            
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
                
        train_loss=train_running_loss/len(train_loader)
        train_acc=100.*train_correct/train_total
            
        train_accus.append(train_acc)
        train_losses.append(train_loss)
        print('Train Loss: %.3f | Train Accuracy: %.3f'%(train_loss,train_acc))
        #----------------------------------------------------------------------
        # Evaluate model by validate dataset

        model.eval()
        valid_running_loss=0
        valid_correct=0
        valid_total=0

        with torch.no_grad():
            for data in tqdm(valid_loader):
                images,labels=data[0], data[1]
                
                outputs=model(images)

                loss= loss_fn(outputs,labels)
                valid_running_loss+=loss.item()
                
                _, predicted = outputs.max(1)
                valid_total += labels.size(0)
                valid_correct += predicted.eq(labels).sum().item()
        
        valid_loss=valid_running_loss/len(valid_loader)
        valid_acc=100.*valid_correct/valid_total

        valid_losses.append(valid_loss)
        valid_accus.append(valid_acc)

        print('Valid Loss: %.3f | Accuracy: %.3f'%(valid_loss,valid_acc))

        # update best_model
        if valid_acc > best_acc:
            best_acc = valid_acc
            best_model = copy.deepcopy(model)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    return best_model,train_losses, train_accus, valid_losses, valid_accus

def test(model,testloader, loss_fn):
    model.eval()
    running_loss=0
    correct=0
    total=0
    eval_losses = []
    eval_accus = []
    with torch.no_grad():
        for data in tqdm(testloader):
            images,labels=data[0], data[1]
            
            outputs=model(images)

            loss= loss_fn(outputs,labels)
            running_loss+=loss.item()
            
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    test_loss=running_loss/len(testloader)
    accu=100.*correct/total

    eval_losses.append(test_loss)
    eval_accus.append(accu)

    print('Test Loss: %.3f | Accuracy: %.3f'%(test_loss,accu)) 
    return eval_losses, eval_accus

# def train_model(model, train_loader, valid_loader, loss_fn, optimizer, num_epochs=20):

#     since = time.time()

#     best_model = model
#     best_acc = 0.0

#     for epoch in range(num_epochs):
#         print('Epoch {}/{}'.format(epoch, num_epochs - 1))
#         print('-' * 10)

#         # Each epoch has a training and validation phase
#         for phase in ['train', 'val']:
#             if phase == 'train':
#                 # optimizer = lr_scheduler(optimizer, epoch)
#                 model.train(True)  # Set model to training mode
#             else:
#                 model.train(False)  # Set model to evaluate mode

#             running_loss = 0.0
#             running_corrects = 0

#             # Iterate over data.
#             for data in tqdm(train_loader):
#                 # get the inputs
#                 inputs, labels = data

#                 # wrap them in Variable
#                 # if use_gpu:
#                 #     inputs, labels = Variable(inputs.cuda()), \
#                 #         Variable(labels.cuda())
#                 # else:
#                 #     inputs, labels = Variable(inputs), Variable(labels)

#                 # zero the parameter gradients
#                 optimizer.zero_grad()

#                 # forward
#                 outputs = model(inputs)
#                 _, preds = torch.max(outputs.data, 1)
#                 loss = loss_fn(outputs, labels)

#                 # backward + optimize only if in training phase
#                 if phase == 'train':
#                     loss.backward()
#                     optimizer.step()

#                 # statistics
#                 running_loss += loss.data[0]
#                 running_corrects += torch.sum(preds == labels.data)

#             epoch_loss = running_loss / len(train_loader)
#             epoch_acc = running_corrects / len(train_loader)

#             print('{} Loss: {:.4f} Acc: {:.4f}'.format(
#                 phase, epoch_loss, epoch_acc))

#             # deep copy the model
#             if phase == 'val' and epoch_acc > best_acc:
#                 best_acc = epoch_acc
#                 best_model = copy.deepcopy(model)

#         print()

#     time_elapsed = time.time() - since
#     print('Training complete in {:.0f}m {:.0f}s'.format(
#         time_elapsed // 60, time_elapsed % 60))
#     print('Best val Acc: {:4f}'.format(best_acc))
#     return best_model

# def train(epoch, model, train_generator, optimizer, loss_fn):
#     train_losses=[]
#     train_accu=[]
#     loss_fn=nn.CrossEntropyLoss()
#     optimizer= torch.optim.Adam(model.parameters(),lr=0.001,weight_decay = 0.0001)

#     print('\nEpoch : %d'%epoch)

#     model.train()

#     running_loss=0
#     correct=0
#     total=0

#     for data in tqdm(train_generator):
        
#         inputs,labels=data[0],data[1]
        
#         # predict classes using images from the training set
#         outputs=model(inputs)
        
#         # compute the loss based on model output and real labels
#         loss=loss_fn(outputs,labels)
        
#         #Replaces pow(2.0) with abs() for L1 regularization
        
#         l2_lambda = 0.001
#         l2_norm = sum(p.pow(2.0).sum()
#                     for p in model.parameters())

#         loss = loss + l2_lambda * l2_norm

#         # zero the parameter gradients
#         optimizer.zero_grad()
#         # backpropagate the loss
#         loss.backward()
#         # adjust parameters based on the calculated gradients
#         optimizer.step()

#         running_loss += loss.item()
        
#         _, predicted = outputs.max(1)
#         total += labels.size(0)
#         correct += predicted.eq(labels).sum().item()
        
#     train_loss=running_loss/len(train_generator)
#     accu=100.*correct/total
    
#     train_accu.append(accu)
#     train_losses.append(train_loss)
#     print('Train Loss: %.3f | Train Accuracy: %.3f'%(train_loss,accu))











