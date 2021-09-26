import numpy as np
from matplotlib import pyplot as plt

def show_plots(file_path, mode):
    '''
    file_path表示8个loss、accs文件所在的目录
    model为step或epoch，表示横坐标
    '''
    if mode == 'step':
        train_losses_step = np.load('train_losses_step.npy', allow_pickle=False)
        train_accs_step = np.load('train_accs_step.npy', allow_pickle=False)

        val_losses_step = np.load('val_losses_step.npy', allow_pickle=False)
        val_accs_step = np.load('val_accs_step.npy', allow_pickle=False)

        plt.subplot(1, 2, 1)
        plt.plot(train_accs_step, label='train')
        plt.plot(val_accs_step, label='val')
        plt.xlabel('Step')
        plt.title('Accuracy')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(train_losses_step, label='train')
        plt.plot(val_losses_step, label='val')
        plt.xlabel('Step')
        plt.title('Loss')
        plt.legend()

        plt.show()

    elif mode == 'epoch':
        train_losses_epoch = np.load('train_losses_epoch.npy', allow_pickle=False)
        train_accs_epoch = np.load('train_accs_epoch.npy', allow_pickle=False)

        val_losses_epoch = np.load('val_losses_epoch.npy', allow_pickle=False)
        val_accs_epoch = np.load('val_accs_epoch.npy', allow_pickle=False)

        plt.subplot(1, 2, 1)
        plt.plot(train_accs_epoch, label='train')
        plt.plot(val_accs_epoch, label='val')
        plt.xlabel('Epoch')
        plt.title('Accuracy')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(train_losses_epoch, label='train')
        plt.plot(val_losses_epoch, label='val')
        plt.xlabel('Epoch')
        plt.title('Loss')
        plt.legend()

        plt.show()

if __name__ == '__main__':
    
    file_path = r'./'
    mode = 'epoch'
    show_plots(file_path, mode)




