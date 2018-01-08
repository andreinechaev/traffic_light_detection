import cv2
import os


def copy(path, name, file):
    cv2.imwrite(path + name, file)


def read_data(path):
    for root, subs, files in os.walk(path):
        if subs:
            for sub in subs:
                read_data(sub)

        for file in files:
            if file.endswith('.jpg') or file.endswith('.png'):
                fp = os.path.join(root, file)
                img = cv2.imread(fp)
                img = cv2.resize(img, (224, 224))
                filename = file.split('/').pop()
                if 'non_tf' in root:
                    copy('resized/none/', filename, img)
                elif 'red' in root:
                    copy('resized/red/', filename, img)
                elif 'yellow' in root:
                    copy('resized/yellow/', filename, img)
                elif 'green' in root:
                    copy('resized/green/', filename, img)


if __name__ == '__main__':
    try:
        os.mkdir('resized')
    except FileExistsError:
        print('Already exists')

    try:
        os.mkdir('resized/red')
    except FileExistsError:
        print('Red Already exists')

    try:
        os.mkdir('resized/yellow')
    except FileExistsError:
        print('Yellow Already exists')

    try:
        os.mkdir('resized/green')
    except FileExistsError:
        print('Green Already exists')

    try:
        os.mkdir('resized/none')
    except FileExistsError:
        print('None Already exists')

    read_data('dataset/simulator')
    read_data('dataset/udacity-sdc')
