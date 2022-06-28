'''
    按照label给的性别列表，将原始数据分成female和male两个文件夹
    被根目录/scripts/init_data.sh脚本调用
'''


import os
import shutil
from tqdm import tqdm

# TODO: 把这坨东西丢到config里面
MALE_LABEL_PATH = './label/male_names.txt'
FEMALE_LABEL_PATH = './label/female_names.txt'
DATA_ROOT = './lfw'
PROCESS_ROOT = './processed'

class ProcessData:
    '''
    Args:
        - male_file: male名字列表文件的路径
        - female_file: female名字列表文件的路径
        - data: 原始数据的路径
        - target: 处理好之后的数据的存储路径
    
    功能: 
    '''
    def __init__(self, male_file, female_file, data, target):

        self.male_file = male_file
        self.female_file = female_file
        self.data = data
        self.target = target

    def check_path(self, label_file):
        if not os.path.isdir(self.target):
            os.mkdir(self.target)
            print('Target path not exists, create it!')
        target_path = os.path.join(self.target, self.get_gender(label_file))
        if not os.path.isdir(target_path):
            os.mkdir(target_path)
            print('Target path not exists, create it!')

    def process_img(self, label_file):
        self.check_path(label_file)
        with open(label_file) as img_file:
            img_list = img_file.readlines()
            pred_name = 'Init Name'
            for img_idx in tqdm(range(len(img_list)), ncols=70):
                img = img_list[img_idx]
                person_name = self.name_format(img)
                if person_name and person_name != pred_name:
                    # new person
                    self.move_img(person_name, self.get_gender(label_file))
                    pred_name = person_name
    

    def name_format(self, name):

        processed_name = ''
        for char in name:
            if char >= '0' and char <= '9':
                break
            else:
                processed_name += char
        return processed_name[:-1]


    def move_img(self, person_name, gender):

        person_dir = os.path.join(self.data, person_name)
        target_path = os.path.join(self.target, gender)
        
        for person_img in os.listdir(person_dir):
            # print(person_img)
            source_path = os.path.join(person_dir, person_img)
            target_img = os.path.join(target_path, person_img)
            # print('copy {} -> {}'.format(source_path, target_img))
            try:
                shutil.copy(source_path, target_img)
            except:
                print('Warning!' + source_path + person_name)
            

    def get_gender(self, file_name):

        if file_name == self.female_file:
            return 'female'
        elif file_name == self.male_file:
            return 'male'



def process_data():
    processer = ProcessData(MALE_LABEL_PATH, FEMALE_LABEL_PATH, DATA_ROOT, PROCESS_ROOT)
    print('Process male list...')
    processer.process_img(processer.male_file)
    print('Process female list...')
    processer.process_img(processer.female_file)
    print('done')



if __name__ == '__main__':
    process_data()
    