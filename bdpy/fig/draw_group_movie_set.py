import os
import numpy as np
import cv2
import copy
from PIL import Image
#from .draw_group_image_set import draw_group_image_set
from . import draw_group_image_set

def load_video(file_path,ret_type='float32', bgr=False, height=224,width=224, interpolation=cv2.INTER_LINEAR):
    """
    ret_type select return as float32 float64, or uint8
    ret_type allows ['float', 'float64']
    """
    cap = cv2.VideoCapture(file_path)
    vid = []
  
    while True:
        
        ret, img = cap.read()
        
        if not ret:
            break
        
        # torchvision model allow RGB image , range of [0,1] and normalised 
        if bgr:
            img = cv2.resize(img, (height, width), interpolation=interpolation)
        else:
            img = cv2.resize(cv2.cvtColor(img , cv2.COLOR_BGR2RGB), (height, width), interpolation=interpolation)
       

        if ret_type == 'float32':
            img = img.astype(np.float32)
        elif ret_type== 'float':
            img = img.astype(np.float)
            
        vid.append(img)

    return np.array(vid)

def save_video(vid, save_name, save_intermidiate_path, bgr=False, fr_rate = 30, codec=None):
    fr, height, width, ch = vid.shape
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    if codec:
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    writer = cv2.VideoWriter(os.path.join(save_intermidiate_path, save_name), fourcc, fr_rate, (width, height))
    for j in range(fr):
        frame = vid[j]
        if bgr == False:
            frame = frame[..., [2, 1, 0]]

        writer.write(frame.astype(np.uint8))
    writer.release()


FONT_PATH ="/usr/share/fonts/truetype/freefont/FreeSans.ttf"
def draw_group_movie_set(movie_condition_list, save_name, save_dir, save_frame_dir=None, insert_first_num=5, insert_last_num=0, fr_rate=16, background_color = (255, 255, 255), 
                    image_size = (160, 160), image_margin = (1, 1, 0, 0), group_margin = (20, 0, 120, 0), max_column_size = 13, 
                    title_fontsize = 20, title_top_padding = 70, title_left_padding = 15, font_family_path = FONT_PATH,
                    id_show = False, id_fontcolor = "black", id_fontsize = 18, image_id_list = []):

    """
    condition_list : list
        Each condition is a dictionary-type object that contains the following information:
        ```
            condition = {
                "title" : string, # Title name
                "title_fontcolor" :  string or list,   # HTML color name or RGB value list 
                "image_filepath_list": array, # movie array list : [Note]movie filepath (e.g. .avi) is not supported yet
            }
        ```
    background_color : list or tuple
        RGB value list like [Red, Green, Blue].
    image_size: list or tuple
        The image size like [Height, Width].
    image_margin: list or tuple
        The margin of an image like [Top, Right, Bottom, Left].
    group_margin : list or tuple
        The margin of the multiple row images as [Top, Right, Bottom, Left].
    max_column_size : int
        Maximum number of images arranged horizontally.
    title_fontsize : int
        The font size of titles.
    title_top_padding : 
        Top margin of the title letter.
    title_left_padding : 
        Left margin of the title letter.
    font_family_path : string or None
        Font file path.
    id_show : bool
        Specifying whether to display id name.
    id_fontcolor : list or tuple
        Font color of id name.
    id_fontsize : int
        Font size of id name.
    image_id_list : list
        List of id names.
        This list is required when `id_show` is True.
    """

    image_pram_dict = {
        "background_color": background_color,
        "image_size": image_size,
        "image_margin": image_margin,
        "group_margin": group_margin,
        "max_column_size": max_column_size,
        "title_fontsize": title_fontsize,
        "title_top_padding": title_top_padding,
        "title_left_padding": title_left_padding,
        "font_family_path": font_family_path,
        "id_show": id_show,
        "id_fontcolor": id_fontcolor,
        "id_fontsize": id_fontsize,
        "image_id_list": image_id_list
    }



    num_fr = len(movie_condition_list[0]['image_filepath_list'][0]) + insert_first_num + insert_last_num

    for fr in range(num_fr):
            
        save_frame_name = f'{fr}.jpg'
        if save_frame_dir == None:
            save_frame_dir = os.path.join(save_dir, 'frame')
        os.makedirs(save_frame_dir, exist_ok=True)

        if fr < insert_first_num:

            show_frame_list = create_frame_condition_list(movie_condition_list, 0)
        elif fr > num_fr + insert_first_num-1:
            show_frame_list = create_frame_condition_list(movie_condition_list, -1)
        else:
            show_frame_list = create_frame_condition_list(movie_condition_list, fr - insert_first_num)


        #create image 
        frame = draw_group_image_set(show_frame_list, **image_pram_dict)
        #save image
        frame.save(os.path.join(save_frame_dir,save_frame_name))
    #save video from their frame
    
    merge_list = []
    for t in range(num_fr):
        file_path = str(t)+ '.jpg'
        ee = Image.open(os.path.join(save_frame_dir, file_path))
        merge_list.append(np.array(ee))

    merge_list = np.array(merge_list)


    save_video(merge_list,save_name, save_dir, bgr=False, fr_rate=fr_rate)
    print(save_name +'done')




def create_frame_condition_list(movie_condition_list, fr):
    return_list = copy.deepcopy(movie_condition_list)#.copy()
    for j, condition_dict in enumerate(movie_condition_list):
        video_array = condition_dict['image_filepath_list']
        #extract 'fr' th frame for each movie
        frame_array = [video_array[i][fr].astype(np.uint8) for i in range(len(video_array))]
        return_list[j]['image_filepath_list'] = frame_array

    return return_list
