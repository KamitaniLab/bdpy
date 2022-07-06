import re
import cv2
import PIL
import copy
import yaml
import pathlib
import argparse
import subprocess
import numpy as np
import PIL.ImageDraw
import PIL.ImageFont
from PIL import Image
from tqdm import tqdm
from natsort import os_sorted
from matplotlib import font_manager

def expand2square(img_array, background_color=(0,0,0)):
    height, width, _ = img_array.shape
    if width == height:
        return img_array
    elif width > height:
        return cv2.copyMakeBorder(img_array, (width - height) // 2, (width - height) // 2, 0, 0, cv2.BORDER_CONSTANT, background_color)
    else:
        return cv2.copyMakeBorder(img_array, 0, 0, (height - width) // 2, (height - width) // 2, cv2.BORDER_CONSTANT, background_color)

def cv2pil(image):
    ''' OpenCV型 -> PIL型 '''
    new_image = image.copy()
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGRA2RGBA)
    new_image = Image.fromarray(new_image)
    return new_image

def pil2cv(image):
    ''' PIL型 -> OpenCV型 '''
    new_image = np.array(image, dtype=np.uint8)
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGBA2BGRA)
    return new_image

def draw_group_image_set(images, labels, font_colors,
                         background_color = (255, 255, 255),
                         image_size = (160, 160), image_margin = (1, 1, 0, 0), group_margin = (20, 0, 20, 0), max_column_size = 13,
                         title_fontsize = 20, title_top_padding = 70, title_left_padding = 15, font_family_path = None,
                         id_show = False, id_fontcolor = "black", id_fontsize = 18, image_id_list = []):
    """
    condition_list : list
        Each condition is a dictionary-type object that contains the following information:
        ```
            condition = {
                "title" : string, # Title name
                "title_fontcolor" :  string or list,   # HTML color name or RGB value list
                "image_list": list, # The list of image filepath, ndarray or PIL.Image object.
            }
        ```
        You can also use "image_filepath_list" instead of "image_list".
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

    #------------------------------------
    # Setting
    #------------------------------------

    assert len(images) == len(labels)
    total_image_size = len(images[0])
    column_size = np.min([max_column_size, total_image_size])

    # create canvas
    turn_num = int(np.ceil(total_image_size / float(column_size)))
    nImg_row = len(images) * turn_num
    nImg_col = 1 + column_size # 1 means title column
    size_x = (image_size[0] + image_margin[0] + image_margin[2]) * nImg_row + (group_margin[0] + group_margin[2]) * turn_num
    size_y = (image_size[1] + image_margin[1] + image_margin[3]) * nImg_col + (group_margin[1] + group_margin[3])
    image = np.ones([size_x, size_y, 3])
    for bi, bc in enumerate(background_color):
        image[:, :, bi] = bc

    # font settings
    if font_family_path is None:
        font = font_manager.FontProperties(family='sans-serif', weight='normal')
        font_family_path = font_manager.findfont(font)

    #------------------------------------
    # Draw image
    #------------------------------------
    for cind, (image_list, title) in enumerate(zip(images, labels)):
        for tind in range(total_image_size):
            image_obj = cv2pil(image_list[tind])
            image_obj = image_obj.convert("RGB")

            # Calc image position
            row_index = cind + (tind // column_size) * len(images)
            column_index = 1 + tind % column_size
            turn_index = tind // column_size
            x = image_margin[0] + group_margin[0] + row_index * (image_size[0] + image_margin[0] + image_margin[2])
            x += turn_index * (group_margin[0] + group_margin[2])
            y = image_margin[3] + group_margin[3] + column_index * (image_size[1] + image_margin[1] + image_margin[3])
            image[ x:(x+image_size[0]), y:(y+image_size[1]), : ] = np.array(image_obj)[:,:,:]

    #------------------------------------
    # Prepare for drawing text
    #------------------------------------
    # cast to unsigned int8
    image = image.astype('uint8')

    # convert ndarray to image object
    image_obj = PIL.Image.fromarray(image)
    draw = PIL.ImageDraw.Draw(image_obj)

    #------------------------------------
    # Draw title name
    #------------------------------------
    draw.font = PIL.ImageFont.truetype(font=font_family_path, size=title_fontsize)
    for cind, (tag, title_fontcolor) in enumerate(zip(labels, font_colors)):
        for turn_index in range(turn_num):
            # Calc text position
            row_index = cind + turn_index * len(images)
            x = image_margin[0] + group_margin[0] + row_index * (image_size[0] + image_margin[0] + image_margin[2])
            x += turn_index * (group_margin[0] + group_margin[2])
            x += title_top_padding
            y = title_left_padding

            # textの座標指定はxとyが逆転するので注意
            # if "title_fontcolor" not in condition.keys():
            #     title_fontcolor = "black"
            # else:
            #     title_fontcolor = condition["title_fontcolor"]
            # title_fontcolor = "black"
            draw.text([y, x], tag, title_fontcolor)

    #------------------------------------
    # Draw image id name
    # * image_id_list variables is necessary
    #------------------------------------

    if id_show:
        draw.font = PIL.ImageFont.truetype(font=font_family_path, size=id_fontsize)
        for tind in range(total_image_size):
            #  Calc text position
            row_index = (tind // column_size) * len(images)
            column_index = 1 + tind % column_size
            turn_index = tind // column_size
            x = image_margin[0] + group_margin[0] + row_index * (image_size[0] + image_margin[0] + image_margin[2])
            x += turn_index * (group_margin[0] + group_margin[2])
            x -= id_fontsize
            y = image_margin[3] + group_margin[3] + column_index * (image_size[1] + image_margin[1] + image_margin[3])

            draw.text([y, x], image_id_list[tind], id_fontcolor)

    return image_obj

def load_one_movie_as_list(movie_path: pathlib.Path, width=224, height=224):
    cap = cv2.VideoCapture(str(movie_path))
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame = expand2square(frame)
            frame = cv2.resize(frame, dsize=(width, height))
            frames.append(frame)
        else:
            break

    return frames

def load_inputs_from_one_source(inputs_list, is_image, width=224, height=224):
    '''
    image: list of pathlib.Path
        [1.jpg, 2.jpg, 3.jpg, ...]
    movie:
        [1.mp4, 2.mp4, 3.mp4, ...]
    ---
    return
        [[1-1, 2-1, 3-1, ...]
         ...
         [1-n, 2-n, 3-n, ...]
    `   ]
    '''
    output = []
    maximum_length = 0
    for target_file in inputs_list:
        if is_image:
            target_image = cv2.imread(str(target_file))
            target_image = expand2square(target_image)
            target_image = cv2.resize(target_image, dsize=(width, height))
            output.append(target_image)
        else:
            target_frames = load_one_movie_as_list(target_file, width=width, height=height)
            if len(target_frames) > maximum_length: maximum_length = len(target_frames)
            output.append(target_frames)
    if not is_image:
        tmp_list = []
        for frames in output:
            if len(frames) < maximum_length:
                tmp_frames = frames + [frames[-1]] * (maximum_length - len(frames))
                tmp_list.append(np.array(tmp_frames))
            else:
                tmp_list.append(np.array(frames))
        output = np.array(tmp_list)
        output = np.swapaxes(output, 0, 1)
    else:
        output = np.array([output])
    return output

def load_all_inputs(inputs_list, to_be_removed, index, n_targets, width=224, height=224):
    '''
    inputs_list: list of source_info dict
    to_be_removed: image/movie ids to be excluded
    index: the current index
    n_targets: the batch size
    ---
    return
        tag_list (s)
        outputs (n x s x k x H x W x C)
    '''
    tags = []
    font_colors = []
    outputs = []
    maximum_length = 0
    target_ids = None
    for source_info in inputs_list:
        # parse information
        source_tag = source_info['tag']
        source_dir = pathlib.Path(source_info['root_dir'])
        inputs_pattern = source_info['inputs_pattern']
        is_image = source_info['is_image']
        font_color = source_info['font_color']

        # list up input files
        inputs_files = list(source_dir.glob(inputs_pattern.replace('<image_id>', '*')))
        # sort
        inputs_files = os_sorted(inputs_files)

        # extract ids
        try:
            ids = [int(re.match(str(source_dir) + '/' + inputs_pattern.replace('*', '.*').replace('<image_id>', '(.*)'), str(path)).groups()[0]) for path in inputs_files]
            # remove unfaborable image/movie
            inputs_files = [input_file for (input_file, id) in zip(inputs_files, ids) if id not in to_be_removed]
            ids = [id for id in ids if id not in to_be_removed]
        except ValueError:
            ids = [re.match(str(source_dir) + '/' + inputs_pattern.replace('*', '.*').replace('<image_id>', '(.*)'), str(path)).groups()[0] for path in inputs_files]
            # remove unfaborable image/movie
            inputs_files = [input_file for (input_file, id) in zip(inputs_files, ids) if id not in to_be_removed]
            ids = [id for id in ids if id not in to_be_removed]

        # check id consistency
        cur_target_ids = ids[n_targets * index:n_targets * (index + 1)]
        if target_ids is None:
            target_ids = cur_target_ids
        else:
            assert target_ids == cur_target_ids, print(target_ids, cur_target_ids)

        inputs_files = inputs_files[n_targets * index:n_targets * (index + 1)]

        # load data
        loaded = load_inputs_from_one_source(inputs_files, is_image, width=width, height=height)
        if maximum_length < len(loaded):
            maximum_length = len(loaded)
        tags.append(source_tag)
        font_colors.append(font_color)
        outputs.append(loaded)
    tmp_outputs = []
    for output in outputs:
        if len(output) < maximum_length:
            tmp_output = list(output) + [output[-1]] * (maximum_length - len(output))
            tmp_outputs.append(tmp_output)
        else:
            tmp_outputs.append(output)
    outputs = np.array(tmp_outputs)
    return np.swapaxes(outputs, 0, 1), tags, font_colors

def removal_conf2id_list(removal_conf):
    out_list = []
    for removal_targets in removal_conf.values():
        for removal_target in removal_targets:
            removal_target_base_path = pathlib.Path(removal_target)
            out_list.append(int(removal_target_base_path.stem))
    return out_list

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf_path', type=str, required=True,
                        help='configuration yaml path')
    args = parser.parse_args()

    conf_path = args.conf_path
    with open(conf_path, 'r') as f:
        conf = yaml.safe_load(f)

    to_be_removed_yaml_path = conf['to_be_removed_yaml']
    with open(to_be_removed_yaml_path, 'r') as f:
        to_be_removed_conf = yaml.safe_load(f)
    to_be_removed_id_list = removal_conf2id_list(to_be_removed_conf)

    height, width = conf['image_height'], conf['image_width']
    output_dir = pathlib.Path(conf['output_root'])

    inputs_list = conf['inputs']
    for index in conf['index']:
        inputs_array_list, tags, font_colors = load_all_inputs(inputs_list, to_be_removed_id_list, index, conf['n_targets'], width=width, height=height)

        output_filename = pathlib.Path(conf['output_filename'].replace('<index>', str(index)))
        output_path = output_dir / output_filename
        output_dir.mkdir(parents=True, exist_ok=True)

        if not output_path.suffix in ['.jpg', '.png', '.jpeg', '.JPEG']: # output movie
            true_output_path = None
            if output_path.suffix != '.avi':
                true_output_path = copy.deepcopy(output_path)
                output_path = output_path.with_suffix('.avi')

            fourcc = cv2.VideoWriter_fourcc(*'DIVX')
            for i, one_frame_data in tqdm(enumerate(inputs_array_list)):
                frame_out = draw_group_image_set(one_frame_data, tags, font_colors, background_color = (255, 255, 255),
                                image_size=(height, width), image_margin=conf['image_margin'], group_margin=conf['group_margin'], max_column_size=conf['max_column_size'],
                                title_fontsize=conf['title_fontsize'], title_top_padding=conf['title_top_padding'], title_left_padding=conf['title_left_padding'], font_family_path=None,
                                id_show=False, id_fontcolor="black", id_fontsize=18, image_id_list=[])
                # if i == 0:
                #     frame_out.save('test.png')
                frame_out = pil2cv(frame_out)
                frame_shape = frame_out.shape
                if i == 0:
                    out = cv2.VideoWriter(str(output_path), fourcc, 100/6, (frame_shape[1], frame_shape[0]))
                out.write(frame_out)

            out.release()

            if true_output_path is not None:
                cmd = 'ffmpeg -y -i {} {}'.format(output_path, true_output_path)
                print(cmd)
                subprocess.run(cmd, shell=True)

            if conf['save_gif']:
                cmd ='ffmpeg -y -i {} {}'.format(output_path, output_path.with_suffix('.gif'))
                print(cmd)
                subprocess.run(cmd, shell=True)

            if true_output_path is not None:
                cmd = 'rm {}'.format(output_path)
                print(cmd)
                subprocess.run(cmd, shell=True)
        else:
            assert len(inputs_array_list) == 1
            one_frame_data = inputs_array_list[0]
            frame_out = draw_group_image_set(one_frame_data, tags, font_colors, background_color = (255, 255, 255),
                                            image_size=(height, width), image_margin=conf['image_margin'], group_margin=conf['group_margin'], max_column_size=conf['max_column_size'],
                                            title_fontsize=conf['title_fontsize'], title_top_padding=conf['title_top_padding'], title_left_padding=conf['title_left_padding'], font_family_path=None,
                                            id_show=False, id_fontcolor="black", id_fontsize=18, image_id_list=[])
            frame_out = pil2cv(frame_out)
            cv2.imwrite(str(output_path), frame_out)

if __name__ == '__main__':
    main()