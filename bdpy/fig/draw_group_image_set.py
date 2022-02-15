# coding:utf-8

import numpy as np
import PIL
import PIL.ImageDraw
import PIL.ImageFont
import matplotlib

def draw_group_image_set(condition_list, background_color = (255, 255, 255), 
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
                "image_filepath_list": list, # Loading image filepath list 
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

    #------------------------------------
    # Setting
    #------------------------------------
    total_image_size = len(condition_list[0]["image_filepath_list"])
    column_size = np.min([max_column_size, total_image_size]) 

    # create canvas
    turn_num = int(np.ceil(total_image_size / float(column_size)))
    nImg_row = len(condition_list) * turn_num 
    nImg_col = 1 + column_size # 1 means title column 
    size_x = (image_size[0] + image_margin[0] + image_margin[2]) * nImg_row + (group_margin[0] + group_margin[2]) * turn_num
    size_y = (image_size[1] + image_margin[1] + image_margin[3]) * nImg_col + (group_margin[1] + group_margin[3])
    image = np.ones([size_x, size_y, 3])
    for bi, bc in enumerate(background_color):
        image[:, :, bi] = bc

    #------------------------------------
    # Draw image
    #------------------------------------
    for cind, condition in enumerate(condition_list):
        title = condition['title']
        image_filepath_list = condition['image_filepath_list']

        for tind in range(total_image_size):
            # Load image
            image_filepath = image_filepath_list[tind]
            if image_filepath is None:
                continue;
            image_obj = PIL.Image.open(image_filepath)
            image_obj = image_obj.convert("RGB")
            image_obj = image_obj.resize((image_size[0], image_size[1]), PIL.Image.LANCZOS)

            # Calc image position
            row_index = cind + (tind // column_size) * len(condition_list) 
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
    if font_family_path is None:
        system_fonts = matplotlib.font_manager.findSystemFonts(fontpaths=None, fontext='ttf')
        draw.font = PIL.ImageFont.truetype(font=system_fonts[0], size=title_fontsize)
    else:
        draw.font = PIL.ImageFont.truetype(font=font_family_path, size=title_fontsize)

    #------------------------------------
    # Draw title name 
    #------------------------------------
    for cind, condition in enumerate(condition_list):
        for turn_index in range(turn_num):
            # Calc text position
            row_index = cind + turn_index * len(condition_list) 
            x = image_margin[0] + group_margin[0] + row_index * (image_size[0] + image_margin[0] + image_margin[2])
            x += turn_index * (group_margin[0] + group_margin[2])
            x += title_top_padding 
            y = title_left_padding

            # textの座標指定はxとyが逆転するので注意
            if "title_fontcolor" not in condition.keys():
                title_fontcolor = "black"
            else:
                title_fontcolor = condition["title_fontcolor"]
            draw.text([y, x], condition["title"], title_fontcolor)

    #------------------------------------
    # Draw image id name 
    # * image_id_list variables is necessary
    #------------------------------------

    if id_show:
        if font_family_path is None:
            system_fonts = matplotlib.font_manager.findSystemFonts(fontpaths=None, fontext='ttf')
            draw.font = PIL.ImageFont.truetype(font=system_fonts[0], size=id_fontsize)
        else:
            draw.font = PIL.ImageFont.truetype(font=font_family_path, size=id_fontsize)

        for tind in range(total_image_size):
            #  Calc text position
            row_index = (tind // column_size) * len(condition_list) 
            column_index = 1 + tind % column_size
            turn_index = tind // column_size            
            x = image_margin[0] + group_margin[0] + row_index * (image_size[0] + image_margin[0] + image_margin[2])
            x += turn_index * (group_margin[0] + group_margin[2])
            x -= id_fontsize
            y = image_margin[3] + group_margin[3] + column_index * (image_size[1] + image_margin[1] + image_margin[3]) 

            draw.text([y, x], image_id_list[tind], id_fontcolor)
    
            
    return image_obj
