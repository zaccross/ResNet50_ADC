## Goal of this file is to augment my data set

## Rotate images

def rotate(image):
    """Takes in 1 image and returns 4 rotated images in incremenets of 90 deg in same dir"""
    orig_image = Image.open(image)

    for i in range(3):
        rotated = orig_image.rotate((i+1)*90)
        rotated.save(image[:-4]+f'r{i}.jpg')
    return None

def invert(image):
    if '_in' in image:
        return None
    orig_image = Image.open(image)
    inverse = ImageOps.invert(orig_image)
    inverse.save(image[:-4]+'_in.jpg')
    return None

def mirror(image):
    if '_mir' in image:
        return None
    orig_image = Image.open(image)
    inverse = ImageOps.mirror(orig_image)
    inverse.save(image[:-4] + '_mir.jpg')

def color_shift(image, value):
    #print(image)
    orig_image = Image.open(image).copy()
    pix_image = orig_image.load()
    width,height = (200,200)
    if 'adjusted' in image:
        return None
    else:
        # go through and shift pixel vals
        for i in range(width):
            for j in range(height):

                pixel_val = pix_image[i,j]
                #print(pixel_val)
                new = []
                for row in pixel_val:
                    if row + value >= 0 and row + value <= 4095:
                        new.append(row+value)
                    else:
                        new.append(row)
                pix_image[i,j] = tuple(new)

        if value < 0:
            value = f'm{value}'
        else:
            value = f'{value}'
        orig_image.save(image[-4]+'_adjusted_by'+value+'.jpg')







from PIL import Image, ImageOps
import os


cwd = './NEU-DET/images'
dirs = os.listdir(cwd)

def main():

    # go through each directory
    for dir in dirs:
        if dir[0] != '.':
            # get images
            images = os.listdir(cwd+'/'+dir)
            print(dir,len(images))
            path = cwd+'/'+dir + '/'
            cnt = 0
            # now go through each image
            for image in images:
                if cnt % 100 == 0:
                    pass
                    # print(cnt, len(images))
                # rotate images
                #rotate(path+image)

                #invert images
                #invert(path+image)
                # Mirror
                #mirror(path+image)

                #move values

                #color_shift(path+image, 100)
                #color_shift(path+image, -100)
                cnt+=1
                pass



main()

