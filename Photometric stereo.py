import os
import glob
import re
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import ndimage
from PIL import Image
import time


start = time.time()

root_path = 'croppedyale/'
subject_name = 'yaleB07'
full_path = '%s%s' % (root_path, subject_name)


def LoadFaceImages(pathname, subject_name, num_images):

    def load_image(fname):
        return np.asarray(Image.open(fname))

    def fname_to_ang(fname):
        yale_name = os.path.basename(fname)
        return int(yale_name[12:16]), int(yale_name[17:20])

    def sph2cart(az, el, r):
        rcos_theta = r * np.cos(el)
        x = rcos_theta * np.cos(az)
        y = rcos_theta * np.sin(az)
        z = r * np.sin(el)
        return x, y, z

    ambimage = load_image(
        os.path.join(pathname, subject_name + '_P00_Ambient.pgm'))
    im_list = glob.glob(os.path.join(pathname, subject_name + '_P00A*.pgm'))
    if num_images <= len(im_list):
        im_sub_list = np.random.choice(im_list, num_images, replace=False)
    else:
        print(
            'Total available images is less than specified.\nProceeding with %d images.\n'
            % len(im_list))
        im_sub_list = im_list
    im_sub_list.sort()
    imarray = np.stack([load_image(fname) for fname in im_sub_list], axis=-1)
    Ang = np.array([fname_to_ang(fname) for fname in im_sub_list])

    x, y, z = sph2cart(Ang[:, 0] / 180.0 * np.pi, Ang[:, 1] / 180.0 * np.pi, 1)
    lightdirs = np.stack([y, z, x], axis=-1)
    return ambimage, imarray, lightdirs


def preprocess(ambient_image, imarray):
    processed_imarray = np.zeros((192, 168, 64))
    for index in range(64):
        processed_imarray[:,:,index] = imarray[:,:,index]-ambient_image

    processed_imarray[processed_imarray < 0] = 0
    processed_imarray = processed_imarray/255

    return processed_imarray


def photometric_stereo(imarray, light_dirs):
    processed_array = preprocess(ambient_image, imarray)
    colne_imarray = np.zeros((64, 32256))
    for i in range(64):
        colne_imarray[i, :] = processed_array[:, :, i].reshape((1, 32256))
    G = np.linalg.lstsq(light_dirs, colne_imarray, rcond=None)[0]
    albedo_image = np.linalg.norm(G, axis=0)
    surface_normals = np.divide(G, np.vstack([albedo_image] * 3))
    surface_normals = surface_normals.T.reshape((192, 168, 3))
    albedo_image = albedo_image.reshape((192, 168))
    return albedo_image, surface_normals


def get_surface(surface_normals, integration_method):
    height_map = np.zeros((192, 168))
    output = np.zeros((192, 168))
    output1 = np.zeros((192, 168))
    normal_x = surface_normals[:,:,0]
    normal_y = surface_normals[:, :, 1]
    normal_z = surface_normals[:, :, 2]
    fx = normal_x/normal_z
    fy = normal_y/normal_z
    X_sum = np.cumsum(normal_x, 1)
    Y_sum = np.cumsum(normal_y,0)
    if integration_method == 'column':
        output[1, 2: 168] = np.cumsum(fx[1, 2: 168], 0)
        output[2: 192,:] = fy[2: 192,:]
        height_map = np.cumsum(output,0)

    if integration_method == 'row':
        output[2: 192, 1] = np.cumsum(fy[2: 192, 1])
        output[:, 2: 168] = fx[:, 2: 168]
        height_map = np.cumsum(output, 1)

    if integration_method == 'average':
        output[2: 192, 1] = np.cumsum(fy[2: 192, 1])
        output[:, 2: 168] = fx[:, 2: 168]
        output1[1, 2: 168] = np.cumsum(fx[1, 2: 168],0)
        output1[2: 192,:] = fy[2: 192,:]
        height_map = (np.cumsum(output1,0) + np.cumsum(output, 1))/2

    if integration_method == 'random':
        height_map[2: 192, 1] = Y_sum[2: 192, 1]
        height_map[1, 2: 168] = X_sum[1, 2: 168]
        for i in range(2,192):
            for j in range(1,168):
                current_value = np.zeros((192,168))
                count = 0
                for p in range(1,i-1):
                    if j - p >= 1:
                        current_value = current_value + Y_sum[1 + p] + X_sum[1 + p, j - p] + Y_sum[i, j - p] - Y_sum[1 + p, j - p] + X_sum[i, j] - X_sum[i, j - p]
                        count = count + 1

                    if (i == 2) or (i == 192) or (p == i ):
                        current_value = current_value+Y_sum[i,1] + X_sum(i,j)
                        count = count+1

                height_map[i,j] = current_value[i,j]/(count+1)

    return height_map


def plot_surface_normals(surface_normals):
    fig = plt.figure()
    ax = plt.subplot(1, 3, 1)
    ax.axis('off')
    ax.set_title('X')
    im = ax.imshow(surface_normals[:,:,0])
    ax = plt.subplot(1, 3, 2)
    ax.axis('off')
    ax.set_title('Y')
    im = ax.imshow(surface_normals[:,:,1])
    ax = plt.subplot(1, 3, 3)
    ax.axis('off')
    ax.set_title('Z')
    im = ax.imshow(surface_normals[:,:,2])


def set_aspect_equal_3d(ax):
    xlim = ax.get_xlim3d()
    ylim = ax.get_ylim3d()
    zlim = ax.get_zlim3d()
    from numpy import mean
    xmean = mean(xlim)
    ymean = mean(ylim)
    zmean = mean(zlim)
    plot_radius = max([
        abs(lim - mean_)
        for lims, mean_ in ((xlim, xmean), (ylim, ymean), (zlim, zmean))
        for lim in lims
    ])
    ax.set_xlim3d([xmean - plot_radius, xmean + plot_radius])
    ax.set_ylim3d([ymean - plot_radius, ymean + plot_radius])
    ax.set_zlim3d([zmean - plot_radius, zmean + plot_radius])


def display_output(albedo_image, height_map):
    fig = plt.figure()
    plt.imshow(albedo_image, cmap='gray')
    plt.axis('off')

    fig = plt.figure(figsize=(10, 10))
    ax = fig.gca(projection='3d')
    ax.view_init(20, 20)
    X = np.arange(albedo_image.shape[0])
    Y = np.arange(albedo_image.shape[1])
    X, Y = np.meshgrid(Y, X)
    H = np.flipud(np.fliplr(height_map))
    A = np.flipud(np.fliplr(albedo_image))
    A = np.stack([A, A, A], axis=-1)
    ax.xaxis.set_ticks([])
    ax.xaxis.set_label_text('Z')
    ax.yaxis.set_ticks([])
    ax.yaxis.set_label_text('X')
    ax.zaxis.set_ticks([])
    ax.yaxis.set_label_text('Y')
    surf = ax.plot_surface(
        X, Y, H, cmap='gray', facecolors=A, linewidth=0, antialiased=False)
    set_aspect_equal_3d(ax)

# H, X, Y
# X, H, Y
# X, Y, H


def save_outputs(subject_name, albedo_image, surface_normals):
    im = Image.fromarray((albedo_image*255).astype(np.uint8))
    im.save("%s_albedo.jpg" % subject_name)
    im = Image.fromarray((surface_normals[:,:,0]*128+128).astype(np.uint8))
    im.save("%s_normals_x.jpg" % subject_name)
    im = Image.fromarray((surface_normals[:,:,1]*128+128).astype(np.uint8))
    im.save("%s_normals_y.jpg" % subject_name)
    im = Image.fromarray((surface_normals[:,:,2]*128+128).astype(np.uint8))
    im.save("%s_normals_z.jpg" % subject_name)


ambient_image, imarray, lightdirs = LoadFaceImages(full_path, subject_name, 64)

albedo_image, surface_normals = photometric_stereo(imarray, lightdirs)

height_map = get_surface(surface_normals, 'average')

display_output(albedo_image, height_map)

plot_surface_normals(surface_normals)

end = time.time()

print(f"Runtime of the program is {end - start}")

# save_outputs(subject_name, albedo_image, surface_normals)

plt.show()





