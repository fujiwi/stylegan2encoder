import os
import pickle
import PIL.Image
import numpy as np
import dnnlib
import dnnlib.tflib as tflib
from encoder.generator_model import Generator
import matplotlib.pyplot as plt
import glob
 
# 预训练好的网络模型，来自NVIDIA
Model = '/content/drive/MyDrive/network/stylegan2-ffhq-config-f.pkl'
_Gs_cache = dict()
 

latent_file = 'latent_representations/star_01.npy'


# 加载StyleGAN已训练好的网络模型
def load_Gs(model):
    if model not in _Gs_cache:
        model_file = glob.glob(Model)
        if len(model_file) == 1:
            model_file = open(model_file[0], "rb")
        else:
            raise Exception('Failed to find the model')

        _G, _D, Gs = pickle.load(model_file)
        # _G = Instantaneous snapshot of the generator. Mainly useful for resuming a previous training run.
        # _D = Instantaneous snapshot of the discriminator. Mainly useful for resuming a previous training run.
        # Gs = Long-term average of the generator. Yields higher-quality results than the instantaneous snapshot.
 
        # Print network details.
        # Gs.print_layers()
 
        _Gs_cache[model] = Gs
    return _Gs_cache[model]
 
# 使用generator生成图片
def generate_image(generator, latent_vector):
    latent_vector = latent_vector.reshape((1, 18, 512))
    generator.set_dlatents(latent_vector)
    img_array = generator.generate_images()[0]
    img = PIL.Image.fromarray(img_array, 'RGB')
    return img
 
def move_and_show(generator, category, latent_vector, direction, coeffs):
    fig,ax = plt.subplots(1, len(coeffs), figsize=(15, 10), dpi=80)
    # 调用coeffs数组，生成一系列的人脸变化图片
    for i, coeff in enumerate(coeffs):
        new_latent_vector = latent_vector.copy()
        # 人脸latent与改变人脸特性/表情的向量相混合，只运算前8层（一共18层）
        new_latent_vector[:8] = (latent_vector + coeff*direction)[:8]
        result = generate_image(generator, new_latent_vector)

        # Save result
        result.save('results/' + str(category) + str(i).zfill(3)+'.png')

        ax[i].imshow(result)
        ax[i].set_title('%s: %0.1f' % (str(category), coeff))

    [x.axis('off') for x in ax]
    # 显示
    plt.show()
 
def main():
    # 初始化
    tflib.init_tf()
    person = np.load(latent_file)

    # 调用预训练模型
    Gs_network = load_Gs(Model)
    generator = Generator(Gs_network, batch_size=1, randomize_noise=False)

    # 读取已训练好的用于改变人脸特性/表情的向量
    # 包括：改变年龄、改变水平角度、改变性别、改变眼睛大小、是否佩戴眼镜、改变笑容等
    age_direction = np.load('ffhq_dataset/latent_directions/age.npy')
    gender_direction = np.load('ffhq_dataset/latent_directions/gender.npy')
    eyes_direction = np.load('ffhq_dataset/latent_directions/eyes_open.npy')
    smile_direction = np.load('ffhq_dataset/latent_directions/smile.npy')

    # 混合人脸和变化向量，生成变化后的图片
    move_and_show(generator, 'age',    person, age_direction,    [-4, -3, -2, 0, 2, 3, 4])
    move_and_show(generator, 'gender', person, gender_direction, [-0.8, -0.6, -0.5, 0, 0.5, 0.6, 0.8])
    move_and_show(generator, 'eyes',   person, eyes_direction,   [-2, -1, -0.5, 0, 0.5, 1, 2])
    move_and_show(generator, 'smile',  person, smile_direction,  [-2, -1, -0.5, 0, 0.5, 1, 2 ])

    direction = np.load('ffhq_dataset/latent_directions/glasses.npy')
    move_and_show(generator, 'glasse',  person, direction,  [-4.2, -4.1, -3, 0, 3, 4, 5])

    direction = np.load('ffhq_dataset/latent_directions/angle_horizontal.npy')
    move_and_show(generator, 'horizontal',  person, direction,  [-12, -8, -5, 0, 5, 8, 12])

    direction = np.load('ffhq_dataset/latent_directions/angle_pitch.npy')
    move_and_show(generator, 'vertical',  person, direction,  [-5, -3, -1, 0, 1, 3, 5 ])

if __name__ == "__main__":
    main()
