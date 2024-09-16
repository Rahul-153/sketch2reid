# import os
from options.test_options import TestOptions
from data import create_dataset
from model import create_model
from util.visualizer import save_images

# class dotdict(dict):
#     """dot.notation access to dictionary attributes"""
#     __getattr__ = dict.get
#     __setattr__ = dict.__setitem__
#     __delattr__ = dict.__delitem__

def art2real(upload_dir:str = 'static',result_dir:str = 'face_reid/faces'):
    # opt = {
    #     'ntest' : float('inf'),
    #     'results_dir' : 'art2real/results/',
    #     'aspect_ratio' : 1.0,
    #     'phase': 'test',
    #     'eval' : False,
    #     'num_test' : 100,
    #     'model' : 'cycle_gan',
    #     'name' : 'art2real/dataset/potrait2photo',
    #     'dataroot' : 'static/testA',
    #     'isTrain' : False,
    #     'dataset_mode' : 'single'
    # }
    opt = TestOptions().parse()
    opt.dataroot = upload_dir
    opt.num_threads = 0
    opt.batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    # opt = dotdict(opt)
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)
    model.eval()
    for i, data in enumerate(dataset):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()     # get image paths
        if i % 5 == 0:  # save images to an HTML file
            print('processing (%04d)-th image... %s' % (i, img_path))
        print(img_path)
        print(result_dir)
        print('------------------**')
        save_images(result_dir, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)
    # webpage.save()  # save the HTML



# if __name__ == '__main__':
#     opt = TestOptions().parse()  # get test options
#     # hard-code some parameters for test
#     opt.num_threads = 0   # test code only supports num_threads = 1
#     opt.batch_size = 1    # test code only supports batch_size = 1
#     opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
#     opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
#     opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
#     dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
#     model = create_model(opt)      # create a model given opt.model and other options
#     model.setup(opt)               # regular setup: load and print networks; create schedulers
#     # create a website
#     web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.epoch))  # define the website directory
#     webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))

#     if opt.eval:
#         model.eval()
#     for i, data in enumerate(dataset):
#         if i >= opt.num_test:  # only apply our model to opt.num_test images.
#             break
#         model.set_input(data)  # unpack data from data loader
#         model.test()           # run inference
#         visuals = model.get_current_visuals()  # get image results
#         img_path = model.get_image_paths()     # get image paths
#         if i % 5 == 0:  # save images to an HTML file
#             print('processing (%04d)-th image... %s' % (i, img_path))
#         save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)
#     webpage.save()  # save the HTML
