import nvisii
import os
import numpy as np 
import cv2
from scipy.spatial.transform import Rotation as R
from torch.nn.functional import pad
from tqdm import tqdm
from generate_occlusion import occlusion
X_RANGE = 3.5
Y_RANGE = 2
def crop_and_save(img_path, crop_size):
    img = cv2.imread(img_path)
    height, width,_ = img.shape

    h_pickable = [i for i in range(crop_size[0]//2, height-crop_size[0]//2)]
    w_pickable = [i for i in range(crop_size[1]//2, width-crop_size[1]//2)]

    loc_h = np.random.choice(h_pickable)
    loc_w = np.random.choice(w_pickable)

    c1 = (loc_h, loc_w) # center 1
    crop_img = img[(c1[0]-crop_size[0]//2):(c1[0]+crop_size[0]//2), (c1[1]-crop_size[1]//2):(c1[1]+crop_size[1]//2)]
    cv2.imwrite(os.path.split(img_path)[0]+'/crop.png', crop_img)
    return os.path.split(img_path)[0]+'/crop.png'


def is_far_enough(xyz_list, xyz, min_dist=1.5):
    for prev_xyz in xyz_list:
        if (prev_xyz[0]-xyz[0])**2+(prev_xyz[1]-xyz[1])**2<min_dist**2:
            # print('F')
            return False
    return True
def degree_to_quat(degree, seq='xyz'):
    quat = R.from_euler(seq,degree, degrees=True).as_quat()
    return (quat[0],quat[1],quat[2],quat[3])

def init_entity(obj_path_list, bg_path, width, height, first=False):

    # camera
    if first:
        camera = nvisii.entity.create(
            name = "camera",
            transform = nvisii.transform.create("camera"),
            camera = nvisii.camera.create(
                name = "camera",  
                aspect = float(width)/float(height)
            )
        )
        
        camera.get_transform().look_at(
            at = (0,0,-1),
            up = (0,1,0),
            eye = (0,0,6),
        )
        nvisii.set_camera_entity(camera)
    else:
        camera = nvisii.entity.get('camera')

    # background
    if first:
        bg_entity = nvisii.entity.create(
            name = "floor",
            mesh = nvisii.mesh.create_plane("mesh_floor"),
            transform = nvisii.transform.create("transform_floor"),
            material = nvisii.material.create("material_floor")
        )
        if width > height:
            width_scale = width / 10**(len(str(width))-1) 
            height_scale = height / 10**(len(str(width))-1)
        else:
            width_scale = width / 10**(len(str(height))-1) 
            height_scale = height / 10**(len(str(height))-1)

        bg_entity.get_transform().set_scale((width_scale*1.5,height_scale*1.5,1))

        bg_material = nvisii.material.get("material_floor")
        
        bg = nvisii.texture.create_from_file("bg",bg_path)
        bg_material.set_base_color_texture(bg)
    else:
        bg_entity = nvisii.entity.get('floor')
        nvisii.texture.clear_all()
        bg = nvisii.texture.create_from_file("bg",bg_path)
        bg_material = nvisii.material.get("material_floor")
        bg_material.set_base_color_texture(bg)

    if not first:
        for i, obj_path in enumerate(obj_path_list):
            nvisii.mesh.remove("obj_"+str(i+1))
            nvisii.material.remove("obj_"+str(i+1))
            nvisii.transform.remove("obj_"+str(i+1))
            nvisii.entity.remove("obj_"+str(i+1))
    # objects
    obj_list = []
    xyz_list = []
    for i, obj_path in enumerate(obj_path_list):
        mesh = nvisii.mesh.create_from_file("obj_"+str(i+1), obj_path)
        obj = nvisii.entity.create(
            name="obj_"+str(i+1),
            mesh = mesh,
            transform = nvisii.transform.create("obj_"+str(i+1)),
            material = nvisii.material.create("obj_"+str(i+1))
        )

        xyz = [np.random.rand()*X_RANGE-X_RANGE/2,np.random.rand()*Y_RANGE-Y_RANGE/2,0.5]
        # xyz = [0.7,0.3,0.5]
        count = 0
        while not is_far_enough(xyz_list, xyz) and count < 500:
            xyz = [np.random.rand()*X_RANGE-X_RANGE/2,np.random.rand()*Y_RANGE-Y_RANGE/2,0.5]
            count += 1
        xyz_list.append(xyz)
        obj.get_transform().set_position(xyz)
        
        obj.get_transform().set_scale((2,2,2))
        # rot_xyz = np.random.rand(3)*360
        # r = degree_to_quat(rot_xyz)
        # obj.get_transform().set_rotation(r)
        gray = np.random.normal(loc=0.5,scale=0.3)
        while gray<0.1 or gray > 0.9:
            gray = np.random.normal(loc=0.5,scale=0.3)
        obj.get_material().set_base_color([gray,gray,gray])
        obj.get_material().set_roughness(1)
        obj.get_material().set_specular(0)
        obj.get_material().set_sheen(0)
        obj_list.append(obj)

    return obj_list, bg_entity, camera

def generate_direction():
    [x,y,z] = np.concatenate([np.random.rand(2)*2-1,[0]])
    length = np.sqrt(x**2+ y**2+ z**2)
    direction = [x/length, y/length, z/length]
    return direction

def generate_random_trajectory(pos, max_distance, max_rotation, num_frame):    
    traj_type = 'linear'# np.random.choice(['linear', 'quadratic'], size=1, p=[0.8,0.2])
    if traj_type == 'linear':
        # dist = max_distance
        # dire = generate_direction()
        dist = np.random.rand(3)*max_distance*2-max_distance
        while pos[0]+dist[0] > X_RANGE/2 or pos[0] + dist[0] < -X_RANGE/2 or pos[1]+dist[1] > Y_RANGE/2 or pos[1] + dist[1] < -Y_RANGE/2:
            dist = np.random.rand(3)*max_distance

        x_traj = [dist[0]/num_frame for i in range(num_frame)]
        y_traj = [dist[1]/num_frame for i in range(num_frame)]
        z_traj = [dist[2]/num_frame for i in range(num_frame)]


    elif traj_type == 'todo':
        pass
    
    # rot_z = max_rotation/num_frame
    rot_z = (np.random.rand()*max_rotation*2-max_rotation)/num_frame
    r = degree_to_quat([0,0,rot_z])
    r_traj = [r for i in range(num_frame)]

    return x_traj,y_traj,z_traj,r_traj

def buffer_to_image(framebuffer,width,height,to_gray=True):
    image = np.array(framebuffer).reshape(height,width,4)[:,:,:3]
    gamma =  1.055*image**(1/2.4)-0.055
    scale = image * 12.92
    image = np.where (image > 0.0031308, gamma, scale)
    image *= 255
    image = np.flip(image, 0)
    if to_gray:
        image = np.mean(image,-1,keepdims=True)
    image = image.transpose(2,0,1)
    image = image.astype(np.uint8)

    return image
    # if to_gray:
    #     image = np.zeros((height, width))
    #     for i in range(height):
    #         for j in range(width):
    #             b = linear_to_srgb(framebuffer[3*(width*i+j)+0])
    #             g = linear_to_srgb(framebuffer[3*(width*i+j)+1])
    #             r = linear_to_srgb(framebuffer[3*(width*i+j)+2])
    #             image[i,j] = (b+g+r)/3
    # else:
    #     image = np.zeros((height, width,3))
    #     for i in range(height):
    #         for j in range(width):
    #             image[i,j,2] = linear_to_srgb(framebuffer[3*(width*i+j)+0])
    #             image[i,j,1] = linear_to_srgb(framebuffer[3*(width*i+j)+1])
    #             image[i,j,0] = linear_to_srgb(framebuffer[3*(width*i+j)+2])
    # image *= 255
    # image = image.astype(np.uint8)
    # return image

def clear_motion(obj_list, camera):
    for obj in obj_list:
        obj.get_transform().clear_motion()
        obj.get_transform().set_position(obj.get_transform().get_position(),previous=True)
        obj.get_transform().set_rotation(obj.get_transform().get_rotation(),previous=True)

    camera.get_transform().clear_motion()
    camera.get_transform().set_position(camera.get_transform().get_position(),previous=True)

def generate_one_scene(obj_paths, bg_path, width, height, out_path, first=True, num_flow=32,warm_up=5,padding=5,spp=100, to_gray=False):
    num_object = len(obj_paths)
    num_frame = warm_up+num_flow+2*padding
    if to_gray:
        result = np.zeros((num_frame, 1, height, width),dtype=np.uint8)
    else:
        result = np.zeros((num_frame, 3, height, width),dtype=np.uint8)
    bg_path = crop_and_save(bg_path, (int(height*1.5),int(width*1.5)))
    print(bg_path)
    obj_list, bg_entity, camera = init_entity(obj_paths, bg_path, width, height, first)
    
    nvisii.sample_time_interval((0,0))

    nvisii.sample_pixel_area(
        x_sample_interval = (.5,.5), 
        y_sample_interval = (.5, .5)
    )
    

    # init pos
    camera.get_transform().set_position(camera.get_transform().get_position(),previous=True)
    camera.get_transform().set_rotation(camera.get_transform().get_rotation(),previous=True)

    for obj in obj_list:
        obj.get_transform().set_position(obj.get_transform().get_position(),previous=True)
        obj.get_transform().set_rotation(obj.get_transform().get_rotation(),previous=True)
    # generate trajectories
    x_trajs, y_trajs, z_trajs,r_trajs = [],[],[],[]
    for obj in obj_list: # object
        x_traj, y_traj, z_traj,r_traj = generate_random_trajectory(obj.get_transform().get_position(),(0.2,0.1,0), 15,num_frame-1)
        x_trajs.append(x_traj)
        y_trajs.append(y_traj)
        z_trajs.append(z_traj)
        r_trajs.append(r_traj)
    else: # camera
        x_traj, y_traj, z_traj,r_traj = generate_random_trajectory(camera.get_transform().get_position(),(0.4,0.2,0), 0,num_frame-1)
        x_trajs.append(x_traj)
        y_trajs.append(y_traj)
        z_trajs.append(z_traj)
        r_trajs.append(r_traj)

    nvisii.sample_time_interval((1,1))

    

    # iteratively render frame
    for f in range(warm_up+padding):
        framebuffer = nvisii.render(
            width=width, 
            height=height, 
            samples_per_pixel=spp,
        )
        result[f] = buffer_to_image(framebuffer, width, height, to_gray)

        for o, obj in enumerate(obj_list):
            obj.get_transform().add_position(nvisii.vec3(x_trajs[o][f],y_trajs[o][f],z_trajs[o][f]))
            obj.get_transform().add_rotation(r_trajs[o][f])
        
        # camera
        camera.get_transform().add_position(nvisii.vec3(x_trajs[-1][f],y_trajs[-1][f],z_trajs[-1][f]))
        camera.get_transform().add_rotation(r_trajs[-1][f])


    
    clear_motion(obj_list, camera)
    
    for f in range(warm_up+padding,warm_up+padding+num_flow):
        framebuffer = nvisii.render(
            width=width, 
            height=height, 
            samples_per_pixel=spp,
        )
        result[f] = buffer_to_image(framebuffer, width, height, to_gray)

        for o, obj in enumerate(obj_list):
            obj.get_transform().add_position(nvisii.vec3(x_trajs[o][f],y_trajs[o][f],z_trajs[o][f]))
            obj.get_transform().add_rotation(r_trajs[o][f])
        
        # camera
        camera.get_transform().add_position(nvisii.vec3(x_trajs[-1][f],y_trajs[-1][f],z_trajs[-1][f]))
        camera.get_transform().add_rotation(r_trajs[-1][f])
    
    motion_vectors_array = nvisii.render_data(
        width=width, 
        height=height, 
        start_frame=0,
        frame_count=1,
        bounce=0,
        options="diffuse_motion_vectors"
    )

    motion_vectors_array = np.array(motion_vectors_array).reshape(height, width,4)
    motion_vectors_array = np.flipud(motion_vectors_array)

    flow_x = -motion_vectors_array[:,:,0:1]*width
    flow_y = motion_vectors_array[:,:,1:2]*height
    flow_b = np.concatenate([flow_x, flow_y], 2)
    flow_b_int = (flow_b*64.0+2**14).astype(np.int16)
    np.save(os.path.join(out_path, 'flow_b.npy'), flow_b_int)

    clear_motion(obj_list, camera)

    for f in range(num_frame-padding,num_frame-1):
        framebuffer = nvisii.render(
            width=width, 
            height=height, 
            samples_per_pixel=spp,
        )
        result[f] = buffer_to_image(framebuffer, width, height, to_gray)

        for o, obj in enumerate(obj_list):
            obj.get_transform().add_position(nvisii.vec3(x_trajs[o][f],y_trajs[o][f],z_trajs[o][f]))
            obj.get_transform().add_rotation(r_trajs[o][f])
        
        # camera
        camera.get_transform().add_position(nvisii.vec3(x_trajs[-1][f],y_trajs[-1][f],z_trajs[-1][f]))
        camera.get_transform().add_rotation(r_trajs[-1][f])
    else:
        framebuffer = nvisii.render(
            width=width, 
            height=height, 
            samples_per_pixel=spp,
        )
        result[-1] = buffer_to_image(framebuffer, width, height, to_gray)
        # nvisii.render_to_file(
        #     width=width, 
        #     height=height, 
        #     samples_per_pixel=spp,
        #     file_path=os.path.join(out_path, '{0:03d}.png'.format(num_frame-2-f))
        # )
    
    np.save(os.path.join(out_path, 'frame.npy'), result[[7,-6]])
    
    
    for f in range(num_frame-2,num_frame-padding-1,-1):
        for o, obj in enumerate(obj_list):
            obj.get_transform().add_position(nvisii.vec3(-x_trajs[o][f],-y_trajs[o][f],-z_trajs[o][f]))
            r_traj = (r_trajs[o][f][0], r_trajs[o][f][1], -r_trajs[o][f][2], r_trajs[o][f][3])
            obj.get_transform().add_rotation(r_traj)
        
        camera.get_transform().add_position(nvisii.vec3(-x_trajs[-1][f],-y_trajs[-1][f],-z_trajs[-1][f]))
        r_traj = (r_trajs[-1][f][0], r_trajs[-1][f][1], -r_trajs[-1][f][2], r_trajs[-1][f][3])
        camera.get_transform().add_rotation(r_traj)
    
    clear_motion(obj_list, camera)

    for f in range(num_frame-padding-1,warm_up+padding-1,-1):
        for o, obj in enumerate(obj_list):
            obj.get_transform().add_position(nvisii.vec3(-x_trajs[o][f],-y_trajs[o][f],-z_trajs[o][f]))
            r_traj = (r_trajs[o][f][0], r_trajs[o][f][1], -r_trajs[o][f][2], r_trajs[o][f][3])
            obj.get_transform().add_rotation(r_traj)
        
        camera.get_transform().add_position(nvisii.vec3(-x_trajs[-1][f],-y_trajs[-1][f],-z_trajs[-1][f]))
        r_traj = (r_trajs[-1][f][0], r_trajs[-1][f][1], -r_trajs[-1][f][2], r_trajs[-1][f][3])
        camera.get_transform().add_rotation(r_traj)

    motion_vectors_array = nvisii.render_data(
        width=width, 
        height=height, 
        start_frame=0,
        frame_count=1,
        bounce=0,
        options="diffuse_motion_vectors"
    )

    motion_vectors_array = np.array(motion_vectors_array).reshape(height, width,4)
    motion_vectors_array = np.flipud(motion_vectors_array)

    flow_x = -motion_vectors_array[:,:,0:1]*width
    flow_y = motion_vectors_array[:,:,1:2]*height
    flow_f = np.concatenate([flow_x, flow_y], 2)
    flow_f_int = (flow_f*64.0+2**14).astype(np.int16)
    np.save(os.path.join(out_path, 'flow_f.npy'), flow_f_int)

    occ_f, occ_b = occlusion(flow_f, flow_b)
    np.save(os.path.join(out_path, 'occ_f.npy'), occ_f)
    np.save(os.path.join(out_path, 'occ_b.npy'), occ_b)
    return result



def integrate_fire(integrator, img, threshold, light_p, sigma: int = 2, reset_type = 0):
    #print(frame.shape)
    noise = np.random.randn(img.shape[0], img.shape[1]) * sigma if sigma != 0 else 0

    integrator += img * light_p + noise
    spike_img = (integrator >= threshold)
    if reset_type == 0:
        integrator -= spike_img * threshold
    else:
        integrator = integrator * (1 - spike_img)  #------way two

    return integrator, spike_img

def image_to_spike(imgs, out_path, height, width, num_frame, warm_up=0):
    num_frame, _, height, width = imgs.shape
    imgs = imgs.mean(1)
    num_frame -= warm_up
    integrator = np.zeros((height, width)) ######
    threshold = 255
    
    light_p = np.random.normal(loc=0.5,scale=0.3)
    while light_p<0.3 or light_p > 1.1:
        light_p = np.random.normal(loc=0.5,scale=0.3)
    # light_p = 0.5
    for i in range(warm_up):
        integrator, _ = integrate_fire(integrator, imgs[i], threshold, light_p, sigma=10)
    spk_imgs = np.zeros((num_frame, height, width), dtype=np.int64)
    for i in range(num_frame):
        integrator, spk_img = integrate_fire(integrator, imgs[warm_up+i], threshold, light_p, sigma=3)
        spk_imgs[i] = spk_img
    spk_imgs = spk_imgs.astype(np.int64)
    
    comp_spk_imgs = np.zeros((1,height,width), dtype=np.int64)
    for i in range(num_frame):
        comp_spk_imgs[0] += spk_imgs[i]<<i
    np.save(os.path.join(out_path,'spike.npy'), comp_spk_imgs)

def image_to_spike_online(imgs):
    _, _, height, width = imgs.shape
    imgs = imgs.mean(1)
    num_frame = 44
    warm_up = 5
    integrator = np.zeros((height, width))
    threshold = 255
    
    # light_p = np.random.normal(loc=0.5,scale=0.3)
    # while light_p<0.3 or light_p > 1.1:
    #     light_p = np.random.normal(loc=0.5,scale=0.3)
    light_p = 1
    for i in range(warm_up):
        integrator, _ = integrate_fire(integrator, imgs[i], threshold, light_p, sigma=10)
    spk_imgs = np.zeros((44, height, width), dtype=np.int64)
    for i in range(num_frame):
        integrator, spk_img = integrate_fire(integrator, imgs[warm_up+i], threshold, light_p, sigma=3)
        spk_imgs[i] = spk_img
    spk_imgs = spk_imgs.astype(np.int64)
    
    return spk_imgs
    
def generate_scenes(obj_root_path, 
                    bg_root_path,
                    out_f_path,
                    width,
                    height,
                    num_object,
                    num_sample,
                    num_flow=32,
                    warm_up=5,
                    padding=5,
                    to_gray=True):

    nvisii.initialize(headless=True, verbose=True)

    nvisii.enable_denoiser()
    obj_paths = []
    for obj_f_path in os.listdir(obj_root_path):
        for obj in os.listdir(os.path.join(obj_root_path, obj_f_path)):
            obj_paths.append(os.path.join(obj_root_path, obj_f_path, obj)) 

    bg_paths = []
    for bg_f_path in os.listdir(bg_root_path):
        for bg in os.listdir(os.path.join(bg_root_path, bg_f_path)):
            if bg == 'crop.png':
                continue
            bg_paths.append(os.path.join(bg_root_path, bg_f_path, bg))
    first = True
    for i in range(num_sample):
        if not os.path.exists(os.path.join(out_f_path, '{0:06d}'.format(i))):
            os.mkdir(os.path.join(out_f_path, '{0:06d}'.format(i)))
        else:
            continue
        imgs = generate_one_scene(np.random.choice(obj_paths, num_object),
                           np.random.choice(bg_paths),
                           width,
                           height,
                           os.path.join(out_f_path, '{0:06d}'.format(i)),
                           first,
                           num_flow,
                           warm_up,
                           padding,
                           to_gray=to_gray)
        
        num_frame = num_flow+2*padding
        image_to_spike(imgs, os.path.join(out_f_path, '{0:06d}'.format(i)), height, width, num_frame, warm_up)
        first = False

    nvisii.deinitialize()


if __name__ == '__main__':
    root_path = '/media/ywqqqqqq/YWQ/Dataset/VidarCity/Simulated/SpikeFlyingThings'
    obj_root_path = os.path.join(root_path, 'object_3d')
    bg_root_path = os.path.join(root_path, 'background')
    out_f_path = os.path.join(root_path, 'dualflow_new')
    merge_path = os.path.join(root_path, 'train')

    if not os.path.exists(obj_root_path) or not os.path.exists(bg_root_path):
        raise FileExistsError
    
    if not os.path.exists(out_f_path):
        os.mkdir(out_f_path)
    generate_scenes(obj_root_path=obj_root_path,
                    bg_root_path=bg_root_path,
                    out_f_path=out_f_path,
                    width=448,
                    height=256,
                    num_object=3,
                    num_sample=5000,
                    num_flow=33,
                    warm_up=2,
                    padding=5)
    
    # if not os.path.exists(merge_path):
    #     os.mkdir(merge_path)
    
    # flows_f = []
    # flows_b = []
    # spikes = []
    # frames = []
    # samples = os.listdir(out_f_path)
    # for sample in tqdm(samples):
    #     sample_path = os.path.join(out_f_path, sample)
    #     flow_f = np.load(os.path.join(sample_path, 'flow_f.npy'))
    #     flow_b = np.load(os.path.join(sample_path, 'flow_b.npy'))
    #     spike = np.load(os.path.join(sample_path, 'spike.npy'))
    #     frame = np.load(os.path.join(sample_path, 'frame.npy'))
    #     flow_f = flow_f.transpose(2,0,1)
    #     flow_b = flow_b.transpose(2,0,1)
    #     frame = frame[[5,36]]
    #     flows_f.append(flow_f)
    #     flows_b.append(flow_b)
    #     spikes.append(spike)
    #     frames.append(frame)
    
    # flows_f = np.array(flows_f)
    # flows_b = np.array(flows_b)
    # spikes = np.array(spikes)
    # frames = np.array(frames)
    # index = [i for i in range(len(flows_f))]
    # np.random.shuffle(index)
    # np.savez(os.path.join(merge_path, 'train.npz'), frame=frames[index[:4500]], spike=spikes[index[:4500]], flow_f=flows_f[index[:4500]], flow_b=flows_b[index[:4500]])
    # np.savez(os.path.join(merge_path, 'val.npz'), frame=frames[index[4500:]], spike=spikes[index[4500:]], flow_f=flows_f[index[4500:]], flow_b=flows_b[index[4500:]])
    # print()

    



